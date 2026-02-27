"""
Segment Selector — Selección inteligente de segmento para short_auto.

Analiza el contenido del video usando solo FFmpeg y el EnhancedFaceDetector
ya existente en el proyecto (sin librerías de pago ni APIs externas).

Estrategia de puntuación por ventana deslizante:
  - Presencia de caras   (peso 0.50): reutiliza EnhancedFaceDetector
  - Actividad de audio   (peso 0.35): ffmpeg astats (RMS por ventana)
  - Penalización escenas (peso 0.15): ffmpeg scdet (evitar cortes internos)

El video se divide en ventanas solapadas de `target_duration` segundos
cada `SAMPLE_STEP` segundos. La ventana con mayor score acumulado gana.

Fallback: si el análisis falla por cualquier razón, se usa el segmento
central (comportamiento original) para garantizar que siempre haya resultado.
"""

import subprocess
import json
import logging
import math
import re
import tempfile
import os
from typing import Tuple, List, Optional

from models.schemas import SHORT_MIN_DURATION_SECONDS


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Constantes de configuración del análisis
# -------------------------------------------------------------------

# Cada cuántos segundos se evalúa una nueva ventana candidata.
# Valor bajo = más candidatos, más precisión, más tiempo de análisis.
# Valor alto = menos candidatos, más rápido, menos precisión.
# 2s es un balance razonable para entrevistas y contenido variado.
SAMPLE_STEP: float = 1.0

# Cada cuántos segundos se samplea un frame para análisis de caras.
# No se analiza frame a frame para mantener la velocidad.
FACE_SAMPLE_EVERY: float = 1.5

# Umbral de cambio de escena (0.0 - 1.0). Valores más altos = menos
# sensible, solo detecta cortes muy abruptos. 0.35 es un buen equilibrio.
SCENE_CHANGE_THRESHOLD: float = 0.35

# Penalización aplicada por cada corte de escena interno en la ventana.
# Hace que se prefieran segmentos sin saltos abruptos de escena.
SCENE_CUT_PENALTY: float = 0.25

# Pesos del score compuesto.
# WEIGHT_FACES + WEIGHT_AUDIO + WEIGHT_MOTION = 1.0 (antes de penalización de escenas)
WEIGHT_FACES: float = 0.40
WEIGHT_AUDIO: float = 0.25
WEIGHT_MOTION: float = 0.25  # Movimiento en frame — valioso para contenido variado y acción
WEIGHT_SCENE: float = 0.10   # Este peso se aplica como reducción, no suma


# -------------------------------------------------------------------
# Clase principal
# -------------------------------------------------------------------

class SegmentSelector:
    """
    Selecciona el mejor segmento de un video para convertir en short.

    Todos los métodos son estáticos para uso directo desde las estrategias
    sin necesidad de instanciar la clase.
    """

    # ----------------------------------------------------------------
    # Método principal — punto de entrada para ShortAutoStrategy
    # ----------------------------------------------------------------

    @staticmethod
    def select_best_segment(
        video_path: str,
        total_duration: float,
        target_duration: int,
        detector=None,
        config=None,
    ) -> Tuple[float, int, str]:
        """
        Selecciona el mejor segmento del video usando análisis de contenido.

        Analiza múltiples ventanas candidatas y elige la que tiene mayor
        score combinando presencia de caras, actividad de audio y
        continuidad de escena.

        Args:
            video_path:      Path local del video.
            total_duration:  Duración total del video en segundos.
            target_duration: Duración deseada del short en segundos.
            detector:        Instancia de EnhancedFaceDetector (opcional).
                             Si es None, el criterio de caras se omite.
            config:          Módulo config_enhanced (necesario para detector).

        Returns:
            Tupla (start_time, actual_duration, strategy_used) donde:
              - start_time:     Segundo de inicio del segmento ganador.
              - actual_duration: Duración efectiva del segmento.
              - strategy_used:  Descripción de la estrategia aplicada
                                (para métricas y logs).
        """
        # Caso borde: video más corto que el target → usar todo
        if total_duration <= target_duration:
            actual_duration = max(SHORT_MIN_DURATION_SECONDS, int(total_duration))
            logger.info(
                "Video corto (%.2fs <= %ds): usando video completo",
                total_duration, target_duration
            )
            return 0.0, actual_duration, "full_video"

        try:
            return SegmentSelector._analyze_and_select(
                video_path=video_path,
                total_duration=total_duration,
                target_duration=target_duration,
                detector=detector,
                config=config,
            )
        except Exception as e:
            # Fallback garantizado: si el análisis falla por cualquier razón
            # (video corrupto, ffprobe no disponible, etc.), usar segmento central.
            logger.warning(
                "Análisis de contenido falló (%s). Usando segmento central como fallback.",
                str(e)
            )
            start_time, actual_duration = SegmentSelector._central_segment(
                total_duration, target_duration
            )
            return start_time, actual_duration, "central_fallback"

    @staticmethod
    def _analyze_and_select(
        video_path: str,
        total_duration: float,
        target_duration: int,
        detector=None,
        config=None,
    ) -> Tuple[float, int, str]:
        """
        Ejecuta el análisis completo y selecciona la ventana ganadora.
        """
        logger.info(
            "Iniciando análisis de contenido | duration=%.2fs | target=%ds",
            total_duration, target_duration
        )

        # 1. Generar candidatos (ventanas solapadas)
        candidates = SegmentSelector._generate_candidates(total_duration, target_duration)
        logger.info("Candidatos generados: %d ventanas", len(candidates))

        if not candidates:
            raise ValueError("No se pudieron generar ventanas candidatas")

        # 2. Analizar audio (rápido — una sola pasada de FFmpeg)
        audio_scores = SegmentSelector._analyze_audio(video_path, total_duration)
        logger.info("Análisis de audio completado")

        # 3. Detectar cortes de escena (una sola pasada de FFmpeg)
        scene_cuts = SegmentSelector._detect_scene_cuts(video_path)
        logger.info("Cortes de escena detectados: %d", len(scene_cuts))

        # 4. Analizar movimiento (una sola pasada de FFmpeg a 1fps)
        motion_scores = SegmentSelector._analyze_motion(video_path, total_duration)
        logger.info("Análisis de movimiento completado | frames=%d", len(motion_scores))

        # 5. Analizar presencia de caras (sampleo por frames)
        face_scores: Optional[dict] = None
        if detector is not None and config is not None:
            try:
                face_scores = SegmentSelector._analyze_faces(
                    video_path, total_duration, detector, config
                )
                logger.info("Análisis de caras completado | frames=%d", len(face_scores) if face_scores else 0)
            except Exception as e:
                logger.warning(
                    "Análisis de caras falló (%s). Continuando sin este criterio.",
                    str(e)
                )

        # 5. Puntuar cada candidato y elegir el mejor
        best_start, best_score, scores_detail = SegmentSelector._score_candidates(
            candidates=candidates,
            target_duration=target_duration,
            audio_scores=audio_scores,
            motion_scores=motion_scores,
            scene_cuts=scene_cuts,
            face_scores=face_scores,
        )

        has_motion = len(motion_scores) > 0
        if face_scores is not None and has_motion:
            strategy_used = "smart_auto"
        elif face_scores is not None:
            strategy_used = "smart_auto_no_motion"
        elif has_motion:
            strategy_used = "smart_auto_no_faces"
        else:
            strategy_used = "smart_auto_audio_only"

        logger.info(
            "Segmento seleccionado | start=%.2fs | score=%.3f | strategy=%s",
            best_start, best_score, strategy_used
        )

        return best_start, target_duration, strategy_used

    # ----------------------------------------------------------------
    # Generación de candidatos
    # ----------------------------------------------------------------

    @staticmethod
    def _generate_candidates(total_duration: float, target_duration: int) -> List[float]:
        """
        Genera los puntos de inicio de las ventanas candidatas.

        Ventanas solapadas que cubren todo el video con paso SAMPLE_STEP.
        La última ventana siempre termina exactamente al final del video.
        """
        candidates = []
        max_start = total_duration - target_duration
        current = 0.0

        while current <= max_start:
            candidates.append(round(current, 3))
            current += SAMPLE_STEP

        # Asegurar que el final del video es siempre un candidato
        if not candidates or candidates[-1] < max_start - 0.1:
            candidates.append(round(max_start, 3))

        return candidates

    # ----------------------------------------------------------------
    # Análisis de audio
    # ----------------------------------------------------------------

    @staticmethod
    def _analyze_audio(video_path: str, total_duration: float) -> dict:
        """
        Mide la actividad de audio por ventana temporal usando ffmpeg astats.

        Divide el audio en segmentos de SAMPLE_STEP segundos y mide el
        RMS (Root Mean Square) de cada uno. RMS alto = más actividad sonora.

        Retorna un dict {tiempo_inicio: score_normalizado (0.0-1.0)}.
        Si el video no tiene audio, retorna dict vacío (el criterio se ignora).
        """
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",                          # Ignorar video
            "-af", f"asetnsamples=n=1024,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parsear salida: "frame:N pts:X pts_time:T lavfi.astats.Overall.RMS_level=V"
            rms_by_time = {}
            pattern = re.compile(
                r"pts_time:(\d+\.?\d*)\s+lavfi\.astats\.Overall\.RMS_level=(-?\d+\.?\d*)"
            )

            for match in pattern.finditer(result.stderr + result.stdout):
                t = float(match.group(1))
                rms_db = float(match.group(2))
                # Convertir dB a valor lineal (silencio = -inf dB → 0.0)
                if rms_db > -100:
                    rms_linear = 10 ** (rms_db / 20)
                else:
                    rms_linear = 0.0
                rms_by_time[t] = rms_linear

            if not rms_by_time:
                logger.debug("No se detectó audio en el video")
                return {}

            # Agregar por ventanas de SAMPLE_STEP segundos
            window_scores = {}
            max_t = max(rms_by_time.keys()) if rms_by_time else 0
            t = 0.0
            while t <= max_t:
                window_end = t + SAMPLE_STEP
                values = [v for ts, v in rms_by_time.items() if t <= ts < window_end]
                window_scores[t] = sum(values) / len(values) if values else 0.0
                t += SAMPLE_STEP

            # Normalizar a 0.0-1.0
            if window_scores:
                max_val = max(window_scores.values())
                if max_val > 0:
                    window_scores = {k: v / max_val for k, v in window_scores.items()}

            return window_scores

        except subprocess.TimeoutExpired:
            logger.warning("Timeout en análisis de audio")
            return {}
        except Exception as e:
            logger.warning("Error en análisis de audio: %s", str(e))
            return {}

    @staticmethod
    def _get_audio_score_for_window(
        audio_scores: dict,
        start_time: float,
        duration: int
    ) -> float:
        """
        Calcula el score de audio promedio para una ventana de tiempo.
        Interpola entre los puntos de medición disponibles.
        """
        if not audio_scores:
            return 0.5  # Sin datos de audio: score neutral

        end_time = start_time + duration
        relevant = [
            score for t, score in audio_scores.items()
            if start_time <= t < end_time
        ]

        if not relevant:
            # Buscar el punto más cercano
            closest = min(audio_scores.keys(), key=lambda t: abs(t - start_time))
            return audio_scores[closest]

        return sum(relevant) / len(relevant)

    # ----------------------------------------------------------------
    # Detección de cortes de escena
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_scene_cuts(video_path: str) -> List[float]:
        """
        Detecta cambios abruptos de escena usando ffmpeg scdet.

        Retorna lista de tiempos (en segundos) donde ocurren cortes.
        Un corte dentro de una ventana candidata penaliza su score.
        """
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"scdet=threshold={SCENE_CHANGE_THRESHOLD}",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parsear "lavfi.scd.time=X.XXX"
            cuts = []
            pattern = re.compile(r"lavfi\.scd\.time=(\d+\.?\d*)")
            for match in pattern.finditer(result.stderr + result.stdout):
                cuts.append(float(match.group(1)))

            logger.debug("Cortes de escena: %s", cuts)
            return cuts

        except subprocess.TimeoutExpired:
            logger.warning("Timeout en detección de escenas")
            return []
        except Exception as e:
            logger.warning("Error en detección de escenas: %s", str(e))
            return []

    @staticmethod
    def _count_scene_cuts_in_window(
        scene_cuts: List[float],
        start_time: float,
        duration: int
    ) -> int:
        """
        Cuenta cuántos cortes de escena hay dentro de una ventana.
        Los cortes justo al inicio o final no penalizan (son el punto de corte natural).
        """
        margin = 0.5  # segundos de margen en los bordes
        end_time = start_time + duration

        return sum(
            1 for cut in scene_cuts
            if (start_time + margin) < cut < (end_time - margin)
        )


    # ----------------------------------------------------------------
    # Análisis de movimiento
    # ----------------------------------------------------------------

    @staticmethod
    def _analyze_motion(video_path: str, total_duration: float) -> dict:
        """
        Mide la actividad de movimiento en el video usando ffmpeg.

        Usa el filtro `mestimate` para calcular vectores de movimiento
        entre frames consecutivos. El SAD (Sum of Absolute Differences)
        promedio de cada frame indica cuánto cambió el contenido visual
        respecto al frame anterior.

        Esta métrica es especialmente útil para:
          - Contenido de deportes y acción (movimiento alto = acción)
          - Distinguir segmentos estáticos (planos fijos, silencios visuales)
            de segmentos dinámicos

        Se samplea 1 frame por segundo para mantener la velocidad de análisis.

        Retorna dict {tiempo: score_normalizado (0.0-1.0)}.
        """
        # Samplear 1 fps es suficiente para medir actividad de movimiento
        # sin decodificar el video completo
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "mestimate=method=ds:search_param=7,metadata=print:key=lavfi.me.sad.avg:file=-",
            "-r", "1",          # 1 frame por segundo
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )

            output = result.stderr + result.stdout

            # Parsear "frame:N pts_time:T lavfi.me.sad.avg=V"
            sad_by_time = {}
            pattern = re.compile(
                r"pts_time:(\d+\.?\d*)\s+lavfi\.me\.sad\.avg=(\d+\.?\d*)"
            )

            for match in pattern.finditer(output):
                t = float(match.group(1))
                sad = float(match.group(2))
                sad_by_time[t] = sad

            if not sad_by_time:
                logger.debug("No se obtuvieron datos de movimiento")
                return {}

            # Normalizar a 0.0-1.0
            max_sad = max(sad_by_time.values())
            if max_sad > 0:
                normalized = {t: v / max_sad for t, v in sad_by_time.items()}
            else:
                normalized = {t: 0.0 for t in sad_by_time}

            # Suavizado con media móvil de 3 puntos para evitar picos aislados
            times = sorted(normalized.keys())
            smoothed = {}
            for i, t in enumerate(times):
                neighbors = [normalized[times[j]] for j in range(
                    max(0, i - 1), min(len(times), i + 2)
                )]
                smoothed[t] = sum(neighbors) / len(neighbors)

            logger.debug(
                "Movimiento analizado | frames=%d | max_sad=%.1f",
                len(smoothed), max_sad
            )

            return smoothed

        except subprocess.TimeoutExpired:
            logger.warning("Timeout en análisis de movimiento")
            return {}
        except Exception as e:
            logger.warning("Error en análisis de movimiento: %s", str(e))
            return {}

    @staticmethod
    def _get_motion_score_for_window(
        motion_scores: dict,
        start_time: float,
        duration: int
    ) -> float:
        """
        Calcula el score de movimiento promedio para una ventana de tiempo.
        """
        if not motion_scores:
            return 0.5  # Sin datos: score neutral

        end_time = start_time + duration
        relevant = [
            score for t, score in motion_scores.items()
            if start_time <= t < end_time
        ]

        if not relevant:
            closest = min(motion_scores.keys(), key=lambda t: abs(t - start_time))
            return motion_scores[closest]

        return sum(relevant) / len(relevant)

    # ----------------------------------------------------------------
    # Análisis de caras
    # ----------------------------------------------------------------

    @staticmethod
    def _analyze_faces(
        video_path: str,
        total_duration: float,
        detector,
        config,
    ) -> dict:
        """
        Mide la presencia de caras en el video samplesando frames.

        Extrae un frame cada FACE_SAMPLE_EVERY segundos usando ffmpeg,
        lo pasa por el EnhancedFaceDetector, y registra si había cara
        detectada. Retorna un dict {tiempo: score_cara (0.0-1.0)}.

        El score de cada tiempo es 1.0 si se detectó cara, 0.0 si no.
        Luego se suaviza con una media móvil para evitar ruido de frames.
        """
        face_scores = {}
        sample_times = []

        t = 0.0
        while t <= total_duration:
            sample_times.append(round(t, 2))
            t += FACE_SAMPLE_EVERY

        if not sample_times:
            return {}

        with tempfile.TemporaryDirectory(prefix="face_analysis_") as tmp_dir:
            for sample_t in sample_times:
                frame_path = os.path.join(tmp_dir, f"frame_{sample_t:.2f}.jpg")

                # Extraer frame en tiempo sample_t
                cmd = [
                    "ffmpeg",
                    "-ss", str(sample_t),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "3",          # Calidad JPEG suficiente para detección
                    "-f", "image2",
                    frame_path,
                    "-y",
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10
                )

                if result.returncode != 0 or not os.path.exists(frame_path):
                    face_scores[sample_t] = 0.0
                    continue

                try:
                    import cv2
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        face_scores[sample_t] = 0.0
                        continue

                    faces = detector.detect_faces(frame)
                    # Score = 1.0 si hay al menos una cara, 0.0 si no
                    # Bonus leve si hay exactamente 1 cara (entrevistas: foco en persona)
                    if len(faces) == 0:
                        face_scores[sample_t] = 0.0
                    elif len(faces) == 1:
                        face_scores[sample_t] = 1.0
                    else:
                        # Múltiples caras: también bueno, pero ligeramente menos
                        # que un plano centrado en una persona
                        face_scores[sample_t] = 0.85

                except Exception as e:
                    logger.debug("Error detectando cara en t=%.2fs: %s", sample_t, str(e))
                    face_scores[sample_t] = 0.0

        # Suavizado con media móvil de 3 puntos para reducir ruido de frames
        smoothed = {}
        times = sorted(face_scores.keys())
        for i, t in enumerate(times):
            neighbors = [face_scores[times[j]] for j in range(
                max(0, i - 1), min(len(times), i + 2)
            )]
            smoothed[t] = sum(neighbors) / len(neighbors)

        return smoothed

    @staticmethod
    def _get_face_score_for_window(
        face_scores: Optional[dict],
        start_time: float,
        duration: int
    ) -> float:
        """
        Calcula el score de caras promedio para una ventana de tiempo.
        """
        if not face_scores:
            return 0.5  # Sin datos: score neutral

        end_time = start_time + duration
        relevant = [
            score for t, score in face_scores.items()
            if start_time <= t < end_time
        ]

        if not relevant:
            closest = min(face_scores.keys(), key=lambda t: abs(t - start_time))
            return face_scores[closest]

        return sum(relevant) / len(relevant)

    # ----------------------------------------------------------------
    # Scoring y selección final
    # ----------------------------------------------------------------

    @staticmethod
    def _score_candidates(
        candidates: List[float],
        target_duration: int,
        audio_scores: dict,
        motion_scores: dict,
        scene_cuts: List[float],
        face_scores: Optional[dict],
    ) -> Tuple[float, float, list]:
        """
        Puntúa cada ventana candidata y retorna la ganadora.

        Score compuesto (pesos configurables en las constantes del módulo):
          base = (WEIGHT_FACES × face_score)
               + (WEIGHT_AUDIO × audio_score)
               + (WEIGHT_MOTION × motion_score)
          penalización = SCENE_CUT_PENALTY × n_cortes_internos
          score_final = max(0.0, base - penalización)

        Si algún criterio no tiene datos, su peso se redistribuye
        proporcionalmente entre los criterios disponibles.

        Returns:
            (mejor_start, mejor_score, detalle_scores)
        """
        has_faces = face_scores is not None and len(face_scores) > 0
        has_audio = len(audio_scores) > 0
        has_motion = len(motion_scores) > 0

        # Calcular pesos efectivos según qué criterios tienen datos.
        # Si un criterio no tiene datos, su peso se redistribuye
        # proporcionalmente entre los disponibles.
        available = []
        if has_faces:
            available.append(('faces', WEIGHT_FACES))
        if has_audio:
            available.append(('audio', WEIGHT_AUDIO))
        if has_motion:
            available.append(('motion', WEIGHT_MOTION))

        total_weight = sum(w for _, w in available) or 1.0
        effective_weights = {name: w / total_weight for name, w in available}

        scores_detail = []
        best_start = candidates[0]
        best_score = -1.0

        for start in candidates:
            face_s = SegmentSelector._get_face_score_for_window(
                face_scores, start, target_duration
            ) if has_faces else None

            audio_s = SegmentSelector._get_audio_score_for_window(
                audio_scores, start, target_duration
            ) if has_audio else None

            motion_s = SegmentSelector._get_motion_score_for_window(
                motion_scores, start, target_duration
            ) if has_motion else None

            n_cuts = SegmentSelector._count_scene_cuts_in_window(
                scene_cuts, start, target_duration
            )

            # Score base: suma ponderada de los criterios disponibles
            if available:
                base = 0.0
                if has_faces:
                    base += effective_weights['faces'] * face_s
                if has_audio:
                    base += effective_weights['audio'] * audio_s
                if has_motion:
                    base += effective_weights['motion'] * motion_s
            else:
                # Sin ningún dato de contenido: score neutral
                # La penalización de escenas sigue aplicando
                base = 0.5

            penalization = SCENE_CUT_PENALTY * n_cuts
            final_score = max(0.0, base - penalization)

            scores_detail.append({
                'start': start,
                'face_score': face_s,
                'audio_score': audio_s,
                'motion_score': motion_s,
                'scene_cuts': n_cuts,
                'final_score': final_score,
            })

            if final_score > best_score:
                best_score = final_score
                best_start = start

        # Log de los top 3 candidatos para diagnóstico
        top3 = sorted(scores_detail, key=lambda x: x['final_score'], reverse=True)[:3]
        for rank, entry in enumerate(top3, 1):
            logger.debug(
                "Top %d | start=%.2fs | face=%.2f | audio=%.2f | motion=%.2f | cuts=%d | score=%.3f",
                rank,
                entry['start'],
                entry['face_score'] or 0,
                entry['audio_score'] or 0,
                entry['motion_score'] or 0,
                entry['scene_cuts'],
                entry['final_score'],
            )

        return best_start, best_score, scores_detail

    # ----------------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------------

    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """
        Obtiene la duración real del video usando ffprobe.

        Args:
            video_path: Path del video local.

        Returns:
            Duración en segundos (float).

        Raises:
            RuntimeError: Si ffprobe no puede leer el archivo.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            info = json.loads(result.stdout)
            duration = float(info["format"]["duration"])

            logger.debug(
                "Duración obtenida | path=%s | duration=%.2fs",
                video_path,
                duration
            )

            return duration

        except subprocess.CalledProcessError as e:
            logger.error("ffprobe falló al leer '%s': %s", video_path, e.stderr)
            raise RuntimeError(
                f"No se pudo obtener la duración del video: {video_path}. "
                "Verifica que el archivo sea un video válido."
            )
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error("Error parseando metadata de '%s': %s", video_path, str(e))
            raise RuntimeError(
                f"Metadata del video inválida o incompleta: {video_path}"
            )

    @staticmethod
    def _central_segment(
        total_duration: float,
        target_duration: int
    ) -> Tuple[float, int]:
        """
        Segmento central — usado como fallback si el análisis falla.
        Mismo comportamiento que la versión original del selector.
        """
        center = total_duration / 2.0
        start_time = max(0.0, center - (target_duration / 2.0))

        if start_time + target_duration > total_duration:
            start_time = total_duration - target_duration

        return round(start_time, 3), target_duration
