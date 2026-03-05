"""
Procesador de Video Ultra-Mejorado
Con análisis de calidad en tiempo real y optimizaciones avanzadas
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


def process_video_enhanced(input_path, config, detector, stabilizer, use_multipass=True, encoder='libx264'):
    """
    Procesar video con sistema mejorado y análisis de calidad

    Args:
        encoder: Encoder de FFmpeg a usar (libx264, h264_nvenc, h264_qsv, etc.)
    """

    verbose = config.PERFORMANCE_SETTINGS.get('verbose', True)

    if verbose:
        logger.info("=" * 60)
        logger.info("PROCESAMIENTO ULTRA-MEJORADO")
        logger.info("=" * 60)
        logger.info(f"Entrada: {input_path}")
        logger.info(f"Modo: {config.CONVERSION_MODE['mode']}")
        logger.info(f"Multi-paso: {'Sí' if use_multipass else 'No'}")
        logger.info(f"Encoder: {encoder}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir el video: {input_path}")
        raise ValueError(f"No se pudo abrir el video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    if verbose:
        logger.info(f"Resolución: {width}x{height}")
        logger.info(f"Duración: {duration:.2f}s ({total_frames} frames @ {fps:.2f} fps)")

    # ============================================================
    # VALIDACIÓN: Detectar si el video ya es vertical
    # ============================================================
    aspect_ratio = width / height
    target_aspect = 9 / 16  # 0.5625
    
    # Si el video ya es vertical (tolerancia de ±5%)
    if 0.53 <= aspect_ratio <= 0.59:  # Rango 9:16 ± 5%
        logger.info(
            "Video ya es vertical | %dx%d | ratio=%.3f | target=%.3f",
            width, height, aspect_ratio, target_aspect
        )
        
        # Si las dimensiones coinciden exactamente con el target, devolver video original
        crop_width = config.CROP_SETTINGS['width']
        crop_height = config.CROP_SETTINGS['height']
        
        if width == crop_width and height == crop_height:
            logger.info(
                "Dimensiones exactas (%dx%d). Retornando video original sin procesamiento.",
                width, height
            )
            cap.release()
            
            # Retornar el video original con métricas básicas
            metrics = {
                'total_frames': total_frames,
                'frames_processed': 0,
                'keyframes': 0,
                'analysis_time': 0,
                'total_time': 0,
                'overall_quality': 1.0,
                'skipped_reason': 'already_vertical_exact_dimensions'
            }
            
            return input_path, metrics
        
        # Si es vertical pero dimensiones diferentes, hacer re-encode simple sin crop
        logger.info(
            "Video vertical pero dimensiones diferentes (%dx%d vs %dx%d). Re-encoding sin crop.",
            width, height, crop_width, crop_height
        )
        cap.release()
        
        return process_already_vertical_video(input_path, config, encoder)
    
    # Si el video NO es vertical pero la resolución de salida es igual o mayor
    # que la de entrada, no se puede hacer crop
    crop_width = config.CROP_SETTINGS['width']
    crop_height = config.CROP_SETTINGS['height']
    
    if width <= crop_width:
        logger.warning(
            "Video más estrecho que el crop deseado | input=%dx%d | crop=%dx%d",
            width, height, crop_width, crop_height
        )
        logger.info("Procesando en modo 'full' (letterbox) en lugar de crop")
        cap.release()
        return process_full_mode_simple(input_path, config, encoder=encoder)
    
    # Video es horizontal, continuar con procesamiento normal
    logger.info(
        "Video horizontal detectado | %dx%d | ratio=%.3f | Aplicando smart crop",
        width, height, aspect_ratio
    )

    mode = config.CONVERSION_MODE['mode']

    if mode == 'full':
        cap.release()
        return process_full_mode_simple(input_path, config, encoder=encoder)

    crop_width = config.CROP_SETTINGS['width']
    crop_height = config.CROP_SETTINGS['height']
    sample_rate = config.PERFORMANCE_SETTINGS['sample_rate']

    if verbose:
        logger.info("Analizando rostros...")
        logger.info(f"Sample rate: 1 de cada {sample_rate} frames")

    positions = []
    quality_metrics = []
    frame_number = 0
    frames_processed = 0
    start_time = time.time()

    tracking_loss_events = []
    low_quality_frames = []

    if use_multipass:
        from app.stabilization_enhanced import MultiPassStabilizer
        multipass = MultiPassStabilizer(config)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_number / fps if fps > 0 else 0

        if frame_number % sample_rate == 0:

            faces = detector.detect(frame)
            primary_face = detector.get_primary_face(faces)

            if primary_face:
                crop_x, crop_y = calculate_optimal_composition(
                    primary_face,
                    (width, height),
                    (crop_width, crop_height),
                    config
                )

                quality = primary_face.get('quality')

                if use_multipass:
                    multipass.add_position(timestamp, crop_x, quality)
                else:
                    if hasattr(stabilizer, 'stabilize'):
                        stabilized_x = stabilizer.stabilize(crop_x, quality)
                    else:
                        stabilized_x = stabilizer.stabilize(crop_x)

                    positions.append((timestamp, stabilized_x))

                if quality:
                    quality_metrics.append({
                        'timestamp': timestamp,
                        'confidence': quality.confidence,
                        'stability': quality.stability,
                        'is_reliable': quality.is_reliable
                    })

                    if not quality.is_reliable:
                        low_quality_frames.append(frame_number)

                        if quality.lost_frames > 3:
                            tracking_loss_events.append({
                                'frame': frame_number,
                                'timestamp': timestamp,
                                'lost_frames': quality.lost_frames
                            })
            else:
                if len(positions) > 0:
                    last_pos = positions[-1][1]
                    if use_multipass:
                        multipass.add_position(timestamp, last_pos, None)
                    else:
                        if hasattr(stabilizer, 'stabilize'):
                            stabilized_x = stabilizer.stabilize(None)
                        else:
                            stabilized_x = last_pos

                        positions.append((timestamp, stabilized_x if stabilized_x else last_pos))
                else:
                    center_x = (width - crop_width) // 2
                    if use_multipass:
                        multipass.add_position(timestamp, center_x, None)
                    else:
                        positions.append((timestamp, center_x))

                tracking_loss_events.append({
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'lost_frames': -1
                })

            frames_processed += 1

            if verbose and frames_processed % 100 == 0:
                progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                elapsed = time.time() - start_time
                fps_analysis = frames_processed / elapsed if elapsed > 0 else 0

                logger.info(
                    f"Progreso: {progress:.1f}% "
                    f"({frames_processed} frames, {fps_analysis:.1f} fps)"
                )

        frame_number += 1

    cap.release()

    if use_multipass and positions:
        if verbose:
            logger.info("Aplicando estabilización multi-paso...")
        positions = multipass.process()

    analysis_time = time.time() - start_time

    if verbose:
        logger.info(f"Análisis completado en {analysis_time:.2f}s")
        logger.info(f"Frames procesados: {frames_processed}")
        logger.info(f"Keyframes generados: {len(positions)}")

        if quality_metrics:
            avg_confidence = np.mean([m['confidence'] for m in quality_metrics])
            avg_stability = np.mean([m['stability'] for m in quality_metrics])
            reliable_frames = sum(1 for m in quality_metrics if m['is_reliable'])
            reliability_rate = (reliable_frames / len(quality_metrics)) * 100

            logger.info("Métricas de Tracking:")
            logger.info(f"Confianza promedio: {avg_confidence * 100:.1f}%")
            logger.info(f"Estabilidad promedio: {avg_stability * 100:.1f}%")
            logger.info(f"Frames confiables: {reliability_rate:.1f}%")

            if tracking_loss_events:
                logger.warning(f"Eventos de pérdida de tracking: {len(tracking_loss_events)}")
        else:
            # No quality metrics means no faces detected at all
            logger.warning("No se detectaron métricas de calidad")

        stats = detector.get_tracking_stats()
        logger.info("Estadísticas del Detector:")
        logger.info(f"Trackers activos: {stats['active_trackers']}")
        logger.info(f"Trackers confiables: {stats['reliable_trackers']}")

        if stats['fallback_activations'] > 0:
            logger.warning(f"Activaciones de fallback: {stats['fallback_activations']}")
    
    # ============================================================
    # VALIDACIÓN: Verificar si se detectaron rostros confiables
    # ============================================================
    reliability_rate = 0.0
    if quality_metrics:
        reliable_frames = sum(1 for m in quality_metrics if m['is_reliable'])
        reliability_rate = reliable_frames / len(quality_metrics)
    
    # Si NO se detectaron rostros confiables (< 10% de frames)
    if reliability_rate < 0.10:
        logger.warning(
            "⚠️  BAJO NIVEL DE DETECCIÓN DE ROSTROS | reliability=%.1f%%",
            reliability_rate * 100
        )
        logger.warning(
            "El video tiene muy pocos rostros detectables. "
            "Se aplicará MODO FULL (letterbox) en lugar de smart crop."
        )
        logger.info("Cambiando automáticamente a modo 'full' para mejor resultado")
        
        cap.release()
        
        # Procesar en modo full (letterbox con fondo)
        return process_full_mode_simple(input_path, config, encoder=encoder)

    if verbose:
        logger.info("Generando video final...")

    input_name = Path(input_path).stem
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    output_filename = f"{input_name}_vertical_{config.CONVERSION_MODE['mode']}_{timestamp_str}.mp4"
    output_path = os.path.join(temp_dir, output_filename)

    from app.ffmpeg_ultra import crop_video_ultra

    success = crop_video_ultra(input_path, output_path, positions, config, encoder=encoder)

    if not success:
        logger.error("Error en el encoding del video")
        raise RuntimeError("Error en el encoding del video")

    total_time = time.time() - start_time

    metrics = {
        'total_frames': total_frames,
        'frames_processed': frames_processed,
        'keyframes': len(positions),
        'analysis_time': analysis_time,
        'total_time': total_time,
        'tracking_loss_events': len(tracking_loss_events),
        'quality_metrics': quality_metrics,
        'overall_quality': 1.0,
        'reliability_rate': 0.0,
    }

    if quality_metrics:
        avg_confidence = np.mean([m['confidence'] for m in quality_metrics])
        avg_stability = np.mean([m['stability'] for m in quality_metrics])
        reliable_frames = sum(1 for m in quality_metrics if m['is_reliable'])
        reliability_rate = (reliable_frames / len(quality_metrics))

        metrics['overall_quality'] = (
            avg_confidence * 0.4 +
            avg_stability * 0.3 +
            reliability_rate * 0.3
        )
        metrics['reliability_rate'] = reliability_rate

    if verbose:
        logger.info("=" * 60)
        logger.info("PROCESAMIENTO COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"Salida: {output_path}")
        logger.info(f"Tiempo total: {total_time:.2f}s")
        logger.info(f"Calidad general: {metrics['overall_quality'] * 100:.1f}%")

        if metrics['overall_quality'] < 0.75:
            logger.warning("Calidad general baja detectada. Revisión recomendada.")

    return output_path, metrics


def process_full_mode_simple(input_path, config, encoder='libx264'):
    """
    Procesa video en modo 'full' (letterbox con fondo).
    
    IMPORTANTE: Si el video ya es vertical (1080x1920), NO hacer crop.
    Solo hacer re-encode si es necesario o retornar el original.
    """
    import subprocess
    
    # Obtener dimensiones del video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # DEBUG: Imprimir dimensiones
    logger.info("=" * 60)
    logger.info("process_full_mode_simple INICIADO")
    logger.info(f"Dimensiones del video: {width}x{height}")
    logger.info("=" * 60)
    
    crop_width = config.CROP_SETTINGS['width']
    crop_height = config.CROP_SETTINGS['height']
    
    logger.info(f"Dimensiones target: {crop_width}x{crop_height}")
    
    # ══════════════════════════════════════════════════════════
    # CASO 1: Video YA es vertical con dimensiones exactas
    # ══════════════════════════════════════════════════════════
    if width == crop_width and height == crop_height:
        logger.info(
            "CASO 1: Video ya tiene dimensiones exactas (%dx%d) en modo full. "
            "Retornando original sin procesamiento.",
            width, height
        )
        
        metrics = {
            'overall_quality': 1.0,
            'mode': 'full',
            'skipped_reason': 'already_exact_dimensions'
        }
        
        return input_path, metrics
    
    # ══════════════════════════════════════════════════════════
    # CASO 2: Video es vertical pero dimensiones diferentes
    # Hacer re-scale + letterbox si es necesario
    # ══════════════════════════════════════════════════════════
    aspect_ratio = width / height
    target_aspect = 9 / 16  # 0.5625
    
    logger.info(f"Aspect ratio calculado: {aspect_ratio:.4f} (target: {target_aspect:.4f})")
    logger.info(f"¿Es vertical? (0.53 <= {aspect_ratio:.4f} <= 0.59): {0.53 <= aspect_ratio <= 0.59}")
    
    if 0.53 <= aspect_ratio <= 0.59:  # Video ya es vertical
        logger.info(
            "CASO 2: Video vertical con dimensiones no estándar (%dx%d). "
            "Re-escalando a %dx%d.",
            width, height, crop_width, crop_height
        )
        
        input_name = Path(input_path).stem
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        temp_dir = os.path.join(project_root, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        output_filename = f"{input_name}_vertical_rescaled_{timestamp_str}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Obtener settings de encoding
        quality_preset = config.ENCODING_SETTINGS['quality_preset']
        settings = config.ENCODING_SETTINGS['presets'][quality_preset]
        
        # Comando FFmpeg para re-scale sin crop
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vf", f"scale={crop_width}:{crop_height}:force_original_aspect_ratio=decrease,"
                   f"pad={crop_width}:{crop_height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", encoder,
            "-preset", settings["preset"],
            "-crf", str(settings["crf"]),
        ]
        
        # Solo agregar profile para software encoders
        if encoder == 'libx264':
            cmd.extend(["-profile:v", settings.get("profile", "high")])
        
        cmd.extend([
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ])
        
        logger.info(f"Re-escalando video vertical a {crop_width}x{crop_height}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("Video re-escalado exitosamente")
            
            metrics = {
                'overall_quality': 1.0,
                'mode': 'full_rescale'
            }
            
            return output_path, metrics
            
        except subprocess.CalledProcessError as e:
            logger.error("Error en FFmpeg durante re-scale")
            logger.error(f"STDERR: {e.stderr[-500:]}")
            raise RuntimeError("Error al re-escalar el video vertical")
    
    # ══════════════════════════════════════════════════════════
    # CASO 3: Video es horizontal - usar crop_video_ultra normal
    # ══════════════════════════════════════════════════════════
    logger.info(
        "CASO 3: Video horizontal (%dx%d). Procesando en modo full con letterbox.",
        width, height
    )
    logger.info(f"Aspect ratio: {aspect_ratio:.4f} (NO está en rango 0.53-0.59)")
    logger.info("=" * 60)
    
    from app.ffmpeg_ultra import crop_video_ultra

    input_name = Path(input_path).stem
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    output_filename = f"{input_name}_vertical_full_{timestamp_str}.mp4"
    output_path = os.path.join(temp_dir, output_filename)

    # ═══════════════════════════════════════════════════════════════
    # FIX CRÍTICO: Cambiar config.CONVERSION_MODE a 'full'
    # Si no, crop_video_ultra intenta hacer crop aunque positions=[]
    # ═══════════════════════════════════════════════════════════════
    original_mode = config.CONVERSION_MODE.get('mode')
    config.CONVERSION_MODE['mode'] = 'full'
    
    logger.info("Modo forzado a 'full' para evitar crop en video horizontal")
    
    try:
        success = crop_video_ultra(input_path, output_path, [], config, encoder=encoder)

        if not success:
            logger.error("Error en el encoding del video")
            raise RuntimeError("Error en el encoding del video")

        metrics = {
            'overall_quality': 1.0,
            'mode': 'full'
        }

        return output_path, metrics
        
    finally:
        # Restaurar modo original
        config.CONVERSION_MODE['mode'] = original_mode
        logger.info(f"Modo restaurado a '{original_mode}'")


def process_already_vertical_video(input_path, config, encoder='libx264'):
    """
    Procesa un video que ya es vertical (9:16) pero con dimensiones diferentes
    a las del target. Hace un re-scale sin crop.
    
    Args:
        input_path: Path del video de entrada
        config: Configuración del sistema
        encoder: Encoder a usar (libx264, h264_nvenc, etc.)
    
    Returns:
        (output_path, metrics): Tupla con path de salida y métricas
    """
    import subprocess
    
    logger.info("Procesando video vertical con dimensiones no estándar")
    
    input_name = Path(input_path).stem
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    output_filename = f"{input_name}_vertical_rescaled_{timestamp_str}.mp4"
    output_path = os.path.join(temp_dir, output_filename)
    
    # Obtener dimensiones target
    crop_width = config.CROP_SETTINGS['width']
    crop_height = config.CROP_SETTINGS['height']
    
    # Obtener settings de encoding
    quality_preset = config.ENCODING_SETTINGS['quality_preset']
    settings = config.ENCODING_SETTINGS['presets'][quality_preset]
    
    # Construir comando FFmpeg para rescale
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"scale={crop_width}:{crop_height}:force_original_aspect_ratio=decrease,"
               f"pad={crop_width}:{crop_height}:(ow-iw)/2:(oh-ih)/2:color=black",
        "-c:v", encoder,
        "-preset", settings["preset"],
        "-crf", str(settings["crf"]),
    ]
    
    # Solo agregar profile para software encoders
    if encoder == 'libx264':
        cmd.extend(["-profile:v", settings.get("profile", "high")])
    
    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ])
    
    logger.info(f"Re-scaling video vertical a {crop_width}x{crop_height}")
    logger.info(f"Encoder: {encoder}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Video re-escalado exitosamente")
        logger.info(f"Output: {output_path}")
        
        metrics = {
            'total_frames': 0,
            'frames_processed': 0,
            'keyframes': 0,
            'analysis_time': 0,
            'total_time': 0,
            'overall_quality': 1.0,
            'mode': 'vertical_rescale'
        }
        
        return output_path, metrics
        
    except subprocess.CalledProcessError as e:
        logger.error("Error en FFmpeg durante re-scale")
        logger.error(f"STDERR: {e.stderr[-500:]}")
        raise RuntimeError("Error al re-escalar el video vertical")


def calculate_optimal_composition(face, frame_size, crop_size, config):

    frame_w, frame_h = frame_size
    crop_w, crop_h = crop_size

    x, y, w, h = face['bbox']
    cx, cy = face['center']

    if config.CROP_SETTINGS_ENHANCED.get(
        'use_rule_of_thirds',
        config.CROP_SETTINGS.get('use_rule_of_thirds', False)
    ):
        face_ratio_x = cx / frame_w
        thirds_offset = config.CROP_SETTINGS_ENHANCED.get(
            'thirds_offset_factor',
            config.CROP_SETTINGS.get('thirds_offset_factor', 0.15)
        )

        if face_ratio_x < 0.35:
            target_offset = crop_w * (0.33 - thirds_offset)
        elif face_ratio_x > 0.65:
            target_offset = crop_w * (0.67 + thirds_offset)
        else:
            target_offset = crop_w * 0.5
    else:
        target_offset = crop_w * 0.5

    headroom = config.CROP_SETTINGS_ENHANCED.get(
        'headroom_ratio',
        config.CROP_SETTINGS.get('headroom_ratio', 0.18)
    ) * crop_h

    face_height_ratio = h / frame_h
    if face_height_ratio > 0.4:
        headroom *= 0.7
    elif face_height_ratio < 0.15:
        headroom *= 1.3

    crop_x = cx - target_offset

    padding = config.CROP_SETTINGS_ENHANCED.get(
        'edge_padding',
        config.CROP_SETTINGS.get('edge_padding', 15)
    )

    crop_x = max(padding, min(crop_x, frame_w - crop_w - padding))

    return int(crop_x), 0