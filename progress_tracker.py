"""
Progress Tracker — Sistema de tracking detallado de progreso y estados.

Versión 2: Agrega fases específicas para los modos short_auto y short_manual:
  - SELECTING_SEGMENT: Determinando qué segmento extraer
  - CUTTING_SEGMENT:   FFmpeg extrayendo el segmento (archivo intermedio)
  - SEGMENT_COMPLETE:  Segmento listo para procesar
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================
# Estados detallados del procesamiento
# ============================================================

class ProcessingPhase(str, Enum):
    """Fases detalladas del procesamiento"""

    # Fase inicial
    QUEUED = "queued"
    VALIDATING = "validating"

    # Fase de descarga
    DOWNLOADING = "downloading"
    DOWNLOAD_COMPLETE = "download_complete"

    # Fases de short (nuevo en v2) ─────────────────────────────
    SELECTING_SEGMENT = "selecting_segment"     # Calculando qué segmento usar
    CUTTING_SEGMENT = "cutting_segment"         # FFmpeg cortando el segmento
    SEGMENT_COMPLETE = "segment_complete"       # Segmento listo
    # ───────────────────────────────────────────────────────────

    # Fase de análisis
    ANALYZING = "analyzing"
    DETECTING_FACES = "detecting_faces"
    ANALYSIS_COMPLETE = "analysis_complete"

    # Fase de procesamiento
    PROCESSING = "processing"
    STABILIZING = "stabilizing"
    CROPPING = "cropping"

    # Fase de encoding
    ENCODING = "encoding"
    ENCODING_COMPLETE = "encoding_complete"

    # Fase de subida
    UPLOADING = "uploading"
    UPLOAD_COMPLETE = "upload_complete"

    # Limpieza y fin
    CLEANING_UP = "cleaning_up"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# Mapeo de fases a porcentajes de progreso
# ============================================================

PHASE_PROGRESS_MAP = {
    # Inicio
    ProcessingPhase.QUEUED:             0,
    ProcessingPhase.VALIDATING:         5,

    # Descarga
    ProcessingPhase.DOWNLOADING:        10,
    ProcessingPhase.DOWNLOAD_COMPLETE:  20,

    # Fases de short (nuevo en v2)
    # Ocupan el hueco entre la descarga y el análisis de rostros.
    # En modo vertical estas fases no se usan, por lo que el progreso
    # salta directamente de DOWNLOAD_COMPLETE a ANALYZING.
    ProcessingPhase.SELECTING_SEGMENT:  22,
    ProcessingPhase.CUTTING_SEGMENT:    28,
    ProcessingPhase.SEGMENT_COMPLETE:   33,

    # Análisis
    ProcessingPhase.ANALYZING:          35,
    ProcessingPhase.DETECTING_FACES:    45,
    ProcessingPhase.ANALYSIS_COMPLETE:  55,

    # Procesamiento
    ProcessingPhase.PROCESSING:         58,
    ProcessingPhase.STABILIZING:        62,
    ProcessingPhase.CROPPING:           65,

    # Encoding
    ProcessingPhase.ENCODING:           70,
    ProcessingPhase.ENCODING_COMPLETE:  85,

    # Subida
    ProcessingPhase.UPLOADING:          88,
    ProcessingPhase.UPLOAD_COMPLETE:    95,

    # Fin
    ProcessingPhase.CLEANING_UP:        98,
    ProcessingPhase.COMPLETED:          100,
    ProcessingPhase.FAILED:             0,
}


# ============================================================
# Mensajes amigables por fase
# ============================================================

PHASE_MESSAGES = {
    ProcessingPhase.QUEUED:             "Video en cola para procesarse",
    ProcessingPhase.VALIDATING:         "Validando el video...",
    ProcessingPhase.DOWNLOADING:        "Descargando video desde Cloudinary...",
    ProcessingPhase.DOWNLOAD_COMPLETE:  "Video descargado correctamente",

    # Mensajes de short (nuevo en v2)
    ProcessingPhase.SELECTING_SEGMENT:  "Seleccionando segmento del video...",
    ProcessingPhase.CUTTING_SEGMENT:    "Cortando segmento del video...",
    ProcessingPhase.SEGMENT_COMPLETE:   "Segmento listo para procesar",

    ProcessingPhase.ANALYZING:          "Analizando contenido del video...",
    ProcessingPhase.DETECTING_FACES:    "Detectando rostros en el video...",
    ProcessingPhase.ANALYSIS_COMPLETE:  "Análisis completado",
    ProcessingPhase.PROCESSING:         "Procesando video...",
    ProcessingPhase.STABILIZING:        "Estabilizando movimiento de cámara...",
    ProcessingPhase.CROPPING:           "Aplicando recorte inteligente...",
    ProcessingPhase.ENCODING:           "Generando video final...",
    ProcessingPhase.ENCODING_COMPLETE:  "Video generado exitosamente",
    ProcessingPhase.UPLOADING:          "Subiendo video procesado...",
    ProcessingPhase.UPLOAD_COMPLETE:    "Video subido correctamente",
    ProcessingPhase.CLEANING_UP:        "Finalizando...",
    ProcessingPhase.COMPLETED:          "Video procesado exitosamente",
    ProcessingPhase.FAILED:             "Error durante el procesamiento",
}


# ============================================================
# Progress Tracker
# ============================================================

class ProgressTracker:
    """
    Tracker de progreso con timestamps y estimación de tiempo restante.
    """

    def __init__(self, job_id: str, update_callback: Optional[Callable] = None):
        """
        Args:
            job_id:           ID del job a trackear.
            update_callback:  Función llamada en cada actualización.
                              Firma: callback(job_id, progress_data)
        """
        self.job_id = job_id
        self.update_callback = update_callback

        self.current_phase: Optional[ProcessingPhase] = None
        self.progress_percentage: int = 0

        self.start_time: Optional[datetime] = None
        self.phase_timestamps: Dict[ProcessingPhase, datetime] = {}
        self.completion_time: Optional[datetime] = None

        self.phases_completed: list = []
        self.total_frames: Optional[int] = None
        self.frames_processed: int = 0

        self.metadata: Dict[str, Any] = {}

    def start(self):
        """Inicia el tracking"""
        self.start_time = datetime.utcnow()
        self.update_phase(ProcessingPhase.QUEUED)

        logger.info(
            "Progress tracking iniciado | job_id=%s | time=%s",
            self.job_id,
            self.start_time.isoformat()
        )

    def update_phase(
        self,
        phase: ProcessingPhase,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Actualiza la fase actual del procesamiento.

        Args:
            phase:    Nueva fase.
            message:  Mensaje personalizado (opcional, sobreescribe el default).
            metadata: Metadata adicional a agregar (opcional).
        """
        self.current_phase = phase
        self.progress_percentage = PHASE_PROGRESS_MAP.get(phase, 0)
        self.phase_timestamps[phase] = datetime.utcnow()

        if phase not in self.phases_completed:
            self.phases_completed.append(phase)

        if metadata:
            self.metadata.update(metadata)

        final_message = message or PHASE_MESSAGES.get(phase, str(phase))

        elapsed = self._get_elapsed_time()
        logger.info(
            "Progreso actualizado | job_id=%s | phase=%s | progress=%d%% | elapsed=%.1fs | message=%s",
            self.job_id,
            phase.value,
            self.progress_percentage,
            elapsed,
            final_message
        )

        if self.update_callback:
            self._trigger_callback(final_message)

    def update_progress(self, percentage: int, message: Optional[str] = None):
        """
        Actualiza el progreso dentro de una fase (granularidad fina).

        Args:
            percentage: Porcentaje de progreso (0-100).
            message:    Mensaje opcional.
        """
        self.progress_percentage = max(0, min(100, percentage))

        if message:
            logger.info(
                "Progreso actualizado | job_id=%s | progress=%d%% | message=%s",
                self.job_id,
                self.progress_percentage,
                message
            )

        if self.update_callback:
            self._trigger_callback(message)

    def update_frames(self, frames_processed: int, total_frames: int):
        """
        Actualiza progreso basado en frames procesados.

        Args:
            frames_processed: Frames procesados hasta ahora.
            total_frames:     Total de frames del segmento.
        """
        self.frames_processed = frames_processed
        self.total_frames = total_frames

        if total_frames > 0:
            frame_percentage = (frames_processed / total_frames) * 100

            if self.current_phase in [ProcessingPhase.ANALYZING, ProcessingPhase.DETECTING_FACES]:
                progress = 35 + (frame_percentage * 0.20)
            elif self.current_phase in [ProcessingPhase.PROCESSING, ProcessingPhase.STABILIZING]:
                progress = 58 + (frame_percentage * 0.12)
            else:
                progress = self.progress_percentage

            self.update_progress(
                int(progress),
                f"Procesando frames: {frames_processed}/{total_frames}"
            )

    def complete(self, success: bool = True):
        """
        Marca el procesamiento como completado o fallido.

        Args:
            success: True si fue exitoso, False si falló.
        """
        self.completion_time = datetime.utcnow()

        if success:
            self.update_phase(ProcessingPhase.COMPLETED)
        else:
            self.current_phase = ProcessingPhase.FAILED
            self.progress_percentage = 0

        total_time = self._get_elapsed_time()

        logger.info(
            "Procesamiento %s | job_id=%s | total_time=%.2fs",
            "completado" if success else "falló",
            self.job_id,
            total_time
        )

        if self.update_callback:
            self._trigger_callback(
                PHASE_MESSAGES[
                    ProcessingPhase.COMPLETED if success else ProcessingPhase.FAILED
                ]
            )

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna el estado completo actual del tracker.

        Returns:
            Dict con toda la información de progreso.
        """
        elapsed = self._get_elapsed_time()
        eta = self._estimate_time_remaining()

        return {
            'job_id': self.job_id,
            'phase': self.current_phase.value if self.current_phase else None,
            'progress': self.progress_percentage,
            'message': PHASE_MESSAGES.get(self.current_phase, "Procesando..."),
            'elapsed_seconds': elapsed,
            'elapsed_formatted': self._format_duration(elapsed),
            'eta_seconds': eta,
            'eta_formatted': self._format_duration(eta) if eta else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'phases_completed': [p.value for p in self.phases_completed],
            'metadata': self.metadata,
        }

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _get_elapsed_time(self) -> float:
        """Calcula tiempo transcurrido en segundos"""
        if not self.start_time:
            return 0.0
        end_time = self.completion_time or datetime.utcnow()
        return (end_time - self.start_time).total_seconds()

    def _estimate_time_remaining(self) -> Optional[float]:
        """Estima tiempo restante basado en progreso actual"""
        if self.progress_percentage == 0 or not self.start_time:
            return None
        if self.progress_percentage >= 100:
            return 0.0
        elapsed = self._get_elapsed_time()
        time_per_percent = elapsed / self.progress_percentage
        remaining_percent = 100 - self.progress_percentage
        return time_per_percent * remaining_percent

    def _format_duration(self, seconds: float) -> str:
        """Formatea duración en formato legible"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _trigger_callback(self, message: Optional[str] = None):
        """Ejecuta el callback de actualización de forma segura"""
        if not self.update_callback:
            return
        try:
            progress_data = self.get_status()
            if message:
                progress_data['message'] = message
            self.update_callback(self.job_id, progress_data)
        except Exception as e:
            logger.error(
                "Error ejecutando callback de progreso | job_id=%s | error=%s",
                self.job_id,
                str(e)
            )


# ============================================================
# Factory function
# ============================================================

def create_progress_tracker(
    job_id: str,
    update_callback: Optional[Callable] = None
) -> ProgressTracker:
    """
    Crea e inicia un tracker de progreso.

    Args:
        job_id:           ID del job.
        update_callback:  Callback opcional para actualizaciones.

    Returns:
        ProgressTracker ya iniciado.
    """
    tracker = ProgressTracker(job_id, update_callback)
    tracker.start()
    return tracker
