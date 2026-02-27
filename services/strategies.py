"""
Processing Strategies — Patrón Strategy para los modos de procesamiento.

Cada estrategia encapsula el flujo completo de un modo:
  - VerticalStrategy:    Video completo convertido a 9:16 (comportamiento original)
  - ShortAutoStrategy:   Corta el segmento central + convierte a 9:16
  - ShortManualStrategy: Corta el segmento indicado por el usuario + convierte a 9:16

El VideoProcessingService selecciona la estrategia según el processing_mode
del request y delega el procesamiento sin conocer los detalles de cada modo.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

from models.schemas import (
    VideoProcessRequest,
    VerticalRequest,
    ShortAutoRequest,
    ShortManualRequest,
    ProcessingMode,
)
from progress_tracker import ProgressTracker, ProcessingPhase


logger = logging.getLogger(__name__)


# ============================================================
# Interfaz base
# ============================================================

class ProcessingStrategy(ABC):
    """
    Interfaz abstracta para las estrategias de procesamiento.

    Contrato:
      - Recibe el video ya descargado en local_input_path.
      - Retorna (local_output_path, metrics).
      - local_output_path es un archivo temporal; el service
        se encarga de subirlo y limpiarlo.
      - metrics es un dict con al menos 'overall_quality'.
        Las estrategias de short agregan 'segment_start' y 'segment_duration'.
    """

    @abstractmethod
    def process(
        self,
        local_input_path: str,
        request: VideoProcessRequest,
        config,
        detector,
        stabilizer,
        encoder: str,
        job_id: str,
        tracker: ProgressTracker,
    ) -> Tuple[str, dict]:
        """
        Ejecuta el procesamiento y retorna (output_path, metrics).

        Args:
            local_input_path: Path del video descargado localmente.
            request:          Request original del usuario.
            config:           Módulo de configuración (config_enhanced).
            detector:         EnhancedFaceDetector configurado.
            stabilizer:       AdaptiveStabilizer configurado.
            encoder:          Nombre del encoder FFmpeg a usar.
            job_id:           ID del job para logging y trazabilidad.
            tracker:          ProgressTracker para reportar fases.

        Returns:
            Tupla (local_output_path, metrics_dict).
        """

    @property
    @abstractmethod
    def mode(self) -> ProcessingMode:
        """Retorna el ProcessingMode que implementa esta estrategia."""


# ============================================================
# Estrategia: Vertical (comportamiento original)
# ============================================================

class VerticalStrategy(ProcessingStrategy):
    """
    Convierte el video completo a formato vertical 9:16.
    Encapsula la lógica que antes vivía directamente en VideoProcessingService.
    """

    @property
    def mode(self) -> ProcessingMode:
        return ProcessingMode.vertical

    def process(
        self,
        local_input_path: str,
        request: VideoProcessRequest,
        config,
        detector,
        stabilizer,
        encoder: str,
        job_id: str,
        tracker: ProgressTracker,
    ) -> Tuple[str, dict]:
        """
        Flujo:
          1. Detectar rostros y calcular posiciones.
          2. Generar video vertical con FFmpeg.
        """
        from app.video_processor_enhanced import process_video_enhanced

        logger.info(
            "VerticalStrategy iniciando | job_id=%s | input=%s",
            job_id,
            local_input_path
        )

        use_multipass = request.quality.value in ['normal', 'high']

        tracker.update_phase(ProcessingPhase.DETECTING_FACES)

        output_path, metrics = process_video_enhanced(
            local_input_path,
            config,
            detector,
            stabilizer,
            use_multipass=use_multipass,
            encoder=encoder
        )

        # VerticalStrategy no tiene segmento específico
        metrics['segment_start'] = None
        metrics['segment_duration'] = None

        logger.info(
            "VerticalStrategy completada | job_id=%s | output=%s | quality=%.2f%%",
            job_id,
            output_path,
            metrics.get('overall_quality', 0) * 100
        )

        return output_path, metrics


# ============================================================
# Estrategia: Short Auto
# ============================================================

class ShortAutoStrategy(ProcessingStrategy):
    """
    Genera un short seleccionando automáticamente el segmento central del video.

    Flujo en dos pasos:
      Paso 1 — Corte: FFmpeg extrae el segmento central en un archivo intermedio.
      Paso 2 — Vertical: process_video_enhanced convierte el intermedio a 9:16.
    """

    @property
    def mode(self) -> ProcessingMode:
        return ProcessingMode.short_auto

    def process(
        self,
        local_input_path: str,
        request: VideoProcessRequest,
        config,
        detector,
        stabilizer,
        encoder: str,
        job_id: str,
        tracker: ProgressTracker,
    ) -> Tuple[str, dict]:
        """
        Flujo:
          1. Obtener duración real del video.
          2. Seleccionar segmento central con SegmentSelector.
          3. Cortar segmento con FFmpeg (archivo intermedio).
          4. Procesar el intermedio a vertical 9:16.
        """
        from app.video_processor_enhanced import process_video_enhanced
        from services.segment_selector import SegmentSelector
        from services.segment_cutter import SegmentCutter

        # El discriminador garantiza que request es ShortAutoRequest aquí,
        # pero usamos isinstance para que el IDE infiera el tipo correctamente.
        assert isinstance(request, ShortAutoRequest), (
            f"ShortAutoStrategy recibió un request de tipo inesperado: {type(request)}"
        )

        logger.info(
            "ShortAutoStrategy iniciando | job_id=%s | target_duration=%ds",
            job_id,
            request.short_auto_duration
        )

        # Paso 1: Obtener duración real
        tracker.update_phase(ProcessingPhase.SELECTING_SEGMENT)

        video_duration = SegmentSelector.get_video_duration(local_input_path)
        logger.info("Duración del video: %.2fs | job_id=%s", video_duration, job_id)

        # Validación de segunda línea con duración real
        from validators import ShortOptionsValidator
        ShortOptionsValidator.validate_short_auto(
            target_duration=request.short_auto_duration,
            video_duration=video_duration
        )

        # Paso 2: Seleccionar el mejor segmento por análisis de contenido
        start_time, actual_duration, selection_strategy = SegmentSelector.select_best_segment(
            video_path=local_input_path,
            total_duration=video_duration,
            target_duration=request.short_auto_duration,
            detector=detector,
            config=config,
        )

        logger.info(
            "Segmento seleccionado | job_id=%s | start=%.2fs | duration=%ds | strategy=%s",
            job_id, start_time, actual_duration, selection_strategy
        )

        # Paso 3: Cortar segmento
        tracker.update_phase(ProcessingPhase.CUTTING_SEGMENT)

        intermediate_path = SegmentCutter.cut_segment(
            input_path=local_input_path,
            start_time=start_time,
            duration=actual_duration,
            job_id=job_id
        )

        logger.info("Segmento cortado | job_id=%s | intermediate=%s", job_id, intermediate_path)

        # Paso 4: Procesar a vertical
        tracker.update_phase(ProcessingPhase.DETECTING_FACES)

        use_multipass = request.quality.value in ['normal', 'high']

        output_path, metrics = process_video_enhanced(
            intermediate_path,
            config,
            detector,
            stabilizer,
            use_multipass=use_multipass,
            encoder=encoder
        )

        _cleanup_intermediate(intermediate_path, job_id)

        metrics['segment_start'] = start_time
        metrics['segment_duration'] = actual_duration
        metrics['selection_strategy'] = selection_strategy
        metrics['original_duration'] = video_duration

        logger.info(
            "ShortAutoStrategy completada | job_id=%s | start=%.2fs | duration=%ds | quality=%.2f%%",
            job_id, start_time, actual_duration, metrics.get('overall_quality', 0) * 100
        )

        return output_path, metrics


# ============================================================
# Estrategia: Short Manual
# ============================================================

class ShortManualStrategy(ProcessingStrategy):
    """
    Genera un short a partir de un segmento definido explícitamente por el usuario.

    Flujo en dos pasos:
      Paso 1 — Corte: FFmpeg extrae el segmento indicado en un archivo intermedio.
      Paso 2 — Vertical: process_video_enhanced convierte el intermedio a 9:16.
    """

    @property
    def mode(self) -> ProcessingMode:
        return ProcessingMode.short_manual

    def process(
        self,
        local_input_path: str,
        request: VideoProcessRequest,
        config,
        detector,
        stabilizer,
        encoder: str,
        job_id: str,
        tracker: ProgressTracker,
    ) -> Tuple[str, dict]:
        """
        Flujo:
          1. Obtener duración real del video.
          2. Validar que el segmento no exceda el video (segunda línea de defensa).
          3. Cortar segmento con FFmpeg (archivo intermedio).
          4. Procesar el intermedio a vertical 9:16.
        """
        from app.video_processor_enhanced import process_video_enhanced
        from services.segment_selector import SegmentSelector
        from services.segment_cutter import SegmentCutter

        # El discriminador garantiza que request es ShortManualRequest aquí.
        assert isinstance(request, ShortManualRequest), (
            f"ShortManualStrategy recibió un request de tipo inesperado: {type(request)}"
        )

        start_time = request.short_options.start_time
        duration = request.short_options.duration

        logger.info(
            "ShortManualStrategy iniciando | job_id=%s | start=%.2fs | duration=%ds",
            job_id, start_time, duration
        )

        # Paso 1: Obtener duración real (necesaria para validación de runtime)
        tracker.update_phase(ProcessingPhase.SELECTING_SEGMENT)

        video_duration = SegmentSelector.get_video_duration(local_input_path)
        logger.info("Duración del video: %.2fs | job_id=%s", video_duration, job_id)

        # Paso 2: Validación de segunda línea con duración real
        from validators import ShortOptionsValidator
        ShortOptionsValidator.validate_short_manual(
            start_time=start_time,
            duration=duration,
            video_duration=video_duration
        )

        # Paso 3: Cortar segmento
        tracker.update_phase(ProcessingPhase.CUTTING_SEGMENT)

        intermediate_path = SegmentCutter.cut_segment(
            input_path=local_input_path,
            start_time=start_time,
            duration=duration,
            job_id=job_id
        )

        logger.info("Segmento cortado | job_id=%s | intermediate=%s", job_id, intermediate_path)

        # Paso 4: Procesar a vertical
        tracker.update_phase(ProcessingPhase.DETECTING_FACES)

        use_multipass = request.quality.value in ['normal', 'high']

        output_path, metrics = process_video_enhanced(
            intermediate_path,
            config,
            detector,
            stabilizer,
            use_multipass=use_multipass,
            encoder=encoder
        )

        _cleanup_intermediate(intermediate_path, job_id)

        metrics['segment_start'] = start_time
        metrics['segment_duration'] = duration
        metrics['selection_strategy'] = 'manual'
        metrics['original_duration'] = video_duration

        logger.info(
            "ShortManualStrategy completada | job_id=%s | start=%.2fs | duration=%ds | quality=%.2f%%",
            job_id, start_time, duration, metrics.get('overall_quality', 0) * 100
        )

        return output_path, metrics


# ============================================================
# Factory — selección de estrategia
# ============================================================

def get_strategy(processing_mode: ProcessingMode) -> ProcessingStrategy:
    """
    Retorna la estrategia correspondiente al modo de procesamiento.

    Args:
        processing_mode: Modo solicitado en el request.

    Returns:
        Instancia de ProcessingStrategy lista para usar.

    Raises:
        ValueError: Si el modo no tiene estrategia implementada.
    """
    strategies = {
        ProcessingMode.vertical: VerticalStrategy,
        ProcessingMode.short_auto: ShortAutoStrategy,
        ProcessingMode.short_manual: ShortManualStrategy,
    }

    strategy_class = strategies.get(processing_mode)

    if strategy_class is None:
        raise ValueError(
            f"No hay estrategia implementada para el modo '{processing_mode}'. "
            f"Modos disponibles: {list(strategies.keys())}"
        )

    instance = strategy_class()

    logger.debug(
        "Estrategia seleccionada | mode=%s | class=%s",
        processing_mode.value,
        strategy_class.__name__
    )

    return instance


# ============================================================
# Helper interno
# ============================================================

def _cleanup_intermediate(intermediate_path: str, job_id: str) -> None:
    """
    Elimina el archivo intermedio del segmento cortado.
    No lanza excepción si falla — es una limpieza no crítica.
    """
    try:
        if intermediate_path and os.path.exists(intermediate_path):
            os.remove(intermediate_path)
            logger.info(
                "Archivo intermedio eliminado | job_id=%s | path=%s",
                job_id,
                intermediate_path
            )
    except Exception as e:
        logger.warning(
            "No se pudo eliminar archivo intermedio | job_id=%s | path=%s | error=%s",
            job_id,
            intermediate_path,
            str(e)
        )
