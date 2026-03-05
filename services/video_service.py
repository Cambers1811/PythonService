"""
Video Service — Orquesta el flujo completo de procesamiento.

Versión 2: Adopta el patrón Strategy para soportar tres modos de procesamiento
(vertical, short_auto, short_manual) sin duplicar lógica de orquestación.

Responsabilidades de este service:
  - Descargar el video desde Cloudinary.
  - Configurar el módulo config_enhanced según el request.
  - Seleccionar y ejecutar la estrategia de procesamiento correcta.
  - Generar thumbnail y preview del output.
  - Subir el resultado a Cloudinary.
  - Limpiar archivos temporales.
  - Registrar métricas y propagar progreso al router.

Lo que NO hace este service:
  - Decidir cómo se selecciona el segmento (SegmentSelector).
  - Decidir cómo se corta el segmento (SegmentCutter).
  - Decidir cómo se procesa cada modo (ProcessingStrategy).
"""

import os
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable

from app import config_enhanced as config
from app.face_detector_enhanced import EnhancedFaceDetector
from app.stabilization_enhanced import AdaptiveStabilizer

from models.schemas import (
    VideoProcessRequest,
    Platform,
    BackgroundMode,
    QualityLevel,
    ProcessingMode,
    QUALITY_TO_PRESET,
    BACKGROUND_TO_CONVERSION_MODE,
    BACKGROUND_TO_BLUR,
)
from storage.cloudinary_service import CloudinaryService
from services.strategies import get_strategy
from error_handler import ErrorHandler, ErrorContext, retry_on_failure
from exceptions import VideoProcessingError
from progress_tracker import ProgressTracker, ProcessingPhase
from cancellation_manager import (
    get_cancellation_manager,
    JobCancelledException,
    CancellableProgressTracker,
    CancellableOperation,
)
from preview_generator import create_preview_generator
from optimization import (
    get_performance_monitor,
    HardwareAccelerationDetector,
)
from services.webhook_service import notify_progress

logger = logging.getLogger(__name__)


class VideoProcessingService:

    def __init__(self, cloudinary_service: CloudinaryService):
        self.cloudinary = cloudinary_service
        self.progress_callback: Optional[Callable] = None
        self.cancellation_manager = get_cancellation_manager()
        self.preview_generator = create_preview_generator(
            temp_dir=cloudinary_service.temp_dir
        )
        self.performance_monitor = get_performance_monitor()
        self.hw_encoder = HardwareAccelerationDetector.get_optimized_ffmpeg_encoder()

        logger.info(
            "VideoProcessingService inicializado | encoder=%s",
            self.hw_encoder
        )

    def set_progress_callback(self, callback: Callable):
        """Establece el callback de progreso para el router"""
        self.progress_callback = callback

    def _create_progress_webhook_callback(self, job_id: str) -> Callable:
        """
        Crea un callback que envía webhooks de progreso a Spring Boot.

        Este callback se ejecuta cada vez que hay un cambio significativo:
        - Progreso cambió ≥ 1%
        - Fase cambió
        - Pasaron ≥ 5 segundos desde última notificación

        Args:
            job_id: ID del job

        Returns:
            Función callback que acepta (job_id, progress_data)
        """

        def progress_callback(job_id_inner: str, progress_data: dict):
            """
            Callback ejecutado en cada cambio significativo de progreso.

            Args:
                job_id_inner: ID del job
                progress_data: Diccionario con datos de progreso del ProgressTracker
            """
            # Enviar webhook a Spring Boot
            notify_progress(job_id_inner, progress_data)

        return progress_callback

    # ============================================================
    # Punto de entrada principal
    # ============================================================

    def process_video(
            self,
            request: VideoProcessRequest,
            job_id: str
    ) -> Tuple[str, dict]:
        """
        Orquesta el flujo completo de procesamiento.

        Flujo:
          1. Descargar video desde Cloudinary.
          2. Configurar parámetros según request.
          3. Seleccionar y ejecutar la estrategia de procesamiento.
          4. Generar thumbnail y preview.
          5. Subir resultado a Cloudinary.
          6. Limpiar archivos temporales.

        Args:
            request: Request validado del usuario.
            job_id:  ID único del job para trazabilidad.

        Returns:
            Tupla (output_url, metrics_dict).

        Raises:
            JobCancelledException: Si el usuario canceló el job.
            VideoProcessingError:  Si ocurre cualquier otro error.
        """
        start_time = time.time()
        local_input_path = None
        local_output_path = None

        perf = self.performance_monitor

        # Crear callback de webhook de progreso
        webhook_callback = self._create_progress_webhook_callback(job_id)

        # Inicializar progress tracker CON el callback de webhooks
        base_tracker = ProgressTracker(job_id, update_callback=webhook_callback)
        base_tracker.start()

        tracker = CancellableProgressTracker(
            base_tracker,
            self.cancellation_manager,
            job_id
        )

        def cleanup():
            try:
                tracker.update_phase(ProcessingPhase.CLEANING_UP)
                if job_id:
                    self.cloudinary.delete_local_files(job_id)
                if local_output_path and os.path.exists(local_output_path):
                    os.remove(local_output_path)
                    logger.info("Cleanup: output eliminado | job_id=%s", job_id)
            except Exception as e:
                logger.warning(
                    "Error durante cleanup | job_id=%s | error=%s",
                    job_id,
                    str(e)
                )

        try:
            logger.info(
                "INICIANDO PROCESAMIENTO | job_id=%s | mode=%s | platform=%s | quality=%s",
                job_id,
                request.processing_mode.value,
                request.platform.value,
                request.quality.value
            )

            # ── PASO 1: Validación ───────────────────────────────────
            tracker.update_phase(ProcessingPhase.VALIDATING)

            # ── PASO 2: Descarga ─────────────────────────────────────
            tracker.update_phase(ProcessingPhase.DOWNLOADING)

            download_start = time.time()
            with CancellableOperation(self.cancellation_manager, job_id, "descarga"):
                with ErrorContext("descarga de video", cleanup_func=cleanup, job_id=job_id):
                    local_input_path = self._download_with_retry(
                        request.cloudinary_input_url,
                        job_id
                    )
            perf.record_metric('download_time', time.time() - download_start)

            tracker.update_phase(ProcessingPhase.DOWNLOAD_COMPLETE)

            # ── PASO 3: Configuración ─────────────────────────────────
            # config_enhanced es un módulo global mutable: siempre se
            # reconfigura por completo antes de cada job para evitar que
            # un job anterior deje estado residual (background_mode, blur, etc.).
            # Cachear "si ya configuré" no es equivalente a tener el módulo
            # en el estado correcto, por lo que el cache de configuración fue removido.
            with ErrorContext("configuración de parámetros", job_id=job_id):
                self._configure_processing(request)
                logger.info(
                    "Configuración aplicada | job_id=%s | background_mode=%s",
                    job_id,
                    request.background_mode.value
                )

            # ── PASO 4: Procesamiento con Strategy ────────────────────
            tracker.update_phase(ProcessingPhase.ANALYZING)

            analysis_start = time.time()
            with ErrorContext("procesamiento de video", cleanup_func=cleanup, job_id=job_id):
                logger.info(
                    "Ejecutando estrategia | job_id=%s | mode=%s",
                    job_id,
                    request.processing_mode.value
                )

                strategy = get_strategy(request.processing_mode)

                detector = EnhancedFaceDetector(config)
                stabilizer = AdaptiveStabilizer(config)

                local_output_path, metrics = strategy.process(
                    local_input_path=local_input_path,
                    request=request,
                    config=config,
                    detector=detector,
                    stabilizer=stabilizer,
                    encoder=self.hw_encoder,
                    job_id=job_id,
                    tracker=tracker,
                )

                if 'frames_processed' in metrics and 'total_frames' in metrics:
                    base_tracker.metadata['frames_processed'] = metrics['frames_processed']
                    base_tracker.metadata['total_frames'] = metrics['total_frames']
                    perf.record_metric('frames_analyzed', metrics.get('frames_processed', 0))

            perf.record_metric('analysis_time', time.time() - analysis_start)

            tracker.update_phase(ProcessingPhase.ENCODING_COMPLETE)

            # ── VERIFICACIÓN CRÍTICA: Cancelación después de procesar ─────
            # Si el job fue cancelado DESPUÉS de completar el procesamiento
            # pero ANTES de subir a Cloudinary, debemos detener aquí para:
            # 1. No desperdiciar ancho de banda subiendo
            # 2. No generar costos de almacenamiento innecesarios
            # 3. No generar thumbnails/previews de un job cancelado
            if self.cancellation_manager.is_cancelled(job_id):
                logger.warning(
                    "Job cancelado después de completar procesamiento | job_id=%s | "
                    "output_local=%s (no subido a Cloudinary)",
                    job_id,
                    local_output_path
                )
                # Cleanup del archivo procesado localmente
                cleanup()
                raise JobCancelledException(job_id)

            # ── PASO 5: Subida ────────────────────────────────────────
            tracker.update_phase(ProcessingPhase.UPLOADING)

            upload_start = time.time()
            with ErrorContext("subida a Cloudinary", cleanup_func=cleanup, job_id=job_id):
                folder = f"processed_{request.platform.value}"

                # Los shorts van a una subcarpeta para organización
                if request.processing_mode in [
                    ProcessingMode.short_auto,
                    ProcessingMode.short_manual
                ]:
                    folder = f"{folder}/shorts"

                output_url = self._upload_with_retry(
                    local_output_path,
                    job_id,
                    folder=folder
                )
            perf.record_metric('upload_time', time.time() - upload_start)

            tracker.update_phase(ProcessingPhase.UPLOAD_COMPLETE)

            # ── PASO 6: Thumbnail y preview ───────────────────────────
            thumbnail_url, preview_url = self._generate_and_upload_previews(
                local_output_path=local_output_path,
                job_id=job_id,
                platform=request.platform.value,
                processing_mode=request.processing_mode
            )

            # ── PASO 7: Limpieza ──────────────────────────────────────
            cleanup()

            # ── Métricas finales ──────────────────────────────────────
            total_time = time.time() - start_time

            metrics['processing_total_time'] = total_time
            metrics['thumbnail_url'] = thumbnail_url
            metrics['preview_url'] = preview_url
            metrics['processing_mode'] = request.processing_mode.value

            perf.record_metric('total_processing_time', total_time)
            if self.hw_encoder != 'libx264':
                perf.record_metric('hw_acceleration_used', True)

            perf.log_summary()
            tracker.complete(success=True)

            logger.info(
                "PROCESAMIENTO COMPLETADO | job_id=%s | mode=%s | "
                "tiempo=%.2fs | calidad=%.1f%% | url=%s",
                job_id,
                request.processing_mode.value,
                total_time,
                metrics.get('overall_quality', 0) * 100,
                output_url
            )

            return output_url, metrics

        except JobCancelledException:
            logger.warning("JOB CANCELADO | job_id=%s", job_id)
            base_tracker.complete(success=False)
            try:
                cleanup()
            except Exception:
                pass
            self.cancellation_manager.remove_cancellation(job_id)
            raise

        except Exception as e:
            logger.exception("ERROR EN PROCESAMIENTO | job_id=%s", job_id)
            tracker.complete(success=False)
            error_info = ErrorHandler.handle_error(
                e,
                job_id=job_id,
                operation="process_video"
            )
            try:
                cleanup()
            except Exception:
                pass
            raise VideoProcessingError(error_info['user_message']) from e

    # ============================================================
    # Helpers internos
    # ============================================================

    def _generate_and_upload_previews(
            self,
            local_output_path: str,
            job_id: str,
            platform: str,
            processing_mode: ProcessingMode
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Genera y sube thumbnail y preview clip.
        No es crítico: si falla, loguea warning y retorna (None, None).
        """
        thumbnail_url = None
        preview_url = None

        try:
            logger.info("Generando thumbnail y preview | job_id=%s", job_id)

            thumbnail_path = self.preview_generator.generate_thumbnail(
                local_output_path,
                timestamp_seconds=1.0,
                width=480
            )
            thumbnail_url = self.cloudinary.upload_image(
                thumbnail_path,
                f"{job_id}_thumb",
                folder=f"processed_{platform}/thumbnails"
            )
            logger.info("Thumbnail subido | job_id=%s | url=%s", job_id, thumbnail_url)

            preview_path = self.preview_generator.generate_preview_clip(
                local_output_path,
                duration_seconds=5,
                start_time=0.0
            )
            preview_url = self.cloudinary.upload_video(
                preview_path,
                f"{job_id}_preview",
                folder=f"processed_{platform}/previews"
            )
            logger.info("Preview subido | job_id=%s | url=%s", job_id, preview_url)

            self.preview_generator.cleanup(thumbnail_path, preview_path)

        except Exception as e:
            logger.warning(
                "Error generando thumbnail/preview (no crítico) | job_id=%s | error=%s",
                job_id,
                str(e)
            )

        return thumbnail_url, preview_url

    @retry_on_failure(max_attempts=3, delay_seconds=2.0)
    def _download_with_retry(self, url: str, job_id: str) -> str:
        return self.cloudinary.download_video(url, job_id)

    @retry_on_failure(max_attempts=3, delay_seconds=2.0)
    def _upload_with_retry(self, local_path: str, job_id: str, folder: str) -> str:
        return self.cloudinary.upload_video(local_path, job_id, folder)

    def _configure_processing(self, request: VideoProcessRequest):
        """
        Aplica configuración al módulo config_enhanced según el request.
        Sin cambios respecto a v1 — la configuración es independiente del modo.
        """
        quality_preset = QUALITY_TO_PRESET[request.quality]

        if request.platform == Platform.tiktok:
            config.apply_preset_enhanced('tiktok')
        elif request.platform == Platform.instagram:
            config.apply_preset_enhanced('instagram')
        elif request.platform == Platform.youtube_shorts:
            config.apply_preset_enhanced('youtube_shorts')

        if request.quality == QualityLevel.fast:
            config.PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 6
            config.PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = False
            config.ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'fast'
        elif request.quality == QualityLevel.normal:
            config.PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 4
            config.PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = False
            config.ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'balanced'
        elif request.quality == QualityLevel.high:
            config.PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 3
            config.PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = True
            config.ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'high'

        conversion_mode = BACKGROUND_TO_CONVERSION_MODE[request.background_mode]
        config.CONVERSION_MODE['mode'] = conversion_mode

        if conversion_mode == 'full':
            use_blur = BACKGROUND_TO_BLUR[request.background_mode]
            config.CONVERSION_MODE['modes']['full']['blur_background'] = use_blur
            logger.info(
                "Modo full | blur=%s | job_id implícito en contexto",
                use_blur
            )
        else:
            logger.info("Modo smart_crop")

        if request.advanced_options:
            adv = request.advanced_options

            if adv.headroom_ratio is not None:
                config.CROP_SETTINGS_ENHANCED['headroom_ratio'] = adv.headroom_ratio

            if adv.smoothing_strength is not None:
                config.STABILIZATION_ENHANCED['exponential_alpha'] = adv.smoothing_strength

            if adv.max_camera_speed is not None:
                config.STABILIZATION_ENHANCED['max_velocity_px_per_frame'] = adv.max_camera_speed

            if adv.apply_sharpening is not None:
                config.ENCODING_SETTINGS_ENHANCED['apply_unsharp'] = adv.apply_sharpening

            if adv.use_rule_of_thirds is not None:
                config.CROP_SETTINGS_ENHANCED['use_rule_of_thirds'] = adv.use_rule_of_thirds

            if adv.edge_padding is not None:
                config.CROP_SETTINGS_ENHANCED['edge_padding'] = adv.edge_padding

            logger.info("Opciones avanzadas aplicadas")

        logger.info(
            "Configuración aplicada | preset=%s | sample_rate=1/%s | "
            "multipass=%s | encoding=%s",
            quality_preset,
            config.PERFORMANCE_SETTINGS_ENHANCED['sample_rate'],
            config.PERFORMANCE_SETTINGS_ENHANCED['use_multipass'],
            config.ENCODING_SETTINGS_ENHANCED['quality_preset']
        )


# ============================================================
# Factory function
# ============================================================

def create_video_service(cloudinary_service: CloudinaryService) -> VideoProcessingService:
    return VideoProcessingService(cloudinary_service)