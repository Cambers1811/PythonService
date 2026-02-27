"""
Validators — Validaciones robustas para todos los inputs del sistema.

Versión 2: Agrega ShortOptionsValidator para los modos short_auto y short_manual.
"""

import re
import requests
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

from models.schemas import (
    VideoProcessRequest,
    BaseVideoRequest,
    VerticalRequest,
    ShortAutoRequest,
    ShortManualRequest,
    Platform,
    BackgroundMode,
    QualityLevel,
    ProcessingMode,
    SHORT_MIN_DURATION_SECONDS,
    SHORT_MAX_DURATION_SECONDS,
)
from exceptions import (
    InvalidURLError,
    VideoFormatError,
    VideoSizeError,
    VideoDurationError,
    UnsupportedPlatformError,
    ValidationError,
)


logger = logging.getLogger(__name__)


# ============================================================
# Configuración de límites
# ============================================================

class VideoLimits:
    """Límites para validación de videos"""

    SUPPORTED_FORMATS = ['mp4', 'mov', 'avi', 'mkv', 'webm']

    MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

    MIN_DURATION_SECONDS = 3
    MAX_DURATION_SECONDS = 600          # 10 minutos

    MIN_WIDTH = 640
    MIN_HEIGHT = 360

    MAX_WIDTH = 7680                    # 8K
    MAX_HEIGHT = 4320


# ============================================================
# Validador de URLs
# ============================================================

class URLValidator:
    """Valida URLs de Cloudinary y su accesibilidad"""

    @staticmethod
    def validate_cloudinary_url(url: str) -> bool:
        """
        Valida que la URL sea de Cloudinary y tenga formato correcto.

        Raises:
            InvalidURLError: Si la URL no es válida
        """
        if not url:
            raise InvalidURLError("URL vacía")

        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise InvalidURLError("URL malformada")
        except Exception as e:
            raise InvalidURLError(f"URL inválida: {str(e)}")

        if 'cloudinary.com' not in parsed.netloc:
            raise InvalidURLError(
                "La URL debe ser de Cloudinary (cloudinary.com). "
                f"Se recibió: {parsed.netloc}"
            )

        if '/video/' not in url and '/raw/' not in url:
            raise InvalidURLError(
                "La URL debe apuntar a un recurso de video en Cloudinary. "
                "Asegúrate de que contenga '/video/' en la ruta."
            )

        url_lower = url.lower()
        has_valid_extension = any(
            url_lower.endswith(f'.{fmt}')
            for fmt in VideoLimits.SUPPORTED_FORMATS
        )

        if not has_valid_extension:
            logger.warning(
                "URL sin extensión reconocida. Formatos soportados: %s",
                ', '.join(VideoLimits.SUPPORTED_FORMATS)
            )

        logger.info("URL de Cloudinary validada correctamente")
        return True

    @staticmethod
    def validate_url_accessible(url: str, timeout: int = 10) -> Tuple[bool, Optional[dict]]:
        """
        Verifica que la URL sea accesible haciendo un HEAD request.

        Returns:
            Tupla (es_accesible, headers_info)

        Raises:
            InvalidURLError: Si la URL no es accesible
        """
        try:
            logger.info("Verificando accesibilidad de URL...")

            response = requests.head(url, timeout=timeout, allow_redirects=True)

            if response.status_code != 200:
                raise InvalidURLError(
                    f"URL no accesible. Status code: {response.status_code}"
                )

            headers_info = {
                'content_type': response.headers.get('Content-Type', ''),
                'content_length': response.headers.get('Content-Length', '0'),
            }

            content_type = headers_info['content_type'].lower()
            if 'video' not in content_type and 'octet-stream' not in content_type:
                logger.warning(
                    "Content-Type inusual: %s. Se esperaba video/*",
                    content_type
                )

            logger.info("URL accesible. Content-Type: %s", headers_info['content_type'])

            return True, headers_info

        except requests.exceptions.Timeout:
            raise InvalidURLError(
                f"Timeout al acceder a la URL (>{timeout}s). "
                "Verifica que la URL sea correcta y accesible."
            )
        except requests.exceptions.ConnectionError:
            raise InvalidURLError(
                "No se pudo conectar a la URL. "
                "Verifica tu conexión a internet y que la URL sea correcta."
            )
        except requests.exceptions.RequestException as e:
            raise InvalidURLError(f"Error al acceder a la URL: {str(e)}")


# ============================================================
# Validador de Video
# ============================================================

class VideoValidator:
    """Valida propiedades del video"""

    @staticmethod
    def validate_video_size(size_bytes: int) -> bool:
        """
        Raises:
            VideoSizeError: Si excede el límite
        """
        if size_bytes <= 0:
            raise VideoSizeError("Tamaño de video inválido")

        if size_bytes > VideoLimits.MAX_SIZE_BYTES:
            size_mb = size_bytes / (1024 * 1024)
            max_mb = VideoLimits.MAX_SIZE_BYTES / (1024 * 1024)
            raise VideoSizeError(
                f"El video es demasiado grande: {size_mb:.2f}MB. "
                f"Tamaño máximo permitido: {max_mb:.0f}MB"
            )

        size_mb = size_bytes / (1024 * 1024)
        logger.info("Tamaño del video validado: %.2f MB", size_mb)

        return True

    @staticmethod
    def validate_video_duration(duration_seconds: float) -> bool:
        """
        Raises:
            VideoDurationError: Si está fuera de límites
        """
        if duration_seconds < VideoLimits.MIN_DURATION_SECONDS:
            raise VideoDurationError(
                f"El video es demasiado corto: {duration_seconds:.1f}s. "
                f"Duración mínima: {VideoLimits.MIN_DURATION_SECONDS}s"
            )

        if duration_seconds > VideoLimits.MAX_DURATION_SECONDS:
            raise VideoDurationError(
                f"El video es demasiado largo: {duration_seconds:.1f}s. "
                f"Duración máxima: {VideoLimits.MAX_DURATION_SECONDS}s "
                f"({VideoLimits.MAX_DURATION_SECONDS // 60} minutos)"
            )

        logger.info("Duración del video validada: %.2f segundos", duration_seconds)

        return True

    @staticmethod
    def validate_video_resolution(width: int, height: int) -> bool:
        """
        Raises:
            VideoFormatError: Si la resolución es inválida
        """
        if width < VideoLimits.MIN_WIDTH or height < VideoLimits.MIN_HEIGHT:
            raise VideoFormatError(
                f"Resolución muy baja: {width}x{height}. "
                f"Mínimo requerido: {VideoLimits.MIN_WIDTH}x{VideoLimits.MIN_HEIGHT}"
            )

        if width > VideoLimits.MAX_WIDTH or height > VideoLimits.MAX_HEIGHT:
            raise VideoFormatError(
                f"Resolución muy alta: {width}x{height}. "
                f"Máximo soportado: {VideoLimits.MAX_WIDTH}x{VideoLimits.MAX_HEIGHT}"
            )

        logger.info("Resolución validada: %dx%d", width, height)

        return True


# ============================================================
# Validador de opciones de short — NUEVO en Versión 2
# ============================================================

class ShortOptionsValidator:
    """
    Valida las opciones específicas de los modos short_auto y short_manual.

    Responsabilidad única: reglas de negocio de segmentos de video.
    Esta clase opera sobre datos ya parseados (floats/ints), no sobre
    el modelo completo, para facilitar el reuso y el testing.
    """

    @staticmethod
    def validate_short_auto(
        target_duration: int,
        video_duration: Optional[float] = None
    ) -> None:
        """
        Valida los parámetros del modo short_auto.

        Args:
            target_duration: Duración deseada del short en segundos.
            video_duration:  Duración real del video en segundos (opcional).
                             Si se provee, se valida que el video sea suficientemente largo.

        Raises:
            VideoDurationError: Si la duración del short está fuera de rango.
            VideoDurationError: Si el video es más corto que la duración mínima de un short.
        """
        # Validar rango de duración del short
        if not (SHORT_MIN_DURATION_SECONDS <= target_duration <= SHORT_MAX_DURATION_SECONDS):
            raise VideoDurationError(
                f"La duración del short debe estar entre "
                f"{SHORT_MIN_DURATION_SECONDS}s y {SHORT_MAX_DURATION_SECONDS}s. "
                f"Se recibió: {target_duration}s"
            )

        # Si tenemos la duración real del video, validar que sea suficiente
        if video_duration is not None:
            if video_duration < SHORT_MIN_DURATION_SECONDS:
                raise VideoDurationError(
                    f"El video es demasiado corto para generar un short: {video_duration:.1f}s. "
                    f"Mínimo requerido: {SHORT_MIN_DURATION_SECONDS}s"
                )

            # Si el video es más corto que el target, se usará la duración completa.
            # No es un error — lo manejará SegmentSelector en la fase de procesamiento.
            if video_duration < target_duration:
                logger.warning(
                    "El video (%0.1fs) es más corto que la duración objetivo del short (%ds). "
                    "Se usará la duración completa del video.",
                    video_duration,
                    target_duration
                )

        logger.info(
            "Opciones de short_auto validadas | target_duration=%ds | video_duration=%s",
            target_duration,
            f"{video_duration:.1f}s" if video_duration is not None else "desconocida"
        )

    @staticmethod
    def validate_short_manual(
        start_time: float,
        duration: int,
        video_duration: Optional[float] = None
    ) -> None:
        """
        Valida los parámetros del modo short_manual.

        Args:
            start_time:      Segundo de inicio del segmento.
            duration:        Duración del segmento en segundos.
            video_duration:  Duración real del video en segundos (opcional).
                             Si se provee, se valida que el segmento no exceda el video.

        Raises:
            ValidationError:     Si start_time es negativo.
            VideoDurationError:  Si la duración del segmento está fuera de rango.
            VideoDurationError:  Si el segmento excede la duración del video.
        """
        # start_time ya está validado como ge=0.0 en el schema,
        # pero hacemos segunda línea de defensa aquí
        if start_time < 0:
            raise ValidationError(
                f"El tiempo de inicio no puede ser negativo. Se recibió: {start_time}s"
            )

        # Rango de duración del segmento
        if not (SHORT_MIN_DURATION_SECONDS <= duration <= SHORT_MAX_DURATION_SECONDS):
            raise VideoDurationError(
                f"La duración del segmento debe estar entre "
                f"{SHORT_MIN_DURATION_SECONDS}s y {SHORT_MAX_DURATION_SECONDS}s. "
                f"Se recibió: {duration}s"
            )

        # Validar contra duración real del video (segunda línea de defensa en runtime)
        if video_duration is not None:
            end_time = start_time + duration

            if start_time >= video_duration:
                raise ValidationError(
                    f"El tiempo de inicio ({start_time}s) es mayor o igual "
                    f"a la duración del video ({video_duration:.1f}s). "
                    "Elige un tiempo de inicio más temprano."
                )

            if end_time > video_duration:
                available = video_duration - start_time
                raise VideoDurationError(
                    f"El segmento solicitado ({start_time}s + {duration}s = {end_time}s) "
                    f"excede la duración del video ({video_duration:.1f}s). "
                    f"Máximo disponible desde {start_time}s: {available:.1f}s. "
                    "Reduce la duración o elige un tiempo de inicio más temprano."
                )

        logger.info(
            "Opciones de short_manual validadas | start_time=%.1fs | duration=%ds | video_duration=%s",
            start_time,
            duration,
            f"{video_duration:.1f}s" if video_duration is not None else "desconocida"
        )


# ============================================================
# Validador de Request
# ============================================================

class RequestValidator:
    """Valida la solicitud completa del usuario"""

    @staticmethod
    def validate_request(request: VideoProcessRequest) -> bool:
        """
        Valida la coherencia de la solicitud completa.

        Usa isinstance() para acceder a campos específicos de cada modo,
        aprovechando la resolución del discriminador que ya hizo Pydantic.

        Raises:
            InvalidURLError:         Si la URL está vacía o es inválida.
            UnsupportedPlatformError: Si la combinación de opciones no es soportada.
        """
        if not request.cloudinary_input_url or not request.cloudinary_input_url.strip():
            raise InvalidURLError("La URL del video es requerida")

        URLValidator.validate_cloudinary_url(request.cloudinary_input_url)

        # Advertencia por calidad fast en producción
        if request.quality == QualityLevel.fast:
            logger.warning(
                "Calidad 'fast' seleccionada. "
                "Recomendado solo para pruebas, no para producción."
            )

        # Advertencia de performance para shorts con fondo difuminado
        if (
            isinstance(request, (ShortAutoRequest, ShortManualRequest))
            and request.background_mode == BackgroundMode.blurred
        ):
            logger.warning(
                "Modo '%s' con fondo 'blurred' puede aumentar el tiempo de procesamiento "
                "debido al paso de corte + blur + encoding.",
                request.processing_mode.value
            )

        logger.info(
            "Request validado | platform=%s | background_mode=%s | processing_mode=%s | quality=%s",
            request.platform.value,
            request.background_mode.value,
            request.processing_mode.value,
            request.quality.value
        )

        return True

    @staticmethod
    def validate_url_and_check_accessible(request: VideoProcessRequest) -> dict:
        """
        Valida la URL y verifica que sea accesible.

        Returns:
            Dict con información del video disponible desde los headers.
        """
        URLValidator.validate_cloudinary_url(request.cloudinary_input_url)

        is_accessible, headers_info = URLValidator.validate_url_accessible(
            request.cloudinary_input_url
        )

        video_info = {}

        if headers_info and 'content_length' in headers_info:
            try:
                size_bytes = int(headers_info['content_length'])
                VideoValidator.validate_video_size(size_bytes)
                video_info['size_bytes'] = size_bytes
                video_info['size_mb'] = size_bytes / (1024 * 1024)
            except (ValueError, TypeError):
                logger.warning("No se pudo obtener el tamaño del video desde headers")

        return video_info


# ============================================================
# Función principal de validación — punto de entrada del router
# ============================================================

def validate_video_request(request: VideoProcessRequest) -> dict:
    """
    Orquesta todas las validaciones para una solicitud de procesamiento.

    Gracias al discriminador de Pydantic, cuando esta función recibe `request`
    ya se sabe exactamente qué tipo concreto es (VerticalRequest, ShortAutoRequest
    o ShortManualRequest). Se usa isinstance() para acceder a campos específicos
    de forma segura y con inferencia correcta del IDE.

    La validación con duración real del video (start_time + duration <= total)
    no puede hacerse aquí porque el video aún no fue descargado. Se ejecuta
    en las estrategias de procesamiento después de la descarga.

    Args:
        request: Request ya parseado y discriminado por Pydantic.

    Returns:
        Dict con información del video disponible en este punto (tamaño si está en headers).

    Raises:
        ValidationError: Si alguna validación falla.
    """
    logger.info(
        "Iniciando validación | processing_mode=%s",
        request.processing_mode.value
    )

    # 1. Validar campos comunes y URL
    RequestValidator.validate_request(request)

    # 2. Verificar accesibilidad y obtener tamaño si está disponible en headers
    video_info = RequestValidator.validate_url_and_check_accessible(request)

    # 3. Validaciones específicas por tipo de request
    #    isinstance() es la forma idiomática de trabajar con Discriminated Union:
    #    el tipo ya fue resuelto por Pydantic, aquí solo lo aprovechamos.
    if isinstance(request, ShortAutoRequest):
        ShortOptionsValidator.validate_short_auto(
            target_duration=request.short_auto_duration,
            video_duration=None  # Se validará con duración real en la estrategia
        )

    elif isinstance(request, ShortManualRequest):
        ShortOptionsValidator.validate_short_manual(
            start_time=request.short_options.start_time,
            duration=request.short_options.duration,
            video_duration=None  # Se validará con duración real en la estrategia
        )

    logger.info(
        "Validación completada | processing_mode=%s | video_info=%s",
        request.processing_mode.value,
        video_info
    )

    return video_info
