"""
Custom Exceptions — Mejora 1
Excepciones personalizadas para mejor manejo de errores.
Ubicación: backend/exceptions.py
"""


class VideoProcessingError(Exception):
    """Excepción base para errores de procesamiento de video"""
    pass


class ValidationError(VideoProcessingError):
    """Error en validación de inputs"""
    pass


class CloudinaryError(VideoProcessingError):
    """Error relacionado con Cloudinary"""
    pass


class VideoFormatError(ValidationError):
    """Formato de video no soportado"""
    pass


class VideoSizeError(ValidationError):
    """Video excede límites de tamaño"""
    pass


class VideoDurationError(ValidationError):
    """Duración del video fuera de límites"""
    pass


class InvalidURLError(ValidationError):
    """URL inválida o no accesible"""
    pass


class UnsupportedPlatformError(ValidationError):
    """Combinación de opciones no soportada"""
    pass
