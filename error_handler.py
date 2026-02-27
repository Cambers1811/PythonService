"""
Error Handler — Mejora 2
Manejo centralizado y robusto de errores.
Ubicación: backend/error_handler.py
"""

import logging
import traceback
import time
from typing import Optional, Callable, Any
from functools import wraps

from exceptions import (
    VideoProcessingError,
    ValidationError,
    CloudinaryError,
    VideoFormatError,
    VideoSizeError,
    VideoDurationError,
    InvalidURLError
)


logger = logging.getLogger(__name__)


# ============================================================
# Categorías de errores
# ============================================================

class ErrorCategory:
    """Categorías de errores para clasificación"""
    
    VALIDATION = "validation"           # Error de validación de input
    NETWORK = "network"                 # Error de red/conectividad
    CLOUDINARY = "cloudinary"           # Error de Cloudinary
    PROCESSING = "processing"           # Error durante procesamiento
    STORAGE = "storage"                 # Error de almacenamiento local
    SYSTEM = "system"                   # Error del sistema
    UNKNOWN = "unknown"                 # Error desconocido


# ============================================================
# Clasificador de errores
# ============================================================

class ErrorClassifier:
    """Clasifica errores en categorías para mejor manejo"""
    
    @staticmethod
    def classify_error(error: Exception) -> str:
        """
        Clasifica un error en una categoría.
        
        Args:
            error: Excepción a clasificar
            
        Returns:
            Categoría del error
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Validación
        if isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION
        
        # Red
        if any(x in error_type.lower() for x in ['timeout', 'connection', 'network']):
            return ErrorCategory.NETWORK
        
        # Cloudinary
        if isinstance(error, CloudinaryError) or 'cloudinary' in error_msg:
            return ErrorCategory.CLOUDINARY
        
        # Procesamiento
        if isinstance(error, VideoProcessingError):
            return ErrorCategory.PROCESSING
        
        # Almacenamiento
        if any(x in error_type.lower() for x in ['filenotfound', 'permission', 'ioerror']):
            return ErrorCategory.STORAGE
        
        # Sistema
        if any(x in error_type.lower() for x in ['memory', 'system', 'os']):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """
        Determina si un error es retryable (temporal).
        
        Args:
            error: Excepción a evaluar
            
        Returns:
            True si se puede reintentar
        """
        category = ErrorClassifier.classify_error(error)
        
        # Errores de red y Cloudinary son típicamente retryables
        if category in [ErrorCategory.NETWORK, ErrorCategory.CLOUDINARY]:
            return True
        
        # Errores de validación NO son retryables
        if category == ErrorCategory.VALIDATION:
            return False
        
        # Algunos errores de sistema son retryables
        error_msg = str(error).lower()
        if any(x in error_msg for x in ['timeout', 'temporary', 'busy']):
            return True
        
        return False


# ============================================================
# Generador de mensajes amigables
# ============================================================

class ErrorMessageGenerator:
    """Genera mensajes de error amigables para el usuario"""
    
    @staticmethod
    def get_user_friendly_message(error: Exception, category: str) -> str:
        """
        Genera un mensaje amigable según el tipo de error.
        
        Args:
            error: Excepción original
            category: Categoría del error
            
        Returns:
            Mensaje amigable para el usuario
        """
        
        if category == ErrorCategory.VALIDATION:
            return str(error)  # Los errores de validación ya son amigables
        
        if category == ErrorCategory.NETWORK:
            return (
                "Error de conexión. Por favor verifica tu internet "
                "e intenta de nuevo en unos momentos."
            )
        
        if category == ErrorCategory.CLOUDINARY:
            if 'not found' in str(error).lower():
                return (
                    "No se pudo encontrar el video en Cloudinary. "
                    "Verifica que la URL sea correcta."
                )
            if 'unauthorized' in str(error).lower():
                return (
                    "Error de autenticación con Cloudinary. "
                    "Contacta al administrador."
                )
            return (
                "Error al comunicarse con Cloudinary. "
                "Por favor intenta de nuevo más tarde."
            )
        
        if category == ErrorCategory.PROCESSING:
            if 'face' in str(error).lower() or 'rostro' in str(error).lower():
                return (
                    "No se pudo detectar un rostro en el video. "
                    "Para el modo 'smart_crop' se requiere que haya al menos "
                    "un rostro visible en el video."
                )
            if 'codec' in str(error).lower() or 'format' in str(error).lower():
                return (
                    "Error al procesar el video. "
                    "El formato del video podría no ser compatible."
                )
            return (
                "Error durante el procesamiento del video. "
                "Por favor verifica que el video sea válido e intenta de nuevo."
            )
        
        if category == ErrorCategory.STORAGE:
            return (
                "Error de almacenamiento temporal. "
                "Puede que el servidor esté sin espacio. Intenta más tarde."
            )
        
        if category == ErrorCategory.SYSTEM:
            return (
                "Error del sistema. "
                "Nuestro equipo ha sido notificado. Intenta más tarde."
            )
        
        # Error desconocido
        return (
            "Ocurrió un error inesperado durante el procesamiento. "
            "Por favor intenta de nuevo. Si el problema persiste, contacta soporte."
        )


# ============================================================
# Retry decorator
# ============================================================

def retry_on_failure(
    max_attempts: int = 3,
    delay_seconds: float = 2.0,
    backoff_multiplier: float = 2.0,
    only_if_retryable: bool = True
):
    """
    Decorator para reintentar operaciones que fallan.
    
    Args:
        max_attempts: Número máximo de intentos
        delay_seconds: Delay inicial entre intentos
        backoff_multiplier: Multiplicador para backoff exponencial
        only_if_retryable: Solo reintentar si el error es retryable
    
    Example:
        @retry_on_failure(max_attempts=3, delay_seconds=1)
        def download_video(url):
            # código que puede fallar
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    # Verificar si es retryable
                    if only_if_retryable and not ErrorClassifier.is_retryable(e):
                        logger.warning(
                            "Error no retryable en %s: %s",
                            func.__name__,
                            str(e)
                        )
                        raise
                    
                    # Si es el último intento, lanzar el error
                    if attempt == max_attempts:
                        logger.error(
                            "Falló después de %d intentos: %s",
                            max_attempts,
                            func.__name__
                        )
                        raise
                    
                    # Log y esperar antes del siguiente intento
                    logger.warning(
                        "Intento %d/%d falló en %s: %s. Reintentando en %.1fs...",
                        attempt,
                        max_attempts,
                        func.__name__,
                        str(e),
                        delay
                    )
                    
                    time.sleep(delay)
                    delay *= backoff_multiplier
            
            # No debería llegar aquí, pero por si acaso
            if last_error:
                raise last_error
        
        return wrapper
    return decorator


# ============================================================
# Error context manager
# ============================================================

class ErrorContext:
    """
    Context manager para manejar errores con cleanup automático.
    
    Example:
        with ErrorContext("descarga de video", cleanup_func=cleanup):
            download_video(url)
    """
    
    def __init__(
        self,
        operation_name: str,
        cleanup_func: Optional[Callable] = None,
        job_id: Optional[str] = None
    ):
        """
        Args:
            operation_name: Nombre de la operación
            cleanup_func: Función a llamar en caso de error
            job_id: ID del job (para logging)
        """
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.job_id = job_id
    
    def __enter__(self):
        logger.info(
            "Iniciando operación: %s%s",
            self.operation_name,
            f" (Job: {self.job_id})" if self.job_id else ""
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Operación exitosa
            logger.info("Operación completada: %s", self.operation_name)
            return False
        
        # Ocurrió un error
        logger.error(
            "Error en operación '%s'%s: %s",
            self.operation_name,
            f" (Job: {self.job_id})" if self.job_id else "",
            str(exc_val)
        )
        
        # Ejecutar cleanup si se proveyó
        if self.cleanup_func:
            try:
                logger.info("Ejecutando cleanup para: %s", self.operation_name)
                self.cleanup_func()
                logger.info("Cleanup completado")
            except Exception as cleanup_error:
                logger.error(
                    "Error durante cleanup: %s",
                    str(cleanup_error)
                )
        
        # No suprimir la excepción
        return False


# ============================================================
# Handler principal
# ============================================================

class ErrorHandler:
    """Handler principal para procesamiento de errores"""
    
    @staticmethod
    def handle_error(
        error: Exception,
        job_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> dict:
        """
        Maneja un error y retorna información estructurada.
        
        Args:
            error: Excepción a manejar
            job_id: ID del job (opcional)
            operation: Nombre de la operación (opcional)
            
        Returns:
            Dict con información del error
        """
        # Clasificar error
        category = ErrorClassifier.classify_error(error)
        is_retryable = ErrorClassifier.is_retryable(error)
        
        # Generar mensaje amigable
        user_message = ErrorMessageGenerator.get_user_friendly_message(
            error,
            category
        )
        
        # Log completo
        logger.error(
            "Error manejado | job_id=%s | operation=%s | category=%s | retryable=%s | error=%s",
            job_id or "N/A",
            operation or "N/A",
            category,
            is_retryable,
            str(error)
        )
        
        # Log traceback solo en desarrollo
        logger.debug("Traceback: %s", traceback.format_exc())
        
        return {
            'error_type': type(error).__name__,
            'error_category': category,
            'is_retryable': is_retryable,
            'user_message': user_message,
            'technical_message': str(error),
            'job_id': job_id
        }
