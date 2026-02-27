"""
Cancellation Manager — Mejora 4
Sistema para cancelar jobs en proceso de forma segura.
Ubicación: backend/cancellation_manager.py
"""

import threading
import logging
from typing import Set, Optional


logger = logging.getLogger(__name__)


# ============================================================
# Cancellation Manager
# ============================================================

class CancellationManager:
    """
    Gestiona las cancelaciones de jobs de forma thread-safe.
    
    Los workers consultan periódicamente si un job ha sido cancelado.
    """
    
    def __init__(self):
        """Inicializa el manager con un set thread-safe"""
        self._cancelled_jobs: Set[str] = set()
        self._lock = threading.Lock()
    
    
    def request_cancellation(self, job_id: str):
        """
        Solicita la cancelación de un job.
        
        Args:
            job_id: ID del job a cancelar
        """
        with self._lock:
            if job_id not in self._cancelled_jobs:
                self._cancelled_jobs.add(job_id)
                logger.info("Cancelación solicitada | job_id=%s", job_id)
            else:
                logger.warning("Job ya estaba marcado para cancelación | job_id=%s", job_id)
    
    
    def is_cancelled(self, job_id: str) -> bool:
        """
        Verifica si un job ha sido cancelado.
        
        Args:
            job_id: ID del job a verificar
            
        Returns:
            True si el job fue cancelado
        """
        with self._lock:
            return job_id in self._cancelled_jobs
    
    
    def remove_cancellation(self, job_id: str):
        """
        Remueve un job del set de cancelados.
        Útil después de que el cleanup se completó.
        
        Args:
            job_id: ID del job
        """
        with self._lock:
            if job_id in self._cancelled_jobs:
                self._cancelled_jobs.discard(job_id)
                logger.info("Job removido de cancelaciones | job_id=%s", job_id)
    
    
    def get_cancelled_jobs(self) -> list:
        """
        Obtiene la lista de jobs cancelados.
        
        Returns:
            Lista de job IDs cancelados
        """
        with self._lock:
            return list(self._cancelled_jobs)
    
    
    def clear_all(self):
        """Limpia todos los jobs cancelados (útil para testing)"""
        with self._lock:
            count = len(self._cancelled_jobs)
            self._cancelled_jobs.clear()
            logger.info("Todos los jobs cancelados limpiados | count=%d", count)


# ============================================================
# Exception para cancelación
# ============================================================

class JobCancelledException(Exception):
    """Excepción lanzada cuando un job es cancelado"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job {job_id} fue cancelado")


# ============================================================
# Decorator para funciones cancelables
# ============================================================

def check_cancellation(cancellation_manager: CancellationManager, job_id: str):
    """
    Verifica si el job ha sido cancelado y lanza excepción si es así.
    
    Args:
        cancellation_manager: Manager de cancelaciones
        job_id: ID del job
        
    Raises:
        JobCancelledException: Si el job fue cancelado
    """
    if cancellation_manager.is_cancelled(job_id):
        logger.warning("Job cancelado detectado | job_id=%s", job_id)
        raise JobCancelledException(job_id)


# ============================================================
# Context manager para operaciones cancelables
# ============================================================

class CancellableOperation:
    """
    Context manager que verifica cancelación al entrar y salir.
    
    Example:
        with CancellableOperation(manager, job_id, "descarga"):
            download_video()  # Si se cancela, se lanza excepción
    """
    
    def __init__(
        self,
        cancellation_manager: CancellationManager,
        job_id: str,
        operation_name: str
    ):
        """
        Args:
            cancellation_manager: Manager de cancelaciones
            job_id: ID del job
            operation_name: Nombre de la operación (para logs)
        """
        self.manager = cancellation_manager
        self.job_id = job_id
        self.operation_name = operation_name
    
    
    def __enter__(self):
        """Verifica cancelación al entrar"""
        check_cancellation(self.manager, self.job_id)
        logger.debug(
            "Iniciando operación cancelable | job_id=%s | operation=%s",
            self.job_id,
            self.operation_name
        )
        return self
    
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Verifica cancelación al salir"""
        # Si ya hay otra excepción, no verificar cancelación
        if exc_type is not None:
            return False
        
        # Verificar una última vez
        check_cancellation(self.manager, self.job_id)
        
        logger.debug(
            "Operación cancelable completada | job_id=%s | operation=%s",
            self.job_id,
            self.operation_name
        )
        
        return False


# ============================================================
# Instancia global (singleton)
# ============================================================

_global_cancellation_manager: Optional[CancellationManager] = None


def get_cancellation_manager() -> CancellationManager:
    """
    Obtiene la instancia global del CancellationManager.
    Crea una si no existe (singleton).
    
    Returns:
        Instancia global de CancellationManager
    """
    global _global_cancellation_manager
    
    if _global_cancellation_manager is None:
        _global_cancellation_manager = CancellationManager()
        logger.info("CancellationManager global inicializado")
    
    return _global_cancellation_manager


# ============================================================
# Helper para integración con ProgressTracker
# ============================================================

class CancellableProgressTracker:
    """
    Wrapper del ProgressTracker que verifica cancelación en cada update.
    """
    
    def __init__(self, progress_tracker, cancellation_manager, job_id):
        """
        Args:
            progress_tracker: ProgressTracker original
            cancellation_manager: Manager de cancelaciones
            job_id: ID del job
        """
        self.tracker = progress_tracker
        self.manager = cancellation_manager
        self.job_id = job_id
    
    
    def update_phase(self, phase, message=None, metadata=None):
        """
        Actualiza fase pero verifica cancelación primero.
        
        Raises:
            JobCancelledException: Si el job fue cancelado
        """
        check_cancellation(self.manager, self.job_id)
        self.tracker.update_phase(phase, message, metadata)
    
    
    def update_progress(self, percentage, message=None):
        """
        Actualiza progreso pero verifica cancelación primero.
        
        Raises:
            JobCancelledException: Si el job fue cancelado
        """
        check_cancellation(self.manager, self.job_id)
        self.tracker.update_progress(percentage, message)
    
    
    def update_frames(self, frames_processed, total_frames):
        """
        Actualiza frames pero verifica cancelación primero.
        
        Raises:
            JobCancelledException: Si el job fue cancelado
        """
        check_cancellation(self.manager, self.job_id)
        self.tracker.update_frames(frames_processed, total_frames)
    
    
    def complete(self, success=True):
        """Completa el tracking (sin verificar cancelación)"""
        self.tracker.complete(success)
    
    
    def get_status(self):
        """Obtiene el estado (sin verificar cancelación)"""
        return self.tracker.get_status()
    
    
    # Exponer atributos del tracker original
    @property
    def current_phase(self):
        return self.tracker.current_phase
    
    @property
    def progress_percentage(self):
        return self.tracker.progress_percentage
    
    @property
    def metadata(self):
        return self.tracker.metadata
