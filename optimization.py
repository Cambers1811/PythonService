"""
Optimization Module — Mejora 7
Optimizaciones de rendimiento para procesamiento de video.
Ubicación: backend/optimization.py
"""

import os
import hashlib
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


# ============================================================
# Configuration Cache
# ============================================================

class ConfigurationCache:
    """
    Cache de configuraciones para evitar recalcular parámetros.
    """
    
    def __init__(self, cache_dir: str = "/tmp/video_processing/cache"):
        """
        Args:
            cache_dir: Directorio para almacenar cache
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("ConfigurationCache inicializado | dir=%s", cache_dir)
    
    
    def _get_cache_key(self, **params) -> str:
        """
        Genera una clave única basada en los parámetros.
        
        Args:
            **params: Parámetros de configuración
            
        Returns:
            Hash MD5 de los parámetros
        """
        # Ordenar params para consistencia
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()
    
    
    def get(self, **params) -> Optional[Dict[str, Any]]:
        """
        Obtiene configuración del cache.
        
        Args:
            **params: Parámetros de búsqueda
            
        Returns:
            Configuración cacheada o None
        """
        cache_key = self._get_cache_key(**params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    logger.debug("Cache hit | key=%s", cache_key[:8])
                    return data
            except Exception as e:
                logger.warning("Error leyendo cache: %s", str(e))
        
        logger.debug("Cache miss | key=%s", cache_key[:8])
        return None
    
    
    def set(self, data: Dict[str, Any], **params):
        """
        Guarda configuración en cache.
        
        Args:
            data: Datos a cachear
            **params: Parámetros clave
        """
        cache_key = self._get_cache_key(**params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.debug("Cache guardado | key=%s", cache_key[:8])
        except Exception as e:
            logger.warning("Error guardando cache: %s", str(e))
    
    
    def clear(self):
        """Limpia todo el cache"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cache limpiado completamente")
        except Exception as e:
            logger.warning("Error limpiando cache: %s", str(e))


# ============================================================
# Frame Sampling Optimizer
# ============================================================

class FrameSamplingOptimizer:
    """
    Optimiza el muestreo de frames para análisis.
    Detecta frames clave y evita procesar frames redundantes.
    """
    
    def __init__(self, threshold: float = 30.0):
        """
        Args:
            threshold: Umbral de diferencia entre frames (0-255)
        """
        self.threshold = threshold
        self.last_frame = None
        self.frames_skipped = 0
        self.frames_processed = 0
    
    
    def should_process_frame(self, frame) -> bool:
        """
        Determina si un frame debe procesarse basado en diferencia con anterior.
        
        Args:
            frame: Frame actual (numpy array)
            
        Returns:
            True si el frame debe procesarse
        """
        import cv2
        import numpy as np
        
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.frames_processed += 1
            return True
        
        # Calcular diferencia
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_current, gray_last)
        mean_diff = np.mean(diff)
        
        # Si la diferencia es significativa, procesar
        if mean_diff > self.threshold:
            self.last_frame = frame.copy()
            self.frames_processed += 1
            return True
        else:
            self.frames_skipped += 1
            return False
    
    
    def get_stats(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de optimización.
        
        Returns:
            Dict con stats
        """
        total = self.frames_processed + self.frames_skipped
        skip_rate = (self.frames_skipped / total * 100) if total > 0 else 0
        
        return {
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'skip_rate_percent': skip_rate
        }
    
    
    def reset(self):
        """Reinicia estadísticas"""
        self.last_frame = None
        self.frames_skipped = 0
        self.frames_processed = 0


# ============================================================
# Hardware Acceleration Detector
# ============================================================

class HardwareAccelerationDetector:
    """
    Detecta y configura aceleración por hardware disponible.
    """
    
    @staticmethod
    def detect_gpu_support() -> Dict[str, Any]:
        """
        Detecta soporte de GPU para encoding.
        
        Returns:
            Dict con info de hardware disponible
        """
        import subprocess
        
        hw_support = {
            'nvidia_cuda': False,
            'nvidia_nvenc': False,
            'intel_qsv': False,
            'amd_vaapi': False,
            'recommended_encoder': 'libx264'  # Default software
        }
        
        try:
            # Verificar NVIDIA
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                hw_support['nvidia_cuda'] = True
                hw_support['nvidia_nvenc'] = True
                hw_support['recommended_encoder'] = 'h264_nvenc'
                logger.info("NVIDIA GPU detectada - usando h264_nvenc")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # Verificar Intel QSV
            result = subprocess.run(
                ['vainfo'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if 'VAProfileH264' in result.stdout:
                hw_support['intel_qsv'] = True
                if not hw_support['nvidia_nvenc']:
                    hw_support['recommended_encoder'] = 'h264_qsv'
                    logger.info("Intel QSV detectado - usando h264_qsv")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        if hw_support['recommended_encoder'] == 'libx264':
            logger.info("No se detectó aceleración por hardware - usando libx264 (software)")
        
        return hw_support
    
    
    @staticmethod
    def get_optimized_ffmpeg_encoder() -> str:
        """
        Obtiene el mejor encoder disponible.
        
        Returns:
            Nombre del encoder a usar
        """
        hw_info = HardwareAccelerationDetector.detect_gpu_support()
        return hw_info['recommended_encoder']


# ============================================================
# Batch Processor
# ============================================================

class BatchProcessor:
    """
    Procesa múltiples operaciones en lote para mejor rendimiento.
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Args:
            batch_size: Tamaño del lote
        """
        self.batch_size = batch_size
        self.batch = []
    
    
    def add(self, item: Any):
        """
        Agrega item al lote.
        
        Args:
            item: Item a procesar
        """
        self.batch.append(item)
    
    
    def should_process(self) -> bool:
        """
        Determina si el lote debe procesarse.
        
        Returns:
            True si el lote está lleno
        """
        return len(self.batch) >= self.batch_size
    
    
    def get_batch(self) -> list:
        """
        Obtiene y limpia el lote.
        
        Returns:
            Lista de items en el lote
        """
        batch = self.batch.copy()
        self.batch = []
        return batch
    
    
    def flush(self) -> list:
        """
        Fuerza el procesamiento del lote actual.
        
        Returns:
            Items restantes
        """
        return self.get_batch()


# ============================================================
# Performance Monitor
# ============================================================

class PerformanceMonitor:
    """
    Monitorea métricas de rendimiento del procesamiento.
    """
    
    def __init__(self):
        """Inicializa monitor"""
        self.metrics = {
            'total_processing_time': 0.0,
            'analysis_time': 0.0,
            'encoding_time': 0.0,
            'upload_time': 0.0,
            'frames_analyzed': 0,
            'frames_skipped': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'hw_acceleration_used': False
        }
    
    
    def record_metric(self, metric_name: str, value: float):
        """
        Registra una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor a registrar
        """
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
        else:
            self.metrics[metric_name] = value
    
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de métricas.
        
        Returns:
            Dict con todas las métricas
        """
        summary = self.metrics.copy()
        
        # Calcular métricas derivadas
        if summary['frames_analyzed'] > 0:
            total_frames = summary['frames_analyzed'] + summary['frames_skipped']
            summary['optimization_rate'] = (
                summary['frames_skipped'] / total_frames * 100
            )
        
        if summary['cache_hits'] + summary['cache_misses'] > 0:
            summary['cache_hit_rate'] = (
                summary['cache_hits'] / 
                (summary['cache_hits'] + summary['cache_misses']) * 100
            )
        
        return summary
    
    
    def log_summary(self):
        """Loguea resumen de métricas"""
        summary = self.get_summary()
        
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info("Total time: %.2fs", summary['total_processing_time'])
        logger.info("  Analysis: %.2fs", summary['analysis_time'])
        logger.info("  Encoding: %.2fs", summary['encoding_time'])
        logger.info("  Upload: %.2fs", summary['upload_time'])
        
        if 'optimization_rate' in summary:
            logger.info("Frames skipped: %d (%.1f%%)", 
                       summary['frames_skipped'],
                       summary['optimization_rate'])
        
        if 'cache_hit_rate' in summary:
            logger.info("Cache hit rate: %.1f%%", summary['cache_hit_rate'])
        
        if summary['hw_acceleration_used']:
            logger.info("Hardware acceleration: ENABLED")


# ============================================================
# Singleton instances
# ============================================================

_config_cache: Optional[ConfigurationCache] = None
_performance_monitor: Optional[PerformanceMonitor] = None


def get_config_cache() -> ConfigurationCache:
    """Obtiene instancia global del cache"""
    global _config_cache
    if _config_cache is None:
        _config_cache = ConfigurationCache()
    return _config_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Obtiene instancia global del monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
