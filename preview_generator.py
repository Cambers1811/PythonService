"""
Preview Generator — Mejora 6
Genera thumbnails y previews de videos procesados.
Ubicación: backend/preview_generator.py
"""

import cv2
import os
import subprocess
import logging
from typing import Optional, Tuple
from pathlib import Path


logger = logging.getLogger(__name__)


# ============================================================
# Preview Generator
# ============================================================

class PreviewGenerator:
    """
    Genera thumbnails y previews de videos.
    """
    
    def __init__(self, temp_dir: str = "/tmp/video_processing"):
        """
        Args:
            temp_dir: Directorio temporal para archivos
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    
    def generate_thumbnail(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        timestamp_seconds: float = 1.0,
        width: int = 480
    ) -> str:
        """
        Genera un thumbnail (imagen) del video.
        
        Args:
            video_path: Path del video
            output_path: Path de salida (opcional, se genera automáticamente)
            timestamp_seconds: Segundo del video para capturar (default: 1.0)
            width: Ancho del thumbnail (mantiene aspect ratio)
        
        Returns:
            Path del thumbnail generado
        
        Raises:
            Exception: Si falla la generación
        """
        try:
            logger.info("Generando thumbnail | video=%s | timestamp=%.1fs", video_path, timestamp_seconds)
            
            # Generar path de salida si no se provee
            if not output_path:
                base_name = Path(video_path).stem
                output_path = os.path.join(self.temp_dir, f"{base_name}_thumb.jpg")
            
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"No se pudo abrir el video: {video_path}")
            
            # Obtener FPS y calcular frame
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp_seconds * fps)
            
            # Ir al frame deseado
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Leer frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception(f"No se pudo leer el frame en {timestamp_seconds}s")
            
            # Redimensionar manteniendo aspect ratio
            h, w = frame.shape[:2]
            aspect = w / h
            new_w = width
            new_h = int(new_w / aspect)
            
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Guardar
            cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            file_size_kb = os.path.getsize(output_path) / 1024
            logger.info(
                "Thumbnail generado | output=%s | size=%dx%d | file_size=%.1f KB",
                output_path,
                new_w,
                new_h,
                file_size_kb
            )
            
            return output_path
            
        except Exception as e:
            logger.error("Error generando thumbnail: %s", str(e))
            raise
    
    
    def generate_preview_clip(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        duration_seconds: int = 5,
        start_time: float = 0.0
    ) -> str:
        """
        Genera un clip corto del video (preview).
        
        Args:
            video_path: Path del video original
            output_path: Path de salida (opcional)
            duration_seconds: Duración del preview en segundos
            start_time: Tiempo de inicio en segundos
        
        Returns:
            Path del preview generado
        
        Raises:
            Exception: Si falla la generación
        """
        try:
            logger.info(
                "Generando preview clip | video=%s | duration=%ds | start=%.1fs",
                video_path,
                duration_seconds,
                start_time
            )
            
            # Generar path de salida
            if not output_path:
                base_name = Path(video_path).stem
                output_path = os.path.join(self.temp_dir, f"{base_name}_preview.mp4")
            
            # Comando FFmpeg para extraer clip
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration_seconds),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]
            
            # Ejecutar
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(
                "Preview generado | output=%s | size=%.2f MB",
                output_path,
                file_size_mb
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error("Error en FFmpeg generando preview: %s", e.stderr[-500:])
            raise Exception("No se pudo generar el preview")
        except Exception as e:
            logger.error("Error generando preview: %s", str(e))
            raise
    
    
    def generate_comparison(
        self,
        original_path: str,
        processed_path: str,
        output_path: Optional[str] = None,
        timestamp_seconds: float = 1.0,
        width: int = 960
    ) -> str:
        """
        Genera una imagen de comparación lado a lado (before/after).
        
        Args:
            original_path: Path del video original
            processed_path: Path del video procesado
            output_path: Path de salida (opcional)
            timestamp_seconds: Segundo para capturar
            width: Ancho total de la imagen (se divide en 2)
        
        Returns:
            Path de la imagen de comparación
        
        Raises:
            Exception: Si falla la generación
        """
        try:
            logger.info(
                "Generando comparación | original=%s | processed=%s",
                original_path,
                processed_path
            )
            
            # Generar path de salida
            if not output_path:
                base_name = Path(processed_path).stem
                output_path = os.path.join(self.temp_dir, f"{base_name}_comparison.jpg")
            
            # Capturar frames de ambos videos
            frame_original = self._capture_frame(original_path, timestamp_seconds)
            frame_processed = self._capture_frame(processed_path, timestamp_seconds)
            
            # Redimensionar ambos al mismo tamaño
            half_width = width // 2
            
            h_orig, w_orig = frame_original.shape[:2]
            aspect_orig = w_orig / h_orig
            h1 = int(half_width / aspect_orig)
            resized_orig = cv2.resize(frame_original, (half_width, h1))
            
            h_proc, w_proc = frame_processed.shape[:2]
            aspect_proc = w_proc / h_proc
            h2 = int(half_width / aspect_proc)
            resized_proc = cv2.resize(frame_processed, (half_width, h2))
            
            # Asegurar que tengan la misma altura
            final_height = min(h1, h2)
            resized_orig = cv2.resize(resized_orig, (half_width, final_height))
            resized_proc = cv2.resize(resized_proc, (half_width, final_height))
            
            # Agregar labels
            import numpy as np
            
            # Label "ORIGINAL"
            label_height = 40
            label_orig = np.zeros((label_height, half_width, 3), dtype=np.uint8)
            cv2.putText(
                label_orig,
                "ORIGINAL",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Label "PROCESSED"
            label_proc = np.zeros((label_height, half_width, 3), dtype=np.uint8)
            cv2.putText(
                label_proc,
                "PROCESSED",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Combinar labels con frames
            left_side = np.vstack([label_orig, resized_orig])
            right_side = np.vstack([label_proc, resized_proc])
            
            # Combinar lado a lado
            comparison = np.hstack([left_side, right_side])
            
            # Guardar
            cv2.imwrite(output_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            file_size_kb = os.path.getsize(output_path) / 1024
            logger.info(
                "Comparación generada | output=%s | size=%.1f KB",
                output_path,
                file_size_kb
            )
            
            return output_path
            
        except Exception as e:
            logger.error("Error generando comparación: %s", str(e))
            raise
    
    
    def _capture_frame(self, video_path: str, timestamp_seconds: float) -> any:
        """
        Captura un frame de un video.
        
        Args:
            video_path: Path del video
            timestamp_seconds: Segundo a capturar
        
        Returns:
            Frame capturado (numpy array)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception(f"No se pudo capturar frame en {timestamp_seconds}s")
        
        return frame
    
    
    def cleanup(self, *file_paths):
        """
        Elimina archivos de preview/thumbnail.
        
        Args:
            *file_paths: Paths de archivos a eliminar
        """
        for path in file_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug("Preview eliminado: %s", path)
            except Exception as e:
                logger.warning("Error eliminando preview %s: %s", path, str(e))


# ============================================================
# Factory function
# ============================================================

def create_preview_generator(temp_dir: str = "/tmp/video_processing") -> PreviewGenerator:
    """
    Crea una instancia de PreviewGenerator.
    
    Args:
        temp_dir: Directorio temporal
    
    Returns:
        PreviewGenerator configurado
    """
    return PreviewGenerator(temp_dir)
