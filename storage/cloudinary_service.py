"""
Cloudinary Service — Tarea 2
Maneja la subida, descarga y gestión de videos en Cloudinary.
"""

import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import requests
import logging
from typing import Optional


logger = logging.getLogger("video-processor-api.cloudinary")


class CloudinaryService:
    """
    Servicio para interactuar con Cloudinary.
    Maneja el upload y download de videos para el procesamiento.
    """

    def __init__(
        self,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        temp_dir: str = "/tmp/video_processing"
    ):
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )

        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        logger.info("CloudinaryService inicializado | temp_dir=%s", temp_dir)

    def download_video(self, cloudinary_url: str, job_id: str) -> str:
        try:
            local_filename = f"{job_id}_input.mp4"
            local_path = os.path.join(self.temp_dir, local_filename)

            logger.info(
                "Descargando video desde Cloudinary | job_id=%s | url=%s",
                job_id,
                cloudinary_url
            )

            response = requests.get(cloudinary_url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)

            logger.info(
                "Video descargado correctamente | job_id=%s | path=%s | size_mb=%.2f",
                job_id,
                local_path,
                file_size_mb
            )

            return local_path

        except Exception as e:
            logger.error(
                "Error descargando video | job_id=%s | url=%s",
                job_id,
                cloudinary_url,
                exc_info=True
            )
            raise Exception(f"No se pudo descargar el video: {str(e)}")
    
    def _compress_video(
        self,
        input_path: str,
        target_size_mb: float,
        job_id: str,
        max_attempts: int = 3
    ) -> str:
        """
        Comprime un video para que quepa dentro del límite de Cloudinary.
        
        Usa estrategia progresiva:
        1. Intento 1: CRF 23 (calidad buena)
        2. Intento 2: CRF 26 (calidad media-alta)
        3. Intento 3: CRF 28 (calidad media)
        
        Args:
            input_path: Path del video original
            target_size_mb: Tamaño objetivo en MB
            job_id: ID del job
            max_attempts: Máximo de intentos
            
        Returns:
            Path del video comprimido
            
        Raises:
            Exception: Si no logra comprimir después de max_attempts
        """
        import subprocess
        
        original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        logger.warning(
            "Video excede límite de Cloudinary | job_id=%s | "
            "size_actual=%.2f MB | limit=%.2f MB | "
            "Comprimiendo automáticamente...",
            job_id,
            original_size_mb,
            target_size_mb
        )
        
        # CRF values: menor = mejor calidad pero más tamaño
        crf_values = [23, 26, 28]
        
        # Timeout adaptativo basado en tamaño del video
        # Base: 6 segundos por MB (177 MB = ~18 minutos)
        # Mínimo: 10 minutos
        base_timeout = max(600, int(original_size_mb * 6))
        
        for attempt in range(max_attempts):
            crf = crf_values[min(attempt, len(crf_values) - 1)]
            
            # Timeout progresivo: incrementa 30% en cada intento
            # Intento 1: base_timeout
            # Intento 2: base_timeout * 1.3
            # Intento 3: base_timeout * 1.69
            timeout_multiplier = 1.0 + (attempt * 0.3)
            timeout_seconds = int(base_timeout * timeout_multiplier)
            
            compressed_filename = f"{job_id}_compressed_attempt{attempt+1}.mp4"
            compressed_path = os.path.join(self.temp_dir, compressed_filename)
            
            logger.info(
                "Intento de compresión %d/%d | job_id=%s | CRF=%d | timeout=%ds (%.1f min)",
                attempt + 1,
                max_attempts,
                job_id,
                crf,
                timeout_seconds,
                timeout_seconds / 60
            )
            
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", input_path,
                    "-c:v", "libx264",
                    "-crf", str(crf),
                    "-preset", "medium",
                    "-profile:v", "main",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-c:a", "aac",
                    "-b:a", "96k",  # Audio más comprimido
                    compressed_path
                ]
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds  # Timeout adaptativo y progresivo
                )
                
                compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                reduction_percent = ((original_size_mb - compressed_size_mb) / original_size_mb) * 100
                
                logger.info(
                    "Compresión completada | job_id=%s | "
                    "size_original=%.2f MB | size_compressed=%.2f MB | "
                    "reducción=%.1f%% | CRF=%d",
                    job_id,
                    original_size_mb,
                    compressed_size_mb,
                    reduction_percent,
                    crf
                )
                
                # Verificar si cumple el objetivo
                if compressed_size_mb <= target_size_mb:
                    logger.info(
                        "✅ Compresión exitosa | job_id=%s | "
                        "Archivo cumple límite (%.2f MB <= %.2f MB)",
                        job_id,
                        compressed_size_mb,
                        target_size_mb
                    )
                    return compressed_path
                else:
                    logger.warning(
                        "⚠️ Aún excede límite | job_id=%s | "
                        "size=%.2f MB > target=%.2f MB | "
                        "Intentando con mayor compresión...",
                        job_id,
                        compressed_size_mb,
                        target_size_mb
                    )
                    # Limpiar intento fallido
                    if os.path.exists(compressed_path):
                        os.remove(compressed_path)
                    
            except subprocess.TimeoutExpired:
                logger.error(
                    "Timeout durante compresión | job_id=%s | attempt=%d",
                    job_id,
                    attempt + 1
                )
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
                    
            except Exception as e:
                logger.error(
                    "Error durante compresión | job_id=%s | attempt=%d | error=%s",
                    job_id,
                    attempt + 1,
                    str(e)
                )
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
        
        # Si llegamos aquí, no logramos comprimir suficiente
        raise Exception(
            f"No se pudo comprimir el video a {target_size_mb} MB después de "
            f"{max_attempts} intentos. Tamaño original: {original_size_mb:.2f} MB. "
            "Considera usar un video más corto o de menor resolución."
        )

    def upload_video(
        self,
        local_path: str,
        job_id: str,
        folder: str = "processed_videos",
        cloudinary_limit_mb: float = 100.0
    ) -> str:
        """
        Sube un video a Cloudinary con compresión automática si excede el límite.
        
        Flujo:
        1. Verifica tamaño del video
        2. Si > cloudinary_limit_mb → comprime automáticamente
        3. Usa chunked upload si es necesario
        4. Sube a Cloudinary
        
        Args:
            local_path: Path local del video
            job_id: ID del job
            folder: Carpeta en Cloudinary
            cloudinary_limit_mb: Límite del plan de Cloudinary (default: 100 MB)
            
        Returns:
            URL segura del video en Cloudinary
            
        Raises:
            Exception: Si falla la subida o compresión
        """
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Archivo no encontrado: {local_path}")

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            original_path = local_path
            was_compressed = False
            
            # ══════════════════════════════════════════════════════════
            # PASO 1: Verificar si necesita compresión
            # ══════════════════════════════════════════════════════════
            # Dejamos margen de 5 MB para evitar casos límite
            if file_size_mb > (cloudinary_limit_mb - 5):
                logger.warning(
                    "Video excede límite de Cloudinary | job_id=%s | "
                    "size=%.2f MB | limit=%.2f MB",
                    job_id,
                    file_size_mb,
                    cloudinary_limit_mb
                )
                
                # Comprimir a un tamaño seguro (90% del límite)
                target_size = cloudinary_limit_mb * 0.90
                
                try:
                    local_path = self._compress_video(
                        original_path,
                        target_size_mb=target_size,
                        job_id=job_id,
                        max_attempts=3
                    )
                    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    was_compressed = True
                    
                    logger.info(
                        "Video comprimido exitosamente | job_id=%s | "
                        "size_final=%.2f MB",
                        job_id,
                        file_size_mb
                    )
                    
                except Exception as compress_error:
                    logger.error(
                        "Fallo la compresión automática | job_id=%s | error=%s",
                        job_id,
                        str(compress_error)
                    )
                    raise Exception(
                        f"El video procesado es demasiado grande ({file_size_mb:.2f} MB) "
                        f"y no se pudo comprimir automáticamente. "
                        f"Límite de Cloudinary: {cloudinary_limit_mb} MB. "
                        f"Error: {str(compress_error)}"
                    )
            
            # ══════════════════════════════════════════════════════════
            # PASO 2: Determinar método de upload
            # ══════════════════════════════════════════════════════════
            # Usar chunked upload solo si está cerca del límite
            use_chunked = file_size_mb > 50  # Threshold: 50 MB

            logger.info(
                "Subiendo video a Cloudinary | job_id=%s | path=%s | "
                "size_mb=%.2f | compressed=%s | chunked=%s",
                job_id,
                local_path,
                file_size_mb,
                was_compressed,
                use_chunked
            )

            public_id = f"{folder}/{job_id}_output"

            # ══════════════════════════════════════════════════════════
            # PASO 3: Upload a Cloudinary
            # ══════════════════════════════════════════════════════════
            if use_chunked:
                logger.info(
                    "Usando chunked upload | job_id=%s | chunks=~%.0f",
                    job_id,
                    file_size_mb / 6
                )
                
                result = cloudinary.uploader.upload_large(
                    local_path,
                    resource_type="video",
                    public_id=public_id,
                    overwrite=True,
                    chunk_size=6000000,
                    timeout=600,
                    eager_async=False,
                )
            else:
                result = cloudinary.uploader.upload(
                    local_path,
                    resource_type="video",
                    public_id=public_id,
                    overwrite=True,
                    timeout=300,
                    notification_url=None,
                    eager_async=False,
                )

            video_url = result.get('secure_url')

            logger.info(
                "Video subido correctamente | job_id=%s | public_id=%s | "
                "url=%s | compressed=%s",
                job_id,
                result.get('public_id'),
                video_url,
                was_compressed
            )
            
            # ══════════════════════════════════════════════════════════
            # PASO 4: Cleanup de archivos temporales de compresión
            # ══════════════════════════════════════════════════════════
            if was_compressed and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logger.debug(
                        "Archivo comprimido temporal eliminado | job_id=%s",
                        job_id
                    )
                except Exception:
                    pass

            return video_url

        except Exception as e:
            error_msg = str(e)
            
            # Limpiar archivos de compresión si hay error
            if was_compressed and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
            
            logger.error(
                "Error subiendo video | job_id=%s | path=%s | size_mb=%.2f | "
                "compressed=%s",
                job_id,
                original_path,
                file_size_mb,
                was_compressed,
                exc_info=True
            )
            
            raise Exception(f"No se pudo subir el video: {error_msg}")

    def upload_image(
        self,
        local_path: str,
        public_id: str,
        folder: str = "thumbnails"
    ) -> str:
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Archivo no encontrado: {local_path}")

            file_size_kb = os.path.getsize(local_path) / 1024

            logger.info(
                "Subiendo imagen a Cloudinary | public_id=%s | path=%s | size_kb=%.2f",
                public_id,
                local_path,
                file_size_kb
            )

            full_public_id = f"{folder}/{public_id}"

            result = cloudinary.uploader.upload(
                local_path,
                resource_type="image",
                public_id=full_public_id,
                overwrite=True
            )

            image_url = result.get('secure_url')

            logger.info(
                "Imagen subida correctamente | public_id=%s | url=%s",
                full_public_id,
                image_url
            )

            return image_url

        except Exception as e:
            logger.error(
                "Error subiendo imagen | public_id=%s | path=%s",
                public_id,
                local_path,
                exc_info=True
            )
            raise Exception(f"No se pudo subir la imagen: {str(e)}")

    def delete_local_files(self, job_id: str):
        try:
            patterns = [
                f"{job_id}_input.mp4",
                f"{job_id}_output.mp4",
                f"{job_id}*.mp4"
            ]

            deleted_count = 0

            for pattern in patterns:
                path = os.path.join(self.temp_dir, pattern)
                if os.path.exists(path):
                    os.remove(path)
                    deleted_count += 1
                    logger.debug(
                        "Archivo temporal eliminado | job_id=%s | path=%s",
                        job_id,
                        path
                    )

            if deleted_count > 0:
                logger.info(
                    "Limpieza completada | job_id=%s | deleted_files=%s",
                    job_id,
                    deleted_count
                )

        except Exception:
            logger.warning(
                "Error durante limpieza de archivos | job_id=%s",
                job_id,
                exc_info=True
            )

    def get_video_info(self, cloudinary_url: str) -> Optional[dict]:
        try:
            parts = cloudinary_url.split('/upload/')
            if len(parts) < 2:
                return None

            public_id = parts[1].split('/')[-1].rsplit('.', 1)[0]

            resource = cloudinary.api.resource(
                public_id,
                resource_type="video"
            )

            return {
                'format': resource.get('format'),
                'duration': resource.get('duration'),
                'width': resource.get('width'),
                'height': resource.get('height'),
                'size_bytes': resource.get('bytes'),
            }

        except Exception:
            logger.warning(
                "No se pudo obtener info del video | url=%s",
                cloudinary_url,
                exc_info=True
            )
            return None


def create_cloudinary_service() -> CloudinaryService:
    cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
    api_key = os.getenv('CLOUDINARY_API_KEY')
    api_secret = os.getenv('CLOUDINARY_API_SECRET')
    temp_dir = os.getenv('CLOUDINARY_TEMP_DIR', '/tmp/video_processing')

    if not all([cloud_name, api_key, api_secret]):
        raise ValueError(
            "Faltan variables de entorno de Cloudinary. "
            "Se requieren: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
        )

    return CloudinaryService(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        temp_dir=temp_dir
    )