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

    def upload_video(
        self,
        local_path: str,
        job_id: str,
        folder: str = "processed_videos"
    ) -> str:
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Archivo no encontrado: {local_path}")

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)

            logger.info(
                "Subiendo video a Cloudinary | job_id=%s | path=%s | size_mb=%.2f",
                job_id,
                local_path,
                file_size_mb
            )

            public_id = f"{folder}/{job_id}_output"

            result = cloudinary.uploader.upload(
                local_path,
                resource_type="video",
                public_id=public_id,
                overwrite=True,
                notification_url=None,
                eager_async=False,
            )

            video_url = result.get('secure_url')

            logger.info(
                "Video subido correctamente | job_id=%s | public_id=%s | url=%s",
                job_id,
                result.get('public_id'),
                video_url
            )

            return video_url

        except Exception as e:
            logger.error(
                "Error subiendo video | job_id=%s | path=%s",
                job_id,
                local_path,
                exc_info=True
            )
            raise Exception(f"No se pudo subir el video: {str(e)}")

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
