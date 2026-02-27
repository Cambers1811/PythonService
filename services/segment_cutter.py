"""
Segment Cutter — Extracción de segmentos de video con FFmpeg.

Responsabilidad única: dado un video, un tiempo de inicio y una duración,
genera un archivo intermedio con el segmento cortado.

El archivo intermedio es un MP4 sin reencoding (stream copy) para máxima
velocidad. El encoding real ocurre en el paso de conversión vertical.
"""

import subprocess
import os
import time
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class SegmentCutter:
    """
    Extrae segmentos de video usando FFmpeg.

    Usa -c copy (stream copy) para evitar reencoding en el corte,
    lo que hace este paso muy rápido. El reencoding ocurre únicamente
    en el paso de conversión vertical.
    """

    @staticmethod
    def cut_segment(
        input_path: str,
        start_time: float,
        duration: int,
        job_id: str,
        temp_dir: Optional[str] = None
    ) -> str:
        """
        Extrae un segmento del video y lo guarda como archivo intermedio.

        Estrategia de corte:
          - Se usa '-ss' antes de '-i' (seek rápido por keyframe).
          - Se usa '-c copy' para evitar reencoding en este paso.
          - El archivo de salida es MP4 con faststart para compatibilidad.

        Nota sobre precisión: '-ss' antes de '-i' hace un seek por keyframe,
        por lo que el corte puede ser impreciso en algunos milisegundos.
        Para videos de short (5-60s) esta imprecisión es aceptable.
        Si se requiere corte frame-accurate en el futuro, cambiar a:
          ffmpeg -i input -ss start -t duration -c copy output

        Args:
            input_path: Path del video original.
            start_time: Tiempo de inicio en segundos.
            duration:   Duración del segmento en segundos.
            job_id:     ID del job para trazabilidad y nombres de archivo.
            temp_dir:   Directorio temporal (default: mismo directorio que input).

        Returns:
            Path del archivo intermedio generado.

        Raises:
            RuntimeError: Si FFmpeg falla al cortar el segmento.
        """
        # Determinar directorio temporal
        if temp_dir is None:
            temp_dir = str(Path(input_path).parent)

        os.makedirs(temp_dir, exist_ok=True)

        # Nombre del archivo intermedio
        intermediate_filename = f"{job_id}_segment_{int(start_time)}s_{duration}s.mp4"
        intermediate_path = os.path.join(temp_dir, intermediate_filename)

        logger.info(
            "Cortando segmento | job_id=%s | start=%.2fs | duration=%ds | output=%s",
            job_id,
            start_time,
            duration,
            intermediate_path
        )

        cmd = [
            "ffmpeg",
            "-y",                           # Sobreescribir si existe
            "-ss", str(start_time),         # Seek ANTES de -i (rápido, por keyframe)
            "-i", input_path,
            "-t", str(duration),            # Duración del segmento
            "-c", "copy",                   # Sin reencoding (stream copy)
            "-movflags", "+faststart",      # Optimización para streaming
            "-avoid_negative_ts", "make_zero",  # Normalizar timestamps
            intermediate_path
        ]

        cut_start = time.time()

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            cut_time = time.time() - cut_start

            if not os.path.exists(intermediate_path):
                raise RuntimeError(
                    f"FFmpeg terminó sin error pero el archivo no fue creado: {intermediate_path}"
                )

            file_size_mb = os.path.getsize(intermediate_path) / (1024 * 1024)

            logger.info(
                "Segmento cortado exitosamente | job_id=%s | "
                "start=%.2fs | duration=%ds | size=%.2fMB | cut_time=%.2fs",
                job_id,
                start_time,
                duration,
                file_size_mb,
                cut_time
            )

            return intermediate_path

        except subprocess.CalledProcessError as e:
            logger.error(
                "FFmpeg falló al cortar segmento | job_id=%s | stderr=%s",
                job_id,
                e.stderr[-500:] if e.stderr else "sin stderr"
            )
            raise RuntimeError(
                f"Error al cortar el segmento del video (start={start_time}s, "
                f"duration={duration}s). "
                "Verifica que el video sea válido y que el segmento esté dentro del rango."
            )
