"""
FFmpeg Runner Ultra-Profesional con Soporte Multi-Modo
Soporta: Full Video (letterbox) y Smart Crop (seguimiento inteligente)
"""

import subprocess
import os
import json
import logging

logger = logging.getLogger(__name__)


def crop_video_ultra(input_path, output_path, positions, config, encoder='libx264'):
    mode = config.CONVERSION_MODE['mode']
    mode_config = config.CONVERSION_MODE['modes'][mode]

    logger.info("Modo de conversion: %s", mode.upper())
    logger.info("Descripcion modo: %s", mode_config['description'])
    logger.info("Encoder: %s", encoder)

    if mode == 'full':
        return process_full_mode(input_path, output_path, config, mode_config, encoder=encoder)
    else:
        return process_smart_crop_mode(input_path, output_path, positions, config, encoder=encoder)


def process_full_mode(input_path, output_path, config, mode_config, encoder='libx264'):
    crop_width = mode_config['width']
    crop_height = mode_config['height']

    if mode_config.get('blur_background', False):
        filter_complex = create_blur_background_filter(crop_width, crop_height)
        logger.info("Usando fondo difuminado")
    else:
        bg_color = mode_config.get('background_color', 'black')
        filter_complex = (
            f"scale={crop_width}:{crop_height}:force_original_aspect_ratio=decrease,"
            f"pad={crop_width}:{crop_height}:(ow-iw)/2:(oh-ih)/2:color={bg_color}"
        )
        logger.info("Usando letterbox con fondo %s", bg_color)

    quality_preset = config.ENCODING_SETTINGS['quality_preset']
    settings = config.ENCODING_SETTINGS['presets'][quality_preset]

    cmd = build_ffmpeg_command_simple(
        input_path,
        output_path,
        filter_complex,
        settings,
        encoder=encoder
    )

    logger.info(
        "Encoding preset=%s | ffmpeg_preset=%s | crf=%s",
        quality_preset,
        settings['preset'],
        settings['crf']
    )

    return execute_ffmpeg(cmd, output_path)


def create_blur_background_filter(width, height):
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=20[blurred];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[scaled];"
        f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2"
    )


def process_smart_crop_mode(input_path, output_path, positions, config, encoder='libx264'):
    crop_width = config.CROP_SETTINGS['width']
    crop_height = config.CROP_SETTINGS['height']

    if config.KEYFRAME_SETTINGS.get('optimize_keyframes', False) and positions:
        logger.info("Optimizando %s keyframes", len(positions))
        positions = optimize_keyframes(positions, config)
        logger.info("Keyframes optimizados a %s", len(positions))

    if not positions:
        crop_filter = f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:0"
    else:
        positions = sorted(positions, key=lambda p: p[0])
        use_easing = config.STABILIZATION.get('use_easing', False)
        crop_x_expr = build_advanced_lerp_expression(positions, use_easing)
        crop_filter = f"crop={crop_width}:{crop_height}:x='{crop_x_expr}':y=0"

    filters = [crop_filter]

    if config.ENCODING_SETTINGS.get('apply_unsharp', False):
        unsharp = config.ENCODING_SETTINGS['unsharp_params']
        filters.append(f"unsharp={unsharp}")

    filter_complex = ",".join(filters)

    quality_preset = config.ENCODING_SETTINGS['quality_preset']
    settings = config.ENCODING_SETTINGS['presets'][quality_preset]

    cmd = build_ffmpeg_command_simple(
        input_path,
        output_path,
        filter_complex,
        settings,
        encoder=encoder
    )

    logger.info(
        "Encoding preset=%s | ffmpeg_preset=%s | crf=%s | profile=%s",
        quality_preset,
        settings['preset'],
        settings['crf'],
        settings.get('profile', 'high')
    )

    return execute_ffmpeg(cmd, output_path)


def build_ffmpeg_command_simple(input_path, output_path, filter_complex, settings, encoder='libx264'):
    """
    Construye comando FFmpeg con encoder configurable.

    Args:
        encoder: Encoder a usar ('libx264', 'h264_nvenc', 'h264_qsv', etc.)
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", filter_complex,
        "-c:v", encoder,
        "-preset", settings["preset"],
        "-crf", str(settings["crf"]),
    ]

    # Solo agregar profile para software encoders
    if encoder == 'libx264':
        cmd.extend(["-profile:v", settings.get("profile", "high")])

    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ])

    return cmd


def optimize_keyframes(positions, config):
    min_movement = config.KEYFRAME_SETTINGS.get("min_movement_threshold", 5)

    if not positions:
        return positions

    optimized = [positions[0]]

    for current in positions[1:]:
        prev = optimized[-1]
        if abs(current[1] - prev[1]) > min_movement:
            optimized.append(current)

    return optimized


def build_advanced_lerp_expression(positions, use_easing):
    if len(positions) == 1:
        return str(int(positions[0][1]))

    expr = ""

    for i in range(len(positions) - 1):
        t1, x1 = positions[i]
        t2, x2 = positions[i + 1]

        duration = t2 - t1
        if duration <= 0:
            continue

        if use_easing:
            segment = (
                f"if(between(t,{t1},{t2}),"
                f"{x1}+({x2}-{x1})*pow((t-{t1})/{duration},2),"
            )
        else:
            segment = (
                f"if(between(t,{t1},{t2}),"
                f"{x1}+({x2}-{x1})*(t-{t1})/{duration},"
            )

        expr += segment

    expr += str(int(positions[-1][1])) + ")" * (len(positions) - 1)

    return expr


def execute_ffmpeg(cmd, output_path):
    try:
        # No capturar stdout/stderr en memoria — FFmpeg escribe mucho en stderr
        # y bufferearlo puede provocar picos de RAM. Los logs van directo al proceso.
        subprocess.run(
            cmd,
            check=True
        )

        logger.info("Video generado exitosamente")

        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("Tamano archivo: %.2f MB", size_mb)
            print_video_info(output_path)

        return True

    except subprocess.CalledProcessError as e:
        logger.error("Error en FFmpeg (código de salida: %d)", e.returncode)
        return False


def print_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    try:
        # ffprobe solo emite JSON pequeño — capture_output aquí es seguro
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        for stream in info.get('streams', []):
            if stream['codec_type'] == 'video':
                logger.info(
                    "Resolucion=%sx%s | codec=%s",
                    stream['width'],
                    stream['height'],
                    stream['codec_name']
                )

                if 'r_frame_rate' in stream:
                    num, den = stream['r_frame_rate'].split('/')
                    fps = float(num) / float(den)
                    logger.info("FPS: %.2f", fps)

                if 'bit_rate' in stream:
                    bitrate_mbps = int(stream['bit_rate']) / 1_000_000
                    logger.info("Bitrate: %.2f Mbps", bitrate_mbps)

        format_info = info.get('format', {})
        if 'duration' in format_info:
            duration = float(format_info['duration'])
            logger.info("Duracion: %.2f s", duration)

    except Exception as e:
        logger.warning("No se pudo obtener metadata del video: %s", str(e))