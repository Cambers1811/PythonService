"""
Procesador de Video Ultra-Mejorado
Con análisis de calidad en tiempo real y optimizaciones avanzadas
"""

import gc
import cv2
import numpy as np
import os
from pathlib import Path
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


def process_video_enhanced(input_path, config, detector, stabilizer, use_multipass=True, encoder='libx264'):
    """
    Procesar video con sistema mejorado y análisis de calidad

    Args:
        encoder: Encoder de FFmpeg a usar (libx264, h264_nvenc, h264_qsv, etc.)
    """

    verbose = config.PERFORMANCE_SETTINGS.get('verbose', True)

    if verbose:
        logger.info("=" * 60)
        logger.info("PROCESAMIENTO ULTRA-MEJORADO")
        logger.info("=" * 60)
        logger.info(f"Entrada: {input_path}")
        logger.info(f"Modo: {config.CONVERSION_MODE['mode']}")
        logger.info(f"Multi-paso: {'Sí' if use_multipass else 'No'}")
        logger.info(f"Encoder: {encoder}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir el video: {input_path}")
        raise ValueError(f"No se pudo abrir el video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    if verbose:
        logger.info(f"Resolución: {width}x{height}")
        logger.info(f"Duración: {duration:.2f}s ({total_frames} frames @ {fps:.2f} fps)")

    mode = config.CONVERSION_MODE['mode']

    if mode == 'full':
        return process_full_mode_simple(input_path, config, encoder=encoder)

    crop_width = config.CROP_SETTINGS['width']
    crop_height = config.CROP_SETTINGS['height']
    sample_rate = config.PERFORMANCE_SETTINGS['sample_rate']

    if verbose:
        logger.info("Analizando rostros...")
        logger.info(f"Sample rate: 1 de cada {sample_rate} frames")

    positions = []
    # Acumuladores en línea — evitan guardar una lista de dicts por frame
    _conf_sum = 0.0
    _stab_sum = 0.0
    _reliable_count = 0
    _quality_sample_count = 0
    frame_number = 0
    frames_processed = 0
    start_time = time.time()

    tracking_loss_count = 0

    if use_multipass:
        from app.stabilization_enhanced import MultiPassStabilizer
        multipass = MultiPassStabilizer(config)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_number / fps if fps > 0 else 0

        if frame_number % sample_rate == 0:

            faces = detector.detect(frame)
            primary_face = detector.get_primary_face(faces)

            if primary_face:
                crop_x, crop_y = calculate_optimal_composition(
                    primary_face,
                    (width, height),
                    (crop_width, crop_height),
                    config
                )

                quality = primary_face.get('quality')

                if use_multipass:
                    multipass.add_position(timestamp, crop_x, quality)
                else:
                    if hasattr(stabilizer, 'stabilize'):
                        stabilized_x = stabilizer.stabilize(crop_x, quality)
                    else:
                        stabilized_x = stabilizer.stabilize(crop_x)

                    positions.append((timestamp, stabilized_x))

                # Acumular métricas sin guardar el dict completo
                if quality:
                    _conf_sum += quality.confidence
                    _stab_sum += quality.stability
                    if quality.is_reliable:
                        _reliable_count += 1
                    _quality_sample_count += 1

                    if quality.lost_frames > 3:
                        tracking_loss_count += 1
            else:
                tracking_loss_count += 1
                if len(positions) > 0:
                    last_pos = positions[-1][1]
                    if use_multipass:
                        multipass.add_position(timestamp, last_pos, None)
                    else:
                        if hasattr(stabilizer, 'stabilize'):
                            stabilized_x = stabilizer.stabilize(None)
                        else:
                            stabilized_x = last_pos
                        positions.append((timestamp, stabilized_x if stabilized_x else last_pos))
                else:
                    center_x = (width - crop_width) // 2
                    if use_multipass:
                        multipass.add_position(timestamp, center_x, None)
                    else:
                        positions.append((timestamp, center_x))

            frames_processed += 1

            if verbose and frames_processed % 100 == 0:
                progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                elapsed = time.time() - start_time
                fps_analysis = frames_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progreso: {progress:.1f}% "
                    f"({frames_processed} frames, {fps_analysis:.1f} fps)"
                )

        # Liberar referencia al frame explícitamente para que el GC
        # recupere memoria antes del siguiente cap.read()
        del frame

        # Ayudar al GC cada 500 frames para prevenir picos prolongados
        if frame_number % 500 == 0:
            gc.collect()

        frame_number += 1

    cap.release()
    gc.collect()

    if use_multipass and positions:
        if verbose:
            logger.info("Aplicando estabilización multi-paso...")
        positions = multipass.process()

    analysis_time = time.time() - start_time

    if verbose:
        logger.info(f"Análisis completado en {analysis_time:.2f}s")
        logger.info(f"Frames procesados: {frames_processed}")
        logger.info(f"Keyframes generados: {len(positions)}")

        if _quality_sample_count > 0:
            avg_confidence = _conf_sum / _quality_sample_count
            avg_stability = _stab_sum / _quality_sample_count
            reliability_rate = _reliable_count / _quality_sample_count

            logger.info("Métricas de Tracking:")
            logger.info(f"Confianza promedio: {avg_confidence * 100:.1f}%")
            logger.info(f"Estabilidad promedio: {avg_stability * 100:.1f}%")
            logger.info(f"Frames confiables: {reliability_rate * 100:.1f}%")

        if tracking_loss_count:
            logger.warning(f"Eventos de pérdida de tracking: {tracking_loss_count}")

        stats = detector.get_tracking_stats()
        logger.info("Estadísticas del Detector:")
        logger.info(f"Trackers activos: {stats['active_trackers']}")
        logger.info(f"Trackers confiables: {stats['reliable_trackers']}")

        if stats['fallback_activations'] > 0:
            logger.warning(f"Activaciones de fallback: {stats['fallback_activations']}")

    if verbose:
        logger.info("Generando video final...")

    input_name = Path(input_path).stem
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    output_filename = f"{input_name}_vertical_{config.CONVERSION_MODE['mode']}_{timestamp_str}.mp4"
    output_path = os.path.join(temp_dir, output_filename)

    from app.ffmpeg_ultra import crop_video_ultra

    success = crop_video_ultra(input_path, output_path, positions, config, encoder=encoder)

    if not success:
        logger.error("Error en el encoding del video")
        raise RuntimeError("Error en el encoding del video")

    total_time = time.time() - start_time

    metrics = {
        'total_frames': total_frames,
        'frames_processed': frames_processed,
        'keyframes': len(positions),
        'analysis_time': analysis_time,
        'total_time': total_time,
        'tracking_loss_events': tracking_loss_count,
        'overall_quality': 1.0,
        'reliability_rate': 0.0,
    }

    if _quality_sample_count > 0:
        avg_confidence = _conf_sum / _quality_sample_count
        avg_stability = _stab_sum / _quality_sample_count
        reliability_rate = _reliable_count / _quality_sample_count

        metrics['overall_quality'] = (
            avg_confidence * 0.4 +
            avg_stability * 0.3 +
            reliability_rate * 0.3
        )
        metrics['reliability_rate'] = reliability_rate

    if verbose:
        logger.info("=" * 60)
        logger.info("PROCESAMIENTO COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"Salida: {output_path}")
        logger.info(f"Tiempo total: {total_time:.2f}s")
        logger.info(f"Calidad general: {metrics['overall_quality'] * 100:.1f}%")

        if metrics['overall_quality'] < 0.75:
            logger.warning("Calidad general baja detectada. Revisión recomendada.")

    return output_path, metrics


def process_full_mode_simple(input_path, config, encoder='libx264'):

    from app.ffmpeg_ultra import crop_video_ultra

    input_name = Path(input_path).stem
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    output_filename = f"{input_name}_vertical_full_{timestamp_str}.mp4"
    output_path = os.path.join(temp_dir, output_filename)

    success = crop_video_ultra(input_path, output_path, [], config, encoder=encoder)

    if not success:
        logger.error("Error en el encoding del video")
        raise RuntimeError("Error en el encoding del video")

    metrics = {
        'overall_quality': 1.0,
        'mode': 'full'
    }

    return output_path, metrics


def calculate_optimal_composition(face, frame_size, crop_size, config):

    frame_w, frame_h = frame_size
    crop_w, crop_h = crop_size

    x, y, w, h = face['bbox']
    cx, cy = face['center']

    if config.CROP_SETTINGS_ENHANCED.get(
        'use_rule_of_thirds',
        config.CROP_SETTINGS.get('use_rule_of_thirds', False)
    ):
        face_ratio_x = cx / frame_w
        thirds_offset = config.CROP_SETTINGS_ENHANCED.get(
            'thirds_offset_factor',
            config.CROP_SETTINGS.get('thirds_offset_factor', 0.15)
        )

        if face_ratio_x < 0.35:
            target_offset = crop_w * (0.33 - thirds_offset)
        elif face_ratio_x > 0.65:
            target_offset = crop_w * (0.67 + thirds_offset)
        else:
            target_offset = crop_w * 0.5
    else:
        target_offset = crop_w * 0.5

    headroom = config.CROP_SETTINGS_ENHANCED.get(
        'headroom_ratio',
        config.CROP_SETTINGS.get('headroom_ratio', 0.18)
    ) * crop_h

    face_height_ratio = h / frame_h
    if face_height_ratio > 0.4:
        headroom *= 0.7
    elif face_height_ratio < 0.15:
        headroom *= 1.3

    crop_x = cx - target_offset

    padding = config.CROP_SETTINGS_ENHANCED.get(
        'edge_padding',
        config.CROP_SETTINGS.get('edge_padding', 15)
    )

    crop_x = max(padding, min(crop_x, frame_w - crop_w - padding))

    return int(crop_x), 0