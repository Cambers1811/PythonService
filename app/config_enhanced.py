"""
Configuración Ultra-Mejorada con Parámetros Optimizados
"""

import logging

logger = logging.getLogger(__name__)

# ==============================
# Detección de rostros mejorada
# ==============================

FACE_DETECTION_ENHANCED = {
    'model_selection': 1,
    'min_confidence': 0.65,
    'min_face_size': 30,
    'max_faces': 1,
    'priority_mode': 'quality',
    'temporal_smoothing_window': 10,
    'redetect_interval_frames': 8,
    'use_fallback_detection': True,
    'fallback_min_area': 2000,
    'tracking_max_lost_frames': 15,
}

# ==============================
# Estabilización mejorada
# ==============================

STABILIZATION_ENHANCED = {
    'method': 'hybrid',
    'kalman': {
        'process_variance': 0.008,
        'measurement_variance': 0.08,
        'estimation_error': 1.0
    },
    'exponential_alpha': 0.88,
    'max_velocity_px_per_frame': 35,
    'max_acceleration_px_per_frame2': 8,
    'deadzone_pixels': 3,
    'use_prediction': True,
    'prediction_frames': 10,
    'prediction_weight': 0.25,
    'use_easing': True,
    'easing_function': 'smooth',
    'adaptive_parameters': True,
    'movement_analysis_window': 15,
}

# ==============================
# Configuración de crop
# ==============================

CROP_SETTINGS_ENHANCED = {
    'width': 1080,
    'height': 1920,
    'use_rule_of_thirds': True,
    'thirds_offset_factor': 0.15,
    'headroom_ratio': 0.18,
    'edge_padding': 15,
    'dynamic_zoom': False,
    'zoom_range': (650, 800),
    'adaptive_composition': True,
    'composition_adjust_speed': 0.3,
}

# ==============================
# Keyframes optimizados
# ==============================

KEYFRAME_SETTINGS_ENHANCED = {
    'optimize_keyframes': True,
    'base_threshold_pixels': 12,
    'max_keyframes': 80,
    'target_keyframes': 50,
    'tolerance': 8,
    'force_keyframe_interval_seconds': 2.5,
    'min_keyframe_distance_seconds': 0.15,
    'quality_based_threshold': True,
}

# ==============================
# Encoding ultra-optimizado
# ==============================

ENCODING_SETTINGS_ENHANCED = {
    'video_codec': 'libx264',
    'audio_codec': 'aac',
    'pix_fmt': 'yuv420p',
    'audio_sample_rate': 48000,
    'b_frames': 2,
    'gop_size': 60,
    'faststart': True,
    'apply_unsharp': True,
    'unsharp_params': '5:5:0.8:5:5:0.0',
    'color_primaries': 'bt709',
    'color_trc': 'bt709',
    'colorspace': 'bt709',
    'quality_preset': 'balanced',
    'presets': {
        'ultra_fast': {
            'preset': 'ultrafast',
            'crf': '28',
            'profile': 'baseline',
            'bitrate_audio': '128k',
        },
        'fast': {
            'preset': 'fast',
            'crf': '24',
            'profile': 'main',
            'bitrate_audio': '160k',
        },
        'balanced': {
            'preset': 'medium',
            'crf': '21',
            'profile': 'high',
            'level': '4.1',
            'bitrate_audio': '192k',
        },
        'high': {
            'preset': 'slow',
            'crf': '18',
            'profile': 'high',
            'level': '4.2',
            'bitrate_audio': '224k',
            'tune': 'film',
        },
        'ultra': {
            'preset': 'veryslow',
            'crf': '16',
            'profile': 'high',
            'level': '5.0',
            'bitrate_audio': '256k',
            'audio_quality': 0,
            'tune': 'film',
        },
        'web': {
            'preset': 'medium',
            'crf': '22',
            'profile': 'high',
            'level': '4.1',
            'maxrate': '5M',
            'bufsize': '10M',
            'bitrate_audio': '160k',
        }
    }
}

# ==============================
# Performance
# ==============================

PERFORMANCE_SETTINGS_ENHANCED = {
    'sample_rate': 3,
    'verbose': True,
    'use_multipass': True,
}

# ==============================
# Métricas de calidad
# ==============================

QUALITY_METRICS_ENHANCED = {
    'calculate_metrics': True,
    'track_quality_issues': True,
    'min_acceptable_quality': 0.70,
}


# ======================================================
# Presets
# ======================================================

def apply_preset_enhanced(preset_name):
    global FACE_DETECTION_ENHANCED, STABILIZATION_ENHANCED, CROP_SETTINGS_ENHANCED
    global KEYFRAME_SETTINGS_ENHANCED, ENCODING_SETTINGS_ENHANCED, PERFORMANCE_SETTINGS_ENHANCED

    if preset_name == 'ultra_quality':
        PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 2
        PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = True
        STABILIZATION_ENHANCED['exponential_alpha'] = 0.92
        STABILIZATION_ENHANCED['max_velocity_px_per_frame'] = 25
        STABILIZATION_ENHANCED['deadzone_pixels'] = 4
        KEYFRAME_SETTINGS_ENHANCED['max_keyframes'] = 100
        KEYFRAME_SETTINGS_ENHANCED['tolerance'] = 5
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'ultra'

    elif preset_name == 'professional':
        PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 3
        PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = True
        STABILIZATION_ENHANCED['exponential_alpha'] = 0.88
        STABILIZATION_ENHANCED['max_velocity_px_per_frame'] = 35
        KEYFRAME_SETTINGS_ENHANCED['max_keyframes'] = 80
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'high'

    elif preset_name == 'balanced':
        PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 4
        PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = False
        STABILIZATION_ENHANCED['exponential_alpha'] = 0.85
        STABILIZATION_ENHANCED['max_velocity_px_per_frame'] = 40
        KEYFRAME_SETTINGS_ENHANCED['max_keyframes'] = 60
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'balanced'

    elif preset_name == 'fast':
        PERFORMANCE_SETTINGS_ENHANCED['sample_rate'] = 6
        PERFORMANCE_SETTINGS_ENHANCED['use_multipass'] = False
        STABILIZATION_ENHANCED['exponential_alpha'] = 0.80
        STABILIZATION_ENHANCED['max_velocity_px_per_frame'] = 50
        STABILIZATION_ENHANCED['deadzone_pixels'] = 2
        KEYFRAME_SETTINGS_ENHANCED['max_keyframes'] = 40
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'fast'
        ENCODING_SETTINGS_ENHANCED['apply_unsharp'] = False

    elif preset_name == 'tiktok':
        apply_preset_enhanced('balanced')
        CROP_SETTINGS_ENHANCED['headroom_ratio'] = 0.16
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'web'

    elif preset_name == 'instagram':
        apply_preset_enhanced('balanced')
        CROP_SETTINGS_ENHANCED['headroom_ratio'] = 0.18
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'balanced'

    elif preset_name == 'youtube_shorts':
        apply_preset_enhanced('professional')
        ENCODING_SETTINGS_ENHANCED['quality_preset'] = 'balanced'

    logger.info(
        "Preset aplicado: %s | Encoding: %s | Sample rate: 1/%s | Multi-paso: %s",
        preset_name,
        ENCODING_SETTINGS_ENHANCED['quality_preset'],
        PERFORMANCE_SETTINGS_ENHANCED['sample_rate'],
        PERFORMANCE_SETTINGS_ENHANCED['use_multipass']
    )


# ======================================================
# Compatibilidad
# ======================================================

FACE_DETECTION = FACE_DETECTION_ENHANCED.copy()
STABILIZATION = STABILIZATION_ENHANCED.copy()
CROP_SETTINGS = CROP_SETTINGS_ENHANCED.copy()
KEYFRAME_SETTINGS = KEYFRAME_SETTINGS_ENHANCED.copy()
ENCODING_SETTINGS = ENCODING_SETTINGS_ENHANCED.copy()
PERFORMANCE_SETTINGS = PERFORMANCE_SETTINGS_ENHANCED.copy()
QUALITY_METRICS = QUALITY_METRICS_ENHANCED.copy()

CONVERSION_MODE = {
    'mode': 'smart_crop',
    'modes': {
        'full': {
            'description': 'Mantiene todo el contenido con letterbox',
            'width': 1080,
            'height': 1920,
            'blur_background': False,
            'background_color': 'black'
        },
        'smart_crop': {
            'description': 'Recorte inteligente siguiendo rostros',
            'width': 1080,
            'height': 1920,
        }
    }
}

SCENE_ANALYSIS = {
    'enabled': False,
    'scene_change_threshold': 0.3,
}


def set_conversion_mode(mode):
    CONVERSION_MODE['mode'] = mode


def print_config():
    logger.info("CONFIGURACIÓN ACTUAL")
    logger.info("Modo: %s", CONVERSION_MODE['mode'])
    logger.info("Encoding preset: %s", ENCODING_SETTINGS['quality_preset'])
    logger.info("Sample rate: 1/%s", PERFORMANCE_SETTINGS['sample_rate'])
    logger.info("Multi-paso: %s", PERFORMANCE_SETTINGS['use_multipass'])


def apply_preset(preset_name):
    apply_preset_enhanced(preset_name)

