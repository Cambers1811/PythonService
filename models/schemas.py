"""
Schemas — Contrato de datos entre el frontend y el backend.

Versión 3: Implementa Discriminated Union en el request.

El campo `processing_mode` actúa como discriminador. Pydantic resuelve
automáticamente qué modelo concreto instanciar según su valor, garantizando
que cada modo exponga exactamente los campos que necesita y ninguno más.

Jerarquía de modelos de request:
  BaseVideoRequest        ← campos comunes a los tres modos
    VerticalRequest       ← processing_mode: Literal["vertical"]
    ShortAutoRequest      ← processing_mode: Literal["short_auto"]  + short_auto_duration
    ShortManualRequest    ← processing_mode: Literal["short_manual"] + short_options

VideoProcessRequest = Union discriminado de los tres modelos concretos.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum
from typing import Annotated, Literal, Optional, Union
from datetime import datetime


# -------------------------------------------------------------------
# Límites para shorts
# Definidos aquí como constantes para ser importados desde validators
# y desde las estrategias de procesamiento sin dependencia circular.
# -------------------------------------------------------------------

SHORT_MIN_DURATION_SECONDS: int = 5
SHORT_MAX_DURATION_SECONDS: int = 60
SHORT_DEFAULT_DURATION_SECONDS: int = 30


# -------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------

class Platform(str, Enum):
    tiktok = "tiktok"
    instagram = "instagram"
    youtube_shorts = "youtube_shorts"


class BackgroundMode(str, Enum):
    smart_crop = "smart_crop"   # Seguir el rostro (recorte dinámico)
    black = "black"             # Fondo negro (letterbox)
    blurred = "blurred"         # Fondo difuminado


class QualityLevel(str, Enum):
    fast = "fast"               # Rápido
    normal = "normal"           # Normal (balanced internamente)
    high = "high"               # Alta calidad (professional internamente)


class ProcessingMode(str, Enum):
    vertical = "vertical"
    short_auto = "short_auto"
    short_manual = "short_manual"


# -------------------------------------------------------------------
# Mapeos internos
# Traducen las opciones del usuario a los valores de config_enhanced
# -------------------------------------------------------------------

QUALITY_TO_PRESET = {
    QualityLevel.fast: "fast",
    QualityLevel.normal: "balanced",
    QualityLevel.high: "professional",
}

BACKGROUND_TO_CONVERSION_MODE = {
    BackgroundMode.smart_crop: "smart_crop",
    BackgroundMode.black: "full",
    BackgroundMode.blurred: "full",
}

BACKGROUND_TO_BLUR = {
    BackgroundMode.black: False,
    BackgroundMode.blurred: True,
    BackgroundMode.smart_crop: None,
}


# -------------------------------------------------------------------
# Modelos anidados reutilizados dentro de los requests
# -------------------------------------------------------------------

class ShortManualOptions(BaseModel):
    """
    Segmento exacto que el usuario quiere convertir.
    Solo existe en ShortManualRequest, por lo que todos sus campos
    son obligatorios: no hay ambigüedad de si aplican o no.
    """

    start_time: float = Field(
        ge=0.0,
        description="Tiempo de inicio del segmento en segundos (>= 0.0)"
    )
    duration: int = Field(
        ge=SHORT_MIN_DURATION_SECONDS,
        le=SHORT_MAX_DURATION_SECONDS,
        description=(
            f"Duración del short en segundos "
            f"({SHORT_MIN_DURATION_SECONDS}-{SHORT_MAX_DURATION_SECONDS}s)"
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {"start_time": 45.0, "duration": 30}
        }
    }


class AdvancedOptions(BaseModel):
    """
    Ajustes opcionales de composición y encoding.
    Presentes en los tres modos porque son independientes del modo.
    """

    headroom_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.5,
        description="Espacio sobre la cabeza (0.0-0.5). Default: 0.18"
    )
    smoothing_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Suavizado del movimiento (0.0-1.0). Default: 0.88"
    )
    max_camera_speed: Optional[int] = Field(
        default=None,
        ge=10,
        le=100,
        description="Velocidad máxima de cámara en px/frame (10-100). Default: 35"
    )
    apply_sharpening: Optional[bool] = Field(
        default=None,
        description="Aplicar sharpening sutil al video final. Default: True"
    )
    use_rule_of_thirds: Optional[bool] = Field(
        default=None,
        description="Usar regla de tercios en composición. Default: True"
    )
    edge_padding: Optional[int] = Field(
        default=None,
        ge=0,
        le=50,
        description="Padding de bordes en píxeles (0-50). Default: 15"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "headroom_ratio": 0.2,
                "smoothing_strength": 0.9,
                "max_camera_speed": 30,
                "apply_sharpening": True,
                "use_rule_of_thirds": True,
                "edge_padding": 20,
            }
        }
    }


# -------------------------------------------------------------------
# Base — campos comunes a los tres modos
#
# No se expone directamente como endpoint. Los tres modelos concretos
# heredan de aquí para no repetir las definiciones de campos comunes.
# -------------------------------------------------------------------

class BaseVideoRequest(BaseModel):
    """
    Campos comunes a todos los modos de procesamiento.
    No instanciar directamente: usar VerticalRequest, ShortAutoRequest
    o ShortManualRequest.
    """

    platform: Platform = Field(
        default=Platform.tiktok,
        description="Plataforma destino del video"
    )
    background_mode: BackgroundMode = Field(
        default=BackgroundMode.smart_crop,
        description="Cómo se verá el fondo del video vertical"
    )
    quality: QualityLevel = Field(
        default=QualityLevel.normal,
        description="Nivel de calidad del procesamiento"
    )
    cloudinary_input_url: str = Field(
        description="URL del video original en Cloudinary"
    )
    advanced_options: Optional[AdvancedOptions] = Field(
        default=None,
        description="Ajustes opcionales de composición y encoding"
    )


# -------------------------------------------------------------------
# Modelos concretos — uno por modo
#
# Cada uno declara `processing_mode` como Literal para que Pydantic
# lo use como discriminador y resuelva el tipo correcto al parsear.
#
# Ventajas del Literal sobre el Enum genérico en la base:
#   - Pydantic puede usar el valor como clave de discriminación.
#   - El schema OpenAPI generado muestra cada modo por separado,
#     lo que permite a Swagger/Redoc mostrar ejemplos correctos.
#   - El IDE infiere el tipo exacto en cada rama del código.
# -------------------------------------------------------------------

class VerticalRequest(BaseVideoRequest):
    """
    Convierte el video completo a formato vertical 9:16.
    El frontend solo envía los campos comunes más el discriminador.
    No hay campos adicionales: el video se procesa completo.
    """

    processing_mode: Literal[ProcessingMode.vertical] = Field(
        default=ProcessingMode.vertical,
        description="Discriminador de modo — debe ser 'vertical'"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "processing_mode": "vertical",
                "platform": "tiktok",
                "background_mode": "smart_crop",
                "quality": "normal",
                "cloudinary_input_url": "https://res.cloudinary.com/demo/video/upload/sample.mp4",
            }
        }
    }


class ShortAutoRequest(BaseVideoRequest):
    """
    Genera un short automático extrayendo el segmento central del video.
    El único campo adicional respecto a los comunes es `short_auto_duration`.
    No se puede confundir con ShortManualRequest porque `short_options`
    directamente no existe en este modelo.
    """

    processing_mode: Literal[ProcessingMode.short_auto] = Field(
        default=ProcessingMode.short_auto,
        description="Discriminador de modo — debe ser 'short_auto'"
    )
    short_auto_duration: int = Field(
        default=SHORT_DEFAULT_DURATION_SECONDS,
        ge=SHORT_MIN_DURATION_SECONDS,
        le=SHORT_MAX_DURATION_SECONDS,
        description=(
            f"Duración deseada del short en segundos "
            f"({SHORT_MIN_DURATION_SECONDS}-{SHORT_MAX_DURATION_SECONDS}s). "
            f"Default: {SHORT_DEFAULT_DURATION_SECONDS}s. "
            "Si el video es más corto que este valor, se usa la duración completa."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "processing_mode": "short_auto",
                "platform": "tiktok",
                "background_mode": "smart_crop",
                "quality": "normal",
                "short_auto_duration": 30,
                "cloudinary_input_url": "https://res.cloudinary.com/demo/video/upload/sample.mp4",
            }
        }
    }


class ShortManualRequest(BaseVideoRequest):
    """
    Genera un short a partir del segmento exacto indicado por el usuario.
    `short_options` es obligatorio (no Optional) porque este modelo
    solo se instancia cuando el usuario eligió este modo específicamente.
    La ausencia de `short_options` es un error de parseo, no de validación.
    """

    processing_mode: Literal[ProcessingMode.short_manual] = Field(
        default=ProcessingMode.short_manual,
        description="Discriminador de modo — debe ser 'short_manual'"
    )
    short_options: ShortManualOptions = Field(
        description="Segmento a extraer: tiempo de inicio y duración"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "processing_mode": "short_manual",
                "platform": "instagram",
                "background_mode": "smart_crop",
                "quality": "high",
                "short_options": {
                    "start_time": 45.0,
                    "duration": 30,
                },
                "cloudinary_input_url": "https://res.cloudinary.com/demo/video/upload/sample.mp4",
            }
        }
    }


# -------------------------------------------------------------------
# VideoProcessRequest — Union discriminada
#
# Este es el único tipo que el endpoint y los validators necesitan importar.
#
# Cómo funciona la resolución:
#   1. Pydantic lee el campo `processing_mode` del JSON entrante.
#   2. Lo compara contra los Literal de cada modelo del Union.
#   3. Instancia el modelo que coincide.
#   4. Si `processing_mode` no coincide con ningún Literal, retorna
#      un ValidationError antes de llegar al endpoint.
#
# Restricción de tipo en el backend:
#   El tipo de `request` en el endpoint es VideoProcessRequest, que
#   es un Union. Para acceder a campos específicos de un modo
#   (como `request.short_options`) se debe usar isinstance():
#
#     if isinstance(request, ShortManualRequest):
#         opts = request.short_options  # el IDE lo infiere correctamente
#
#   Los validators y strategies ya están diseñados de esta forma.
# -------------------------------------------------------------------

VideoProcessRequest = Annotated[
    Union[
        VerticalRequest,
        ShortAutoRequest,
        ShortManualRequest,
    ],
    Field(discriminator="processing_mode"),
]


# -------------------------------------------------------------------
# Job Status
# -------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# -------------------------------------------------------------------
# Response models
# -------------------------------------------------------------------

class VideoProcessResponse(BaseModel):
    """Respuesta inmediata al crear un job (202 Accepted)."""

    job_id: str = Field(description="ID único del job")
    status: JobStatus = Field(description="Estado actual del job")
    message: str = Field(description="Mensaje descriptivo")
    processing_mode: ProcessingMode = Field(
        description="Modo de procesamiento solicitado"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
                "status": "pending",
                "message": "El video está en cola para procesarse",
                "processing_mode": "short_auto",
            }
        }
    }


class JobStatusResponse(BaseModel):
    """Estado completo de un job, incluyendo progreso y resultado."""

    job_id: str
    status: JobStatus
    message: str
    processing_mode: Optional[ProcessingMode] = Field(
        default=None,
        description="Modo de procesamiento del job"
    )
    progress: Optional[int] = Field(
        default=None,
        description="Porcentaje de progreso (0-100)"
    )
    phase: Optional[str] = Field(
        default=None,
        description="Fase detallada del procesamiento"
    )
    elapsed_seconds: Optional[float] = Field(
        default=None,
        description="Segundos transcurridos desde el inicio"
    )
    eta_seconds: Optional[float] = Field(
        default=None,
        description="Segundos estimados restantes"
    )
    eta_formatted: Optional[str] = Field(
        default=None,
        description="Tiempo estimado restante en formato legible (ej: 2m 30s)"
    )
    output_url: Optional[str] = Field(
        default=None,
        description="URL del video procesado en Cloudinary"
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="URL del thumbnail en Cloudinary"
    )
    preview_url: Optional[str] = Field(
        default=None,
        description="URL del preview clip en Cloudinary"
    )
    quality_score: Optional[float] = Field(
        default=None,
        description="Score de calidad del tracking (0.0-1.0)"
    )
    # Campos de short — solo poblados para short_auto y short_manual
    segment_start: Optional[float] = Field(
        default=None,
        description="Segundo de inicio del segmento procesado"
    )
    segment_duration: Optional[int] = Field(
        default=None,
        description="Duración en segundos del segmento procesado"
    )
    output_duration_seconds: Optional[float] = Field(
        default=None,
        description="Duración real del video de salida en segundos"
    )
    error_detail: Optional[str] = Field(
        default=None,
        description="Detalle del error cuando status=failed"
    )
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
                "status": "processing",
                "processing_mode": "short_auto",
                "message": "Cortando segmento del video...",
                "progress": 28,
                "phase": "cutting_segment",
                "elapsed_seconds": 12.4,
                "eta_seconds": 35.0,
                "eta_formatted": "35s",
                "output_url": None,
                "quality_score": None,
                "segment_start": None,
                "segment_duration": None,
                "output_duration_seconds": None,
                "error_detail": None,
                "created_at": "2026-02-20T10:00:00",
                "completed_at": None,
            }
        }
    }
