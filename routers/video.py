"""
Video Router — Endpoints de procesamiento de videos.

Documentación completa en /docs (Swagger UI) y /redoc.

Flujo de uso:
  1. El cliente sube el video a Cloudinary y obtiene una URL.
  2. Llama a POST /api/video/process con el modo deseado.
  3. Recibe un job_id y hace polling a GET /api/video/status/{job_id}.
  4. Cuando status=completed, usa output_url directamente desde Cloudinary.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.openapi.utils import get_openapi

from models.schemas import (
    VideoProcessRequest,
    VerticalRequest,
    ShortAutoRequest,
    ShortManualRequest,
    VideoProcessResponse,
    JobStatusResponse,
    JobStatus,
    ProcessingMode,
)
from services.video_service import VideoProcessingService
from storage.cloudinary_service import CloudinaryService
from validators import validate_video_request
from exceptions import ValidationError, VideoProcessingError
from error_handler import ErrorHandler
from cancellation_manager import get_cancellation_manager, JobCancelledException
from auth import ServiceTokenData, require_service_token, verify_job_ownership
from services.webhook_service import notify_job_completed, notify_job_failed, notify_job_cancelled


logger = logging.getLogger("video-processor-api.video")


# ============================================================
# Almacenamiento en memoria
# ============================================================

jobs_db: Dict[str, dict] = {}


# ============================================================
# Cancellation Manager
# ============================================================

cancellation_manager = get_cancellation_manager()


# ============================================================
# Modelos de respuesta de error para documentación
# ============================================================

_ERROR_400 = {
    "description": "Request inválido",
    "content": {
        "application/json": {
            "examples": {
                "url_invalida": {
                    "summary": "URL de Cloudinary inválida",
                    "value": {"detail": "La URL debe ser de Cloudinary (cloudinary.com)."}
                },
                "short_manual_sin_opciones": {
                    "summary": "short_manual sin short_options",
                    "value": {"detail": "El modo 'short_manual' requiere el campo 'short_options'."}
                },
                "segmento_excede_video": {
                    "summary": "Segmento fuera de rango",
                    "value": {
                        "detail": "El segmento solicitado (120s + 30s = 150s) excede "
                                  "la duración del video (95.0s)."
                    }
                },
                "job_no_cancelable": {
                    "summary": "Job ya terminado",
                    "value": {"detail": "El job ya está en estado 'completed' y no puede ser cancelado."}
                },
            }
        }
    },
}

_ERROR_404 = {
    "description": "Job no encontrado",
    "content": {
        "application/json": {
            "example": {"detail": "Job a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0 no encontrado"}
        }
    },
}

_ERROR_401 = {
    "description": "Token de servicio ausente o inválido",
    "content": {
        "application/json": {
            "examples": {
                "sin_token": {
                    "summary": "Header Authorization ausente",
                    "value": {"detail": "Autenticación requerida."}
                },
                "token_expirado": {
                    "summary": "Token expirado",
                    "value": {"detail": "Token expirado."}
                },
                "token_invalido": {
                    "summary": "Firma inválida o token malformado",
                    "value": {"detail": "Token inválido."}
                },
            }
        }
    },
}

_ERROR_403 = {
    "description": "El job pertenece a otro usuario",
    "content": {
        "application/json": {
            "example": {"detail": "No tienes permiso para acceder a este job."}
        }
    },
}

_ERROR_500 = {
    "description": "Error interno del servidor",
    "content": {
        "application/json": {
            "example": {"detail": "Error al validar el request"}
        }
    },
}


# ============================================================
# Ejemplos de request para POST /process
# ============================================================

_PROCESS_REQUEST_EXAMPLES = {
    "vertical": {
        "summary": "Modo vertical — video completo a 9:16",
        "description": (
            "Convierte el video completo a formato vertical 9:16. "
            "No requiere campos adicionales más allá de los comunes."
        ),
        "value": {
            "processing_mode": "vertical",
            "platform": "tiktok",
            "background_mode": "smart_crop",
            "quality": "normal",
            "cloudinary_input_url": "https://res.cloudinary.com/mi-cloud/video/upload/v123/video.mp4",
        },
    },
    "vertical_con_opciones_avanzadas": {
        "summary": "Modo vertical — con ajustes de composición",
        "description": (
            "Igual que el modo vertical pero con control fino sobre "
            "headroom, suavizado de cámara y sharpening."
        ),
        "value": {
            "processing_mode": "vertical",
            "platform": "instagram",
            "background_mode": "blurred",
            "quality": "high",
            "cloudinary_input_url": "https://res.cloudinary.com/mi-cloud/video/upload/v123/video.mp4",
            "advanced_options": {
                "headroom_ratio": 0.2,
                "smoothing_strength": 0.9,
                "max_camera_speed": 25,
                "apply_sharpening": True,
                "use_rule_of_thirds": True,
                "edge_padding": 20,
            },
        },
    },
    "short_auto_30s": {
        "summary": "Modo short_auto — 30 segundos del centro",
        "description": (
            "Selecciona automáticamente el segmento central del video "
            "y lo convierte a vertical. Ideal para contenido sin un "
            "momento específico a destacar."
        ),
        "value": {
            "processing_mode": "short_auto",
            "platform": "tiktok",
            "background_mode": "smart_crop",
            "quality": "normal",
            "short_auto_duration": 30,
            "cloudinary_input_url": "https://res.cloudinary.com/mi-cloud/video/upload/v123/video.mp4",
        },
    },
    "short_auto_15s": {
        "summary": "Modo short_auto — 15 segundos del centro",
        "description": "Igual que short_auto pero extrayendo solo 15 segundos.",
        "value": {
            "processing_mode": "short_auto",
            "platform": "youtube_shorts",
            "background_mode": "smart_crop",
            "quality": "high",
            "short_auto_duration": 15,
            "cloudinary_input_url": "https://res.cloudinary.com/mi-cloud/video/upload/v123/video.mp4",
        },
    },
    "short_manual": {
        "summary": "Modo short_manual — segmento exacto del usuario",
        "description": (
            "El usuario define exactamente qué parte del video convertir. "
            "`start_time` es el segundo de inicio y `duration` la duración "
            "del clip. El segmento no puede exceder la duración del video."
        ),
        "value": {
            "processing_mode": "short_manual",
            "platform": "instagram",
            "background_mode": "smart_crop",
            "quality": "high",
            "short_options": {
                "start_time": 45.0,
                "duration": 30,
            },
            "cloudinary_input_url": "https://res.cloudinary.com/mi-cloud/video/upload/v123/video.mp4",
        },
    },
    "short_manual_desde_inicio": {
        "summary": "Modo short_manual — desde el inicio del video",
        "description": "Extrae los primeros 20 segundos del video.",
        "value": {
            "processing_mode": "short_manual",
            "platform": "tiktok",
            "background_mode": "black",
            "quality": "normal",
            "short_options": {
                "start_time": 0.0,
                "duration": 20,
            },
            "cloudinary_input_url": "https://res.cloudinary.com/mi-cloud/video/upload/v123/video.mp4",
        },
    },
}


# ============================================================
# Ejemplos de response para GET /status/{job_id}
# ============================================================

_STATUS_RESPONSE_EXAMPLES = {
    "pending": {
        "summary": "Job en cola",
        "value": {
            "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
            "status": "pending",
            "processing_mode": "short_auto",
            "message": "El video está en cola para procesarse",
            "progress": 0,
            "phase": "queued",
            "elapsed_seconds": 0.5,
            "eta_seconds": None,
            "eta_formatted": None,
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": None,
            "created_at": "2026-02-20T10:00:00",
            "completed_at": None,
        },
    },
    "procesando_descargando": {
        "summary": "Procesando — descargando video",
        "value": {
            "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
            "status": "processing",
            "processing_mode": "short_auto",
            "message": "Descargando video desde Cloudinary...",
            "progress": 10,
            "phase": "downloading",
            "elapsed_seconds": 2.1,
            "eta_seconds": 75.0,
            "eta_formatted": "1m 15s",
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": None,
            "created_at": "2026-02-20T10:00:00",
            "completed_at": None,
        },
    },
    "procesando_cortando_segmento": {
        "summary": "Procesando — cortando segmento (modos short)",
        "description": "Esta fase solo aparece en short_auto y short_manual.",
        "value": {
            "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
            "status": "processing",
            "processing_mode": "short_auto",
            "message": "Cortando segmento del video...",
            "progress": 28,
            "phase": "cutting_segment",
            "elapsed_seconds": 12.4,
            "eta_seconds": 55.0,
            "eta_formatted": "55s",
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": None,
            "created_at": "2026-02-20T10:00:00",
            "completed_at": None,
        },
    },
    "procesando_detectando_rostros": {
        "summary": "Procesando — detectando rostros",
        "value": {
            "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
            "status": "processing",
            "processing_mode": "vertical",
            "message": "Detectando rostros en el video...",
            "progress": 45,
            "phase": "detecting_faces",
            "elapsed_seconds": 28.3,
            "eta_seconds": 34.6,
            "eta_formatted": "34s",
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": None,
            "created_at": "2026-02-20T10:00:00",
            "completed_at": None,
        },
    },
    "procesando_encoding": {
        "summary": "Procesando — generando video final",
        "value": {
            "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
            "status": "processing",
            "processing_mode": "short_manual",
            "message": "Generando video final...",
            "progress": 70,
            "phase": "encoding",
            "elapsed_seconds": 51.0,
            "eta_seconds": 21.9,
            "eta_formatted": "21s",
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": None,
            "created_at": "2026-02-20T10:00:00",
            "completed_at": None,
        },
    },
    "completado_vertical": {
        "summary": "Completado — modo vertical",
        "value": {
            "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
            "status": "completed",
            "processing_mode": "vertical",
            "message": "Video procesado exitosamente",
            "progress": 100,
            "phase": "completed",
            "elapsed_seconds": 87.3,
            "eta_seconds": 0.0,
            "eta_formatted": "0s",
            "output_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_tiktok/a3f2c1d4_output.mp4",
            "thumbnail_url": "https://res.cloudinary.com/mi-cloud/image/upload/processed_tiktok/thumbnails/a3f2c1d4_thumb.jpg",
            "preview_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_tiktok/previews/a3f2c1d4_preview.mp4",
            "quality_score": 0.91,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": 125.4,
            "error_detail": None,
            "created_at": "2026-02-20T10:00:00",
            "completed_at": "2026-02-20T10:01:27",
        },
    },
    "completado_short_auto": {
        "summary": "Completado — modo short_auto",
        "description": (
            "En modos short, segment_start y segment_duration indican "
            "qué parte del video original fue procesada."
        ),
        "value": {
            "job_id": "b7e1d2f3-9c0a-4b2e-cd34-e6f7a8b9c0d1",
            "status": "completed",
            "processing_mode": "short_auto",
            "message": "Video procesado exitosamente",
            "progress": 100,
            "phase": "completed",
            "elapsed_seconds": 62.1,
            "eta_seconds": 0.0,
            "eta_formatted": "0s",
            "output_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_tiktok/shorts/b7e1d2f3_output.mp4",
            "thumbnail_url": "https://res.cloudinary.com/mi-cloud/image/upload/processed_tiktok/thumbnails/b7e1d2f3_thumb.jpg",
            "preview_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_tiktok/previews/b7e1d2f3_preview.mp4",
            "quality_score": 0.87,
            "segment_start": 47.5,
            "segment_duration": 30,
            "output_duration_seconds": 30.0,
            "error_detail": None,
            "created_at": "2026-02-20T10:05:00",
            "completed_at": "2026-02-20T10:06:02",
        },
    },
    "completado_short_manual": {
        "summary": "Completado — modo short_manual",
        "value": {
            "job_id": "c9f3e4a5-0d1b-5c3f-de45-f7a8b9c0d1e2",
            "status": "completed",
            "processing_mode": "short_manual",
            "message": "Video procesado exitosamente",
            "progress": 100,
            "phase": "completed",
            "elapsed_seconds": 58.7,
            "eta_seconds": 0.0,
            "eta_formatted": "0s",
            "output_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_instagram/shorts/c9f3e4a5_output.mp4",
            "thumbnail_url": "https://res.cloudinary.com/mi-cloud/image/upload/processed_instagram/thumbnails/c9f3e4a5_thumb.jpg",
            "preview_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_instagram/previews/c9f3e4a5_preview.mp4",
            "quality_score": 0.93,
            "segment_start": 45.0,
            "segment_duration": 30,
            "output_duration_seconds": 30.0,
            "error_detail": None,
            "created_at": "2026-02-20T10:10:00",
            "completed_at": "2026-02-20T10:10:58",
        },
    },
    "fallido": {
        "summary": "Fallido — error durante el procesamiento",
        "value": {
            "job_id": "d1a4f5b6-1e2c-6d4g-ef56-a8b9c0d1e2f3",
            "status": "failed",
            "processing_mode": "short_manual",
            "message": "Error durante el procesamiento del video.",
            "progress": 0,
            "phase": "failed",
            "elapsed_seconds": 8.2,
            "eta_seconds": None,
            "eta_formatted": None,
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": (
                "El segmento solicitado (120.0s + 30s = 150.0s) "
                "excede la duración del video (95.0s). "
                "Reduce la duración o elige un tiempo de inicio más temprano."
            ),
            "created_at": "2026-02-20T10:15:00",
            "completed_at": "2026-02-20T10:15:08",
        },
    },
    "cancelado": {
        "summary": "Cancelado por el usuario",
        "value": {
            "job_id": "e2b5a6c7-2f3d-7e5h-fg67-b9c0d1e2f3a4",
            "status": "cancelled",
            "processing_mode": "vertical",
            "message": "Procesamiento cancelado por el usuario",
            "progress": 0,
            "phase": None,
            "elapsed_seconds": 34.1,
            "eta_seconds": None,
            "eta_formatted": None,
            "output_url": None,
            "thumbnail_url": None,
            "preview_url": None,
            "quality_score": None,
            "segment_start": None,
            "segment_duration": None,
            "output_duration_seconds": None,
            "error_detail": None,
            "created_at": "2026-02-20T10:20:00",
            "completed_at": "2026-02-20T10:20:34",
        },
    },
}


# ============================================================
# Router
# Tags separados por grupo de funcionalidad
# ============================================================

router = APIRouter(
    prefix="/api/video",
    tags=["Procesamiento de Video"],
)


# ============================================================
# Dependencias (inyectadas desde main)
# ============================================================

cloudinary_service = None
video_service = None


def set_services(cloudinary_svc: CloudinaryService, video_svc: VideoProcessingService):
    global cloudinary_service, video_service
    cloudinary_service = cloudinary_svc
    video_service = video_svc

    def progress_callback(job_id: str, progress_data: dict):
        if job_id in jobs_db:
            jobs_db[job_id].update({
                'progress': progress_data.get('progress', 0),
                'message': progress_data.get('message', ''),
                'phase': progress_data.get('phase'),
                'elapsed_seconds': progress_data.get('elapsed_seconds'),
                'eta_seconds': progress_data.get('eta_seconds'),
                'eta_formatted': progress_data.get('eta_formatted'),
            })

    video_service.set_progress_callback(progress_callback)


# ============================================================
# Background Task
# ============================================================

def process_video_task(job_id: str, request: VideoProcessRequest):
    try:
        logger.info(
            "Job iniciado | job_id=%s | mode=%s",
            job_id,
            request.processing_mode.value
        )

        jobs_db[job_id]['status'] = JobStatus.processing
        jobs_db[job_id]['message'] = 'Procesando el video...'
        jobs_db[job_id]['progress'] = 10

        output_url, metrics = video_service.process_video(request, job_id)

        jobs_db[job_id]['status'] = JobStatus.completed
        jobs_db[job_id]['message'] = 'Video procesado exitosamente'
        jobs_db[job_id]['progress'] = 100
        jobs_db[job_id]['output_url'] = output_url
        jobs_db[job_id]['thumbnail_url'] = metrics.get('thumbnail_url')
        jobs_db[job_id]['preview_url'] = metrics.get('preview_url')
        jobs_db[job_id]['quality_score'] = metrics.get('overall_quality')
        jobs_db[job_id]['completed_at'] = datetime.utcnow()
        jobs_db[job_id]['segment_start'] = metrics.get('segment_start')
        jobs_db[job_id]['segment_duration'] = metrics.get('segment_duration')
        jobs_db[job_id]['output_duration_seconds'] = metrics.get('output_duration_seconds')

        logger.info(
            "Job completado | job_id=%s | mode=%s | url=%s",
            job_id,
            request.processing_mode.value,
            output_url
        )

        # Notificar a Spring Boot — no bloquea ni falla el job si el webhook falla
        notify_job_completed(
            job_id=job_id,
            output_url=output_url,
            metrics=metrics,
            job_data=jobs_db[job_id],
        )

    except JobCancelledException:
        jobs_db[job_id]['status'] = JobStatus.cancelled
        jobs_db[job_id]['message'] = 'Procesamiento cancelado por el usuario'
        jobs_db[job_id]['progress'] = 0
        jobs_db[job_id]['completed_at'] = datetime.utcnow()
        logger.info("Job cancelado | job_id=%s", job_id)

        notify_job_cancelled(job_id=job_id, job_data=jobs_db[job_id])

    except VideoProcessingError as e:
        error_info = ErrorHandler.handle_error(e, job_id=job_id, operation="background_task")
        jobs_db[job_id]['status'] = JobStatus.failed
        jobs_db[job_id]['message'] = error_info['user_message']
        jobs_db[job_id]['error_detail'] = error_info['user_message']
        jobs_db[job_id]['progress'] = 0
        jobs_db[job_id]['completed_at'] = datetime.utcnow()
        logger.error(
            "Job falló (VideoProcessingError) | job_id=%s | error=%s",
            job_id,
            error_info['user_message']
        )

        notify_job_failed(
            job_id=job_id,
            error_message=error_info['user_message'],
            job_data=jobs_db[job_id],
        )

    except Exception as e:
        error_info = ErrorHandler.handle_error(e, job_id=job_id, operation="background_task")
        jobs_db[job_id]['status'] = JobStatus.failed
        jobs_db[job_id]['message'] = 'Error inesperado durante el procesamiento'
        jobs_db[job_id]['error_detail'] = error_info['user_message']
        jobs_db[job_id]['progress'] = 0
        jobs_db[job_id]['completed_at'] = datetime.utcnow()
        logger.exception("Job falló con error inesperado | job_id=%s", job_id)

        notify_job_failed(
            job_id=job_id,
            error_message=error_info['user_message'],
            job_data=jobs_db[job_id],
        )


# ============================================================
# ENDPOINT: POST /process
# ============================================================

@router.post(
    "/process",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Crear job de procesamiento",
    description="Envía un video a procesar en segundo plano y devuelve un `job_id` para hacer seguimiento.",
    responses={
        202: {
            "description": "Job creado exitosamente. Usar `job_id` para hacer polling del estado.",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
                        "status": "pending",
                        "message": "El video está en cola para procesarse",
                        "processing_mode": "short_auto",
                    }
                }
            },
        },
        400: _ERROR_400,
        401: _ERROR_401,
        500: _ERROR_500,
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "examples": _PROCESS_REQUEST_EXAMPLES
                }
            },
        }
    },
)
async def process_video(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks,
    token_data: ServiceTokenData = Depends(require_service_token),
):
    try:
        logger.info(
            "Validando request | mode=%s | platform=%s",
            request.processing_mode.value,
            request.platform.value
        )
        video_info = validate_video_request(request)
        logger.info("Request validado correctamente")

    except ValidationError as e:
        logger.warning("Validación falló: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Error inesperado durante validación")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al validar el request"
        )

    job_id = str(uuid.uuid4())

    jobs_db[job_id] = {
        'job_id': job_id,
        'user_id': token_data.user_id,
        'status': JobStatus.pending,
        'message': 'El video está en cola para procesarse',
        'processing_mode': request.processing_mode,
        'progress': 0,
        'output_url': None,
        'thumbnail_url': None,  
        'preview_url': None,
        'quality_score': None,
        'error_detail': None,
        'segment_start': None,
        'segment_duration': None,
        'output_duration_seconds': None,
        'created_at': datetime.utcnow(),
        'completed_at': None,
        'request': request.model_dump(),
        'video_info': video_info,
    }

    background_tasks.add_task(process_video_task, job_id, request)

    logger.info(
        "Job creado | job_id=%s | user_id=%s | mode=%s | platform=%s | quality=%s",
        job_id,
        token_data.user_id,
        request.processing_mode.value,
        request.platform.value,
        request.quality.value
    )

    return VideoProcessResponse(
        job_id=job_id,
        status=JobStatus.pending,
        message="El video está en cola para procesarse",
        processing_mode=request.processing_mode,
    )


# ============================================================
# ENDPOINT: GET /status/{job_id}
# ============================================================

@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Consultar estado del job",
    description="""
Retorna el estado actual del job incluyendo progreso, fase y resultado.

## Polling recomendado

```
Intervalo sugerido: cada 3 segundos mientras status = "processing"
Detener polling cuando: status = "completed", "failed" o "cancelled"
```

## Campos de short

Los campos `segment_start` y `segment_duration` solo tienen valor
cuando `processing_mode` es `short_auto` o `short_manual` y `status = completed`.
Indican qué parte del video original fue procesada.

## quality_score

Valor entre 0.0 y 1.0 que indica la calidad del tracking de rostros.
- `>= 0.85` → Excelente
- `0.70 - 0.84` → Bueno
- `< 0.70` → Revisar resultado manualmente
""",
    responses={
        200: {
            "description": "Estado del job",
            "content": {
                "application/json": {
                    "examples": _STATUS_RESPONSE_EXAMPLES
                }
            },
        },
        401: _ERROR_401,
        403: _ERROR_403,
        404: _ERROR_404,
    },
)
async def get_job_status(
    job_id: str,
    token_data: ServiceTokenData = Depends(require_service_token),
):
    job_data = jobs_db.get(job_id)
    verify_job_ownership(job_data, token_data, job_id)

    return JobStatusResponse(
        job_id=job_data['job_id'],
        status=job_data['status'],
        message=job_data['message'],
        processing_mode=job_data.get('processing_mode'),
        progress=job_data.get('progress'),
        phase=job_data.get('phase'),
        elapsed_seconds=job_data.get('elapsed_seconds'),
        eta_seconds=job_data.get('eta_seconds'),
        eta_formatted=job_data.get('eta_formatted'),
        output_url=job_data.get('output_url'),
        thumbnail_url=job_data.get('thumbnail_url'),
        preview_url=job_data.get('preview_url'),
        quality_score=job_data.get('quality_score'),
        segment_start=job_data.get('segment_start'),
        segment_duration=job_data.get('segment_duration'),
        output_duration_seconds=job_data.get('output_duration_seconds'),
        error_detail=job_data.get('error_detail'),
        created_at=job_data.get('created_at'),
        completed_at=job_data.get('completed_at'),
    )


# ============================================================
# ENDPOINT: GET /download/{job_id}
# ============================================================

@router.get(
    "/download/{job_id}",
    summary="Obtener URL de descarga",
    description="""
Retorna la URL del video procesado cuando el job está completado.

**Nota:** La URL apunta directamente a Cloudinary. No es necesario
descargar el archivo a través del backend; el cliente puede acceder
a la URL directamente.
""",
    responses={
        200: {
            "description": "URL de descarga disponible",
            "content": {
                "application/json": {
                    "examples": {
                        "vertical": {
                            "summary": "Descarga de modo vertical",
                            "value": {
                                "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
                                "status": "completed",
                                "processing_mode": "vertical",
                                "video_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_tiktok/a3f2c1d4_output.mp4",
                                "quality_score": 0.91,
                                "segment_start": None,
                                "segment_duration": None,
                            },
                        },
                        "short": {
                            "summary": "Descarga de short (auto o manual)",
                            "value": {
                                "job_id": "b7e1d2f3-9c0a-4b2e-cd34-e6f7a8b9c0d1",
                                "status": "completed",
                                "processing_mode": "short_auto",
                                "video_url": "https://res.cloudinary.com/mi-cloud/video/upload/processed_tiktok/shorts/b7e1d2f3_output.mp4",
                                "quality_score": 0.87,
                                "segment_start": 47.5,
                                "segment_duration": 30,
                            },
                        },
                    }
                }
            },
        },
        400: {
            "description": "El video aún no está listo",
            "content": {
                "application/json": {
                    "example": {"detail": "El video no está listo para descarga"}
                }
            },
        },
        401: _ERROR_401,
        403: _ERROR_403,
        404: _ERROR_404,
    },
)
async def download_video(
    job_id: str,
    token_data: ServiceTokenData = Depends(require_service_token),
):
    job_data = jobs_db.get(job_id)
    verify_job_ownership(job_data, token_data, job_id)

    if job_data['status'] != JobStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El video no está listo para descarga"
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "processing_mode": job_data.get('processing_mode'),
        "video_url": job_data['output_url'],
        "quality_score": job_data.get('quality_score'),
        "segment_start": job_data.get('segment_start'),
        "segment_duration": job_data.get('segment_duration'),
    }


# ============================================================
# ENDPOINT: POST /jobs/{job_id}/cancel
# ============================================================

@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancelar job en proceso",
    description="""
Solicita la cancelación de un job que está en proceso.

El procesamiento se detiene de forma segura en el próximo punto de
control interno. La respuesta es inmediata pero la cancelación
efectiva puede tardar unos segundos.

Solo se pueden cancelar jobs con `status = pending` o `status = processing`.
""",
    responses={
        200: {
            "description": "Cancelación solicitada",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
                        "message": "Cancelación solicitada. El procesamiento se detendrá pronto.",
                        "previous_status": "processing",
                    }
                }
            },
        },
        400: _ERROR_400,
        401: _ERROR_401,
        403: _ERROR_403,
        404: _ERROR_404,
    },
)
async def cancel_job(
    job_id: str,
    token_data: ServiceTokenData = Depends(require_service_token),
):
    job_data = jobs_db.get(job_id)
    verify_job_ownership(job_data, token_data, job_id)
    current_status = job_data['status']

    if current_status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El job ya está en estado '{current_status}' y no puede ser cancelado"
        )

    cancellation_manager.request_cancellation(job_id)
    jobs_db[job_id]['message'] = 'Cancelando procesamiento...'

    logger.info(
        "Cancelación solicitada | job_id=%s | current_status=%s",
        job_id,
        current_status
    )

    return {
        "job_id": job_id,
        "message": "Cancelación solicitada. El procesamiento se detendrá pronto.",
        "previous_status": current_status,
    }


# ============================================================
# ENDPOINTS DE UTILIDAD
# ============================================================

@router.get(
    "/jobs",
    summary="Listar todos los jobs",
    description="Retorna el listado de todos los jobs en memoria con su estado actual.",
    tags=["Utilidades"],
    responses={
        200: {
            "description": "Listado de jobs",
            "content": {
                "application/json": {
                    "example": {
                        "total_jobs": 2,
                        "jobs": [
                            {
                                "job_id": "a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0",
                                "status": "completed",
                                "processing_mode": "short_auto",
                                "created_at": "2026-02-20T10:00:00",
                                "platform": "tiktok",
                                "quality": "normal",
                            },
                            {
                                "job_id": "b7e1d2f3-9c0a-4b2e-cd34-e6f7a8b9c0d1",
                                "status": "processing",
                                "processing_mode": "vertical",
                                "created_at": "2026-02-20T10:05:00",
                                "platform": "instagram",
                                "quality": "high",
                            },
                        ],
                    }
                }
            },
        }
    },
)
async def list_jobs(
    token_data: ServiceTokenData = Depends(require_service_token),
):
    logger.debug("Listado de jobs solicitado | user_id=%s", token_data.user_id)

    user_jobs = [
        {
            "job_id": job_id,
            "status": data['status'],
            "processing_mode": data.get('processing_mode'),
            "created_at": data['created_at'],
            "platform": data['request']['platform'],
            "quality": data['request']['quality'],
        }
        for job_id, data in jobs_db.items()
        if data.get('user_id') == token_data.user_id
    ]

    return {
        "total_jobs": len(user_jobs),
        "jobs": user_jobs,
    }


@router.delete(
    "/jobs/{job_id}",
    summary="Eliminar job",
    description="Elimina un job del almacenamiento en memoria. No afecta los archivos en Cloudinary.",
    tags=["Utilidades"],
    responses={
        200: {
            "description": "Job eliminado",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Job a3f2c1d4-8b9e-4f1a-bc23-d5e6f7a8b9c0 eliminado exitosamente"
                    }
                }
            },
        },
        404: _ERROR_404,
    },
)
async def delete_job(
    job_id: str,
    token_data: ServiceTokenData = Depends(require_service_token),
):
    job_data = jobs_db.get(job_id)
    verify_job_ownership(job_data, token_data, job_id)

    del jobs_db[job_id]
    logger.info("Job eliminado | job_id=%s | user_id=%s", job_id, token_data.user_id)

    return {"message": f"Job {job_id} eliminado exitosamente"}
