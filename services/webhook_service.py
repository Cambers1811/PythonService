"""
Webhook Service — Notifica a Spring Boot sobre el estado de los jobs.

Versión 2: Agrega notificación de progreso en tiempo real.

Responsabilidades:
  - Notificar cuando un job completa/falla/cancela (existente)
  - Notificar progreso en tiempo real durante el procesamiento (nuevo)

Configuración requerida (variables de entorno):
  SPRING_BOOT_WEBHOOK_URL  — URL para notificaciones finales
                              Ejemplo: https://api.com/api/internal/processing-jobs/webhook
  SPRING_BOOT_PROGRESS_WEBHOOK_URL — URL para notificaciones de progreso
                              Ejemplo: https://api.com/api/webhook/progress
  SERVICE_API_KEY          — API key compartido con Spring Boot

Comportamiento ante fallos:
  - Notificaciones finales: Reintenta 3 veces con backoff.
  - Notificaciones de progreso: 1 solo intento (no crítico).
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------

WEBHOOK_URL: str = os.getenv("SPRING_BOOT_WEBHOOK_URL", "")
PROGRESS_WEBHOOK_URL: str = os.getenv("SPRING_BOOT_PROGRESS_WEBHOOK_URL", "")
SERVICE_API_KEY: str = os.getenv("SERVICE_API_KEY", "")

# Reintentos con backoff exponencial (solo para webhooks finales)
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 1.0
WEBHOOK_TIMEOUT: int = 10

# Timeout para webhooks de progreso (no se reintenta)
PROGRESS_WEBHOOK_TIMEOUT: int = 3  # 3 segundos

if not WEBHOOK_URL:
    logger.warning(
        "SPRING_BOOT_WEBHOOK_URL no configurada. "
        "Las notificaciones de jobs completados están DESACTIVADAS."
    )

if not PROGRESS_WEBHOOK_URL:
    logger.warning(
        "SPRING_BOOT_PROGRESS_WEBHOOK_URL no configurada. "
        "Las notificaciones de progreso en tiempo real están DESACTIVADAS."
    )

if not SERVICE_API_KEY:
    logger.warning(
        "SERVICE_API_KEY no configurada. "
        "Spring Boot rechazará las notificaciones del webhook."
    )


# -------------------------------------------------------------------
# Webhooks de notificación FINAL (completado/fallido/cancelado)
# -------------------------------------------------------------------

def notify_job_completed(
    job_id: str,
    output_url: str,
    metrics: dict,
    job_data: dict,
) -> bool:
    """
    Notifica a Spring Boot que un job terminó exitosamente.

    Args:
        job_id:     ID del job en Python.
        output_url: URL del video procesado en Cloudinary.
        metrics:    Diccionario de métricas retornado por video_service.
        job_data:   Entrada de jobs_db para obtener timestamps y modo.

    Returns:
        True si la notificación fue exitosa, False si falló tras reintentos.
    """
    if not WEBHOOK_URL:
        logger.debug("Webhook desactivado — omitiendo notificación | job_id=%s", job_id)
        return False

    payload = _build_completed_payload(job_id, output_url, metrics, job_data)
    return _send_with_retry(job_id, payload, WEBHOOK_URL)


def notify_job_failed(
    job_id: str,
    error_message: str,
    job_data: dict,
) -> bool:
    """
    Notifica a Spring Boot que un job falló.

    Args:
        job_id:        ID del job en Python.
        error_message: Mensaje de error para mostrar al usuario.
        job_data:      Entrada de jobs_db para obtener timestamps y modo.

    Returns:
        True si la notificación fue exitosa, False si falló tras reintentos.
    """
    if not WEBHOOK_URL:
        logger.debug("Webhook desactivado — omitiendo notificación | job_id=%s", job_id)
        return False

    payload = _build_failed_payload(job_id, error_message, job_data)
    return _send_with_retry(job_id, payload, WEBHOOK_URL)


def notify_job_cancelled(job_id: str, job_data: dict) -> bool:
    """
    Notifica a Spring Boot que un job fue cancelado.
    """
    if not WEBHOOK_URL:
        return False

    payload = _build_cancelled_payload(job_id, job_data)
    return _send_with_retry(job_id, payload, WEBHOOK_URL)


# -------------------------------------------------------------------
# Webhook de PROGRESO en tiempo real (nuevo)
# -------------------------------------------------------------------

def notify_progress(job_id: str, progress_data: Dict[str, Any]) -> bool:
    """
    Notifica a Spring Boot sobre el progreso actual del job.
    
    Este webhook NO se reintenta — es best-effort.
    Si falla, el progreso simplemente no se actualiza en esa iteración.
    
    Args:
        job_id:        ID del job.
        progress_data: Diccionario con progreso actual del ProgressTracker.
        
    Returns:
        True si se notificó exitosamente, False si falló.
    """
    if not PROGRESS_WEBHOOK_URL:
        return False
    
    payload = _build_progress_payload(job_id, progress_data)
    return _send_progress_webhook(job_id, payload)


# -------------------------------------------------------------------
# Constructores de payload
# -------------------------------------------------------------------

def _build_completed_payload(
    job_id: str,
    output_url: str,
    metrics: dict,
    job_data: dict,
) -> dict:
    """
    Construye el payload para un job completado.
    Los campos coinciden exactamente con ProcessingJobWebhookRequest en Spring Boot.
    """
    return {
        "job_id":                   job_id,
        "status":                   "completed",
        "processing_mode":          metrics.get("processing_mode", job_data.get("request", {}).get("processing_mode")),
        "elapsed_seconds":          metrics.get("processing_total_time"),
        "phase":                    "completed",
        "completed_at":             datetime.now(timezone.utc).isoformat(),

        # Resultado del video
        "output_url":               output_url,
        "thumbnail_url":            metrics.get("thumbnail_url"),
        "preview_url":              metrics.get("preview_url"),
        "quality_score":            metrics.get("overall_quality"),
        "output_duration_seconds":  metrics.get("output_duration_seconds"),

        # Campos de segmento — solo para shorts
        "segment_start":            metrics.get("segment_start"),
        "segment_duration":         metrics.get("segment_duration"),

        # Sin error
        "error_detail":             None,
    }


def _build_failed_payload(
    job_id: str,
    error_message: str,
    job_data: dict,
) -> dict:
    return {
        "job_id":           job_id,
        "status":           "failed",
        "processing_mode":  job_data.get("request", {}).get("processing_mode"),
        "elapsed_seconds":  None,
        "phase":            "failed",
        "completed_at":     datetime.now(timezone.utc).isoformat(),

        # Sin resultado
        "output_url":       None,
        "thumbnail_url":    None,
        "preview_url":      None,
        "quality_score":    None,
        "output_duration_seconds": None,
        "segment_start":    None,
        "segment_duration": None,

        "error_detail":     error_message,
    }


def _build_cancelled_payload(job_id: str, job_data: dict) -> dict:
    return {
        "job_id":           job_id,
        "status":           "cancelled",
        "processing_mode":  job_data.get("request", {}).get("processing_mode"),
        "elapsed_seconds":  None,
        "phase":            "cancelled",
        "completed_at":     datetime.now(timezone.utc).isoformat(),

        "output_url":       None,
        "thumbnail_url":    None,
        "preview_url":      None,
        "quality_score":    None,
        "output_duration_seconds": None,
        "segment_start":    None,
        "segment_duration": None,

        "error_detail":     None,
    }


def _build_progress_payload(job_id: str, progress_data: Dict[str, Any]) -> dict:
    """
    Construye el payload para notificación de progreso.
    
    Campos enviados a Spring Boot:
      - job_id
      - status (siempre "processing")
      - progress (0-100)
      - phase (string)
      - eta_seconds
      - elapsed_seconds
      - message
    """
    return {
        "job_id":           job_id,
        "status":           "processing",  # Siempre "processing" para webhooks de progreso
        "progress":         progress_data.get("progress", 0),
        "phase":            progress_data.get("phase"),
        "eta_seconds":      progress_data.get("eta_seconds"),
        "elapsed_seconds":  progress_data.get("elapsed_seconds"),
        "message":          progress_data.get("message", "Procesando..."),
    }


# -------------------------------------------------------------------
# HTTP con reintentos (solo para webhooks finales)
# -------------------------------------------------------------------

def _send_with_retry(job_id: str, payload: dict, webhook_url: str) -> bool:
    """
    Envía el webhook con reintentos y backoff exponencial.

    Intento 1: inmediato
    Intento 2: espera 1s
    Intento 3: espera 2s
    Si los 3 fallan: loguea error, retorna False (no lanza excepción)
    """
    headers = {
        "Content-Type":  "application/json",
        "X-Service-Key": SERVICE_API_KEY,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=WEBHOOK_TIMEOUT,
            )

            if response.status_code in (200, 201, 204):
                logger.info(
                    "Webhook enviado exitosamente | job_id=%s | status=%d | attempt=%d",
                    job_id, response.status_code, attempt,
                )
                return True

            # 4xx — error del cliente, no tiene caso reintentar
            if 400 <= response.status_code < 500:
                logger.error(
                    "Webhook rechazado por Spring Boot (no se reintenta) | "
                    "job_id=%s | status=%d | body=%s",
                    job_id, response.status_code, response.text[:200],
                )
                return False

            # 5xx — error del servidor, puede recuperarse
            logger.warning(
                "Webhook falló con error de servidor | "
                "job_id=%s | status=%d | attempt=%d/%d",
                job_id, response.status_code, attempt, MAX_RETRIES,
            )

        except requests.Timeout:
            logger.warning(
                "Webhook timeout | job_id=%s | attempt=%d/%d",
                job_id, attempt, MAX_RETRIES,
            )
        except requests.ConnectionError:
            logger.warning(
                "Webhook connection error | job_id=%s | attempt=%d/%d",
                job_id, attempt, MAX_RETRIES,
            )
        except Exception as e:
            logger.warning(
                "Webhook error inesperado | job_id=%s | attempt=%d/%d | error=%s",
                job_id, attempt, MAX_RETRIES, str(e),
            )

        # Backoff exponencial antes del siguiente intento
        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 1s, 2s
            logger.debug("Reintentando webhook en %.1fs | job_id=%s", delay, job_id)
            time.sleep(delay)

    logger.error(
        "Webhook falló tras %d intentos — Spring Boot no fue notificado | job_id=%s",
        MAX_RETRIES, job_id,
    )
    return False


# -------------------------------------------------------------------
# HTTP sin reintentos (solo para webhooks de progreso)
# -------------------------------------------------------------------

def _send_progress_webhook(job_id: str, payload: dict) -> bool:
    """
    Envía webhook de progreso sin reintentos (best-effort).
    
    Si falla, solo loguea warning — no es crítico.
    El progreso se puede recuperar en la siguiente actualización.
    """
    headers = {
        "Content-Type":  "application/json",
        "X-Service-Key": SERVICE_API_KEY,
    }

    try:
        response = requests.post(
            PROGRESS_WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=PROGRESS_WEBHOOK_TIMEOUT,
        )

        if response.status_code in (200, 201, 204):
            logger.debug(
                "Webhook de progreso enviado | job_id=%s | progress=%d%% | phase=%s",
                job_id,
                payload.get("progress", 0),
                payload.get("phase")
            )
            return True
        else:
            logger.warning(
                "Webhook de progreso rechazado | job_id=%s | status=%d",
                job_id,
                response.status_code
            )
            return False

    except requests.Timeout:
        logger.warning(
            "Webhook de progreso timeout | job_id=%s",
            job_id
        )
        return False
    except requests.ConnectionError:
        logger.warning(
            "Webhook de progreso connection error | job_id=%s",
            job_id
        )
        return False
    except Exception as e:
        logger.warning(
            "Webhook de progreso error | job_id=%s | error=%s",
            job_id,
            str(e)
        )
        return False
