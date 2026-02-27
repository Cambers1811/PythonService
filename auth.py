"""
Auth — Validación del JWT de servicio emitido por Spring Boot.

Arquitectura (Opción B — microservicio interno, servidores separados):

  Frontend → [HTTPS] → Spring Boot → [HTTPS + JWT de servicio] → Microservicio Python

Estructura del JWT que emite Spring Boot:
  {
    "sub":        "41c1d3c1-e75c-49eb-bca4-b6fee3867444",  ← user_id
    "iat":        1772017361,
    "exp":        1772017661,
    "aud":        "python-service",    ← identifica al destinatario
    "token_type": "DELEGATED_SERVICE", ← distingue del token de usuario
    "scope":      "PYTHON_SERVICE"     ← confirma el propósito
  }

Capas de seguridad (en orden de ejecución):
  1. X-Service-Key    — middleware.py rechaza si no coincide (antes del JWT)
  2. JWT presente     — header Authorization: Bearer <token> requerido
  3. Firma válida     — HS256 con SERVICE_JWT_SECRET compartida con Spring Boot
  4. No expirado      — campo exp verificado por PyJWT
  5. Vida máxima      — iat verificado: token más antiguo que MAX_AGE → rechazado
  6. Audience válido  — aud debe ser "python-service"
  7. token_type válido — debe ser "DELEGATED_SERVICE" (no un token de usuario)
  8. scope válido     — debe ser "PYTHON_SERVICE"
  9. sub presente     — es el user_id para aislar jobs por usuario

Por qué verificar token_type y scope además de aud:
  Si alguien obtiene un JWT de usuario (aud distinto) no puede usarlo aquí
  porque aud fallaría. Pero token_type y scope son una segunda garantía:
  confirman que Spring Boot emitió este token específicamente para Python,
  no que sea un token de usuario que accidentalmente pasó el filtro de aud.

Configuración requerida (variables de entorno):
  SERVICE_JWT_SECRET  — secret HS256 compartida con Spring Boot (jwt.secret en Spring)
  SERVICE_JWT_MAX_AGE — vida máxima del token en segundos (default: 300 = 5 min)

Uso en endpoints:
  @router.post("/process")
  async def process_video(
      request: VideoProcessRequest,
      token_data: ServiceTokenData = Depends(require_service_token),
  ):
      user_id = token_data.user_id  # UUID del usuario, extraído del sub del JWT
      ...
"""

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Valores esperados del JWT — deben coincidir con lo que emite Spring Boot
# -------------------------------------------------------------------

# Valor del campo "aud" — identifica a este microservicio como destinatario
EXPECTED_AUDIENCE = "python-service"

# Valor del campo "token_type" — distingue del JWT de usuario
EXPECTED_TOKEN_TYPE = "DELEGATED_SERVICE"

# Valor del campo "scope" — confirma el propósito del token
EXPECTED_SCOPE = "PYTHON_SERVICE"


# -------------------------------------------------------------------
# Configuración — leída desde variables de entorno
# -------------------------------------------------------------------

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Variable de entorno requerida no configurada: '{name}'. "
            f"Agrega esta variable al entorno del proceso."
        )
    return value


try:
    _secret_b64: str = _get_required_env("SERVICE_JWT_SECRET")
    # Spring Boot hace Decoders.BASE64.decode(secretKey) antes de crear la key.
    # PyJWT usa el string directamente como bytes, así que hay que decodificar
    # el Base64 aquí para que ambos lados trabajen con los mismos bytes.
    SERVICE_JWT_SECRET: bytes = base64.b64decode(_secret_b64)
    SERVICE_JWT_ALGORITHM: str = "HS256"
    SERVICE_JWT_MAX_AGE_SECONDS: int = int(os.getenv("SERVICE_JWT_MAX_AGE", "300"))
except EnvironmentError as e:
    logger.error("CONFIGURACIÓN INCOMPLETA: %s", str(e))
    SERVICE_JWT_SECRET = b""
    SERVICE_JWT_ALGORITHM = "HS256"
    SERVICE_JWT_MAX_AGE_SECONDS = 300


# -------------------------------------------------------------------
# Modelo de datos del token validado
# -------------------------------------------------------------------

class ServiceTokenData:
    """
    Datos extraídos del JWT después de validarlo correctamente.

    user_id proviene del campo "sub" del JWT — es el UUID del usuario
    que Spring Boot autenticó antes de llamar al microservicio.

    Python usa user_id para:
      1. Asociar cada job al usuario que lo creó (guardado en jobs_db).
      2. Filtrar GET /api/video/jobs — solo retorna jobs del usuario.
      3. Verificar propiedad en status, cancel y delete.

    Python NO usa user_id para consultar la BD — no tiene acceso a ella.
    La autorización sobre el video ya la hizo Spring Boot antes de llamar.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

    def __repr__(self) -> str:
        return f"ServiceTokenData(user_id={self.user_id!r})"


# -------------------------------------------------------------------
# Extractor del header Authorization
# -------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


# -------------------------------------------------------------------
# Dependencia principal
# -------------------------------------------------------------------

async def require_service_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> ServiceTokenData:
    """
    Dependencia de FastAPI que valida el JWT de servicio.

    Orden de verificación:
      1. Header Authorization presente.
      2. Firma HS256 válida con SERVICE_JWT_SECRET.
      3. No expirado (campo exp).
      4. Audience correcto (aud == "python-service").
      5. Vida máxima no superada (iat + MAX_AGE > ahora).
      6. token_type == "DELEGATED_SERVICE".
      7. scope == "PYTHON_SERVICE".
      8. sub presente (user_id del usuario).

    Todos los errores retornan 401 con mensaje genérico —
    no se revela al cliente qué parte de la validación falló.
    """
    if not SERVICE_JWT_SECRET:
        logger.error("SERVICE_JWT_SECRET no configurada o inválida — todos los requests serán rechazados.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio no disponible.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 1. Header presente
    if credentials is None:
        logger.warning("Request rechazado: header Authorization ausente")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticación requerida.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # 2-4. Firma, expiración y audience en una sola llamada a PyJWT
    try:
        payload = jwt.decode(
            token,
            SERVICE_JWT_SECRET,
            algorithms=[SERVICE_JWT_ALGORITHM],
            audience=EXPECTED_AUDIENCE,   # verifica campo "aud"
            options={
                "verify_exp": True,
                "verify_iat": True,
                "require": ["sub", "exp", "iat", "aud", "token_type", "scope"],
            },
        )

    except jwt.ExpiredSignatureError:
        logger.warning("Token de servicio expirado")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        # aud no es "python-service" — token no destinado a este microservicio
        logger.warning("Token con audience inválido — esperado: '%s'", EXPECTED_AUDIENCE)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token no autorizado.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.MissingRequiredClaimError as e:
        logger.warning("Token con campo requerido ausente: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning("Token de servicio inválido: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 5. Vida máxima — segunda capa contra replay attacks
    iat = payload.get("iat")
    if iat is not None:
        token_age = (
            datetime.now(timezone.utc) - datetime.fromtimestamp(iat, tz=timezone.utc)
        ).total_seconds()

        if token_age > SERVICE_JWT_MAX_AGE_SECONDS:
            logger.warning(
                "Token demasiado antiguo | age=%.0fs | max=%ds",
                token_age, SERVICE_JWT_MAX_AGE_SECONDS,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expirado.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 6. token_type — confirma que es un token de servicio delegado
    token_type = payload.get("token_type")
    if token_type != EXPECTED_TOKEN_TYPE:
        logger.warning(
            "token_type inválido | recibido='%s' | esperado='%s'",
            token_type, EXPECTED_TOKEN_TYPE,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token no autorizado.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 7. scope — confirma el propósito específico del token
    scope = payload.get("scope")
    if scope != EXPECTED_SCOPE:
        logger.warning(
            "scope inválido | recibido='%s' | esperado='%s'",
            scope, EXPECTED_SCOPE,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token no autorizado.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 8. Extraer user_id del sub (ya verificado como presente por "require")
    user_id = str(payload["sub"])

    logger.debug("Token de servicio válido | user_id=%s", user_id)

    return ServiceTokenData(user_id=user_id)


# -------------------------------------------------------------------
# Helper para verificar propiedad de un job
# -------------------------------------------------------------------

def verify_job_ownership(job_data: dict, token_data: ServiceTokenData, job_id: str):
    """
    Verifica que el job pertenece al usuario del token.

    Spring Boot ya verificó que el usuario tiene acceso al video antes
    de llamar al microservicio. Esta verificación es una segunda capa:
    garantiza que un request de Spring Boot en nombre del usuario A
    no pueda acceder a jobs creados por el usuario B, incluso si
    ambos requests llegan con tokens válidos.

    Retorna 403 (no 404) cuando el job existe pero pertenece a otro
    usuario — hace la distinción explícita entre "no existe" y
    "existe pero no es tuyo".
    """
    if job_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} no encontrado.",
        )

    job_owner = job_data.get("user_id")

    if job_owner != token_data.user_id:
        logger.warning(
            "Acceso denegado a job | job_id=%s | owner=%s | requester=%s",
            job_id, job_owner, token_data.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para acceder a este job.",
        )
