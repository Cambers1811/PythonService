"""
Middleware de seguridad — Primera línea de defensa para servidores separados.

Con Spring Boot y el microservicio Python en servidores distintos sin red privada
(dos cuentas Render en free tier), la protección se basa en dos capas:

  1. ServiceKeyMiddleware    — verifica header X-Service-Key antes del JWT
  2. SecurityHeadersMiddleware — headers HTTP de seguridad en todas las respuestas

Flujo de validación por capas (orden de ejecución):

  Request entrante
    │
    ▼
  ServiceKeyMiddleware
    │  ¿Header X-Service-Key presente y correcto?
    │  NO → 401 inmediato (nunca llega al JWT)
    │  SÍ → continúa
    ▼
  require_service_token (auth.py)
    │  ¿JWT válido, no expirado, issuer correcto, user_id presente?
    │  NO → 401
    │  SÍ → continúa
    ▼
  Endpoint (lógica de negocio)

Por qué X-Service-Key + JWT y no solo JWT:
  - El JWT prueba "quién es el usuario y que Spring Boot autorizó la operación"
  - X-Service-Key prueba "que quien llama es Spring Boot" (identidad del servicio)
  - Son capas ortogonales: si alguien obtiene el JWT de un usuario pero no
    conoce X-Service-Key, no puede llamar al microservicio directamente.
  - La comparación usa secrets.compare_digest para evitar timing attacks.

Configuración requerida (variables de entorno):
  SERVICE_API_KEY  — valor secreto compartido con Spring Boot
                     Genera uno con: openssl rand -hex 32
                     Si está vacío, el middleware loguea advertencia
                     pero deja pasar (útil en desarrollo local).

Cómo registrar los middlewares en main.py:
  from middleware import ServiceKeyMiddleware, SecurityHeadersMiddleware

  app.add_middleware(SecurityHeadersMiddleware)
  app.add_middleware(ServiceKeyMiddleware)

  # FastAPI ejecuta middlewares en orden inverso al registro.
  # Con el orden de arriba: ServiceKey corre primero, SecurityHeaders después.

Qué configurar en Spring Boot:
  Agregar el header en cada request al microservicio:
    headers.set("X-Service-Key", System.getenv("SERVICE_API_KEY"));
  La variable SERVICE_API_KEY debe tener el mismo valor en ambos servicios.
"""

import logging
import os
import secrets

from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

load_dotenv()

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Nombre del header — definido como constante para no tener strings
# dispersos por el código. Si se cambia el nombre, se cambia aquí.
# -------------------------------------------------------------------

SERVICE_KEY_HEADER = "X-Service-Key"


# -------------------------------------------------------------------
# Configuración — leída una vez al importar el módulo
# -------------------------------------------------------------------

_SERVICE_API_KEY: str = os.getenv("SERVICE_API_KEY", "").strip()

if not _SERVICE_API_KEY:
    logger.warning(
        "SERVICE_API_KEY no configurada. "
        "La verificación de X-Service-Key está DESACTIVADA. "
        "Configura esta variable en producción antes de desplegar."
    )
else:
    # No logueamos el valor — solo que está configurado
    logger.info("Service key configurada y activa | header=%s", SERVICE_KEY_HEADER)


# -------------------------------------------------------------------
# Rutas excluidas de la verificación
# El health check no requiere autenticación — lo usa Render internamente
# para saber si el servicio está vivo.
# -------------------------------------------------------------------

_EXCLUDED_PATHS = {
    "/health",
    "/",
}


# -------------------------------------------------------------------
# Middleware 1: Service Key
# -------------------------------------------------------------------

class ServiceKeyMiddleware(BaseHTTPMiddleware):
    """
    Verifica que el header X-Service-Key esté presente y sea correcto.

    Es la primera capa de defensa — corre antes de que el JWT sea
    evaluado. Un atacante que no conozca SERVICE_API_KEY no puede
    llegar a los endpoints aunque tenga un JWT válido.

    Comportamiento según configuración:
      - Sin SERVICE_API_KEY configurada: deja pasar todo con advertencia.
        Útil en desarrollo local donde no hay Spring Boot real.
      - Con SERVICE_API_KEY: rechaza con 401 si el header está ausente
        o tiene un valor incorrecto.

    La comparación usa secrets.compare_digest en lugar de == para
    evitar timing attacks — ataques que miden el tiempo de respuesta
    para adivinar el valor del secret carácter a carácter.

    Por qué 401 y no 403:
      401 significa "necesitas autenticarte". El header X-Service-Key
      es una forma de autenticación del servicio (no del usuario), así
      que semánticamente 401 es más correcto aquí que 403.
    """

    async def dispatch(self, request: Request, call_next):
        # Paths excluidos (health check, raíz)
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        # Si no hay key configurada, dejar pasar (modo desarrollo)
        if not _SERVICE_API_KEY:
            return await call_next(request)

        # Verificar presencia del header
        incoming_key = request.headers.get(SERVICE_KEY_HEADER, "")

        if not incoming_key:
            logger.warning(
                "Request rechazado: header %s ausente | path=%s | method=%s",
                SERVICE_KEY_HEADER,
                request.url.path,
                request.method,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Autenticación requerida."},
            )

        # Comparación en tiempo constante para evitar timing attacks
        # secrets.compare_digest tarda lo mismo independientemente de
        # en qué carácter difieren los dos strings.
        if not secrets.compare_digest(incoming_key, _SERVICE_API_KEY):
            logger.warning(
                "Request rechazado: %s inválido | path=%s | method=%s",
                SERVICE_KEY_HEADER,
                request.url.path,
                request.method,
                # Nota: nunca logueamos el valor recibido — podría ser
                # un intento de ataque y loguear el valor sería un riesgo.
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Autenticación requerida."},
            )

        logger.debug(
            "Service key válida | path=%s",
            request.url.path,
        )

        return await call_next(request)


# -------------------------------------------------------------------
# Middleware 2: Security Headers
# -------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Agrega headers HTTP de seguridad a todas las respuestas.

    Estos headers son estándar en APIs de producción y protegen contra
    ataques comunes como clickjacking, MIME sniffing y más.

    Como este microservicio es una API (no sirve HTML), los headers
    están ajustados para ese contexto.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Evita que el navegador infiera el Content-Type
        # Protege contra ataques de MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Evita que la respuesta se muestre en un iframe
        response.headers["X-Frame-Options"] = "DENY"

        # Fuerza HTTPS en el cliente una vez que lo contactaron por HTTPS
        # max-age=31536000 = 1 año
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        # No enviar URL de origen en requests externos
        response.headers["Referrer-Policy"] = "no-referrer"

        # Desactiva features del navegador que una API no necesita
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # Eliminar el header que revela el framework
        # Un atacante no necesita saber que usamos Starlette/FastAPI
        response.headers.pop("server", None)

        return response
