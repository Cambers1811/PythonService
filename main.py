"""
Main.py — Punto de entrada de la API
Ensambla todos los componentes y arranca el servidor FastAPI.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import video
from storage.cloudinary_service import create_cloudinary_service
from services.video_service import create_video_service


# ============================================================
# Configuración de logging
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("video-processor-api")


# ============================================================
# Crear app de FastAPI
# ============================================================

app = FastAPI(
    title="Video Processor API",
    description="API para convertir videos horizontales a verticales para redes sociales",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================
# Configurar CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Inicializar servicios
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando Video Processor API")

    # Crear servicio de Cloudinary
    try:
        cloudinary_service = create_cloudinary_service()
        logger.info("Cloudinary service inicializado")
        logger.info("Cloud: %s", os.getenv("CLOUDINARY_CLOUD_NAME"))
        logger.info("Temp dir: %s", cloudinary_service.temp_dir)

    except ValueError as e:
        logger.exception("Error inicializando Cloudinary")
        raise

    # Crear servicio de procesamiento de video
    video_service = create_video_service(cloudinary_service)
    logger.info("Video processing service inicializado")

    # Inyectar servicios en el router
    video.set_services(cloudinary_service, video_service)
    logger.info("Servicios inyectados en routers")

    logger.info("API lista")
    logger.info("Documentación: /docs | /redoc")


# ============================================================
# Incluir routers
# ============================================================

app.include_router(video.router)


# ============================================================
# Endpoints base
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "Video Processor API está funcionando",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "process": "/api/video/process",
            "status": "/api/video/status/{job_id}",
            "download": "/api/video/download/{job_id}"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME"))
    }


# ============================================================
# Ejecutar con uvicorn
# ============================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload_enabled = os.getenv("RELOAD", "true").lower() == "true"

    logger.info("Arrancando servidor en %s:%s | reload=%s", host, port, reload_enabled)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    import logging
    logging.getLogger("validation").error("422 detail: %s", exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})