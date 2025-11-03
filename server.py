import os
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

# Try to get your Render public URL from environment
PUBLIC_BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")

app = FastAPI(
    title="Professor Aurelius Actions API",
    description=(
        "Backend för Jana/Analys-aktionsendpoints: "
        "kalender, flashcards, tentastatistik, simulera tenta, exportera bevis, "
        "veckorapport, kunskapsluckor och quiz-rättare."
    ),
    version="1.0.0",
)

# ---------- Force correct OpenAPI servers entry ----------
def custom_openapi():
    """
    Rebuild OpenAPI schema so GPT Builder sees the public Render URL.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    if PUBLIC_BASE_URL:
        openapi_schema["servers"] = [{"url": PUBLIC_BASE_URL}]
    else:
        # fallback if environment var missing
        openapi_schema["servers"] = [{"url": "/"}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
# ---------------------------------------------------------
