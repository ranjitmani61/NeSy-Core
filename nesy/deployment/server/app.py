"""
nesy/deployment/server/app.py
==============================
FastAPI inference server for NeSy-Core.
Exposes /reason, /learn, /explain as REST endpoints.
Requires: pip install fastapi uvicorn
"""
from __future__ import annotations
import logging
import uvicorn
from fastapi import FastAPI
from nesy.deployment.server.routes import router
from nesy.deployment.server.middleware import setup_middleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeSy-Core Inference Server",
    description="Neuro-Symbolic AI with Meta-Cognition and Negative Space Intelligence",
    version="0.1.0",
)

setup_middleware(app)
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "framework": "nesy-core", "version": "0.1.0"}


def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    uvicorn.run("nesy.deployment.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
