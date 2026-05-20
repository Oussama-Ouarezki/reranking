"""FastAPI entry point."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from . import deps
from .routers import queries as queries_router
from .routers import chat as chat_router
from .routers import eval as eval_router
from .routers import generation as generation_router
from .routers import statistical as statistical_router
from .routers import failure as failure_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag-app")
# show per-query margin diagnostics from the dynamic cascade
logging.getLogger("application.backend.rerankers.cascade").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading BM25 index...")
    deps.get_bm25()
    log.info("Loading corpus (~30s)...")
    corpus = deps.get_corpus()
    log.info("Corpus loaded: %d docs", len(corpus))
    log.info("Loading queries + qrels...")
    deps.get_queries()
    deps.get_qrels()
    log.info("Startup complete.")
    yield


app = FastAPI(title="BioRAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(queries_router.router, prefix="/api")
app.include_router(chat_router.router, prefix="/api")
app.include_router(eval_router.router, prefix="/api")
app.include_router(generation_router.router, prefix="/api")
app.include_router(statistical_router.router, prefix="/api")
app.include_router(failure_router.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


_dist = Path(__file__).parent.parent / "frontend" / "dist"
app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    return FileResponse(_dist / "index.html")
