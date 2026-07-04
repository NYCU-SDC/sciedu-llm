import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import build_rag_pipeline, get_settings, validate_allowed_models
from app.routers import chat, health, title

load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # Forces loading of settings
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    allowed_models = await validate_allowed_models()
    logger.info("Allowed chat models: %s", allowed_models)

    app.state.rag_pipeline = await build_rag_pipeline()
    if app.state.rag_pipeline is not None:
        logger.info(
            "RAG pipeline built from corpus datasets: %s",
            settings.rag_corpus_dataset_names,
        )
    else:
        logger.info("RAG disabled — no corpus datasets configured (RAG_CORPUS_DATASETS)")

    logger.info("Application successfully started")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(title.router)
