import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import (
    build_rag_pipeline,
    get_langfuse_client,
    get_openai_client,
    get_settings,
    validate_allowed_models,
)
from app.presets import PresetRegistry, ensure_default_presets
from app.routers import admin, agents, chat, health, title
from judge import EvalRunner

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
        logger.info(
            "RAG disabled — no corpus datasets configured (RAG_CORPUS_DATASETS) "
            "and none discovered under the corpus folder"
        )

    # Put the shipped defaults in the preset dataset so an operator can see and
    # edit them. Existing items are left exactly as they are — a default that
    # has already been tuned in Langfuse keeps its tuning.
    try:
        created = await ensure_default_presets(get_langfuse_client(), settings)
        if created:
            logger.info(
                "Seeded %d default preset(s) into '%s': %s",
                len(created),
                settings.presets_dataset_name,
                ", ".join(created),
            )
    except Exception:
        logger.exception(
            "Could not seed the default presets into '%s'; they are served from "
            "code regardless",
            settings.presets_dataset_name,
        )

    # Warm the preset registry so the first request does not pay for the dataset
    # fetch. Best-effort on purpose: the registry serves the code defaults from
    # the moment it is constructed, so a Langfuse outage must not stop the app
    # from booting — it only means the dataset overrides are missing until the
    # TTL brings the next attempt round.
    registry = PresetRegistry(langfuse=get_langfuse_client(), settings=settings)
    try:
        report = await registry.refresh()
        logger.info(
            "Presets loaded (%d): %s%s",
            len(report.loaded),
            ", ".join(report.loaded),
            f" — {len(report.errors)} item(s) skipped" if report.errors else "",
        )
    except Exception:
        logger.exception(
            "Could not warm the preset registry; serving the code defaults only"
        )
    app.state.preset_registry = registry

    # Evaluation runs are scheduled onto this loop and outlive the request that
    # started them, so the runner has to be born with the app and told to stop
    # with it — otherwise shutdown would strand in-flight runs mid-experiment.
    app.state.eval_runner = EvalRunner(get_openai_client(), get_langfuse_client())

    logger.info("Application successfully started")
    yield

    app.state.eval_runner.shutdown()


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
app.include_router(agents.router)
app.include_router(admin.router)
