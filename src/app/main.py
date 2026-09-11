import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

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
from app.rag_builds import RagBuildManager
from app.routers import admin, agents, chat, health, title
from judge import EvalRunner

load_dotenv()

# `level` is not optional here, however tempting the default looks: without it
# the root logger stays at WARNING and every INFO line this service writes — the
# index-build progress, the eval-run phases, "application started" — is dropped
# before it reaches a handler. Read from LOG_LEVEL (via .env too, loaded above)
# because logging has to be configured at import, long before Settings exists.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

# httpx logs one INFO line per request. At our level that is a line per embedding
# batch — hundreds during a rebuild, drowning the progress lines that are meant
# to be read. Errors still come through, and the app reports its own upstream
# failures anyway.
logging.getLogger("httpx").setLevel(logging.WARNING)

# jieba sets its *own* logger to DEBUG, so its dictionary-loading chatter reaches
# our handler however high the root level is. It has nothing to say to an
# operator; the first build of a process would otherwise open with it.
logging.getLogger("jieba").setLevel(logging.WARNING)


async def _build_rag_in_background(app: FastAPI) -> None:
    """Build and publish the initial RAG pipeline after the app starts serving."""
    logger = logging.getLogger(__name__)
    try:
        pipeline = await build_rag_pipeline()
    except asyncio.CancelledError:
        logger.info("Initial RAG index build cancelled by shutdown")
        raise
    except Exception:
        # RAG is optional. An embedding/backend failure must not take down an
        # otherwise healthy chat service after startup has already completed.
        logger.exception("Initial RAG index build failed; RAG remains disabled")
        return

    if pipeline is None:
        logger.info(
            "RAG remains disabled — no corpus datasets configured "
            "(RAG_CORPUS_DATASETS) and none discovered under the corpus folder"
        )
        return

    # Publish the manager first so the moment request dependencies can see the
    # pipeline they can also see the matching manager. There is no interval in
    # which a half-built pipeline is exposed: RAGPipeline.build installs its
    # indexes only after every embedding has completed.
    app.state.rag_build_manager = RagBuildManager(pipeline)
    app.state.rag_pipeline = pipeline
    logger.info(
        "RAG enabled after background index build from corpus datasets: %s",
        ", ".join(pipeline.corpus_dataset_names),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # Forces loading of settings
    logger = logging.getLogger(__name__)

    # Requests initially observe RAG as disabled. The corpus is deliberately
    # built only after startup yields control to Uvicorn, then published as one
    # fully-built pipeline by `_build_rag_in_background`.
    app.state.rag_pipeline = None
    app.state.rag_build_manager = None
    app.state.rag_startup_task = None

    allowed_models = await validate_allowed_models()
    if allowed_models:
        logger.info("Allowed chat models: %s", allowed_models)
    else:
        logger.info("Allowed chat models: all upstream models")

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
    app.state.eval_runner = EvalRunner(await get_openai_client(), get_langfuse_client())

    app.state.rag_startup_task = asyncio.create_task(
        _build_rag_in_background(app), name="rag-initial-build"
    )
    logger.info("Application successfully started; RAG is initially disabled")
    yield

    app.state.eval_runner.shutdown()
    rag_startup_task = app.state.rag_startup_task
    if rag_startup_task is not None:
        if not rag_startup_task.done():
            rag_startup_task.cancel()
        with suppress(asyncio.CancelledError):
            await rag_startup_task
    if app.state.rag_build_manager is not None:
        app.state.rag_build_manager.shutdown()


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
