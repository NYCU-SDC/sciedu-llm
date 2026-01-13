import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.dependencies import get_settings
from app.routers import chat, health

load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # Forces loading of settings
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("Application successfully started")
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(health.router)
app.include_router(chat.router)
