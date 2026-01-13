from dotenv import load_dotenv
from fastapi import FastAPI

from app.dependencies import get_settings
from app.routers import chat, health

load_dotenv()

app = FastAPI()
settings = get_settings()  # Forces loading of settings

app.include_router(health.router)
app.include_router(chat.router)
