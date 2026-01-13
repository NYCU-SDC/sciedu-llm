from typing import Literal
from pydantic import BaseModel


class HealthzRespoonse(BaseModel):
    status: Literal["ok"]
