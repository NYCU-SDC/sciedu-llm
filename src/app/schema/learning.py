"""Optional course-chat contract. Ordinary /agents callers remain unchanged."""

import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, StrictBool, model_validator

Stage = Literal["claim", "warrant", "backing", "rebuttal"]
Action = Literal["start", "message", "discuss", "continue", "finish_discussion"]


class TeachingState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Stage = "claim"
    phase: Literal["awaiting_answer", "choice", "discussion", "completed"] = (
        "awaiting_answer"
    )

    @model_validator(mode="after")
    def valid_phase(self):
        if self.stage == "claim" and self.phase != "awaiting_answer":
            raise ValueError("Claim has no discussion or completion phase")
        if self.phase == "completed" and self.stage != "rebuttal":
            raise ValueError("only Rebuttal can complete the workflow")
        return self


class LearningContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    experimentId: str
    courseId: str
    pageId: str
    courseTitle: str
    pageTitle: str
    lessonStage: str
    text: str
    imageDescriptions: list[str]
    questions: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    simulationState: dict[str, StrictBool] | None = None

    @model_validator(mode="after")
    def size_limit(self):
        if len(json.dumps(self.model_dump(), ensure_ascii=False).encode()) > 65536:
            raise ValueError("learning context exceeds 65536 UTF-8 bytes")
        if self.simulationState is not None and set(self.simulationState) != {
            "led",
            "partition",
        }:
            raise ValueError("simulation state must contain led and partition only")
        return self
