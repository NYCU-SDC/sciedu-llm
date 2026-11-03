"""Loopback-only integration harness: real /agents workflow, deterministic model.
It records model input for role/context verification. No Langfuse or cloud writes.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from fastapi import FastAPI
import uvicorn
from app.main import app
from app.dependencies import (
    get_openai_client,
    get_langfuse_client,
    get_settings,
    get_preset_registry,
    get_rag_pipeline,
    Settings,
)
from app.presets import Preset

ROOT = Path(os.environ["COURSE_CHAT_WORKSPACE"]).resolve()


class Prompt:
    def __init__(self, name):
        self.name = name

    def compile(self, **kw):
        if kw:
            return [
                {"role": "system", "content": "Bunny：中立的同儕，不能評價學生。"},
                {"role": "user", "content": kw.get("task", "")},
            ]
        return "Teacher：引導學生進行科學論證，不代寫答案。"


class Langfuse:
    def get_prompt(self, name, **kw):
        return Prompt(name)

    @contextmanager
    def start_as_current_observation(self, **kw):
        yield SimpleNamespace(update=lambda **kw: None)

    def update_current_generation(self, **kw):
        pass


class Stream:
    def __init__(self, text):
        self.text = text

    async def __aiter__(self):
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=self.text), finish_reason="stop"
                )
            ]
        )


class Completions:
    async def create(self, **kw):
        with (ROOT / "evidence/role-inputs.fixture.jsonl").open("a") as out:
            out.write(json.dumps({"fixture": True, **kw}, ensure_ascii=False) + "\n")
        if not kw.get("stream"):
            latest = next(
                (
                    m.get("content", "")
                    for m in reversed(kw["messages"])
                    if m.get("role") == "user"
                ),
                "",
            )
            passed = len(latest) > 8 and "天氣" not in latest
            answer = json.dumps(
                {
                    "passed": passed,
                    "feedback": "請把你的想法與暗箱中光的路徑連結起來。",
                },
                ensure_ascii=False,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
            )
        systems = "\n".join(
            str(m.get("content", ""))
            for m in kw["messages"]
            if m.get("role") == "system"
        )
        if "以 Bunny" in systems or "最多40字" in systems:
            text = "如果隔板擋住光，玩偶反射到眼睛的光會有什麼改變？"
        elif "總結 Claim" in systems:
            text = "你已提出可討論的主張。接著進入 Warrant：你的理由如何支持這個主張？"
        elif "總結 warrant" in systems:
            text = "接著進入 Backing：有哪些觀察或依據可以支持你的理由？"
        elif "總結 backing" in systems:
            text = "接著進入 Rebuttal：你如何回應不同的看法或可能的反例？"
        elif "總結本頁四階段" in systems:
            text = "本頁論證已完成。你已練習主張、理由、依據與回應反例，可以繼續教材。"
        else:
            text = "我們從 Claim 開始：你支持哪一位學生的看法？請提出自己的主張。"
        return Stream(text)


class Registry:
    async def get(self, name):
        if name != "augmentation_teacher_and_peer":
            from app.presets import PresetNotFoundError

            raise PresetNotFoundError(name)
        return preset

    def names(self):
        return [preset.name]

    def snapshot(self):
        return {preset.name: preset}


preset = Preset.model_validate(
    {
        "name": "augmentation_teacher_and_peer",
        "teacher_prompt_name": "fixture/teacher",
        "tools": {
            "rag": {"enable_tool": False, "force": False},
            "subagents": {
                "enable_tool": True,
                "character_forcing": True,
                "personas": [
                    {
                        "id": "peer",
                        "display_name": "Bunny",
                        "prompt_name": "fixture/peer",
                    }
                ],
            },
        },
    }
)
app.dependency_overrides[get_openai_client] = lambda: SimpleNamespace(
    chat=SimpleNamespace(completions=Completions())
)
app.dependency_overrides[get_langfuse_client] = lambda: Langfuse()
app.dependency_overrides[get_preset_registry] = lambda: Registry()
app.dependency_overrides[get_rag_pipeline] = lambda: None
app.dependency_overrides[get_settings] = lambda: Settings(
    openai_api_key="local-fixture", allowed_models="gpt-oss-120b"
)
# Wrapper avoids the production application's startup connections. The actual
# routers, validation, workflow, role engine and SSE serialization are unchanged.
local = FastAPI()
local.router.routes.extend(app.router.routes)
app.dependency_overrides_provider = app
local.dependency_overrides = app.dependency_overrides
if __name__ == "__main__":
    uvicorn.run(local, host="127.0.0.1", port=58087, log_level="warning")
