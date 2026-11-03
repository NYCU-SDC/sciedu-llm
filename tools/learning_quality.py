"""Run synthetic assessment cases against an explicitly selected LLM /chat service.
This checks the new rubric with a real model; it does not certify the deployment's
private Langfuse prompts, RAG tools, or the new /agents integration.
"""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
from app.agents.learning import assess
from app.schema.learning import LearningContext, TeachingState


class Model:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.chat = SimpleNamespace(completions=self)

    @staticmethod
    def json_content(raw):
        # /chat cannot forward OpenAI response_format. Normalize only the exact
        # enclosing code fence for this probe; production assess() stays strict.
        text = raw.strip()
        if text.startswith("```json\n") and text.endswith("\n```"):
            return text[8:-4]
        return text

    async def create(self, **kwargs):
        payload = {"messages": kwargs["messages"], "stream": False, "enable_rag": False}

        def request():
            with urlopen(
                Request(
                    self.base + "/chat",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=50,
            ) as response:
                return json.load(response)

        response = await asyncio.to_thread(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.json_content(response["content"])
                    )
                )
            ]
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    context = LearningContext(
        experimentId="synthetic",
        courseId="dark-room",
        pageId="page-3",
        courseTitle="黑暗房間中的玩偶",
        pageTitle="Simulation 與論證",
        lessonStage="simulation",
        text="學生論證在無光、LED 開啟、LED 與玩偶間有隔板的情況，能否看見玩偶。",
        imageDescriptions=[
            "LED、玩偶、觀察者。隔板可阻擋 LED 照射玩偶。模型忽略箱壁二次反射。"
        ],
        questions=[],
        answers=[],
        simulationState={"led": True, "partition": False},
    )
    cases = [
        (
            "claim",
            "我支持小安的看法，玩偶需要光照射並反射進眼睛才會被看見。",
            True,
            "明確主張",
        ),
        (
            "claim",
            "我認為眼睛適應黑暗後會主動發光，所以三種情況都看得見玩偶。",
            True,
            "有科學迷思但仍是完整主張",
        ),
        ("claim", "今天午餐很好吃，我想放學去打籃球。", False, "偏題"),
        ("warrant", "我支持小安，我就是支持小安。", False, "重複主張，未說明理由連結"),
        (
            "backing",
            "我觀察到關掉 LED 時看不見，開燈沒有隔板時能看見；這支持需要光的理由。",
            True,
            "學生明確敘述自己的觀察",
        ),
        (
            "rebuttal",
            "對於適應黑暗就能看見的說法，若完全沒有光，適應後仍沒有反射光進入眼睛，所以不能看見。",
            True,
            "回應不同觀點",
        ),
    ]
    model = Model(args.llm_base)
    rows = []
    for stage, text, expected, label in cases:
        try:
            result = await assess(
                model,
                SimpleNamespace(model="default"),
                [{"role": "user", "content": text}],
                context,
                TeachingState(stage=stage),
            )
            row = {
                "stage": stage,
                "input": text,
                "case": label,
                "expectedPassed": expected,
                **result.model_dump(),
                "matchesExpectation": result.passed == expected,
            }
        except Exception as error:
            row = {
                "stage": stage,
                "case": label,
                "error": str(error),
                "matchesExpectation": False,
            }
        rows.append(row)
        Path(args.output).write_text(
            json.dumps(
                {
                    "scope": "real model rubric probe through legacy /chat; exact JSON fences normalized only by probe; no response_format, private prompt or RAG certification",
                    "base": args.llm_base,
                    "cases": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(label, row.get("passed", row.get("error")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
