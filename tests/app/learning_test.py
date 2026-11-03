from types import SimpleNamespace
import pytest
from app.agents import learning
from app.agents.cast import build_cast
from app.agents.events import Part, PartEndEvent, DoneEvent
from app.agents.run import PreparedRun
from app.presets import DEFAULT_PRESETS
from app.schema.learning import LearningContext, TeachingState


@pytest.fixture
def context():
    return LearningContext(
        experimentId="e",
        courseId="c",
        pageId="p",
        courseTitle="科學論證",
        pageTitle="模擬",
        lessonStage="simulation",
        text="正文",
        imageDescriptions=["圖說"],
        questions=[],
        answers=[{"questionId": "q", "draft": "我的草稿", "submitted": "初始回答"}],
        simulationState={"led": False, "partition": True},
    )


@pytest.fixture
def prepared():
    preset = DEFAULT_PRESETS["default-agents"]
    return PreparedRun(
        preset=preset,
        cast=build_cast(preset),
        tools=[],
        model="test",
        system_message=None,
    )


async def run(monkeypatch, prepared, context, action, state=None, passed=True):
    calls = []

    async def assess(*args):
        return learning.Assessment(passed=passed, feedback="請說明你的主張。")

    async def say(**kwargs):
        calls.append(kwargs)
        return (
            "同儕的可檢驗觀點"
            if kwargs["speaker"] != prepared.cast.orchestrator.id
            else "老師的引導"
        )

    monkeypatch.setattr(learning, "assess", assess)
    monkeypatch.setattr(learning, "say", say)
    events = [
        e.payload()
        async for e in learning.run_learning(
            prepared=prepared,
            context=context,
            state=state,
            action=action,
            messages=[{"role": "user", "content": "學生訊息"}],
            openai=None,
            langfuse=None,
            settings=None,
            rag_pipeline=None,
        )
    ]
    return events, calls


@pytest.mark.asyncio
async def test_claim_pass_goes_to_warrant_without_discussion(
    monkeypatch, prepared, context
):
    events, calls = await run(
        monkeypatch, prepared, context, "message", TeachingState()
    )
    assert [c["speaker"] for c in calls] == [
        prepared.cast.summon_target_ids[0],
        prepared.cast.orchestrator.id,
    ]
    assert events[-2] == {
        "type": "teaching_state",
        "state": {"stage": "warrant", "phase": "awaiting_answer"},
    }
    assert all(c["context"] == context for c in calls)


@pytest.mark.asyncio
async def test_incomplete_answer_has_scaffolding_and_no_peer(
    monkeypatch, prepared, context
):
    events, calls = await run(
        monkeypatch, prepared, context, "message", TeachingState(), False
    )
    assert calls == []
    assert [e["part"]["agent"] for e in events if e["type"] == "part_end"] == [
        prepared.cast.orchestrator.id
    ]
    assert events[-2]["state"] == {"stage": "claim", "phase": "awaiting_answer"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,next_stage",
    [("warrant", "backing"), ("backing", "rebuttal"), ("rebuttal", "rebuttal")],
)
async def test_choice_discussion_loop_and_completion(
    monkeypatch, prepared, context, stage, next_stage
):
    events, _ = await run(
        monkeypatch, prepared, context, "message", TeachingState(stage=stage)
    )
    assert events[-2]["state"]["phase"] == "choice"
    events, _ = await run(
        monkeypatch,
        prepared,
        context,
        "discuss",
        TeachingState(stage=stage, phase="choice"),
    )
    assert events[-2]["state"]["phase"] == "discussion"
    events, calls = await run(
        monkeypatch,
        prepared,
        context,
        "message",
        TeachingState(stage=stage, phase="discussion"),
    )
    assert len(calls) == 1 and calls[0]["speaker"] == prepared.cast.summon_target_ids[0]
    assert events[-2]["state"]["phase"] == "discussion"
    for action, phase in [("continue", "choice"), ("finish_discussion", "discussion")]:
        events, calls = await run(
            monkeypatch,
            prepared,
            context,
            action,
            TeachingState(stage=stage, phase=phase),
        )
        assert calls[0]["speaker"] == prepared.cast.orchestrator.id
        assert events[-2]["state"] == {
            "stage": next_stage,
            "phase": "completed" if stage == "rebuttal" else "awaiting_answer",
        }


@pytest.mark.asyncio
async def test_invalid_control_and_failure_do_not_advance(
    monkeypatch, prepared, context
):
    with pytest.raises(ValueError):
        await run(monkeypatch, prepared, context, "continue", TeachingState())

    async def broken(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(learning, "say", broken)
    events = [
        e.payload()
        async for e in learning.run_learning(
            prepared=prepared,
            context=context,
            state=None,
            action="start",
            messages=[],
            openai=None,
            langfuse=None,
            settings=None,
            rag_pipeline=None,
        )
    ]
    assert [e["type"] for e in events] == ["error"]


@pytest.mark.asyncio
async def test_peer_uses_same_snapshot_and_total_40_character_limit(
    monkeypatch, prepared, context
):
    seen = []

    async def fake_run(**kwargs):
        seen.append(kwargs["messages"])
        yield PartEndEvent(
            0, Part(type="text", id="a", agent="student", text="中立的觀點" * 20)
        )
        yield PartEndEvent(
            1, Part(type="text", id="b", agent="student", text="補充" * 20)
        )
        yield DoneEvent()

    monkeypatch.setattr(learning, "run_preset", fake_run)
    langfuse = SimpleNamespace(
        get_prompt=lambda *a, **k: SimpleNamespace(
            compile=lambda **k: [{"role": "system", "content": "同儕設定"}]
        )
    )
    text = await learning.say(
        prepared=prepared,
        speaker=prepared.cast.summon_target_ids[0],
        instruction="Bunny中立",
        messages=[],
        context=context,
        openai=None,
        langfuse=langfuse,
        settings=None,
        rag_pipeline=None,
    )
    assert len(text) == 40
    assert (
        "我的草稿" in seen[0][1]["content"]
        and "simulationState" in seen[0][1]["content"]
    )
    assert "當下模擬設定" not in text


@pytest.mark.asyncio
async def test_partial_role_failure_publishes_no_peer_or_transition(
    monkeypatch, prepared, context
):
    async def passed(*args):
        return learning.Assessment(passed=True, feedback="")

    async def say(**kwargs):
        if kwargs["speaker"] == prepared.cast.orchestrator.id:
            raise RuntimeError("teacher failed after peer generated")
        return "中立的同儕觀點"

    monkeypatch.setattr(learning, "assess", passed)
    monkeypatch.setattr(learning, "say", say)
    events = [
        e.payload()
        async for e in learning.run_learning(
            prepared=prepared,
            context=context,
            state=TeachingState(),
            action="message",
            messages=[{"role": "user", "content": "主張"}],
            openai=None,
            langfuse=None,
            settings=None,
            rag_pipeline=None,
        )
    ]
    assert [e["type"] for e in events] == ["error"]


def test_assessment_does_not_coerce_model_text_to_boolean():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        learning.Assessment.model_validate_json('{"passed":"true","feedback":""}')
    with pytest.raises(ValidationError):
        learning.Assessment.model_validate_json(
            '{"passed":true,"feedback":"","nextStage":"rebuttal"}'
        )
