"""The whole stack, over a real socket.

Every other test in `tests/app` drives the routers through `TestClient`, which
calls the ASGI app directly. That covers the logic but not the transport: the
SSE frames are never actually chunked onto a socket, uvicorn never touches them,
and nothing proves a client reading the stream incrementally sees what the spec
says it should.

So this boots the real `app.main:app` — wired to the fakes from
`scripts/fake_stack.py`, the same ones its `--demo` mode drives — on uvicorn on
an ephemeral port, and talks to it over HTTP. One `teacher-student` run (a
dataset preset the fake Langfuse serves) through /agents backs most of the
assertions; the rest cover /chat's legacy framing and both non-streaming
shapes.

The fakes live in the script rather than here on purpose: `--demo` and this test
must exercise the *same* stack, or the script stops being evidence of anything.
"""

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

# `scripts/` is not a package (it is a directory of runnable files), so it goes
# on the path by hand rather than being imported as `scripts.fake_stack`.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import fake_stack  # noqa: E402

pytestmark = pytest.mark.integration

QUESTION = "幫我根據課本第三章解釋光反應"

# The fake upstream sleeps between chunks and the fake retriever for 0.4s, so a
# whole run is a couple of seconds; anything past this is a hang, not slowness.
TIMEOUT_SECONDS = 30.0


# --- the server -------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url():
    """Serve the fake stack on a real socket for the duration of this module.

    `build_fake_app` overrides dependencies on the process-wide app, so the
    previous state is put back afterwards — otherwise the fakes would follow the
    rest of the suite around.
    """
    app = fake_stack.app
    saved_overrides = dict(app.dependency_overrides)
    saved_lifespan = app.router.lifespan_context
    saved_registry = getattr(app.state, "preset_registry", None)

    config = uvicorn.Config(
        fake_stack.build_fake_app(),
        host="127.0.0.1",
        # Port 0 lets the kernel pick a free one, so a developer already serving
        # the fake stack on :8000 does not collide with the suite.
        port=0,
        log_level="warning",
        # The app's lifespan builds a RAG index and validates models upstream.
        # `build_fake_app` already neutralises it; this makes sure uvicorn does
        # not run one at all.
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10.0
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("uvicorn exited before it began serving")
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start within 10s")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved_overrides)
        app.router.lifespan_context = saved_lifespan
        app.state.preset_registry = saved_registry


# --- wire helpers -----------------------------------------------------------


def post_stream(base_url: str, path: str, body: dict):
    """POST and drain an SSE response, keeping the bytes exactly as they arrived."""
    with (
        httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS) as client,
        client.stream("POST", path, json=body) as response,
    ):
        raw = b"".join(response.iter_raw())
        return SimpleNamespace(
            status_code=response.status_code,
            content_type=response.headers["content-type"],
            raw=raw,
            frames=parse_frames(raw),
        )


def post_json(base_url: str, path: str, body: dict):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS) as client:
        return client.post(path, json=body)


def get_json(base_url: str, path: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS) as client:
        return client.get(path)


def parse_frames(raw: bytes) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in raw.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


def of_type(frames: list[dict], type_: str) -> list[dict]:
    return [frame for frame in frames if frame.get("type") == type_]


# --- one agentic run, asserted from several angles --------------------------


@pytest.fixture(scope="module")
def agents_stream(base_url):
    return post_stream(
        base_url,
        "/agents",
        {
            "messages": [{"role": "user", "content": QUESTION}],
            "stream": True,
            "preset": "teacher-student",
        },
    )


def test_agents_streams_events_over_the_socket(agents_stream):
    assert agents_stream.status_code == 200
    assert agents_stream.content_type.startswith("text/event-stream")
    # Every frame is a `data:` line terminated by a blank line, which is what
    # makes an EventSource client emit a message per event rather than one at
    # the end.
    assert agents_stream.raw.endswith(b"\n\n")
    assert len(agents_stream.frames) > 10


def test_agents_stream_opens_with_the_cast(agents_stream):
    first = agents_stream.frames[0]
    assert first["type"] == "cast"
    assert [character["id"] for character in first["characters"]] == [
        "teacher",
        "student",
    ]
    assert all(
        character["displayName"] and character["role"]
        for character in first["characters"]
    )
    # Exactly once, at the head — a client sets up its name tags off this.
    assert len(of_type(agents_stream.frames, "cast")) == 1


def test_agents_stream_frames_every_speaker(agents_stream):
    frames = agents_stream.frames
    speaker_events = [
        frame
        for frame in frames
        if frame.get("type") in ("agent_start", "agent_end", "part_start")
    ]
    assert speaker_events[0] == {"type": "agent_start", "agent": "teacher"}

    summon = next(
        frame["part"]
        for frame in of_type(frames, "part_start")
        if frame["part"].get("name") == "summon_subagent"
    )
    student_start = next(
        frame for frame in of_type(frames, "agent_start") if frame["agent"] == "student"
    )
    # The summoned character is a real speaker, and the events say who called it
    # in and on which tool call — that pairing is what lets a client nest the
    # student's turn inside the teacher's.
    assert student_start["parent"] == "teacher"
    assert student_start["summonedBy"] == summon["tool_call_id"]

    starts = [frame["agent"] for frame in of_type(frames, "agent_start")]
    ends = [frame["agent"] for frame in of_type(frames, "agent_end")]
    assert starts == ["teacher", "student", "teacher"]
    assert ends == ["student", "teacher"]


def test_agents_stream_hides_the_summon_machinery(agents_stream):
    parts = [frame["part"] for frame in of_type(agents_stream.frames, "part_end")]
    by_name = {part.get("name"): part for part in parts if part.get("name")}

    # The summon call and the tool_result carrying the student's answer back are
    # plumbing, and are flagged so a client can hide them...
    summon = by_name["summon_subagent"]
    assert summon["internal"] is True
    summon_result = next(
        part
        for part in parts
        if part["type"] == "tool_result"
        and part["tool_call_id"] == summon["tool_call_id"]
    )
    assert summon_result["internal"] is True

    # ...while an ordinary tool call is not internal, and neither is anything
    # the student actually says.
    assert "internal" not in by_name["rag_search"]
    student_text = [
        part for part in parts if part["agent"] == "student" and part["type"] == "text"
    ]
    assert student_text and all("internal" not in part for part in student_text)


def test_agents_stream_indices_stay_aligned(agents_stream):
    frames = agents_stream.frames
    starts = of_type(frames, "part_start")
    ends = of_type(frames, "part_end")

    # Parts are numbered 0, 1, 2, … in the order they open, and each closes on
    # the index it opened with: this is the only handle a client has for
    # attaching a delta to the right part.
    assert [frame["index"] for frame in starts] == list(range(len(starts)))
    assert [frame["index"] for frame in ends] == [frame["index"] for frame in starts]

    open_indices: set[int] = set()
    for frame in frames:
        type_ = frame.get("type")
        if type_ == "part_start":
            open_indices.add(frame["index"])
        elif type_ == "delta":
            assert frame["index"] in open_indices, "delta for a part never started"
        elif type_ == "part_end":
            open_indices.discard(frame["index"])
    assert not open_indices, "a part was left open"


def test_agents_streams_the_student_answer_token_by_token(agents_stream):
    frames = agents_stream.frames
    index = next(
        frame["index"]
        for frame in of_type(frames, "part_start")
        if frame["part"]["agent"] == "student" and frame["part"]["type"] == "text"
    )
    deltas = [
        frame["delta"] for frame in of_type(frames, "delta") if frame["index"] == index
    ]
    end = next(
        frame for frame in of_type(frames, "part_end") if frame["index"] == index
    )

    assert len(deltas) > 1, "the student's answer arrived in one lump"
    assert "".join(deltas) == end["part"]["text"]
    # /agents does not escape non-ASCII (unlike /chat), so the Chinese text is
    # on the wire verbatim.
    assert end["part"]["text"].encode("utf-8") in agents_stream.raw


def test_agents_stream_ends_completed(agents_stream):
    frames = agents_stream.frames
    assert not of_type(frames, "error")
    assert frames[-1] == {"type": "done", "finishReason": "stop", "status": "completed"}


def test_agents_serves_the_preset_from_the_dataset(base_url):
    """The run above came out of Langfuse, not out of `DEFAULT_PRESETS`.

    The fake Langfuse serves a `teacher-student` dataset item — a preset this
    server does not ship, which is the ordinary way a deployment adds a
    behaviour without a release. It is served alongside the code defaults, and
    the admin view is where the difference shows.
    """
    report = post_json(base_url, "/admin/presets/refresh", {})
    assert report.status_code == 200
    loaded = report.json()["loaded"]
    assert "teacher-student" in loaded
    assert {"default-agents", "default-chat", "default-chat-plain"} <= set(loaded)
    assert report.json()["errors"] == {}

    detail = get_json(base_url, "/admin/presets/teacher-student")
    assert detail.status_code == 200
    body = detail.json()
    # Dataset-defined only: no code default of this name, so nothing to shadow —
    # and, unlike a code default, it can be deleted.
    assert body["builtin"] is False
    assert body["shadowed_builtin"] is False
    assert body["description"] == "老師先查課本，召喚學生作答，再補充訂正。"
    assert body["definition"]["orchestrator"] == "teacher"


def test_agents_non_streaming_folds_the_same_parts(base_url, agents_stream):
    response = post_json(
        base_url,
        "/agents",
        {
            "messages": [{"role": "user", "content": QUESTION}],
            "stream": False,
            "preset": "teacher-student",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert body["role"] == "assistant"
    assert body["status"] == "completed"
    assert body["finishReason"] == "stop"
    assert [character["id"] for character in body["cast"]] == ["teacher", "student"]

    streamed = [frame["part"] for frame in of_type(agents_stream.frames, "part_end")]
    assert [(part["type"], part["agent"]) for part in body["parts"]] == [
        (part["type"], part["agent"]) for part in streamed
    ]


# --- the legacy endpoint ----------------------------------------------------


@pytest.fixture(scope="module")
def chat_stream(base_url):
    return post_stream(
        base_url,
        "/chat",
        {"messages": [{"role": "user", "content": QUESTION}], "stream": True},
    )


def test_chat_stream_keeps_the_legacy_frames(chat_stream):
    assert chat_stream.status_code == 200
    assert chat_stream.content_type.startswith("text/event-stream")

    frames = chat_stream.frames
    assert frames, "no frames at all"
    # The old contract, byte for byte: two keys, nothing else, ever. In
    # particular none of the typed protocol (`type`, `index`, `part`, `agent`)
    # may leak through the translation.
    assert all(set(frame) == {"delta", "isFinished"} for frame in frames)
    assert all(isinstance(frame["delta"], str) for frame in frames)
    assert frames[-1] == {"delta": "", "isFinished": True}
    assert all(frame["isFinished"] is False for frame in frames[:-1])


def test_chat_stream_escapes_non_ascii_on_the_wire(chat_stream):
    # The old implementation used the JSON default (`ensure_ascii=True`), so a
    # client decoding frames by hand would break on raw UTF-8. The bytes stay
    # ASCII; the decoded text does not.
    assert chat_stream.raw.isascii()
    assert rb"\u" in chat_stream.raw
    text = "".join(frame["delta"] for frame in chat_stream.frames)
    assert text and not text.isascii()


def test_chat_non_streaming_returns_the_whole_message(base_url, chat_stream):
    response = post_json(
        base_url,
        "/chat",
        {"messages": [{"role": "user", "content": QUESTION}], "stream": False},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert set(body) == {"content", "finishReason"}
    assert body["finishReason"] == "stop"
    assert body["content"] == "".join(frame["delta"] for frame in chat_stream.frames)
