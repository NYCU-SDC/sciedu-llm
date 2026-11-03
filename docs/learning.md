# Learning context and argumentation-v1

`/agents` accepts optional `learning_context`, `teaching_state`, and `teaching_action`. Existing callers are unchanged. Context alone augments general presets and is explicitly inherited by summoned peers. Context plus a typed action runs the deterministic teaching workflow; controls without context or impossible state combinations return 400/422.

For argumentation, use a preset with one configured peer prompt, such as the existing `augmentation_teacher_and_peer`. Existing teacher/peer prompts and model/tool configuration remain in service. The workflow adds a higher-level teaching policy; it disables autonomous summon_subagent for each controlled speaker. Teacher RAG remains available when the preset enables it. No live preset or Langfuse prompt is rewritten by this branch.

Context is bounded to 65536 UTF-8 bytes and includes course/page/lesson stage, visible text, questions, hidden image descriptions, own learner answers labelled submitted/draft/unanswered, and optional current LED/partition settings. It is reference data, not role instructions. No image pixels, answer keys, grades, other learners or credentials should be supplied. The SciEdu backend builds and authorizes it; this internal LLM endpoint trusts that boundary.

Each role receives the same JSON context. The completeness evaluator also receives it and must output strictly validated `{passed:boolean, feedback:string}`. Strings such as `"true"`, extra fields, prose or malformed JSON fail the turn. Scientific misconception alone does not make a clear Claim incomplete; no formal grade is produced. Current simulation settings are not evidence that the learner performed an experiment.

## State protocol

- Start → Teacher introduces Claim / awaiting_answer.
- Message in awaiting_answer → structured assessment. Failed/off-topic → Teacher scaffold, same state, no peer.
- Claim pass → Bunny view + Teacher summary and Warrant introduction, without discussion.
- Other stage pass → Bunny view / choice.
- Choice → discuss enters discussion; continue has Teacher summarize and advance.
- Discussion message → Bunny only, no advance. finish_discussion → Teacher summary/advance.
- Rebuttal continue/finish → Teacher summary / completed.

Bunny is prompted neutral, without student evaluation, and the entire visible response is capped to 40 non-whitespace Unicode characters (39 plus ellipsis when truncated). This is a hard bound; natural language quality still needs review.

All required role outputs are buffered before publishing any. A partial generation failure yields only an error, no transition. SSE emits private `teaching_state` before `done`; the backend consumes it, commits reply/state atomically and exposes state as persisted reply metadata. Non-stream responses include `teachingState`. The backend owns stored progress, branch and original snapshot selection; the LLM does not persist it or use a tool to choose stages.

## Verification and rollout

`uv run pytest` and `uv run ruff check . && uv run ruff format --check .`.

`tools/course_chat_fixture.py` is a loopback-only integration app with the actual routers, role engine and workflow, deterministic completions, no production startup/Langfuse/RAG connections. Set COURSE_CHAT_WORKSPACE to the integration root. It writes synthetic model inputs under evidence for comparing role snapshots. Never use this fixture in a deployment.

`PYTHONPATH=src uv run python tools/learning_quality.py --llm-base <explicit LLM URL> --output <file>` sends six synthetic completeness cases to a real model through the legacy `/chat`. Because that endpoint cannot forward response_format, only this probe normalizes an exact JSON code fence. Production assess remains strict. This probe checks rubric quality, not the private teacher/peer prompts, real RAG or the new endpoint's deployed JSON mode.

Deploy this service before the new backend/frontend. The latter will send optional controls and validate returned state; an older LLM ignores them and cannot complete a teaching turn. After all services are deployed, verify the actual augmentation preset and RAG with a dedicated dev experiment before enrolling real learners. Do not silently substitute code defaults for a missing custom preset.
