import logging
from types import SimpleNamespace

from eval_ui.main import list_judge_prompt_names, list_model_ids


def _prompts_page(names: list[str], page: int, total_pages: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(name=n) for n in names],
        meta=SimpleNamespace(page=page, total_pages=total_pages),
    )


def test_list_judge_prompt_names_filters_by_prefix_and_paginates():
    pages = {
        1: _prompts_page(
            ["judge-zeta", "extract-score", "judge-alpha", "rag-generator"],
            page=1,
            total_pages=2,
        ),
        2: _prompts_page(["judge-mu", "judge-beta"], page=2, total_pages=2),
    }

    def list_prompts(*, page, limit):  # noqa: ARG001
        return pages[page]

    langfuse = SimpleNamespace(api=SimpleNamespace(prompts=SimpleNamespace(list=list_prompts)))

    assert list_judge_prompt_names(langfuse) == [
        "judge-alpha",
        "judge-beta",
        "judge-mu",
        "judge-zeta",
    ]


def test_list_judge_prompt_names_returns_empty_on_failure(caplog):
    def boom(*, page, limit):  # noqa: ARG001
        raise RuntimeError("langfuse exploded")

    langfuse = SimpleNamespace(api=SimpleNamespace(prompts=SimpleNamespace(list=boom)))

    with caplog.at_level(logging.ERROR, logger="eval_ui.main"):
        assert list_judge_prompt_names(langfuse) == []
    assert any("Failed to list Langfuse prompts" in r.message for r in caplog.records)


async def _async_iter(models):
    for model in models:
        yield model


def test_list_model_ids_returns_sorted_ids():
    # `openai.models.list()` returns an AsyncPaginator (not a coroutine);
    # the implementation must iterate it with `async for`.
    def models_list():
        return _async_iter(
            [
                SimpleNamespace(id="zeta-7b"),
                SimpleNamespace(id="bge-m3"),
                SimpleNamespace(id="alpha-1"),
            ]
        )

    openai = SimpleNamespace(models=SimpleNamespace(list=models_list))

    assert list_model_ids(openai) == ["alpha-1", "bge-m3", "zeta-7b"]


def test_list_model_ids_returns_empty_on_failure(caplog):
    async def boom():
        raise RuntimeError("upstream 503")
        yield  # pragma: no cover — make this an async generator

    def models_list():
        return boom()

    openai = SimpleNamespace(models=SimpleNamespace(list=models_list))

    with caplog.at_level(logging.ERROR, logger="eval_ui.main"):
        assert list_model_ids(openai) == []
    assert any("Failed to list OpenAI models" in r.message for r in caplog.records)
