"""The refusal branches around selecting the llamacpp backend.

Every one of these is a path a human hits by editing the TOML for a bake-off
run, and every one of them fails in a way that would otherwise look like a
broken model rather than a misconfiguration -- which is the specific shape of
the `num_predict = 512` scar this project keeps naming.
"""

from __future__ import annotations

import pytest

from glados.brain.llm.llamacpp import LlamaCppLLM
from glados.core.config import LLMConfig, RouterConfig
from glados.core.server import _build_llm, _build_specialist_llm


def test_llamacpp_refuses_the_text_parser() -> None:
    """Under --jinja the same model returns structure, so leaving the parser on
    would keep the spoken channel dispatchable for no benefit."""
    with pytest.raises(ValueError, match="text_tool_format must be unset"):
        LLMConfig(
            backend="llamacpp",
            model="ministral3:8b-instruct",
            text_tool_format="mistral_v13",
        )


def test_llamacpp_accepts_ministral_without_the_text_parser() -> None:
    """The inverse arm must NOT fire: on ollama this model REQUIRES the parser,
    and a backend-blind validator would refuse the whole llamacpp config."""
    cfg = LLMConfig(
        backend="llamacpp", model="ministral3:8b-instruct", text_tool_format=None
    )
    assert cfg.backend == "llamacpp"


def test_build_llm_constructs_the_adapter() -> None:
    llm = _build_llm(
        LLMConfig(backend="llamacpp", model="ministral", text_tool_format=None)
    )
    assert isinstance(llm, LlamaCppLLM)


def test_build_llm_refuses_an_unknown_backend() -> None:
    """No fall-through to FakeLLM: a server answering every turn with canned
    text reads as a broken model, not a missing branch."""
    cfg = LLMConfig(backend="fake")
    object.__setattr__(cfg, "backend", "nonesuch")
    with pytest.raises(ValueError, match="has no builder"):
        _build_llm(cfg)


def test_specialist_refuses_a_mixed_runtime() -> None:
    """A distinct local_smart_model builds an OllamaLLM unconditionally, so
    under llamacpp SOME turns would be served by Ollama -- a mixed-runtime run
    that looks clean in the logs and silently corrupts the measurement."""
    llm_cfg = LLMConfig(backend="llamacpp", model="ministral", text_tool_format=None)
    router = RouterConfig(enabled=True, provider="local", local_smart_model="other:14b")
    with pytest.raises(ValueError, match="mixed|backend"):
        _build_specialist_llm(router, llm_cfg, primary_llm=object())


def test_specialist_aliases_the_primary_under_llamacpp() -> None:
    """The supported shape: an empty local_smart_model reuses the primary brain,
    so routing still exercises with a single runtime."""
    llm_cfg = LLMConfig(backend="llamacpp", model="ministral", text_tool_format=None)
    router = RouterConfig(enabled=True, provider="local", local_smart_model="")
    primary = object()
    assert _build_specialist_llm(router, llm_cfg, primary_llm=primary) is primary


def test_env_cannot_select_llamacpp(monkeypatch) -> None:
    """It is a measurement backend: reaching it should take an edited TOML a
    human read, not an exported variable."""
    from glados.core import config as config_mod

    monkeypatch.setenv("GLADOS_LLM_BACKEND", "llamacpp")
    with pytest.raises(ValueError, match="not selectable by environment"):
        config_mod._apply_env_overrides(config_mod.GladosConfig())


def test_env_rejects_garbage_instead_of_falling_back(monkeypatch) -> None:
    from glados.core import config as config_mod

    monkeypatch.setenv("GLADOS_LLM_BACKEND", "olama")
    with pytest.raises(ValueError, match="not selectable by environment"):
        config_mod._apply_env_overrides(config_mod.GladosConfig())
