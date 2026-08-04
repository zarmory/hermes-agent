"""Regression guard: ``/model`` must rebuild the Bedrock SDK client.

``agent.agent_runtime_helpers.switch_model`` performs the runtime swap behind
the ``/model`` command (CLI, gateway and TUI all funnel into it). Its
``anthropic_messages`` branch called ``build_anthropic_client()``
unconditionally, with no provider dispatch — unlike ``agent_init``,
``run_agent._rebuild_anthropic_client`` and
``run_agent._create_request_anthropic_client``, which all special-case
``provider == "bedrock"``.

Bedrock-hosted Claude speaks the Anthropic Messages protocol but
authenticates through the AWS SDK, against a base_url that has no
``/v1/messages`` route. So a ``/model`` switch on Bedrock replaced a working
AnthropicBedrock client with a direct Anthropic one and every call after the
switch failed.

The central test asserts the cross-site invariant (switch_model and
_rebuild_anthropic_client must agree on which SDK a provider gets) rather
than snapshotting either implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

BEDROCK_BASE = "https://bedrock-runtime.eu-central-1.amazonaws.com"
NATIVE_BASE = "https://api.anthropic.com"


@pytest.fixture
def spies(monkeypatch):
    """Record which Anthropic client factory each code path calls."""
    calls = {"direct": [], "bedrock": []}

    import agent.anthropic_adapter as adapter

    def _build_direct(api_key, base_url=None, timeout=None, **kw):
        calls["direct"].append({"api_key": api_key, "base_url": base_url})
        c = MagicMock(name="Anthropic")
        c.base_url = base_url
        return c

    def _build_bedrock(region, **kw):
        calls["bedrock"].append({"region": region})
        return MagicMock(name="AnthropicBedrock")

    monkeypatch.setattr(adapter, "build_anthropic_client", _build_direct)
    monkeypatch.setattr(adapter, "build_anthropic_bedrock_client", _build_bedrock)
    monkeypatch.setattr(adapter, "resolve_anthropic_token", lambda: "", raising=False)
    monkeypatch.setattr(adapter, "_is_oauth_token", lambda k: False, raising=False)
    return calls


def _agent(provider, base_url, *, bedrock_region="eu-central-1"):
    """Bare AIAgent carrying only what the swap path reads.

    switch_model explicitly supports this shape — see its ``_MISSING``
    sentinel comment about tests constructing agents via ``__new__``.
    """
    from run_agent import AIAgent

    a = object.__new__(AIAgent)
    a.model = "claude-opus-4-8"
    a.provider = provider
    a.requested_provider = provider
    a.api_mode = "anthropic_messages"
    a.base_url = base_url
    a.api_key = "placeholder"
    a.client = None
    a._client_kwargs = {}
    a._anthropic_client = MagicMock(name="pre-existing client")
    a._anthropic_api_key = "placeholder"
    a._anthropic_base_url = base_url
    a._is_anthropic_oauth = False
    a._config_context_length = None
    a._oauth_1m_beta_disabled = False
    a._credential_pool = None
    a._credential_pool_entry_id = None
    a._bedrock_region = bedrock_region
    a._primary_runtime = {}
    a._fallback_activated = False
    a._fallback_index = 0
    return a


def _switch(agent, new_model, new_provider, base_url):
    """Drive switch_model far enough to observe the client rebuild.

    Everything after the rebuild (context-length probing, cache policy,
    compressor refresh) is unrelated to provider dispatch; stub it so a bare
    agent can get through. If the tail still raises we swallow it — the
    dispatch has already happened and the spies have recorded it.
    """
    from agent import agent_runtime_helpers as arh
    from run_agent import AIAgent

    with (
        patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", lambda self, *a, **k: None, create=True),
        patch.object(AIAgent, "_lmstudio_load_was_unverified", lambda self, *a, **k: False, create=True),
        patch.object(AIAgent, "_effective_lmstudio_context_length", lambda self, *a, **k: None, create=True),
        patch.object(AIAgent, "_anthropic_prompt_cache_policy", lambda self, *a, **k: (False, False), create=True),
        patch.object(AIAgent, "_apply_client_headers_for_base_url", lambda self, *a, **k: None, create=True),
        patch.object(AIAgent, "_create_openai_client", lambda self, *a, **k: MagicMock(), create=True),
        patch("agent.credential_pool.load_pool", lambda *a, **k: None, create=True),
    ):
        try:
            arh.switch_model(
                agent, new_model, new_provider,
                api_key="", base_url=base_url, api_mode="anthropic_messages",
            )
        except Exception:
            # Post-rebuild tail is out of scope for these assertions.
            pass


def test_switch_on_bedrock_uses_bedrock_sdk(spies):
    """/model to another Claude on Bedrock must rebuild via AnthropicBedrock."""
    agent = _agent("bedrock", BEDROCK_BASE)
    _switch(agent, "claude-sonnet-4-5", "bedrock", BEDROCK_BASE)

    assert spies["bedrock"], (
        "regression: /model switch on Bedrock rebuilt via the direct Anthropic "
        f"adapter ({spies['direct']}) — every later call would fail against "
        f"{BEDROCK_BASE}/v1/messages"
    )
    assert spies["direct"] == []


def test_bedrock_region_comes_from_the_destination_endpoint(spies):
    """Region is read from the base_url being switched TO, not a stale attr."""
    agent = _agent("bedrock", BEDROCK_BASE, bedrock_region="us-west-2")
    _switch(agent, "claude-sonnet-4-5", "bedrock", BEDROCK_BASE)

    assert spies["bedrock"][0]["region"] == "eu-central-1"
    assert agent._bedrock_region == "eu-central-1"


def test_bedrock_region_falls_back_when_base_url_has_no_region(spies):
    """An unparseable base_url falls back to the stashed region, then default."""
    agent = _agent("bedrock", "https://example.invalid", bedrock_region="ap-south-1")
    _switch(agent, "claude-sonnet-4-5", "bedrock", "https://example.invalid")
    assert spies["bedrock"][0]["region"] == "ap-south-1"

    agent2 = _agent("bedrock", "https://example.invalid")
    del agent2._bedrock_region
    _switch(agent2, "claude-sonnet-4-5", "bedrock", "https://example.invalid")
    assert spies["bedrock"][-1]["region"] == "us-east-1"


def test_switch_on_native_anthropic_still_uses_direct_sdk(spies):
    """Guard the other direction — native Anthropic must not regress."""
    agent = _agent("anthropic", NATIVE_BASE)
    _switch(agent, "claude-opus-4-8", "anthropic", NATIVE_BASE)

    assert spies["direct"], "native Anthropic must use the direct adapter"
    assert spies["bedrock"] == []


def test_switch_model_agrees_with_rebuild_anthropic_client(spies):
    """The invariant that broke: rebuild paths must agree per provider."""
    for provider, base in (("bedrock", BEDROCK_BASE), ("anthropic", NATIVE_BASE)):
        for key in spies:
            spies[key].clear()
        _switch(_agent(provider, base), "claude-sonnet-4-5", provider, base)
        via_switch = {k: len(v) > 0 for k, v in spies.items()}

        for key in spies:
            spies[key].clear()
        _agent(provider, base)._rebuild_anthropic_client()
        via_rebuild = {k: len(v) > 0 for k, v in spies.items()}

        assert via_switch == via_rebuild, (
            f"provider={provider!r}: switch_model and "
            f"_rebuild_anthropic_client disagree on SDK dispatch "
            f"({via_switch} vs {via_rebuild})"
        )
