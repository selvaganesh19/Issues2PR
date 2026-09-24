"""Issue2PR test suite.

All tests here run fully offline: no network, no Docker, no Redis, and no live
LLM. The agent loop is exercised with an injected fake LLM and a real
:class:`~app.sandbox.runner.LocalRunner`.
"""
