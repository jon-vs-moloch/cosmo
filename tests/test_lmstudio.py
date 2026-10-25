import json

import pytest

from cosmo import lmstudio
from cosmo.lmstudio import LMStudioClient, LMStudioError


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_chat_separates_reasoning_and_content(monkeypatch):
    monkeypatch.setattr(
        lmstudio._OPENER,
        "open",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "model_instance_id": "test",
                "output": [
                    {"type": "reasoning", "content": "private scratch"},
                    {"type": "message", "content": "FINAL: 42"},
                ],
                "stats": {"reasoning_output_tokens": 3, "total_output_tokens": 5},
            }
        ),
    )
    generation = LMStudioClient().chat(
        model="test",
        role="draft",
        system_prompt="system",
        prompt="prompt",
        reasoning="on",
        temperature=0,
        max_output_tokens=10,
    )
    assert generation.content == "FINAL: 42"
    assert generation.reasoning_hash
    assert "private scratch" not in repr(generation)


def test_chat_falls_back_when_native_endpoint_rejects_response_format(monkeypatch):
    client = LMStudioClient()
    calls = []

    def request(_method, _path, body):
        calls.append(dict(body))
        if "response_format" in body:
            raise LMStudioError(
                "LM Studio HTTP 400: Unrecognized key(s) in object: "
                "'response_format'"
            )
        return {
            "model_instance_id": "test",
            "output": [{"type": "message", "content": "{}"}],
            "stats": {"total_output_tokens": 1},
        }

    monkeypatch.setattr(client, "_request", request)
    generation = client.chat(
        model="test",
        role="review",
        system_prompt="system",
        prompt="prompt",
        reasoning="off",
        temperature=0,
        max_output_tokens=10,
        response_format={"type": "json_schema"},
    )

    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert generation.stats["cosmo_response_format_fallback"] == (
        "prompt_only_after_lmstudio_rejection"
    )


def test_requires_loaded_reasoning_model(monkeypatch):
    client = LMStudioClient()
    monkeypatch.setattr(
        client,
        "models",
        lambda: [
            {
                "key": "m",
                "loaded_instances": [{"id": "instance-m"}],
                "capabilities": {"reasoning": {"allowed_options": ["off", "on"]}},
            }
        ],
    )
    assert client.require_loaded_reasoning_model("m")["key"] == "m"
    assert client.require_loaded_reasoning_model("instance-m")["key"] == "m"


def test_rejects_non_loopback_endpoint():
    with pytest.raises(LMStudioError, match="local-only"):
        LMStudioClient(base_url="https://api.example.com/v1")


def test_redirect_off_loopback_is_refused_without_contacting_target(monkeypatch):
    import http.client
    import http.server
    import threading

    seen = []

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(("loopback", self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "http://outside.invalid/api/v1/models")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    connected_hosts = []
    original_connect = http.client.HTTPConnection.connect

    def recording_connect(self):
        connected_hosts.append(self.host)
        return original_connect(self)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", recording_connect)
    # A proxy in the environment must not be used for loopback either.
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")

    server = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = LMStudioClient(
            base_url=f"http://127.0.0.1:{server.server_address[1]}",
            timeout_seconds=5,
            api_token="test-token",
        )
        with pytest.raises(lmstudio.LMStudioRedirectRefused, match="outside.invalid"):
            client.models()
    finally:
        server.shutdown()
        server.server_close()

    assert seen == [("loopback", "/api/v1/models", "Bearer test-token")]
    assert connected_hosts == ["127.0.0.1"]
