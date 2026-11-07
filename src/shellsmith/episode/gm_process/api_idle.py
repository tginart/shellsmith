"""Loopback-only HTTP relay with a real upstream socket inactivity timeout.

No response/header/body or credential logging. Each API exchange has its own
timer; time spent by Codex running local tools has no upstream socket to time out.
"""
from __future__ import annotations

from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import socket
import threading
import tomllib
from urllib.parse import urlsplit

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "host"}


class ApiIdleRelay:
    def __init__(self, upstream: str, idle_seconds: float):
        self.upstream = urlsplit(upstream)
        if self.upstream.scheme not in {"http", "https"} or not self.upstream.hostname or self.upstream.username or self.upstream.query or self.upstream.fragment:
            raise ValueError("API relay requires an explicit HTTP(S) provider origin")
        if isinstance(idle_seconds, bool) or not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("API idle timeout must be finite and positive")
        self.idle_seconds, self.timeouts = idle_seconds, []
        self.prefix = "/" + secrets.token_hex(24)
        self.sockets, self.lock = set(), threading.Lock()
        self.closed = threading.Event()
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.forward()

            def do_GET(self):
                self.forward()

            def forward(self):
                if not self.path.startswith(relay.prefix + "/"):
                    self.send_error(404)
                    return
                if self.headers.get("Transfer-Encoding"):
                    self.send_error(400, "Chunked request bodies are unsupported")
                    return
                connection_type = http.client.HTTPSConnection if relay.upstream.scheme == "https" else http.client.HTTPConnection
                connection = connection_type(relay.upstream.hostname, relay.upstream.port, timeout=relay.idle_seconds)
                upstream_socket = None
                with relay.lock:
                    relay.sockets.add(self.connection)
                sent_headers, phase = False, "response_headers"
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length) if length else None
                    headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
                    path = relay.upstream.path.rstrip("/") + self.path[len(relay.prefix):]
                    connection.connect()
                    upstream_socket = connection.sock
                    with relay.lock:
                        relay.sockets.add(upstream_socket)
                    if relay.closed.is_set():
                        return
                    connection.request(self.command, path, body=body, headers=headers)
                    response = connection.getresponse()
                    self.send_response(response.status)
                    for key, value in response.getheaders():
                        if key.lower() not in HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    sent_headers, phase = True, "response_stream"
                    while chunk := response.read1(64 * 1024):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (TimeoutError, socket.timeout):
                    if relay.closed.is_set():
                        return
                    with relay.lock:
                        relay.timeouts.append({"phase": phase, "idle_seconds": relay.idle_seconds})
                    if not sent_headers:
                        body = b'{"error":{"type":"api_idle_timeout","message":"Upstream API produced no activity within its idle timeout"}}'
                        self.send_response(504)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(body)
                except (OSError, http.client.HTTPException):
                    if not sent_headers:
                        self.send_error(502, "Upstream API connection failed")
                finally:
                    self.close_connection = True
                    connection.close()
                    if upstream_socket is not None:
                        upstream_socket.close()
                    with relay.lock:
                        relay.sockets.discard(upstream_socket)
                        relay.sockets.discard(self.connection)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        # Never print handler tracebacks: a transport exception must not expose
        # request headers, body, or authentication through an audit log.
        self.server.handle_error = lambda *args: None
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}{self.prefix}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.closed.set()
        self.server.shutdown()
        with self.lock:
            sockets = list(self.sockets)
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self.server.server_close()
        self.thread.join(timeout=1)


def provider_origin(command: list[str]) -> tuple[str, str]:
    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = value

    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    settings, overrides = {}, {}
    if "--ignore-user-config" not in command:
        try:
            settings = tomllib.loads((home / "config.toml").read_text())
        except FileNotFoundError:
            pass
    for index, arg in enumerate(command[:-1]):
        if arg in {"-c", "--config"}:
            merge(overrides, tomllib.loads(command[index + 1]))
    profile = overrides.get("profile", settings.get("profile"))
    for index, arg in enumerate(command[:-1]):
        if arg in {"-p", "--profile"}:
            profile = command[index + 1]
    if profile:
        merge(settings, settings.get("profiles", {}).get(profile, {}))
    merge(settings, overrides)
    provider = settings.get("model_provider", "openai")
    configured = settings.get("model_providers", {}).get(provider, {}).get("base_url")
    if configured:
        return provider, configured
    if provider != "openai":
        raise ValueError("API idle protection requires an explicit custom provider base_url")
    if base := settings.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL"):
        return provider, base
    try:
        chatgpt = json.loads((home / "auth.json").read_text()).get("auth_mode") == "chatgpt"
    except (OSError, ValueError):
        chatgpt = False
    return provider, "https://chatgpt.com/backend-api/codex" if chatgpt else "https://api.openai.com/v1"


@contextmanager
def api_idle_command(command: list[str], *, idle_seconds: float, turn_seconds: float):
    if isinstance(turn_seconds, bool) or not math.isfinite(turn_seconds) or turn_seconds <= 0:
        raise ValueError("GM turn timeout must be finite and positive")
    provider, upstream = provider_origin(command)
    with ApiIdleRelay(upstream, idle_seconds) as relay:
        # The relay counts all received bytes, including keepalives. Codex's
        # native SSE timer counts parsed events and must not reject a healthy
        # heartbeat-only interval before the whole-invocation limit.
        extra = ["-c", f"model_providers.{provider}.base_url={json.dumps(relay.base_url)}",
                 "-c", f"model_providers.{provider}.stream_idle_timeout_ms={int(turn_seconds * 1000) + 1000}",
                 "-c", f"model_providers.{provider}.supports_websockets=false"]
        insertion = len(command) - 1 if command[-1:] == ["-"] else len(command)
        yield command[:insertion] + extra + command[insertion:], relay
