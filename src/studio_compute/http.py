"""Loopback HTTP front end. Numeric execution always remains on Service.owner."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import socket
from urllib.parse import urlsplit, parse_qs, unquote

from .protocol import Conflict, canonical, MAX_REQUEST_BYTES


def loopback_host(value):
    if value == "localhost":
        return "127.0.0.1"
    try:
        host = ipaddress.ip_address(value)
    except ValueError as error:
        raise ValueError("Unauthenticated Studio service requires a loopback IP or localhost.") from error
    mapped = getattr(host, "ipv4_mapped", None)
    if not (host.is_loopback or (mapped and mapped.is_loopback)):
        raise ValueError("Unauthenticated Studio service requires a loopback bind; use an SSH tunnel.")
    return str(host)


def server(service, address=("127.0.0.1", 8766)):
    host = loopback_host(address[0])
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value):
            data = canonical(value)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/v1/capabilities":
                    value = service.capabilities
                elif parsed.path == "/v1/metrics":
                    client = parse_qs(parsed.query).get("client", [None])[0]
                    value = service.telemetry(client_id=client) if client else service.telemetry()
                elif parsed.path == "/v1/tokenizer":
                    value = service.engine.tokenizer_bundle  # immutable CPU assets
                elif parsed.path.startswith("/v1/clients/"):
                    value = service.client_info(unquote(parsed.path.removeprefix("/v1/clients/")))
                elif parsed.path.startswith("/v1/policies/"):
                    value = service.policy_info(unquote(parsed.path.removeprefix("/v1/policies/")))
                elif parsed.path.startswith("/v1/operations/"):
                    request_id = unquote(parsed.path.removeprefix("/v1/operations/"))
                    timeout = min(
                        30.0,
                        max(0.0, float(parse_qs(parsed.query).get("wait", [0])[0])),
                    )
                    value = service.wait(request_id, timeout)
                else:
                    raise KeyError(parsed.path)
                self.reply(200, value)
            except KeyError as error:
                self.reply(404, {"error": str(error)})
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            try:
                if self.path.startswith("/v1/operations/") and self.path.endswith("/cancel"):
                    request_id = unquote(self.path[len("/v1/operations/"):-len("/cancel")])
                    self.reply(202, service.cancel(request_id))
                    return
                if self.path != "/v1/operations":
                    self.reply(404, {"error": "Unknown endpoint"})
                    return
                count = int(self.headers.get("Content-Length", 0))
                if not 0 < count <= MAX_REQUEST_BYTES:
                    raise ValueError("Request size exceeds the compute boundary limit.")
                body = json.loads(self.rfile.read(count))
                self.reply(
                    202,
                    service.submit(
                        body["id"], body["client"], body["kind"], body["body"]
                    ),
                )
            except Conflict as error:
                self.reply(409, {"error": str(error)})
            except (KeyError, ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except RuntimeError as error:
                self.reply(503, {"error": str(error)})

    class LoopbackServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET

    return LoopbackServer((host, address[1]), Handler)
