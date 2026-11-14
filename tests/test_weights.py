"""The node's weights: its list of files, and --download-only, --no-download and a start with the model missing,
against a local HTTP server (no internet). The download itself (locks, checks, interrupts) is xblick.weights' and
tested in the core."""

import contextlib
import hashlib
import http.server
import os
import signal
import socket
import threading

import pytest
from xblick import weights as store
from xblick.session import open_session

from xblick_yolox import node, weights

DATA = os.urandom(3 * 1024 * 1024 + 17)  # a stand-in model file
DATA_SHA = hashlib.sha256(DATA).hexdigest()


@pytest.fixture
def server():
    """A local HTTP server of DATA at every path; records each GET."""
    gets = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            gets.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(DATA)))
            self.end_headers()
            self.wfile.write(DATA)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", gets
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def home(tmp_path, monkeypatch, server):
    """The node's STORE in tmp_path; every model a stand-in on the local server."""
    root = tmp_path / "weights"
    monkeypatch.setattr(weights, "STORE", store.Store(weights.PROG, root, weights.TERMS, (), repo=tmp_path))
    url, _ = server
    for name in weights.MODELS:
        monkeypatch.setitem(weights.MODELS, name, (f"{url}/yolox_{name}.onnx", len(DATA), DATA_SHA))
    return root


def test_the_files_are_those_of_yolox_release_0_1_1rc0():
    release = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
    assert {name: (w.path.name, w.source.url, w.size, w.sha256) for name in ("nano", "tiny", "s")
            for w in weights.needed(name)} == {  # measured from the downloaded files, 2026-11-14
        "nano": ("yolox_nano.onnx", f"{release}/yolox_nano.onnx", 3_659_407,
                 "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
        "tiny": ("yolox_tiny.onnx", f"{release}/yolox_tiny.onnx", 20_219_662,
                 "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
        "s": ("yolox_s.onnx", f"{release}/yolox_s.onnx", 35_858_002,
              "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063")}
    assert weights.needed("nano")[0].path == weights.WEIGHTS / "yolox_nano.onnx"
    assert weights.STORE.root == weights.WEIGHTS


def test_download_only_downloads_the_model_option_and_then_has_nothing_to_do(home, server, capsys):
    _, gets = server
    assert node.main(["--download-only", "--model", "s"]) == 0
    out = capsys.readouterr().out
    assert (home / "yolox_s.onnx").read_bytes() == DATA and not (home / "yolox_tiny.onnx").exists()
    assert "YOLOX is the work of Megvii" in out and "weights complete: 1 downloaded" in out
    assert node.main(["--download-only", "--model", "s"]) == 0 and gets == ["/yolox_s.onnx"]
    assert "nothing to download" in capsys.readouterr().out


def test_download_only_refuses_a_changed_file(home, monkeypatch, capsys):
    url, size, _ = weights.MODELS["tiny"]
    monkeypatch.setitem(weights.MODELS, "tiny", (url, size, "0" * 64))
    assert node.main(["--download-only"]) == 1
    assert "wrong SHA-256" in capsys.readouterr().err and not (home / "yolox_tiny.onnx").exists()


class Started(Exception):
    """The node got past its weights."""


@pytest.fixture
def start(monkeypatch):
    """A start's options with a Zenoh session to connect to; the model replaced by raising Started."""
    def stop(*args):
        raise Started

    monkeypatch.setattr(node, "load_detector", stop)
    kept = []  # the node's parameters and session: a start that raised Started left them open

    class Kept(node.node.Parameters):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kept.append(self)

    monkeypatch.setattr(node.node, "Parameters", Kept)
    opened = node.open_from_args
    monkeypatch.setattr(node, "open_from_args", lambda args: kept.append(opened(args)) or kept[-1])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    sigterm = signal.getsignal(signal.SIGTERM)  # node.start sets its own
    yield ["--camera", "test/camera", "--connect", f"127.0.0.1:{port}"]
    for closable in kept:
        with contextlib.suppress(Exception):  # closed already by a node that ended
            closable.close()
    signal.signal(signal.SIGTERM, sigterm)
    session.close()


def test_no_download_with_the_model_missing_ends_with_one_line(home, server, start, capsys):
    _, gets = server
    assert node.main([*start, "--no-download"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "yolox_tiny.onnx" in err[0] and "--download-only" in err[0], err
    assert gets == [] and not (home / "yolox_tiny.onnx").exists()


def test_a_start_with_the_model_missing_downloads_it_first(home, server, start, capsys):
    with pytest.raises(Started):
        node.main(start)
    assert (home / "yolox_tiny.onnx").read_bytes() == DATA
    assert "YOLOX is the work of Megvii" in capsys.readouterr().out
