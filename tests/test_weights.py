"""The node's weights: its list of files, and --download-only, --no-download and a start with the model missing,
against a local HTTP server (no internet). The download itself (locks, checks, interrupts) is xblick.weights' and
tested in the core."""

import contextlib
import hashlib
import http.server
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from xblick import paths
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
    """The node's STORE in tmp_path (no former place), XBLICK_HOME too (nothing reaches this PC's cache); every model a
    stand-in on the local server."""
    root = tmp_path / "weights"
    monkeypatch.setenv("XBLICK_HOME", str(tmp_path / "xblick-home"))
    monkeypatch.setattr(weights, "STORE", store.Store(weights.PROG, root, weights.TERMS, (), repo=tmp_path))
    url, _ = server
    for name in weights.MODELS:
        monkeypatch.setitem(weights.MODELS, name, (f"{url}/yolox_{name}.onnx", len(DATA), DATA_SHA))
    return root


def test_the_files_are_those_of_yolox_release_0_1_1rc0():
    release = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
    assert {name: (w.path.name, w.source.url, w.size, w.sha256) for name in weights.MODELS
            for w in weights.needed(name)} == {  # measured from the downloaded files, 2026-11-14 and -15
        "nano": ("yolox_nano.onnx", f"{release}/yolox_nano.onnx", 3_659_407,
                 "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
        "tiny": ("yolox_tiny.onnx", f"{release}/yolox_tiny.onnx", 20_219_662,
                 "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
        "s": ("yolox_s.onnx", f"{release}/yolox_s.onnx", 35_858_002,
              "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
        "m": ("yolox_m.onnx", f"{release}/yolox_m.onnx", 101_259_744,
              "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f"),
        "l": ("yolox_l.onnx", f"{release}/yolox_l.onnx", 216_746_733,
              "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"),
        "x": ("yolox_x.onnx", f"{release}/yolox_x.onnx", 396_142_663,
              "c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55"),
        "darknet53": ("yolox_darknet.onnx", f"{release}/yolox_darknet.onnx", 254_814_364,
                      "10ccc527877c6ee5596929c4b3eee5ad6a78038647260bc9abdfce9a0df55d05")}
    assert weights.needed("nano")[0].path == Path("yolox_nano.onnx")  # below the store's root


def test_the_store_is_the_cache_directory_with_the_checkout_as_former_place(tmp_path):
    assert weights.STORE.root == paths.cache_dir("yolox") and weights.STORE.node == weights.NODE == "yolox"
    assert weights.STORE.former == (weights.REPO / "weights",) and weights.FORMER == weights.REPO / "weights"
    assert weights.path_of("s") == paths.cache_dir("yolox") / "yolox_s.onnx"
    env = {**os.environ, "XBLICK_HOME": str(tmp_path)}  # moves it, and asking creates nothing
    root = subprocess.run([sys.executable, "-c", "from xblick_yolox.weights import STORE; print(STORE.root)"],
                          env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert root == str(tmp_path / "cache" / "yolox") and not list(tmp_path.iterdir())



def test_weights_dir_is_a_read_only_parameter_and_download_only_none():
    from xblick.parameters import Parameters

    parser = node.make_parser()
    args = store.parse_args(parser, ["--camera", "test/camera", "--weights-dir", "/w"])
    params = Parameters(None, "yolox", "test/camera")
    params.configure(args)
    params.declare_options(parser, args, live=node.LIVE, leave_out=("camera", "node", "download_only"))
    assert params.descriptor("weights_dir").read_only and params["weights_dir"] == "/w"
    assert "download_only" not in params.names()

def test_weights_dir_is_where_the_model_goes(home, tmp_path, capsys):
    there = tmp_path / "elsewhere"
    assert node.main(["--download-only", "--model", "nano", "--weights-dir", str(there)]) == 0
    assert (there / "yolox_nano.onnx").read_bytes() == DATA and not home.exists()
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--weights-dir", "${XBLICK_NO_SUCH_VARIABLE}/w"])
    assert exit_.value.code == 2 and "--weights-dir" in capsys.readouterr().err


def test_a_model_at_the_former_place_is_named_not_used_and_its_command_moves_it(home, tmp_path, monkeypatch,
                                                                                 server, capsys):
    _, gets = server
    old = tmp_path / "checkout" / "weights"
    old.mkdir(parents=True)
    (old / "yolox_tiny.onnx").write_bytes(DATA)
    monkeypatch.setattr(weights, "STORE", store.Store(weights.PROG, home, weights.TERMS, (), former=[old]))
    with pytest.raises(store.WeightsError, match="xblick-yolox --download-only --model tiny"):
        weights.STORE.ensure(weights.needed("tiny"), download=False, options=["--model", "tiny"])
    lines = capsys.readouterr().out.splitlines()
    assert "still at the node's former place. They are not used from there." in lines[0]
    assert lines[1].strip() == f"mkdir -p {home.parent} && mv {old} {home}"
    subprocess.run(["bash", "-c", lines[1]], check=True)  # the command as printed
    assert node.main(["--download-only"]) == 0 and gets == [] and not old.exists()
    assert "nothing to download" in capsys.readouterr().out


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
    assert len(err) == 1 and "yolox_tiny.onnx" in err[0] and "xblick-yolox --download-only --model tiny" in err[0], err
    assert gets == [] and not (home / "yolox_tiny.onnx").exists()


def test_a_start_with_the_model_missing_downloads_it_first(home, server, start, capsys):
    with pytest.raises(Started):
        node.main(start)
    assert (home / "yolox_tiny.onnx").read_bytes() == DATA
    assert "YOLOX is the work of Megvii" in capsys.readouterr().out


def test_the_model_table_is_complete_and_consistent():
    from xblick_yolox import model

    assert list(weights.MODELS) == ["nano", "tiny", "s", "m", "l", "x", "darknet53"]  # small to large
    assert set(model.INPUT_SIZE) == set(weights.MODELS)
    assert len({weights.file_name(name) for name in weights.MODELS}) == len(weights.MODELS)
    assert all(url.startswith(weights.RELEASE + "/") and len(sha) == 64 and size > 0
               for url, size, sha in weights.MODELS.values())
    options = node.make_parser()._option_string_actions["--model"]
    assert options.choices == tuple(weights.MODELS) and options.default == "tiny"
