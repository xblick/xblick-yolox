"""runtime.py without a GPU, without ONNX Runtime for CUDA and without the internet: the shared GPU runtime's place,
the one-line errors, the setup's rule for "a GPU is present" with a stand-in nvidia-smi, the install with a stand-in
uv, and a start that is stopped while it prepares the GPU. The same in every ONNX Runtime node."""

import importlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# the node's package: the one directory below src/ (this file is the same in every ONNX Runtime node)
PACKAGE = next(p.name for p in (Path(__file__).parents[1] / "src").iterdir() if (p / "runtime.py").is_file())
runtime = importlib.import_module(f"{PACKAGE}.runtime")
TAG = f"cp{sys.version_info[0]}{sys.version_info[1]}"


@pytest.fixture(autouse=True)
def any_python(monkeypatch):
    """Every Python counts as one with a wheel (the tests run on 3.10 too); the test of the check sets its own."""
    monkeypatch.setattr(runtime, "PYTHONS", ((3, 0), (3, 99)))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """XBLICK_HOME in tmp_path: the shared runtime goes to <home>/cache/onnxruntime."""
    monkeypatch.setenv("XBLICK_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def fake_runtime(place: Path, tag: str = TAG):
    """A directory that looks like an installed GPU runtime (dist-info and the module's library name)."""
    (place / f"onnxruntime_gpu-{runtime.ONNXRUNTIME_GPU}.dist-info").mkdir(parents=True)
    capi = place / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    (capi / f"onnxruntime_pybind11_state.cpython-{tag[2:]}-x86_64-linux-gnu.so").write_bytes(b"")


def test_the_shared_place_has_the_version_and_the_python(home):
    assert runtime.runtime_dir() == home / "cache" / "onnxruntime" / f"gpu-{runtime.ONNXRUNTIME_GPU}-{TAG}"


def test_cuda_without_the_runtime_names_the_one_step(home):
    with pytest.raises(RuntimeError) as error:
        runtime.onnxruntime_for("cuda:0", "xblick-x")
    message = str(error.value)
    assert message.startswith("--device cuda:0: no GPU runtime in " + str(runtime.runtime_dir()))
    assert "xblick-x-gpu-runtime install" in message and "tools/setup.sh --gpu" in message
    assert "CUDA 13" in message and "\n" not in message


def test_a_runtime_at_a_former_place_gives_the_exact_mv(home, tmp_path):
    old = tmp_path / "old" / "onnxruntime-gpu"
    fake_runtime(old)
    with pytest.raises(RuntimeError, match=f"mv {old} {runtime.runtime_dir()}"):
        runtime.onnxruntime_for("cuda:0", "xblick-x", [old])
    assert runtime.install("uv-never-called", "xblick-x", [old]) == 1
    fake_runtime(tmp_path / "other", tag="cp39")  # another Python's: not this one's to move
    assert runtime.former_runtime([tmp_path / "other"]) is None


def test_python_3_10_has_no_gpu_runtime(home, monkeypatch):
    monkeypatch.setattr(runtime, "PYTHONS", ((3, 99), (3, 99)))  # as on a Python without a wheel
    with pytest.raises(RuntimeError, match="has no wheel for Python"):
        runtime.onnxruntime_for("cuda:0", "xblick-x")
    assert runtime.onnxruntime_for("cpu", "xblick-x").__name__ == "onnxruntime"  # the CPU is unaffected


def stand_in(directory: Path, name: str, script: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(script))
    path.chmod(0o755)
    return path


@pytest.fixture
def smi(tmp_path, monkeypatch):
    """A stand-in nvidia-smi on PATH (nothing else there): smi(body) sets its script."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    return lambda body: stand_in(bin_dir, "nvidia-smi", body)


@pytest.mark.parametrize(("body", "state", "text"), [
    (None, "none", "no nvidia-smi on PATH"),
    ("echo 'NVIDIA-SMI has failed' >&2; exit 9", "none", "exited 9: NVIDIA-SMI has failed"),
    ("exit 0", "none", "lists none"),
    ("echo 'NVIDIA GeForce RTX 3090, 580.95.05'", "gpu", "found 1 NVIDIA GPU(s): NVIDIA GeForce RTX 3090, driver 580"),
    ("echo 'NVIDIA GeForce RTX 3090, 535.230.02'", "old", "needs driver 580 or newer"),
    ("printf 'NVIDIA A, 580.1\\nNVIDIA A, 580.1\\nNVIDIA B, 581.2\\n'", "gpu",
     "found 3 NVIDIA GPU(s): 2 x NVIDIA A, NVIDIA B"),
])
def test_the_rule_for_a_gpu(smi, body, state, text):
    if body is not None:
        smi(body)
    got, line = runtime.detect()
    assert got == state and text in line, line


@pytest.fixture
def uv(tmp_path):
    """A stand-in uv: records its arguments, makes the --target directory look installed."""
    log = tmp_path / "uv.log"
    path = stand_in(tmp_path, "uv", f"""\
        echo "$@" >> {log}
        while [ $# -gt 0 ]; do
            if [ "$1" = --target ]; then target=$2; fi
            shift
        done
        /bin/mkdir -p "$target/onnxruntime_gpu-{runtime.ONNXRUNTIME_GPU}.dist-info" "$target/onnxruntime"
        """)
    return path, log


def test_setup_without_a_gpu_installs_nothing_and_says_how(home, smi, uv, capsys):
    path, log = uv
    assert runtime.setup(str(path), "xblick-x", "auto") == 0
    out = capsys.readouterr().out
    assert "no NVIDIA GPU found" in out and "the node computes on the CPU" in out and "tools/setup.sh --gpu" in out
    assert not log.exists() and not runtime.runtime_dir().exists()


def test_setup_with_an_old_driver_installs_nothing(home, smi, uv, capsys):
    smi("echo 'Tesla T4, 470.256.02'")
    path, log = uv
    assert runtime.setup(str(path), "xblick-x", "auto") == 0
    assert "needs driver 580 or newer, so it was not installed" in capsys.readouterr().out and not log.exists()


def test_setup_with_a_gpu_announces_installs_once_and_then_reuses(home, smi, uv, capsys):
    smi("echo 'NVIDIA RTX PRO 6000, 580.105.08'")
    path, log = uv
    assert runtime.setup(str(path), "xblick-x", "auto") == 0
    out = capsys.readouterr().out
    announced = out.index("installing the GPU runtime for --device cuda")
    assert "to skip it: tools/setup.sh --no-gpu" in out and announced < out.index("GPU runtime in place")
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and f"--no-deps onnxruntime-gpu=={runtime.ONNXRUNTIME_GPU}" in calls[0]
    assert all(lib in calls[1] for lib in runtime.CUDA_LIBRARIES) and f"--python {sys.executable}" in calls[1]
    assert (runtime.runtime_dir() / "onnxruntime").is_dir() and not list(runtime.runtime_dir().parent.glob("*.part"))
    assert runtime.setup(str(path), "xblick-x", "auto") == 0  # the second call: nothing for the runtime
    assert "in place already" in capsys.readouterr().out and len(log.read_text().splitlines()) == 2


def test_setup_with_a_gpu_on_a_python_without_a_wheel_says_so_once(home, smi, uv, monkeypatch, capsys):
    smi("echo 'NVIDIA RTX PRO 6000, 580.105.08'")
    monkeypatch.setattr(runtime, "PYTHONS", ((3, 99), (3, 99)))
    path, log = uv
    assert runtime.setup(str(path), "xblick-x", "auto") == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith("xblick-x: found 1 NVIDIA GPU(s)")
    assert lines[0].count("xblick-x:") == 1
    assert "has no wheel for Python" in lines[0] and not log.exists()


def test_gpu_installs_without_a_gpu_and_no_gpu_never(home, smi, uv, capsys):
    path, log = uv
    assert runtime.setup(str(path), "xblick-x", "off") == 0
    assert "--no-gpu" in capsys.readouterr().out and not log.exists()
    assert runtime.setup(str(path), "xblick-x", "on") == 0  # no nvidia-smi at all
    assert (runtime.runtime_dir() / "onnxruntime").is_dir()


def test_an_interrupted_install_never_looks_complete(home, uv, capsys):
    path, _ = uv
    part = runtime.runtime_dir().with_name(runtime.runtime_dir().name + ".part")
    part.mkdir(parents=True)
    (part / "half").write_bytes(b"x")
    assert runtime.install(str(path), "xblick-x") == 0
    assert "left by an interrupted install" in capsys.readouterr().out
    assert not part.exists() and not (runtime.runtime_dir() / "half").exists()
    failing = stand_in(path.parent, "uv-fails", "exit 3\n")
    other = runtime.runtime_dir()
    os.rename(other, other.with_name("done"))  # as if missing again
    assert runtime.install(str(failing), "xblick-x") == 1 and not other.exists()


def test_the_command_after_uv_tool_install_installs_and_names_the_place(home, uv, monkeypatch, capsys):
    path, log = uv
    prog = PACKAGE.replace("_", "-")
    assert runtime.tool_main(["where"]) == 0 and capsys.readouterr().out.strip() == str(runtime.runtime_dir())
    monkeypatch.setenv("PATH", str(path.parent))  # the stand-in uv, as "uv" on PATH
    os.rename(path, path.parent / "uv-stand-in")
    stand_in(path.parent, "uv", f'exec "{path.parent / "uv-stand-in"}" "$@"\n')
    assert runtime.tool_main(["install"]) == 0
    assert "GPU runtime in place" in capsys.readouterr().out and len(log.read_text().splitlines()) == 2
    assert runtime.tool_main(["install"]) == 0 and "in place already" in capsys.readouterr().out
    assert runtime.tool_main(["install", "--uv", "no-such-uv"]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"{prog}-gpu-runtime: no uv (no-such-uv)") and "astral.sh/uv/install.sh" in err


def test_the_command_is_a_console_script_of_the_node():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    prog = PACKAGE.replace("_", "-")
    assert f'{prog}-gpu-runtime = "{PACKAGE}.runtime:tool_main"' in pyproject


PREPARING = f"""
import sys, time
from {PACKAGE}.runtime import prepare
prepare(lambda: time.sleep(60), "cuda:0", "xblick-x")
print("not stopped", flush=True)
"""


@pytest.mark.parametrize(("sig", "code"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_a_start_is_stopped_while_it_prepares_the_gpu(sig, code):
    process = subprocess.Popen([sys.executable, "-c", PREPARING], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        line = process.stdout.readline()  # printed at once, before the work
        assert "preparing the GPU (cuda:0)" in line
        time.sleep(0.3)
        started = time.monotonic()
        process.send_signal(sig)  # SIGINT ignored at the start, as for a background job of a script
        _, err = process.communicate(timeout=10)
    finally:
        process.kill()
    assert process.returncode == code and time.monotonic() - started < 5
    assert "stopped while preparing the GPU" in err
