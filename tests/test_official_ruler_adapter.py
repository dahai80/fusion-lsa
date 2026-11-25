import json
import logging
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "official_ruler"))

from map_results import (
    PINNED_RULER_COMMIT,
    map_prediction_records,
    read_prediction_jsonl,
    write_mapped_results,
)
from run_official_ruler import build_commands, verify_server

logger = logging.getLogger(__name__)


def test_map_prediction_records_counts_hits():
    records = [
        {"pred": "The magic number is 123-456.", "golden": "123-456"},
        {"pred": "I do not know.", "golden": "123-456"},
        {"pred": "It is 42.", "golden": "42"},
    ]
    result = map_prediction_records(records, "niah", 32000)
    assert result["hits"] == 2
    assert result["total"] == 3
    assert abs(result["accuracy"] - 2 / 3) < 1e-9
    assert len(result["details"]) == 3


def test_map_accepts_official_field_variants():
    for pred_field, gold_field in (("prediction", "answer"), ("output", "gold"), ("response", "label")):
        result = map_prediction_records(
            [{"pred_field": "x", "gold_field": "y", pred_field: "value one", gold_field: "one"}],
            "vt", 32000,
        )
        assert result["hits"] == 1, f"field variant {pred_field}/{gold_field} not recognized"
    with pytest.raises(ValueError, match="prediction"):
        map_prediction_records([{"golden": "42"}], "niah", 32000)
    with pytest.raises(ValueError, match="golden"):
        map_prediction_records([{"pred": "42"}], "niah", 32000)
    with pytest.raises(ValueError, match="no prediction records"):
        map_prediction_records([], "niah", 32000)


def test_read_prediction_jsonl_rejects_bad_lines(tmp_path):
    path = tmp_path / "pred.jsonl"
    path.write_text('{"pred": "a", "golden": "a"}\nnot json\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        read_prediction_jsonl(path)


def test_write_mapped_results_schema(tmp_path):
    results = [map_prediction_records([{"pred": "g", "golden": "g"}], "niah", 32000)]
    path = write_mapped_results(results, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["harness"] == "NVIDIA/RULER (official)"
    assert payload["pinned_commit"] == PINNED_RULER_COMMIT
    assert payload["tasks"][0]["hits"] == 1
    assert payload["tasks"][0]["total"] == 1
    assert payload["mean_accuracy"] == 1.0


def test_build_commands_wires_pinned_pipeline():
    from pathlib import Path as P

    plan = build_commands(
        P("/tmp/RULER"), "http://localhost:11555/v1", 32000, 42,
        ["niah", "vt", "fwe"], "models/qwen38-27b", num_samples=50,
    )
    assert len(plan) == 7, "prepare + call_api per task, then one evaluate"
    for step in plan:
        assert step["cwd"] == "/tmp/RULER/scripts", "pinned scripts need cwd=<RULER>/scripts"
        assert step["env"]["OPENAI_API_BASE"] == "http://localhost:11555/v1"
        assert step["env"]["OPENAI_BASE_URL"] == "http://localhost:11555/v1"
        assert step["env"]["HF_MIRROR"] == "https://hf-mirror.com"
        assert "ruler_compat" in step["env"]["PYTHONPATH"], "nemo shim must be on PYTHONPATH"
    prepare = plan[0]["argv"]
    assert prepare[1:3] == ["data/prepare.py", "--save_dir"]
    assert "--task" in prepare and prepare[prepare.index("--task") + 1] == "niah_single_1"
    assert "--tokenizer_type" in prepare and prepare[prepare.index("--tokenizer_type") + 1] == "hf"
    assert "--num_samples" in prepare and prepare[prepare.index("--num_samples") + 1] == "50"
    call = plan[1]["argv"]
    assert call[1] == "pred/call_api.py"
    assert call[call.index("--server_type") + 1] == "openai"
    assert call[call.index("--temperature") + 1] == "0", "greedy decoding matches our bench"
    assert plan[-1]["argv"][1:3] == ["eval/evaluate.py", "--data_dir"]
    assert plan[-1]["argv"][3] .endswith("pred")
    with pytest.raises(SystemExit):
        build_commands(
            P("/tmp/RULER"), "http://localhost:11555/v1", 32000, 42, ["nope"], "tok"
        )


def test_emitted_paths_exist_in_pinned_checkout():
    # Regression for issue #17: every script path the adapter emits must
    # exist at the pinned commit. Skips when no pinned clone is present.
    repo = Path.home() / "work" / "RULER"
    if not (repo / "scripts" / "data" / "prepare.py").exists():
        pytest.skip("no pinned RULER checkout at ~/work/RULER")
    plan = build_commands(
        repo, "http://localhost:11555/v1", 32000, 42, ["niah", "vt", "fwe", "qa"], "tok",
    )
    for step in plan:
        script = Path(step["cwd"]) / step["argv"][1]
        assert script.exists(), f"emitted script missing at pin: {script}"


def test_nemo_shim_manifest_roundtrip(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "official_ruler" / "ruler_compat"))
    try:
        from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
    finally:
        sys.path.pop(0)
    path = tmp_path / "m.jsonl"
    write_manifest(path, [{"index": 0, "input": "a"}, {"index": 1, "input": "b"}])
    assert read_manifest(path) == [{"index": 0, "input": "a"}, {"index": 1, "input": "b"}]
    path.write_text('{"index": 2}\nnot json\n\n')
    assert read_manifest(path) == [{"index": 2}]
    assert read_manifest(path, fields=["index"]) == [{"index": 2}]


class _MockTierHandler(BaseHTTPRequestHandler):
    # Minimal OpenAI-compatible /models surface for the smoke test. When
    # server.required_key is set, /models 401s without a matching Bearer
    # header (issue #18: key-required tier servers).

    def do_GET(self):
        if self.path.endswith("/models"):
            required_key = getattr(self.server, "required_key", None)
            if required_key is not None:
                auth = self.headers.get("Authorization", "")
                if auth != "Bearer " + required_key:
                    self.send_response(401)
                    self.end_headers()
                    return
            body = json.dumps({"data": [{"id": "qwen38-27b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # silence per-request stderr
        pass


def test_smoke_adapter_against_mock_tier_server():
    server = HTTPServer(("127.0.0.1", 0), _MockTierHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        payload = verify_server(base_url)
        assert payload["data"][0]["id"] == "qwen38-27b"
        with pytest.raises(SystemExit, match="not serving"):
            verify_server("http://127.0.0.1:1/v1")
    finally:
        server.shutdown()


def test_verify_server_sends_bearer_key(monkeypatch):
    # Issue #18: the /models probe must authenticate on key-required tier
    # servers using FUSION_LSA_BENCH_KEY, same as the RULER subprocesses.
    monkeypatch.setenv("FUSION_LSA_BENCH_KEY", "secret-bench-key")
    server = HTTPServer(("127.0.0.1", 0), _MockTierHandler)
    server.required_key = "secret-bench-key"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        payload = verify_server(base_url)
        assert payload["data"][0]["id"] == "qwen38-27b"
        monkeypatch.delenv("FUSION_LSA_BENCH_KEY")
        with pytest.raises(SystemExit, match="401"):
            verify_server(base_url)
    finally:
        server.shutdown()


def test_adapter_module_dry_run_smoke():
    # The dry-run mode must complete without a RULER repo, a server, or any
    # network access: it verifies the pin (skipped here via a stubbed repo
    # path only when the repo exists) and prints the plan. We assert the
    # module exits 0 on --help-style usage of the plan builder instead of
    # requiring a clone: the pure plan builder is covered above, so here we
    # only pin that the script parses and exposes --dry-run.
    script = Path(__file__).resolve().parent.parent / "scripts" / "official_ruler" / "run_official_ruler.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=True
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
