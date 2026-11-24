import json
import logging
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "official_ruler"))

from map_results import (  # noqa: E402
    PINNED_RULER_COMMIT,
    map_prediction_records,
    read_prediction_jsonl,
    write_mapped_results,
)
from run_official_ruler import build_commands, verify_server  # noqa: E402

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


def test_build_commands_wires_tier_server_env():
    from pathlib import Path as P

    plan = build_commands(
        P("/tmp/RULER"), "http://localhost:11555/v1", "qwen38-27b", 32000, 42,
        ["niah", "vt", "fwe"], 512, "models/qwen38-27b",
    )
    assert len(plan) == 4, "one data-gen per task + one eval step"
    for step in plan:
        assert step["env"]["OPENAI_API_BASE"] == "http://localhost:11555/v1"
        assert step["env"]["MODEL_NAME"] == "qwen38-27b"
        assert step["env"]["HF_MIRROR"] == "https://hf-mirror.com"
    assert plan[-1]["argv"][-1] == "512", "eval must receive --max-tokens 512"
    with pytest.raises(SystemExit):
        build_commands(
            P("/tmp/RULER"), "http://localhost:11555/v1", "m", 32000, 42, ["qa"], 512, "tok"
        )


class _MockTierHandler(BaseHTTPRequestHandler):
    # Minimal OpenAI-compatible /models surface for the smoke test.

    def do_GET(self):  # noqa: N802
        if self.path.endswith("/models"):
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


def test_adapter_module_dry_run_smoke():
    # The dry-run mode must complete without a RULER repo, a server, or any
    # network access: it verifies the pin (skipped here via a stubbed repo
    # path only when the repo exists) and prints the plan. We assert the
    # module exits 0 on --help-style usage of the plan builder instead of
    # requiring a clone: the pure plan builder is covered above, so here we
    # only pin that the script parses and exposes --dry-run.
    script = Path(__file__).resolve().parent.parent / "scripts" / "official_ruler" / "run_official_ruler.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
