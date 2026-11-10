"""The benchmark scripts run end to end and write self-describing results files."""

import json

from benchmarks import benchmark_failover, benchmark_overhead

MACHINE_KEYS = {
    "timestamp_utc", "git_sha", "git_dirty", "platform", "arch", "cpu_model", "logical_cpus",
    "total_ram_bytes", "python_version", "python_implementation", "packages",
    "load_avg_1_5_15_before", "load_avg_1_5_15_after", "command",
}


def test_overhead_inprocess_records_every_pair_per_run(tmp_path):
    out = tmp_path / "overhead.json"
    argv = ["--iterations", "4", "--runs", "2", "--warmup", "1", "--out", str(out)]
    assert benchmark_overhead.main(argv) == 0

    doc = json.loads(out.read_text())
    assert MACHINE_KEYS <= doc["machine"].keys()
    assert {"fastapi", "starlette", "httpx", "uvicorn", "pydantic"} <= doc["machine"]["packages"].keys()
    assert doc["config"]["mode"] == "inprocess" and doc["config"]["mock"]["added_ttft_delay_ms"] == 0
    summary = doc["summary"]
    assert summary["errors"] == 0
    assert summary["pooled"]["overhead_ms"]["n"] == 8
    assert {"p50", "p90", "p99", "mean", "stdev"} <= summary["pooled"]["overhead_ms"].keys()
    assert [r["overhead_ms"]["n"] for r in summary["per_run"]] == [4, 4]
    assert [s["gateway_first"] for s in doc["samples"][:4]] == [False, True, False, True], "order alternates"


def test_overhead_loopback_streams_over_tcp(tmp_path):
    out = tmp_path / "overhead-loopback.json"
    argv = ["--mode", "loopback", "--iterations", "3", "--runs", "1", "--warmup", "1", "--out", str(out)]
    assert benchmark_overhead.main(argv) == 0

    summary = json.loads(out.read_text())["summary"]
    assert summary["mode"] == "loopback" and summary["errors"] == 0
    assert summary["pooled"]["overhead_ms"]["n"] == 3


def test_failover_serves_every_request_and_records_the_breaker_cycle(tmp_path):
    out = tmp_path / "failover.json"
    argv = ["--normal", "2", "--outage", "8", "--concurrency", "4", "--recovery", "3",
            "--recovery-timeout-s", "0.1", "--out", str(out)]
    assert benchmark_failover.main(argv) == 0

    doc = json.loads(out.read_text())
    assert MACHINE_KEYS <= doc["machine"].keys()
    summary = doc["summary"]
    assert summary["non_200"] == 0 and summary["incomplete_streams"] == 0
    assert summary["status_counts"] == {"200": 13}
    assert summary["phases"]["outage"]["served_by_fallback"] == 8
    assert summary["phases"]["recovery"]["served_by_primary"] == 3
    assert summary["primary_state_sequence"] == ["CLOSED", "OPEN", "HALF-OPEN", "CLOSED"]
    assert summary["final_states"]["groq-primary"] == "CLOSED"
    assert len(doc["samples"]) == 13


def test_a_stream_ended_by_an_error_frame_is_not_complete():
    from benchmarks._report import stream_completed
    from gateway.stream.error_frames import generate_sse_error_frame

    good = (b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n\n'
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            b"data: [DONE]\n\n")
    broken = b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n\n' + \
        generate_sse_error_frame("peer closed", "upstream_connection_broken", 502, "groq")
    assert stream_completed(good)
    assert not stream_completed(broken), "the error frame ends in [DONE] too"
    assert not stream_completed(good.replace(b'"stop"', b"null")), "no finish_reason"
