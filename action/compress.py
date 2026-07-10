"""GitHub Action entrypoint: submit a compression job and gate CI on fidelity.

Reads its inputs from TP_* env vars (set by action.yml), submits the job to the
TurboPress control plane, polls until it finishes, prints a summary, and exits
non-zero if any fidelity gate failed (unless fail-on-gate is false). Pure stdlib
so no pip install is needed on the runner.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _parse_gates(raw: str) -> dict[str, float]:
    gates: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            _die(f"malformed gate '{part}' (expected key=value)")
        key, value = part.split("=", 1)
        gates[key.strip()] = float(value.strip())
    return gates


def _request(method: str, url: str, api_key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        _die(f"{method} {url} -> HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        _die(f"{method} {url} failed: {exc}")
    return {}  # unreachable


def _set_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def _die(message: str) -> None:
    print(f"::error::{message}")
    sys.exit(1)


def main() -> None:
    api_url = _env("TP_API_URL", "https://api.turbopress.ai").rstrip("/")
    api_key = _env("TP_API_KEY")
    model = _env("TP_MODEL")
    if not api_key or not model:
        _die("api-key and model are required")
    targets = [t.strip() for t in _env("TP_TARGETS", "4bit").split(",") if t.strip()]
    gates = _parse_gates(_env("TP_GATES"))
    private = _env("TP_PRIVATE", "true").lower() != "false"
    fail_on_gate = _env("TP_FAIL_ON_GATE", "true").lower() != "false"
    timeout = int(_env("TP_TIMEOUT_SECONDS", "3600"))

    print(f"Submitting {model} -> {targets} with gates {gates or '(none)'}")
    job = _request(
        "POST",
        f"{api_url}/jobs",
        api_key,
        {"model_ref": model, "targets": targets, "fidelity_gates": gates,
         "private": private},
    )
    job_id = job["id"]
    _set_output("job-id", job_id)
    print(f"job {job_id} status={job['status']}")

    deadline = time.time() + timeout
    while job["status"] in ("queued", "running"):
        if time.time() > deadline:
            _die(f"timed out after {timeout}s waiting for job {job_id}")
        time.sleep(5)
        job = _request("GET", f"{api_url}/jobs/{job_id}", api_key)

    if job["status"] != "succeeded":
        _die(f"job {job_id} {job['status']}: {job.get('error')}")

    print(f"\nResults for job {job_id}:")
    any_failed = False
    for cert in job.get("certificates", []):
        m = cert["metrics"]
        status = "PASS" if cert["gates_passed"] else "FAIL"
        print(
            f"  [{status}] {cert['target']:>5}  bits/w={cert['bits_per_weight']:.3f}  "
            f"KL={m.get('mean_kl')}  top1={m.get('top1_agreement')}  ppl={m.get('ppl_q')}"
        )
        for failure in cert.get("gate_failures", []):
            print(f"      ::warning::gate failed: {failure}")
        any_failed = any_failed or not cert["gates_passed"]

    gates_passed = not any_failed
    _set_output("gates-passed", "true" if gates_passed else "false")

    if any_failed and gates and fail_on_gate:
        _die(f"job {job_id} did not meet fidelity gates")
    print(f"\nDone: gates-passed={gates_passed}")


if __name__ == "__main__":
    main()
