"""End-to-end control-plane flow with the inline runner (no GPU, no network).

Proves the exit-criterion loop: a user signs up, activates billing, submits a
private compression job, and gets back succeeded jobs with signed, gate-checked
certificates — with no human in the loop.
"""

from service.jobs import evaluate_gates
from turbopress.certificate import verify_certificate


def _signup(client, email="stranger@example.com"):
    r = client.post("/signup", json={"email": email})
    assert r.status_code == 200, r.text
    body = r.json()
    return body["api_key"], {"Authorization": f"Bearer {body['api_key']}"}


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_auth_required(client):
    assert client.post("/jobs", json={"model_ref": "x"}).status_code == 401
    _, headers = _signup(client)
    # Malformed key rejected.
    assert client.get(
        "/jobs/none", headers={"Authorization": "Bearer nope"}
    ).status_code == 401


def test_full_pay_and_compress_flow(client):
    _, headers = _signup(client)

    # "Pay" (dev shortcut; the Stripe webhook does this in production).
    assert client.post("/billing/dev-activate", headers=headers).json() == {
        "billing_active": True
    }

    # Submit a private job with two targets and fidelity gates.
    r = client.post(
        "/jobs",
        headers=headers,
        json={
            "model_ref": "Qwen/Qwen3-0.6B",
            "targets": ["4bit", "2bit"],
            "fidelity_gates": {"mean_kl_max": 0.10, "top1_agreement_min": 0.70},
            "private": True,
        },
    )
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "succeeded"
    assert len(job["certificates"]) == 2
    assert job["gates_passed"] is False  # the 2-bit target violates the KL gate

    by_target = {c["target"]: c for c in job["certificates"]}
    assert by_target["4bit"]["gates_passed"] is True
    assert by_target["2bit"]["gates_passed"] is False
    assert by_target["2bit"]["gate_failures"]

    # Certificates are signed and verify.
    for cert in job["certificates"]:
        assert verify_certificate(cert["certificate"]) is True

    # The job is retrievable and the public certificate endpoint works.
    got = client.get(f"/jobs/{job['id']}", headers=headers)
    assert got.status_code == 200
    cert_id = by_target["4bit"]["id"]
    pub = client.get(f"/certificates/{cert_id}")
    assert pub.status_code == 200
    assert verify_certificate(pub.json()) is True
    assert pub.json()["manifest"]["bits"] == 4


def test_job_ownership_isolation(client):
    _, alice = _signup(client, "alice@example.com")
    r = client.post(
        "/jobs", headers=alice, json={"model_ref": "m", "targets": ["4bit"]}
    )
    job_id = r.json()["id"]
    _, bob = _signup(client, "bob@example.com")
    assert client.get(f"/jobs/{job_id}", headers=bob).status_code == 404


def test_gate_evaluation_unit():
    ok, fails = evaluate_gates(
        {"mean_kl": 0.08, "top1_agreement": 0.84},
        {"mean_kl_max": 0.10, "top1_agreement_min": 0.70},
    )
    assert ok and not fails
    ok, fails = evaluate_gates({"mean_kl": 1.2}, {"mean_kl_max": 0.10})
    assert not ok and fails
    ok, fails = evaluate_gates({}, {"mean_kl_max": 0.10})  # missing metric fails
    assert not ok and fails
