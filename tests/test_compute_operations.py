from __future__ import annotations

import json
from datetime import UTC, datetime

from app.compute.service import ComputeOperationsService
from app.core.clock import FrozenClock
from app.database import get_connection, transaction


TEMPLATE = {
    "code": "solver-a",
    "name": "方程求解模板",
    "algorithm": "solver-a",
    "parameter_schema": {
        "iterations": {"type": "integer", "required": True, "minimum": 1, "maximum": 10000},
        "tolerance": {"type": "number", "required": False, "minimum": 0.0, "maximum": 1.0},
        "mode": {"type": "string", "required": True, "choices": ["fast", "accurate"]},
    },
    "default_parameters": {"tolerance": 0.001},
    "max_runtime_seconds": 300,
    "max_attempts": 2,
}


def submit_payload(key: str, *, user: str = "researcher-1", priority: int = 50) -> dict:
    return {
        "template_code": "solver-a",
        "project_code": "project-a",
        "requested_by": user,
        "parameters": {"iterations": 100, "mode": "accurate"},
        "priority": priority,
        "idempotency_key": key,
    }


def create_template(client) -> None:
    response = client.post("/api/compute/templates?actor=administrator", json=TEMPLATE)
    assert response.status_code == 201, response.text


def test_template_submission_idempotency_and_parameter_validation(client):
    create_template(client)
    first = client.post("/api/compute/tasks", json=submit_payload("request-000001"))
    second = client.post("/api/compute/tasks", json=submit_payload("request-000001"))
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    invalid = submit_payload("request-000002")
    invalid["parameters"]["iterations"] = 20000
    rejected = client.post("/api/compute/tasks", json=invalid)
    assert rejected.status_code == 422


def test_priority_capability_claim_and_result_version(client):
    create_template(client)
    low = client.post("/api/compute/tasks", json=submit_payload("priority-low", priority=10)).json()
    high = client.post("/api/compute/tasks", json=submit_payload("priority-high", priority=90)).json()
    no_match = client.post("/api/compute/tasks/claim", json={"worker_id": "w0", "capabilities": ["other"], "lease_seconds": 60})
    assert no_match.status_code == 200 and no_match.json()["task"] is None
    claimed = client.post("/api/compute/tasks/claim", json={"worker_id": "w1", "capabilities": ["solver-a"], "lease_seconds": 60})
    assert claimed.status_code == 200
    assert claimed.json()["task"]["id"] == high["id"]
    completed = client.post(
        f"/api/compute/tasks/{high['id']}/complete",
        json={"worker_id": "w1", "result": {"value": 3.14}, "metrics": {"seconds": 2}},
    )
    assert completed.status_code == 200
    details = client.get(f"/api/compute/task-details/{high['id']}").json()
    assert details["status"] == "succeeded"
    assert details["current_result_version"] == 1
    assert len(details["results"]) == 1
    assert low["status"] == "queued"


def test_quota_cancel_retry_priority_and_batch_interventions(client):
    create_template(client)
    quota = client.put(
        "/api/compute/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "limited", "max_queued": 1, "max_running": 1, "daily_submissions": 2},
    )
    assert quota.status_code == 200
    one = client.post("/api/compute/tasks", json=submit_payload("quota-one", user="limited")).json()
    blocked = client.post("/api/compute/tasks", json=submit_payload("quota-two", user="limited"))
    assert blocked.status_code == 409
    cancelled = client.post(f"/api/compute/tasks/{one['id']}/cancel", json={"actor": "administrator", "reason": "项目暂停"})
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    retried = client.post(f"/api/compute/tasks/{one['id']}/retry", json={"actor": "administrator", "reason": "项目恢复", "priority": 95})
    assert retried.status_code == 200 and retried.json()["priority"] == 95
    other = client.post("/api/compute/tasks", json=submit_payload("batch-other", user="other-user")).json()
    batch = client.post(
        "/api/compute/tasks/batch",
        json={"task_ids": [one["id"], other["id"]], "operation": "priority", "actor": "administrator", "reason": "紧急算例", "priority": 99},
    )
    assert batch.status_code == 200
    assert len(batch.json()["succeeded"]) == 2
    details = client.get(f"/api/compute/task-details/{one['id']}").json()
    assert [item["action"] for item in details["interventions"]] == ["cancel", "retry", "priority"]


def test_failure_backoff_and_expired_lease_recovery(client):
    from app.database import init_db

    init_db()
    clock = FrozenClock(datetime(2026, 9, 26, 2, 0, tzinfo=UTC))
    service = ComputeOperationsService(get_connection(), clock)
    service.create_template(TEMPLATE, "administrator")
    first = service.submit(submit_payload("failure-000001"))
    claimed = service.claim("worker-a", ["solver-a"], 10)
    assert claimed and claimed["id"] == first["id"]
    failed = service.fail(first["id"], "worker-a", "numeric_error", "数值不收敛", True)
    assert failed["status"] == "queued"
    assert failed["available_at"] > failed["updated_at"]
    clock.advance(seconds=2)
    claimed_again = service.claim("worker-a", ["solver-a"], 10)
    assert claimed_again and claimed_again["attempt_count"] == 2
    clock.advance(seconds=11)
    recovered = service.recover_expired()
    assert recovered["exhausted"] == [first["id"]]
    details = service.get_task(first["id"])
    assert details["status"] == "failed"
    assert details["interventions"][-1]["action"] == "lease_recovery"


def bad_template(**overrides) -> dict:
    template = {
        "code": "recycle-water-a",
        "name": "再生水循环用量模板",
        "algorithm": "recycle-water",
        "parameter_schema": {
            "cycle_count": {"type": "integer", "required": True, "minimum": 1, "maximum": 100},
            "reuse_rate": {"type": "number", "required": False, "minimum": 0.0, "maximum": 1.0},
            "mode": {"type": "string", "required": True, "choices": ["fast", "accurate"]},
        },
        "default_parameters": {"reuse_rate": 0.5},
        "max_runtime_seconds": 300,
        "max_attempts": 2,
    }
    template.update(overrides)
    return template


def assert_no_templates(client) -> None:
    listed = client.get("/api/compute/templates")
    assert listed.status_code == 200
    assert listed.json()["items"] == []


def test_create_template_rejects_invalid_default_type_and_rolls_back(client):
    response = client.post(
        "/api/compute/templates?actor=operator-1",
        json=bad_template(default_parameters={"reuse_rate": "half"}),
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["context"]["parameter"] == "reuse_rate"
    assert_no_templates(client)
    # 回滚后同一编码可以立即重新登记为合法模板
    fixed = client.post("/api/compute/templates?actor=operator-1", json=bad_template())
    assert fixed.status_code == 201, fixed.text


def test_create_template_rejects_out_of_range_and_choice_defaults_and_rolls_back(client):
    too_large = client.post(
        "/api/compute/templates?actor=operator-1",
        json=bad_template(default_parameters={"reuse_rate": 1.5}),
    )
    assert too_large.status_code == 422
    assert too_large.json()["error"]["context"]["parameter"] == "reuse_rate"
    assert_no_templates(client)

    bad_choice = client.post(
        "/api/compute/templates?actor=operator-1",
        json=bad_template(default_parameters={"mode": "turbo"}),
    )
    assert bad_choice.status_code == 422
    assert bad_choice.json()["error"]["context"]["parameter"] == "mode"
    assert_no_templates(client)


def test_create_template_rejects_invalid_rule_declarations(client):
    inverted = bad_template(
        parameter_schema={"cycle_count": {"type": "integer", "minimum": 100, "maximum": 1}}
    )
    response = client.post("/api/compute/templates?actor=operator-1", json=inverted)
    assert response.status_code == 422
    assert response.json()["error"]["context"]["parameter"] == "cycle_count"

    bound_type = bad_template(
        parameter_schema={"reuse_rate": {"type": "number", "minimum": "oops"}}
    )
    response = client.post("/api/compute/templates?actor=operator-1", json=bound_type)
    assert response.status_code == 422

    choice_type = bad_template(
        parameter_schema={"mode": {"type": "string", "choices": ["fast", 7]}}
    )
    response = client.post("/api/compute/templates?actor=operator-1", json=choice_type)
    assert response.status_code == 422

    unknown_default = bad_template(default_parameters={"unknown": 1})
    response = client.post("/api/compute/templates?actor=operator-1", json=unknown_default)
    assert response.status_code == 422
    assert_no_templates(client)


def test_failed_creation_then_valid_creation_and_submission_flow(client):
    bad = bad_template(default_parameters={"reuse_rate": False})
    rejected = client.post("/api/compute/templates?actor=operator-1", json=bad)
    assert rejected.status_code == 422
    assert_no_templates(client)

    created = client.post("/api/compute/templates?actor=operator-1", json=bad_template())
    assert created.status_code == 201, created.text

    payload = {
        "template_code": "recycle-water-a",
        "project_code": "reuse-project",
        "requested_by": "researcher-1",
        "parameters": {"cycle_count": 8, "mode": "accurate"},
        "priority": 50,
        "idempotency_key": "water-request-001",
    }
    first = client.post("/api/compute/tasks", json=payload)
    assert first.status_code == 202, first.text
    # 默认值 reuse_rate=0.5 随模板合并生效
    stored_parameters = json.loads(first.json()["parameters_json"])
    assert stored_parameters == {"cycle_count": 8, "mode": "accurate", "reuse_rate": 0.5}
    second = client.post("/api/compute/tasks", json=payload)
    assert second.status_code == 202
    assert second.json()["id"] == first.json()["id"]

    claimed = client.post(
        "/api/compute/tasks/claim",
        json={"worker_id": "w1", "capabilities": ["recycle-water"], "lease_seconds": 60},
    )
    assert claimed.status_code == 200
    task_id = claimed.json()["task"]["id"]
    completed = client.post(
        f"/api/compute/tasks/{task_id}/complete",
        json={"worker_id": "w1", "result": {"saved": 42}, "metrics": {"seconds": 3}},
    )
    assert completed.status_code == 200
    details = client.get(f"/api/compute/task-details/{task_id}").json()
    assert details["status"] == "succeeded"
    assert details["current_result_version"] == 1
    assert len(details["results"]) == 1

    # 模板登记成功后，提交时仍执行相同的参数校验
    invalid = {**payload, "idempotency_key": "water-request-002", "parameters": {"cycle_count": 8, "mode": "turbo"}}
    rejected_task = client.post("/api/compute/tasks", json=invalid)
    assert rejected_task.status_code == 422
    assert rejected_task.json()["error"]["context"]["parameter"] == "mode"

