import pytest
from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.auth import bearer_matches, load_service_token

TOKEN = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def test_constant_time_bearer_shapes_and_token_loading(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n")
    token_file.chmod(0o600)
    assert load_service_token(token_file=token_file) == TOKEN
    assert load_service_token(environ={"NXML_CONTROL_TOKEN": TOKEN}) == TOKEN
    assert bearer_matches(f"Bearer {TOKEN}", TOKEN)
    assert bearer_matches(f"bearer {TOKEN}", TOKEN)
    assert not bearer_matches(None, TOKEN)
    assert not bearer_matches("Basic nope", TOKEN)
    assert not bearer_matches("Bearer wrong", TOKEN)
    token_file.chmod(0o640)
    assert load_service_token(token_file=token_file) == TOKEN
    token_file.chmod(0o644)
    assert load_service_token(token_file=token_file) == TOKEN
    token_file.chmod(0o660)
    with pytest.raises(PermissionError, match="not group/world-writable"):
        load_service_token(token_file=token_file)
    with pytest.raises(ValueError, match="at least 32"):
        load_service_token(environ={"NXML_CONTROL_TOKEN": "short"})


def test_health_public_but_all_control_endpoints_require_correct_token(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path, auth_token=TOKEN))
    health = client.get("/healthz")
    assert health.status_code == 200 and health.json() == {"status": "ready"}
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}):
        response = client.get("/v1/deployments", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    authorized = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/v1/deployments", headers=authorized).status_code == 200
    assert client.get("/openapi.json").status_code == 401
    spec = client.get("/openapi.json", headers=authorized).json()
    # REST is control-plane only; 30fps ZMQ+frames inference stays separate.
    assert spec["components"]["securitySchemes"]["bearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert "security" not in spec["paths"]["/healthz"]["get"]
    assert spec["paths"]["/v1/deployments"]["get"]["security"] == [{"bearerAuth": []}]
    assert not any("predict" in path or "frame" in path for path in spec["paths"])


def test_short_programmatic_auth_token_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least 32"):
        create_app(state_dir=tmp_path, auth_token="short")
