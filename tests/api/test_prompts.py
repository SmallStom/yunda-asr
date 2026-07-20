"""Prompt 管理 API 测试."""


def test_list_prompts(api_client):
    response = api_client.get("/api/v1/prompts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert "v1" in [v["version"] for v in data]


def test_get_prompt(api_client):
    response = api_client.get("/api/v1/prompts/v1")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "user_template" in data


def test_switch_default_prompt(api_client):
    response = api_client.post("/api/v1/prompts/v2/set-default")
    assert response.status_code == 200
    assert response.json()["default_version"] == "v2"

    # 恢复
    response = api_client.post("/api/v1/prompts/v1/set-default")
    assert response.status_code == 200
