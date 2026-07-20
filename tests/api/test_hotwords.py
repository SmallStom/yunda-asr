"""热词管理 API 测试."""


def test_list_hotwords(api_client):
    response = api_client.get("/api/v1/hotwords")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data


def test_create_update_delete_hotword(api_client):
    # 创建
    response = api_client.post(
        "/api/v1/hotwords",
        json={"word": "测试热词ABC", "category": "test", "enabled": True},
    )
    assert response.status_code == 201
    item = response.json()
    hw_id = item["id"]
    assert item["word"] == "测试热词ABC"

    # 更新
    response = api_client.put(
        f"/api/v1/hotwords/{hw_id}",
        json={"enabled": False},
    )
    assert response.status_code == 200
    assert response.json()["enabled"] is False

    # 删除
    response = api_client.delete(f"/api/v1/hotwords/{hw_id}")
    assert response.status_code == 204

    # 确认已删除
    response = api_client.get(f"/api/v1/hotwords/{hw_id}")
    assert response.status_code == 404
