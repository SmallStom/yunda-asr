"""纠错 API 测试."""


def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_correct_basic(api_client):
    response = api_client.post(
        "/api/v1/correct",
        json={"text": "十八号道差开通反位，信号好了", "layers": [1, 2, 3]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original"] == "十八号道差开通反位，信号好了"
    assert "18号道岔" in data["corrected"]
    assert "layer1" in data["layer_outputs"]
    assert "layer2" in data["layer_outputs"]


def test_correct_batch(api_client):
    response = api_client.post(
        "/api/v1/correct/batch",
        json={
            "items": [
                {"text": "十八号道差开通反位", "layers": [1, 2, 3]},
                {"text": "点击送人节按钮", "layers": [1, 2, 3]},
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2


def test_correct_invalid_input(api_client):
    response = api_client.post("/api/v1/correct", json={})
    assert response.status_code == 422
