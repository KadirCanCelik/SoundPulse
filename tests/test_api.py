import pytest

from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

from src.api.main import app

pytestmark = pytest.mark.asyncio

@patch("src.api.routes.chain")
async def test_analyze_audio_endpoint(mock_chain):
    """
    Ensures the /analyze endpoint correctly receives a file,
    triggers the Celery chain, and returns a Composite ID.
    """
    mock_task_b = MagicMock()
    mock_task_b.id = "mock_cpu_id"

    mock_task_a = MagicMock()
    mock_task_a.id = "mock_gpu_id"

    mock_task_b.parent = mock_task_a

    mock_chain_instance = MagicMock()
    mock_chain_instance.apply_async.return_value = mock_task_b
    mock_chain.return_value = mock_chain_instance

    file_content = b"fake_audio_data"
    files = {"file": ("test_audio.wav", file_content, "audio/wav")}
    data = {"num_speakers": 2}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/v1/analyze", files=files, data=data)

    assert response.status_code == 200
    response_data = response.json()
    
    assert response_data["status"] == "PENDING"
    assert "mock_gpu_id::mock_cpu_id" == response_data["job_id"]


@patch("src.api.routes.celery_client.AsyncResult")
async def test_status_endpoint_invalid_id(mock_async_result):
    """
    Ensures the /status endpoint handles invalid or missing IDs gracefully.
    """
    mock_instance = MagicMock()
    mock_instance.state = "PENDING"
    mock_async_result.return_value = mock_instance

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/api/v1/status/invalid_id::test")

    assert response.status_code == 200, "API did not return a successful 200 OK status."
    assert response.json()["status"] == "PENDING", "API failed to fallback gracefully to PENDING."