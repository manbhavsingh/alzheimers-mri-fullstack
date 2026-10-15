import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import httpx
import numpy as np

from app.gradcam_note import gradcam_note
from app.main import app
from app.report import Report


def test_api_root_is_discoverable():
    async def request_root():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/")

    response = asyncio.run(request_root())

    assert response.status_code == 200
    assert response.json()["service"] == "Neurio API"
    assert response.json()["health"] == "/health"
    assert response.json()["docs"] == "/docs"

    async def request_head():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.head("/")

    head_response = asyncio.run(request_head())
    assert head_response.status_code == 200




def test_gradcam_note_focused_upper_left():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0:2, 0:2] = 0.9

    note = gradcam_note(cam)

    assert "upper-left" in note
    assert "focused" in note
    assert "4.0%" in note
    assert "relative to the image, not anatomical" in note


def test_gradcam_note_wide():
    cam = np.full((10, 10), 0.9, dtype=np.float32)

    note = gradcam_note(cam)

    assert "wide" in note
    assert "100.0%" in note
    assert "middle-center" in note


def test_gradcam_note_empty():
    cam = np.zeros((10, 10), dtype=np.float32)

    assert gradcam_note(cam) == "No strongly activated region."


def test_report_endpoint_with_gemini_mocked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    expected = Report(
        summary="The model returned Mild Demented as its predicted category.",
        confidence_note="These are model scores, not clinical probabilities.",
        caveats=[
            "This output cannot diagnose Alzheimer's disease.",
            "The Grad-CAM overlay does not prove a medical finding.",
        ],
        next_steps=["Discuss the result with a qualified doctor."],
    )
    mock_client = Mock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=expected,
        text=expected.model_dump_json(),
    )

    async def request_report():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/report",
                json={
                    "label": "Mild Demented",
                    "probs": {
                        "Non-Demented": 0.05,
                        "Very Mild Demented": 0.15,
                        "Mild Demented": 0.70,
                        "Moderate Demented": 0.10,
                    },
                    "gradcam_note": "Activation centroid: middle-center. Spread: moderate.",
                },
            )

    with patch("app.report.genai.Client", return_value=mock_client) as client_factory:
        response = asyncio.run(request_report())

    assert response.status_code == 200
    assert response.json() == expected.model_dump()
    client_factory.assert_called_once_with(api_key="test-only-key", http_options=ANY)
    mock_client.models.generate_content.assert_called_once()
    mock_client.close.assert_called_once()
