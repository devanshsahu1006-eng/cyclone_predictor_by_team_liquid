import os
import io
import PIL.Image as Image
import numpy as np
from fastapi.testclient import TestClient
from app import app

def run_all_tests():
    with TestClient(app) as client:
        # Test health
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        print("HEALTH RESPONSE:", data)
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        print("[PASS] HEALTH TEST PASSED")

        # Test samples
        response = client.get("/api/samples")
        assert response.status_code == 200
        samples = response.json()
        assert len(samples) > 0
        print("[PASS] SAMPLES TEST PASSED:", [s["id"] for s in samples])

        # Test predict with synthetic image
        img = Image.fromarray((np.random.rand(224, 224) * 255).astype(np.uint8))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {"file": ("test.png", img_byte_arr, "image/png")}
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        pred_data = response.json()
        assert "vmax" in pred_data
        assert "vmax_kmh" in pred_data
        assert "category" in pred_data
        assert "severity" in pred_data
        assert "pressure_hpa" in pred_data
        assert "confidence" in pred_data
        print("[PASS] PREDICT TEST PASSED:", pred_data)

if __name__ == "__main__":
    run_all_tests()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
