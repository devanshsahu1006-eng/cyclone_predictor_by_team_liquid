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
        assert "model_mode" in data
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
        form_data = {
            "sea_surface_temp_c": "29.2",
            "mean_sea_level_pressure_hpa": "990",
            "vertical_wind_shear_kt": "12",
            "relative_humidity_pct": "78",
            "data_age_minutes": "20",
        }
        response = client.post("/predict", files=files, data=form_data)
        assert response.status_code == 200
        pred_data = response.json()
        assert "vmax" in pred_data
        assert "vmax_kmh" in pred_data
        assert "category" in pred_data
        assert "severity" in pred_data
        assert "pressure_hpa" in pred_data
        assert "confidence" in pred_data
        assert "environmental_summary" in pred_data
        assert "explanation" in pred_data
        assert "data_quality" in pred_data
        print("[PASS] PREDICT TEST PASSED:", pred_data)

        # --- Added: impact & risk assessment fields (from server3) ---
        assert "property_damage_prediction" in pred_data
        assert "risk_level" in pred_data["property_damage_prediction"]
        assert "damage_score" in pred_data["property_damage_prediction"]
        assert "calming_time_prediction" in pred_data
        assert "estimated_hours" in pred_data["calming_time_prediction"]
        assert "mortality_prediction" in pred_data
        assert "risk_score" in pred_data["mortality_prediction"]
        assert "death_rate_prediction" in pred_data
        assert "estimated_death_rate_percent" in pred_data["death_rate_prediction"]
        assert "overall_risk" in pred_data
        assert "overall_risk_score" in pred_data["overall_risk"]
        assert "affected_area_prediction" in pred_data
        assert "impact_radius_km" in pred_data["affected_area_prediction"]
        print("[PASS] IMPACT & RISK ASSESSMENT TEST PASSED")

        # Same predict call, now with optional eye coordinates for the bounding box
        img_byte_arr.seek(0)
        files_geo = {"file": ("test.png", img_byte_arr, "image/png")}
        form_data_geo = {**form_data, "latitude": "16.2", "longitude": "87.3"}
        response = client.post("/predict", files=files_geo, data=form_data_geo)
        assert response.status_code == 200
        geo_data = response.json()
        assert "affected_bounding_box" in geo_data["affected_area_prediction"]
        assert "min_latitude" in geo_data["affected_area_prediction"]["affected_bounding_box"]
        print("[PASS] AFFECTED AREA GEO BOUNDING BOX TEST PASSED")

        response = client.get("/api/demo-track")
        assert response.status_code == 200
        outlook = response.json()
        assert len(outlook["track"]) == 4
        assert "Synthetic" in outlook["source"]
        print("[PASS] DEMO OUTLOOK TEST PASSED")

if __name__ == "__main__":
    run_all_tests()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
