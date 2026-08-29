# CycloneLens India — MVP

CycloneLens India is an **explainable tropical-cyclone decision-support prototype**. It combines a grayscale satellite-image intensity estimate with entered environmental context (sea-surface temperature, mean sea-level pressure, vertical wind shear, and humidity). It presents an intensity assessment, transparent fusion contributors, data-quality information, and a synthetic forecast-cone interface preview.

> This project is not an operational forecasting system and must not be used in place of IMD warnings.

## What is implemented

- Satellite-image upload and three demo images.
- ResNet18 intensity inference when `cyclone_resnet18.pth` is supplied.
- A clearly disclosed image-statistical demo fallback when the model weight file is absent.
- Environmental late fusion with visible per-variable contributions.
- IMD intensity category, simplified severity, pressure, gale-radius, freshness, and confidence indicators.
- Synthetic 24-hour track-cone UI preview, explicitly marked non-operational.
- API health, samples, inference, and demo-track endpoints.

## Run locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: place a trained `cyclone_resnet18.pth` in this folder, or set `MODEL_PATH` to its location.
4. Start the server:

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

5. Open `http://127.0.0.1:8000`.

## Tests

```bash
python test_app.py
```

## API summary

- `GET /health` — model status and active inference mode.
- `GET /api/samples` — bundled satellite-image samples.
- `POST /predict` — multipart image plus optional environmental values.
- `GET /api/demo-track` — synthetic, non-operational track data used only to demonstrate the forecast-cone interface.

## Next production-oriented steps

Replace manual environmental inputs with aligned MOSDAC/ERA5 ingestion, train and evaluate against storm-wise IBTrACS splits, use calibrated probabilistic forecasts, add PostGIS district overlays, and introduce analyst authentication/audit logs before any institutional trial.
