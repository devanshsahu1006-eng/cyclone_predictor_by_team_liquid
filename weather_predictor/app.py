"""
Cyclone Intensity Prediction — FastAPI Backend & Command Center
----------------------------------------------------------------
Serves the trained ResNet18 model (cyclone_resnet18.pth) via a /predict endpoint.
Returns predicted wind speed (Vmax in knots and km/h), IMD category, estimated central pressure,
gale wind radius, and severity telemetry. Serves modern command center frontend at root /.

Run locally with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import re
import time
from datetime import datetime, timezone
from html import unescape
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import model as model_module
import risk_assessment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "cyclone_resnet18.pth")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = model_module.DEVICE

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CycloneLens India API",
    description="Explainable, confidence-aware tropical cyclone decision-support prototype",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: nn.Module | None = None
model_load_time: float = 0.0
LIVE_BULLETIN_URL = "https://mausam.imd.gov.in/Forecast/satellite_bulletin_view.php"
LIVE_CYCLONE_URL = "https://mausam.imd.gov.in/responsive/cycloneinformation.php?lang=en"
LIVE_CACHE_TTL_SECONDS = 600
_live_cache: dict[str, object] = {"updated_at": 0.0, "payload": None, "image_url": None}


@app.on_event("startup")
def startup_event():
    global model, model_load_time
    start = time.time()
    try:
        model = model_module.load_model(MODEL_PATH)
        model_load_time = round(time.time() - start, 3)
        print(f"Model loaded successfully from '{MODEL_PATH}' on device '{device}' in {model_load_time}s.")
    except Exception as e:
        print(f"ERROR loading model '{MODEL_PATH}': {e}")


# ---------------------------------------------------------------------------
# Preprocessing — exact satellite IR grayscale pipeline
# ---------------------------------------------------------------------------
def preprocess_image(raw_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")  # 1-channel grayscale
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0  # normalize [0, 1]
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor


def _fetch_url(url: str, accept: str) -> tuple[bytes, str]:
    """Fetch only a fixed official IMD resource with a short timeout."""
    request = Request(url, headers={"User-Agent": "CycloneLens-Student-Prototype/1.0", "Accept": accept})
    with urlopen(request, timeout=15) as response:  # nosec B310: URL is fixed/validated below
        return response.read(), response.headers.get_content_type()


def _strip_html(markup: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", markup, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _image_url_from_bulletin(markup: str) -> str | None:
    candidates = re.findall(r"<img[^>]+src=[\"']([^\"']+)[\"']", markup, flags=re.I)
    absolute = [urljoin(LIVE_BULLETIN_URL, source) for source in candidates]
    preferred = [source for source in absolute if any(key in source.lower() for key in ("sat", "insat", "image", "pic"))]
    for source in preferred + absolute:
        parsed = urlparse(source)
        if parsed.scheme == "https" and parsed.hostname and parsed.hostname.endswith("imd.gov.in"):
            return source
    return None


def _get_live_bulletin() -> tuple[dict[str, object], str | None]:
    now = time.time()
    cached = _live_cache.get("payload")
    if cached and now - float(_live_cache["updated_at"]) < LIVE_CACHE_TTL_SECONDS:
        return cached, _live_cache.get("image_url")  # type: ignore[return-value]

    raw, _ = _fetch_url(LIVE_BULLETIN_URL, "text/html")
    markup = raw.decode("utf-8", errors="replace")
    plain_text = _strip_html(markup)
    timestamp = re.search(r"Date:\s*([^T]{1,40})\s*Time:\s*([^<]{1,30})", plain_text, flags=re.I)
    salient = re.search(r"SALIENT FEATURES:\s*(.{0,1600}?)(?:CLOUD DESCRIPTION|LEGEND)", plain_text, flags=re.I)
    payload: dict[str, object] = {
        "source": "India Meteorological Department (IMD)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "bulletin_time": (f"{timestamp.group(1).strip()} {timestamp.group(2).strip()}" if timestamp else "Timestamp not found in source bulletin"),
        "summary": (salient.group(1).strip() if salient else plain_text[:900]),
        "official_bulletin_url": LIVE_BULLETIN_URL,
        "official_cyclone_url": LIVE_CYCLONE_URL,
        "refresh_seconds": LIVE_CACHE_TTL_SECONDS,
        "status": "Official bulletin available — review the IMD advisory before taking action.",
    }
    image_url = _image_url_from_bulletin(markup)
    _live_cache.update({"updated_at": now, "payload": payload, "image_url": image_url})
    return payload, image_url


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    vmax: float  # Wind speed in knots (kt)
    vmax_kmh: float  # Wind speed in km/h
    vmax_mph: float  # Wind speed in mph
    category: str  # IMD Category
    severity: str  # LOW, MODERATE, SEVERE, EXTREME
    pressure_hpa: float  # Estimated central pressure (hPa)
    radius_km: int  # Estimated storm radius (km)
    confidence: float  # Model inference confidence rating (%)
    inference_time_ms: float  # Latency in ms
    model_mode: str
    advisory: str
    environmental_summary: dict[str, float | str]
    explanation: dict[str, str | list[str]]
    data_quality: dict[str, str | int]
    # --- Added from server3(1): impact & risk assessment ---
    property_damage_prediction: dict[str, str | int] | None = None
    calming_time_prediction: dict[str, str | int | float] | None = None
    mortality_prediction: dict[str, str | int] | None = None
    death_rate_prediction: dict[str, str | float] | None = None
    overall_risk: dict[str, str | float] | None = None
    affected_area_prediction: dict | None = None


class TrackPoint(BaseModel):
    hour: int
    latitude: float
    longitude: float
    uncertainty_km: int
    wind_kt: float


class TrackOutlookResponse(BaseModel):
    source: str
    generated_at: str
    track: list[TrackPoint]
    affected_districts: list[dict[str, str | int]]
    disclaimer: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Cyclone Prediction Command Center API is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "FastAPI",
        "device": str(device),
        "model_loaded": model is not None,
        "model_file": MODEL_PATH,
        "model_architecture": "ResNet18 (1-channel IR input) + environmental late fusion",
        "model_load_time_s": model_load_time,
        "model_mode": "resnet18" if model is not None else "image-statistical-demo",
        "warning": None if model is not None else "No trained weight file is present; image-only demo fallback is active."
    }


@app.get("/api/live-satellite")
def live_satellite_status():
    """Near-real-time IMD satellite bulletin metadata, cached to respect the source service."""
    try:
        payload, image_url = _get_live_bulletin()
        return {**payload, "image_available": image_url is not None,
                "image_proxy_url": "/api/live-satellite/image" if image_url else None}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The official IMD live bulletin is temporarily unavailable. Please use the IMD link directly and try again later.",
        ) from exc


@app.get("/api/live-satellite/image")
def live_satellite_image():
    """Safely proxy the imagery referenced by the IMD bulletin to avoid browser CORS issues."""
    try:
        _, image_url = _get_live_bulletin()
        if not image_url:
            raise HTTPException(status_code=404, detail="The current IMD bulletin does not expose a satellite image link.")
        parsed = urlparse(image_url)
        if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith("imd.gov.in"):
            raise HTTPException(status_code=502, detail="The current imagery source could not be safely verified.")
        image, content_type = _fetch_url(image_url, "image/avif,image/webp,image/png,image/jpeg,*/*")
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=502, detail="The referenced IMD asset was not an image.")
        return StreamingResponse(io.BytesIO(image), media_type=content_type,
                                 headers={"Cache-Control": "public, max-age=600"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Live satellite imagery is temporarily unavailable.") from exc


@app.get("/api/samples")
def list_samples():
    samples_dir = os.path.join(BASE_DIR, "samples")
    if not os.path.exists(samples_dir):
        return []
    
    samples_info = [
        {
            "id": "super_cyclone_amphan",
            "name": "Super Cyclone Amphan (Bay of Bengal)",
            "filename": "super_cyclone_amphan.png",
            "type": "Super Cyclonic Storm",
            "url": "/api/samples/super_cyclone_amphan.png"
        },
        {
            "id": "severe_cyclone_mocha",
            "name": "Severe Cyclone Mocha (North Indian Ocean)",
            "filename": "severe_cyclone_mocha.png",
            "type": "Severe Cyclonic Storm",
            "url": "/api/samples/severe_cyclone_mocha.png"
        },
        {
            "id": "depression_bob01",
            "name": "Tropical Depression BOB-01",
            "filename": "depression_bob01.png",
            "type": "Depression",
            "url": "/api/samples/depression_bob01.png"
        }
    ]
    return samples_info


@app.get("/api/samples/{filename}")
def get_sample_file(filename: str):
    file_path = os.path.join(BASE_DIR, "samples", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(file_path, media_type="image/png")


def _fallback_intensity(tensor: torch.Tensor) -> float:
    """A visible demo fallback, only used when trained weights are unavailable.

    It intentionally uses simple image statistics and is not a meteorological model.
    Keeping it here makes the dashboard demonstrable without fabricating a model result.
    """
    image = tensor.squeeze().cpu().numpy()
    brightness = float(image.mean())
    texture = float(image.std())
    return 18.0 + 92.0 * (0.45 * brightness + 0.55 * texture)


def _fuse_environment(image_vmax: float, sst_c: float, pressure_hpa: float,
                      shear_kt: float, humidity_pct: float) -> tuple[float, list[str]]:
    """Late-fusion adjustment designed for a student prototype, not operational use."""
    warm_ocean = max(-4.0, min(6.0, (sst_c - 27.0) * 1.5))
    low_pressure = max(-5.0, min(7.0, (1010.0 - pressure_hpa) * 0.45))
    low_shear = max(-7.0, min(5.0, (18.0 - shear_kt) * 0.40))
    moisture = max(-3.0, min(3.0, (humidity_pct - 65.0) * 0.10))
    contributors = [
        f"Sea-surface temperature contribution: {warm_ocean:+.1f} kt",
        f"Pressure contribution: {low_pressure:+.1f} kt",
        f"Vertical wind-shear contribution: {low_shear:+.1f} kt",
        f"Humidity contribution: {moisture:+.1f} kt",
    ]
    return image_vmax + warm_ocean + low_pressure + low_shear + moisture, contributors


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    sea_surface_temp_c: float = Form(29.0, ge=20.0, le=38.0),
    mean_sea_level_pressure_hpa: float = Form(995.0, ge=850.0, le=1040.0),
    vertical_wind_shear_kt: float = Form(15.0, ge=0.0, le=100.0),
    relative_humidity_pct: float = Form(75.0, ge=0.0, le=100.0),
    data_age_minutes: int = Form(30, ge=0, le=10080),
    latitude: float | None = Form(None, ge=-90.0, le=90.0),
    longitude: float | None = Form(None, ge=-180.0, le=180.0),
):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid satellite image.")

    start_time = time.time()
    raw_bytes = await file.read()
    tensor = preprocess_image(raw_bytes).to(device)

    if model is not None:
        with torch.no_grad():
            output = model(tensor)
            image_vmax = float(output.item())
        model_mode = "resnet18 + environmental late fusion"
    else:
        image_vmax = _fallback_intensity(tensor)
        model_mode = "image-statistical-demo + environmental late fusion"

    fused_vmax, contributors = _fuse_environment(
        image_vmax, sea_surface_temp_c, mean_sea_level_pressure_hpa,
        vertical_wind_shear_kt, relative_humidity_pct,
    )
    vmax_clean = max(5.0, min(180.0, round(fused_vmax, 2)))
    
    vmax_kmh = round(vmax_clean * 1.852, 2)
    vmax_mph = round(vmax_clean * 1.15078, 2)
    category = model_module.get_imd_category(vmax_clean)
    severity = model_module.get_severity(category)
    pressure_hpa = model_module.estimate_pressure_hpa(vmax_clean)
    radius_km = model_module.estimate_gale_radius_km(vmax_clean)
    
    # Confidence represents prototype data coverage/quality, not forecast probability.
    freshness_penalty = min(28.0, max(0.0, (data_age_minutes - 30) * 0.08))
    confidence = round(min(92.0, max(45.0, 84.0 - freshness_penalty - abs(vmax_clean - 70.0) * 0.10)), 1)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # --- Added from server3(1): impact & risk assessment ---
    # Reuses the already-fused vmax_clean (kt) instead of re-running inference,
    # so this stays consistent with the category/pressure/radius already computed above.
    property_damage = risk_assessment.estimate_property_damage(vmax_clean)
    dissipation_time = risk_assessment.estimate_dissipation_time(vmax_clean)
    mortality_risk = risk_assessment.estimate_mortality_risk(vmax_clean)
    death_rate = risk_assessment.estimate_death_rate(vmax_clean)
    overall_risk = risk_assessment.calculate_overall_risk(
        vmax_clean,
        property_damage["damage_score"],
        mortality_risk["risk_score"],
    )
    affected_area = risk_assessment.estimate_affected_area(vmax_clean, latitude, longitude)

    return PredictionResponse(
        vmax=vmax_clean,
        vmax_kmh=vmax_kmh,
        vmax_mph=vmax_mph,
        category=category,
        severity=severity,
        pressure_hpa=pressure_hpa,
        radius_km=radius_km,
        confidence=confidence,
        inference_time_ms=latency_ms,
        model_mode=model_mode,
        advisory=(
            "High-convective-risk pattern: verify with current IMD guidance and analyst review."
            if severity in {"SEVERE", "EXTREME"}
            else "Prototype assessment only; compare with current IMD guidance before action."
        ),
        environmental_summary={
            "sea_surface_temp_c": sea_surface_temp_c,
            "mean_sea_level_pressure_hpa": mean_sea_level_pressure_hpa,
            "vertical_wind_shear_kt": vertical_wind_shear_kt,
            "relative_humidity_pct": relative_humidity_pct,
        },
        explanation={
            "method": "Late fusion of image intensity estimate and user-supplied environmental context.",
            "top_contributors": contributors,
            "interpretation": "Contributions are prototype model components, not causal proof.",
        },
        data_quality={
            "image": "available",
            "environmental_context": "user-supplied",
            "freshness": "current" if data_age_minutes <= 60 else "stale",
            "data_age_minutes": data_age_minutes,
        },
        property_damage_prediction=property_damage,
        calming_time_prediction=dissipation_time,
        mortality_prediction=mortality_risk,
        death_rate_prediction=death_rate,
        overall_risk=overall_risk,
        affected_area_prediction=affected_area,
    )


@app.get("/api/demo-track", response_model=TrackOutlookResponse)
def demo_track(latitude: float = 16.2, longitude: float = 87.3, wind_kt: float = 65.0):
    """Return a synthetic outlook for UI demonstration, clearly labelled as non-operational."""
    from datetime import datetime, timezone

    drift = [(0, 0.0, 0.0, 45), (6, 0.35, -0.22, 65), (12, 0.82, -0.52, 90),
             (24, 1.55, -1.05, 145)]
    track = [TrackPoint(
        hour=hour,
        latitude=round(latitude + dlat, 2),
        longitude=round(longitude + dlon, 2),
        uncertainty_km=uncertainty,
        wind_kt=round(max(15, wind_kt - hour * 0.35), 1),
    ) for hour, dlat, dlon, uncertainty in drift]
    return TrackOutlookResponse(
        source="Synthetic UI demonstration — not a live forecast",
        generated_at=datetime.now(timezone.utc).isoformat(),
        track=track,
        affected_districts=[
            {"name": "Demo coastal district A", "risk": "Watch", "distance_km": 128},
            {"name": "Demo coastal district B", "risk": "Monitor", "distance_km": 214},
        ],
        disclaimer="This route visualizes the planned forecast-cone interface using synthetic positions. It must not be used for decisions."
    )


# Serve static assets if directory exists
samples_path = os.path.join(BASE_DIR, "samples")
if os.path.exists(samples_path):
    app.mount("/samples", StaticFiles(directory=samples_path), name="samples")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
