"""AquaSentinel AI: FastAPI backend.  Run:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.config import Settings
from app.ml import features as F
from app.services.engine import Conflict, Engine, NotFound

settings = Settings()
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = Engine(settings)
    yield


app = FastAPI(title="AquaSentinel AI", version="1.0.0", lifespan=lifespan,
              description="Predict the stress. Save the water. Protect the harvest.")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])


def engine() -> Engine:
    return state["engine"]


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Missing or invalid X-API-Key")


@app.exception_handler(NotFound)
async def _nf(_, e):
    return JSONResponse({"detail": str(e)}, status_code=404)


@app.exception_handler(Conflict)
async def _cf(_, e):
    return JSONResponse({"detail": str(e)}, status_code=409)


@app.exception_handler(ValueError)
async def _ve(_, e):
    return JSONResponse({"detail": str(e)}, status_code=422)


# ----------------------------------------------------------------- schemas
class ZoneIn(BaseModel):
    id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,32}$")
    name: str
    crop: str
    stage: str = "initial"
    area_m2: float = Field(gt=0)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    device_id: str
    pump_flow_lpm: float = Field(default=20.0, gt=0)
    weight: float = Field(default=1.0, gt=0)
    tz_offset_min: int = 330
    auto_mode: bool = False


class ZonePatch(BaseModel):
    name: Optional[str] = None
    crop: Optional[str] = None
    stage: Optional[str] = None
    area_m2: Optional[float] = Field(default=None, gt=0)
    lat: Optional[float] = None
    lon: Optional[float] = None
    pump_flow_lpm: Optional[float] = Field(default=None, gt=0)
    weight: Optional[float] = Field(default=None, gt=0)
    tz_offset_min: Optional[int] = None
    auto_mode: Optional[bool] = None


class Telemetry(BaseModel):
    device_id: str
    soil_moisture: float = Field(ge=0, le=100)
    air_temp: float = Field(ge=-20, le=70)
    humidity: float = Field(ge=0, le=100)
    tank_pct: Optional[float] = Field(default=None, ge=0, le=100)
    flow_lpm: float = Field(default=0.0, ge=0)
    pump_on: bool = False
    ts: Optional[float] = None


class BudgetIn(BaseModel):
    available_l: float = Field(ge=0)
    apply: bool = False


class ActuateIn(BaseModel):
    volume_l: Optional[float] = Field(default=None, gt=0)
    duration_min: Optional[float] = Field(default=None, gt=0)


class CompleteIn(BaseModel):
    actual_volume_l: float = Field(ge=0)
    duration_s: Optional[float] = None
    status: Literal["done", "failed"] = "done"


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/app")


@app.get("/app", include_in_schema=False)
def frontend():
    return FileResponse("app/static/aquasentinel.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "model": engine().model.meta, "weather": settings.weather_provider}


@app.get("/api/crops")
def crops():
    return {"crops": {k: {"critical_moisture_pct": v["critical"], "kc": v["kc"]} for k, v in F.CROPS.items()},
            "stages": list(F.STAGES)}


# zones
@app.post("/api/zones", status_code=201, dependencies=[Depends(require_key)])
def create_zone(z: ZoneIn):
    return engine().create_zone(z.model_dump())


@app.get("/api/zones")
def list_zones():
    return engine().list_zones()


@app.get("/api/zones/{zid}")
def get_zone(zid: str):
    return engine().get_zone(zid)


@app.patch("/api/zones/{zid}", dependencies=[Depends(require_key)])
def patch_zone(zid: str, p: ZonePatch):
    return engine().update_zone(zid, p.model_dump(exclude_none=True))


# device ingestion (ESP32): returns the next pending command in the same round trip
@app.post("/api/telemetry", dependencies=[Depends(require_key)])
def telemetry(t: Telemetry):
    return engine().ingest(t.model_dump())


@app.post("/api/commands/{cid}/complete", dependencies=[Depends(require_key)])
def complete(cid: int, c: CompleteIn):
    return engine().complete_command(cid, c.actual_volume_l, c.duration_s, c.status)


# dashboard + intelligence
@app.get("/api/dashboard")
def dashboard():
    return engine().dashboard()


@app.get("/api/zones/{zid}/readings")
def readings(zid: str, hours: float = Query(24, gt=0, le=720)):
    return engine().readings(zid, hours)


@app.get("/api/zones/{zid}/recommendation")
def recommendation(zid: str):
    return engine().recommend(zid)


@app.get("/api/zones/{zid}/fingerprint")
def fingerprint(zid: str):
    return engine().fingerprint(zid)


@app.get("/api/zones/{zid}/feedback")
def feedback(zid: str):
    return engine().feedback(zid)


@app.post("/api/zones/{zid}/simulate")
def simulate(zid: str):
    return engine().simulate(zid)


@app.post("/api/budget/allocate", dependencies=[Depends(require_key)])
def budget(b: BudgetIn):
    return engine().allocate_budget(b.available_l, b.apply)


# actuation
@app.post("/api/zones/{zid}/actuate", status_code=202, dependencies=[Depends(require_key)])
def actuate(zid: str, a: ActuateIn = ActuateIn()):
    return engine().actuate(zid, a.volume_l, a.duration_min)


@app.post("/api/zones/{zid}/stop", status_code=202, dependencies=[Depends(require_key)])
def stop(zid: str):
    return engine().stop(zid)


# measurement + model
@app.get("/api/metrics")
def metrics():
    return engine().metrics()


@app.post("/api/model/retrain", dependencies=[Depends(require_key)])
def retrain():
    return engine().retrain()
