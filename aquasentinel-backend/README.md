# AquaSentinel AI — Backend

> Predict the stress. Save the water. Protect the harvest.

FastAPI + scikit-learn backend for the architecture in the pitch deck:

```
Field sensors → ESP32 → /api/telemetry → stress model + weather → decision engine → pump/valve
        ▲                                                                              │
        └──────────── closed loop: measured litres + soil response ◄───────────────────┘
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set AQUA_API_KEY; use AQUA_WEATHER=mock to work offline
uvicorn app.main:app --reload   # docs at http://localhost:8000/docs
python scripts/simulate_device.py   # fake ESP32 (in another terminal)
python -m tests.test_core           # 12 logic tests
```
The first start trains the model (~6 s) and caches it in `models/`. Docker: `docker build -t aquasentinel . && docker run -p 8000:8000 aquasentinel`.

## Deck feature → where it lives

| Deck feature | Implementation |
|---|---|
| Predictive water stress | `app/ml/model.py` — Random Forest predicts 6 h stress risk (0–1) and water need (mm) |
| Rain-aware decision | `app/services/decision.py` — WAIT when probability-weighted rain (Open-Meteo) covers the need; volume is net of rain |
| Explainable AI | `factors[]` on every recommendation (local attribution per factor group) + plain-English `reasons[]` |
| Water Budget Mode | `app/services/budget.py` → `POST /api/budget/allocate` ranks zones by risk × weight, partial/deferred allocation |
| What-If Simulator | `POST /api/zones/{id}/simulate` — baseline / heatwave / rain scenarios + WAIT vs REDUCE vs IRRIGATE moisture projection |
| Crop Stress Fingerprint | `GET /api/zones/{id}/fingerprint` — per-field moisture stats, day/night dry-down rates, "drying faster than usual" flag |
| Closed-loop feedback | commands record planned vs actual litres; ~30 min after irrigation the soil response is measured and a per-zone volume calibration (0.7–1.5×) kicks in after 3 samples |
| Practical & measurable | `GET /api/metrics` — actual water used, prediction MAE vs realised stress, decision change rate, pump accuracy. **No savings are assumed.** |

## API

Reads are open; writes need header `X-API-Key`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/zones` · PATCH `/api/zones/{id}` | create / edit zone (crop, stage, area, lat/lon, `device_id`, `auto_mode`) |
| GET | `/api/zones`, `/api/zones/{id}`, `/api/crops` | list / detail / supported crops |
| **POST** | **`/api/telemetry`** | **ESP32 uploads a reading; response carries the next command** |
| POST | `/api/commands/{id}/complete` | ESP32 reports litres actually pumped |
| GET | `/api/dashboard` | all zones: latest reading + recommendation + summary |
| GET | `/api/zones/{id}/recommendation` | fresh WAIT/REDUCE/IRRIGATE with reasons and factors |
| GET | `/api/zones/{id}/readings?hours=24` | time series for charts |
| POST | `/api/zones/{id}/simulate` | what-if scenarios |
| POST | `/api/budget/allocate` `{available_l, apply}` | water budget mode |
| POST | `/api/zones/{id}/actuate` · `/stop` | manual pump command / emergency stop |
| GET | `/api/zones/{id}/fingerprint`, `/feedback` | baseline + closed-loop records |
| GET | `/api/metrics` · POST `/api/model/retrain` | validation + retrain from field outcomes |

### ESP32 contract

```http
POST /api/telemetry          X-API-Key: <key>
{"device_id":"esp32-north","soil_moisture":21.5,"air_temp":34.2,"humidity":35,
 "tank_pct":72,"flow_lpm":0,"pump_on":false}

200 → {"stored":true,"zone_id":"north",
       "command":{"id":7,"kind":"IRRIGATE","volume_l":1200.0,"duration_s":3600}}   // or "command": null
```
Run the pump for `duration_s` (stop early on `kind:"STOP"`), then
`POST /api/commands/7/complete {"actual_volume_l": 1135}` using the flow-meter total. Post telemetry every 5–15 min.

### Sample recommendation (trimmed)

```json
{"action":"IRRIGATE","stress_risk":0.627,"stress_risk_without_rain":0.627,"net_need_mm":28.87,
 "volume_l":1200.0,"duration_min":60.0,"pump_inhibited":false,
 "reasons":["Predicted stress risk 63% within 6 h; soil 21.5% vs threshold 23%.","Capped at 60 min of pumping per run."],
 "factors":[{"factor":"Current soil moisture","impact":0.475,"effect":"raises risk","detail":"21.5% vs stress threshold 23%"}]}
```
`factors[].impact` is the change in risk versus typical conditions for that factor group.

## Safety interlocks
Tank ≤ 5 % inhibits the pump · each run capped at 60 min · one open command per zone · auto-mode waits ≥ 3 h between irrigations · unclaimed commands expire after 30 min · stale sensor data (> 3 h) is flagged · `STOP` overrides everything. Tune in `app/config.py`.

## Be honest in your presentation — current limitations
1. **The model is bootstrapped on physics-based synthetic data.** Its ~0.98 R² is on a synthetic holdout and only shows it learned the crop-water balance, **not** that it predicts real fields. Real accuracy appears in `/api/metrics → prediction_quality.stress_risk_mae` once ≥ 6.5 h of real readings exist; `POST /api/model/retrain` then blends field outcomes in.
2. Crop thresholds, Kc values and the 300 mm root zone are textbook-style defaults; calibrate per soil/crop with your agronomist. Soil-moisture units assume volumetric % — calibrate the capacitive sensor first.
3. Not yet included: user accounts/roles (only a shared API key), MQTT transport, Postgres (SQLite is used), push/SMS alerts, satellite/leaf-temperature inputs.
4. The FastAPI layer (`app/main.py`) is thin and compiles, but I could not run it in my sandbox (no network to install FastAPI). Everything beneath it — model, decisions, budget, simulator, closed loop, DB — is covered by `tests/test_core.py`. Run `uvicorn` and open `/docs` to confirm on your machine.
