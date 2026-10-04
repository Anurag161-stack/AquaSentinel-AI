# 🌊 AquaSentinel AI

### Predict the Stress. Save the Water. Protect the Harvest.

AquaSentinel AI is a predictive irrigation prototype designed to move irrigation from **reactive watering** to **predictive water intelligence**.

Instead of asking only **"Is the soil dry right now?"**, AquaSentinel asks **"Is this crop likely to experience water stress soon?"**

## 🎯 Problem

Climate change is making irrigation more difficult through irregular rainfall, heatwaves and water scarcity. Traditional irrigation often reacts to current soil dryness instead of anticipating upcoming crop stress.

AquaSentinel provides a predictive decision layer that combines field conditions, crop context and weather information to recommend:

**WAIT → REDUCE → IRRIGATE**

## ⚙️ How It Works

```text
FIELD CONDITIONS
      ↓
    SENSE
      ↓
   TRANSMIT
      ↓
AI STRESS PREDICTION
      ↓
   WATER RISK
      ↓
WAIT / REDUCE / IRRIGATE
      ↓
 SMART VALVE / PUMP
      ↓
 FIELD RESPONSE
      ↓
    LEARN
      ↺
```

### 1. Sense

The prototype works with soil moisture, temperature, humidity, weather/rainfall, crop type, crop stage and water availability.

### 2. Transmit

Telemetry is sent to the local **FastAPI backend**.

### 3. Predict

The AI engine estimates upcoming crop water-stress conditions rather than reacting only to current dryness.

### 4. Decide

| Decision | Meaning |
|---|---|
| 🟢 **WAIT** | Conditions are sufficient or expected rain can cover the requirement |
| 🟡 **REDUCE** | Some stress is expected, but full irrigation is unnecessary |
| 🔴 **IRRIGATE** | Significant water stress is expected and irrigation is required |

### 5. Act

The prototype represents the recommendation through smart-valve behavior and planned water application.

### 6. Learn

The architecture supports feedback between predicted conditions, water use and field response.

## 🧠 X-Factors

- **Predictive Water Stress** — forecast crop stress instead of only detecting dryness.
- **Rain-Aware Irrigation** — wait when useful rainfall is expected.
- **Water Budget** — prioritize plots when available water is limited.
- **Explainable AI** — show factors behind the recommendation.
- **Closed-Loop Feedback** — compare predicted conditions with actual response.

## 🏗️ Architecture

```text
Field Conditions
       ↓
   IoT / ESP32
       ↓
 FastAPI Backend
       ↓
 AI Prediction
       ↓
 Decision Engine
       ↓
WAIT / REDUCE / IRRIGATE
       ↓
 Valve / Pump
       ↓
 Field Response
       ↺
   Feedback
```

## 🖥️ Working Prototype

The current prototype includes:

- Interactive AquaSentinel dashboard
- Four demonstration plots
- Simulated field telemetry
- AI stress prediction
- WAIT / REDUCE / IRRIGATE recommendations
- Water-budget visualization
- Weather context
- Explainable decision factors
- Valve simulation
- FastAPI backend
- Local API endpoints
- Device simulation support
- Trained model artifact

### Current software flow

```text
Browser Simulation → FastAPI Backend → AI / Decision Engine
→ Recommendation → Dashboard → Valve Simulation
```

## 🚀 Run Locally

### 1. Create the environment

```powershell
cd aquasentinel-backend
py -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Start the backend

```powershell
$env:AQUA_API_KEY="dev-key-change-me"; $env:AQUA_WEATHER="mock"; $env:AQUA_CORS="http://127.0.0.1:5500,http://localhost:5500"; .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Open the dashboard

**http://127.0.0.1:8000/app**

Health check: **http://127.0.0.1:8000/health**  
API docs: **http://127.0.0.1:8000/docs**

## 📁 Project Structure

```text
AquaSentinel-AI/
├── aquasentinel-backend/
│   ├── app/
│   │   ├── ml/
│   │   ├── services/
│   │   ├── static/
│   │   │   └── aquasentinel.html
│   │   ├── config.py
│   │   ├── db.py
│   │   └── main.py
│   ├── models/
│   │   └── aquasentinel.joblib
│   ├── scripts/
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
├── START_HERE.txt
└── .gitignore
```

## ⚠️ Prototype Status

This is a **working local prototype / demonstration**. Browser sensor values are simulated for the demo and the current weather configuration uses mock data. The prototype demonstrates the software decision pipeline but does **not** claim measured real-world water savings or field-validated accuracy.

A real deployment can connect the architecture to ESP32 hardware, soil-moisture sensors, temperature/humidity sensors, flow sensors, tank-level monitoring, pumps/valves and real weather services.

## 🔭 Future Scope

- Physical ESP32 sensor integration
- Real-time weather API integration
- Real field telemetry
- Satellite and remote-sensing data
- Solar-powered sensor nodes
- Multilingual farmer alerts
- Multi-farm deployment
- Continuous model validation

## 🌍 Vision

Transform irrigation from **reactive watering** into **predictive water intelligence**.

> **We don't just detect dry soil. We predict crop stress before it becomes visible.**

## 🏆 Hackathon

**AquaSentinel AI**  
**Theme:** AI for Climate Change  
**Core concept:** Sense → Predict → Decide → Act → Learn  
**Team:** Team Nirman  
**Event:** PCCOE International Grand Challenge 2026

## 📌 Disclaimer

This project is a prototype for demonstration and experimentation. Sensor readings and weather information used in the current demo may be simulated. Real agricultural deployment requires hardware integration, field testing, model validation and appropriate safety controls.
