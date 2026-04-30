# Barjeel AI — Precision Cooling Intelligence

**Ancient wisdom · Modern intelligence**

DEWA CleanTech Hackathon 2026 — Track 3: AI & Digitalisation

**Team:** Sameeha Siddiqui & Shahed Marashdeh | **Location:** United Arab Emirates

---

## The Problem

Air conditioning consumes up to 70% of summer electricity in UAE buildings. Most cooling systems run on fixed schedules – blasting cold air whether a building is full or empty, day or night. This wastes enormous amounts of energy, burns fossil fuels, and increases CO₂ emissions.

Existing AI cooling solutions exist only for data centers and district cooling plants, not for the thousands of individual malls, hotels, hospitals, and office towers across Dubai and Abu Dhabi.

## The Solution

Barjeel AI is an artificial intelligence system that predicts a building's cooling demand 24 hours in advance using only two data inputs: weather forecast and building occupancy patterns. The AI (Isolation Forest + Gradient Boosting) then automatically optimizes zone-level cooling dispatch.

**Inspired by the Barjeel** — the traditional UAE wind tower that cooled homes for 1,000 years without electricity. We rebuilt that same adaptive intelligence, now powered by AI.

## Key Metrics

| Metric | Value |
|--------|-------|
| Average energy reduction | 28% |
| Reduction during low-occupancy hours | Up to 48% |
| Model accuracy | R² = 0.97 |
| Prediction error | MAPE = 3-4% |
| Payback period | ~20 months |

## Live Demo

[Launch Barjeel AI Dashboard](https://barjeelai-zkevqpx7vlz8dtwbsxflyc.streamlit.app/)

## Video Demo

[Watch Demo Video](https://youtu.be/_ttkc3l5ADQ)

## Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Scikit-learn (Gradient Boosting, Isolation Forest)
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy

## Features

- ✅ 24-hour cooling demand prediction
- ✅ 9-zone precision dispatch (3 floors × 3 zones)
- ✅ Digital Twin engine with scenario simulation
- ✅ 6 adaptive control modes (Standby, Extreme Heat, Peak Demand, Dehumidification, Demand Response, Night Setback)
- ✅ Real-time CO₂ tracking
- ✅ ROI calculator

## How It Works

1. **Input:** Weather forecast + building occupancy data
2. **AI Model:** Predicts cooling load for each zone
3. **Decision Engine:** Selects optimal control mode
4. **Output:** Zone-level airflow dispatch
5. **Result:** 28% less energy wasted

## Run Locally

```bash
git clone https://github.com/Shahd698/BarjeelAI.git
cd BarjeelAI
pip install -r requirements.txt
streamlit run app.py
