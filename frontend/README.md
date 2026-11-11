# PredictX AI — React Frontend

Premium dark industrial command-center interface built with React, Vite, Recharts and Lucide icons.

## Run locally

1. Extract this folder into your PredictX project root (or copy its files into a new `frontend/` folder).
2. Open a terminal in this folder.
3. Install Node.js LTS if it is not already installed.
4. Run:

```bash
npm install
npm run dev
```

5. Open the URL Vite prints, usually `http://localhost:5173`.

## Build for production

```bash
npm run build
npm run preview
```

## Data integration

The UI tries `GET http://localhost:8000/api/dashboard`.

Expected JSON shape:

```json
{
  "summary": {
    "total_engines": 100,
    "high_priority": 19,
    "medium_priority": 81,
    "low_priority": 0,
    "average_predicted_rul": 85.8,
    "average_sensor_health": 82.8,
    "mean_interval_width": 174.8,
    "test_coverage": 99,
    "mae": 21.78,
    "rmse": 30.34
  },
  "engines": [
    {
      "unit_id": 1,
      "predicted_rul": 201.4,
      "lower_rul": 97.7,
      "upper_rul": 305.1,
      "interval_width": 207.4,
      "average_sensor_health": 81.9,
      "unhealthy_sensor_count": 9,
      "maintenance_priority": "High",
      "sensor_health_status": "Warning",
      "prediction_trust": "Use with caution",
      "recommended_action": "Prioritise inspection and verify sensor readings"
    }
  ]
}
```

If the API is unavailable, the UI uses clearly labeled demo-preview data. Do not present preview data as live model output.

Set a different API base with `.env`:

```env
VITE_API_BASE=http://localhost:8000
```

## Current scope

- Responsive command center, fleet table and engine inspection.
- Charts and KPI cards.
- CSV export.
- API-ready data loading with demo fallback.
- No model inference is performed in the browser.
