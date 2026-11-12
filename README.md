# PredictX AI

### Reliability-Aware and Uncertainty-Driven Predictive Maintenance System

PredictX AI is a machine learning-based predictive maintenance system designed to estimate Remaining Useful Life (RUL) and assess prediction reliability under imperfect sensor conditions.

## Key Features

- Remaining Useful Life prediction
- Sensor reliability assessment
- Robust training under sensor corruption
- Uncertainty estimation using conformal prediction
- SHAP-based explainability
- Maintenance decision support
- FastAPI backend and React dashboard
- Docker-based deployment

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python, JavaScript |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | React, Vite, Recharts |
| Deployment | Docker, Docker Compose |
| Testing | Pytest |

## Dataset

NASA C-MAPSS turbofan engine degradation dataset, FD001 subset.

This is simulated data, not real industrial sensor data.

## Baseline Model Results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Random Forest | 23.52 | 31.85 | 0.765 |
| XGBoost | 23.98 | 32.16 | 0.760 |
| Linear Regression | 25.18 | 31.69 | 0.767 |

Robust training experiments show scenario-dependent trade-offs between clean-data accuracy and performance under sensor corruption.

## Project Structure

- app/ — FastAPI backend
- frontend/ — React dashboard
- src/ — ML source code
- data/ — Dataset and processed data
- models/ — Locally stored trained models
- reports/ — Evaluation and explainability reports
- notebook/ — Experiment notebooks
- tests/ — API tests

## Run Locally

### Backend

```bash
python -m uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

### Docker

```bash
docker compose up -d --build
```

Frontend: http://localhost:5173

API documentation: http://localhost:5173/docs

## Testing

```bash
python -m pytest tests/ -v
```

## Limitations

- Experiments use the FD001 subset.
- Sensor health scores are rule-based indicators, not calibrated physical failure probabilities.
- Maintenance thresholds are prototype rules.
- Uncertainty coverage depends on the calibration and evaluation protocol.
- This is a research and engineering prototype, not a certified industrial maintenance system.

## Future Improvements

- Independent calibration and evaluation
- Additional operating conditions and datasets
- Model drift monitoring
- Production-grade deployment and security

## Author

AI/ML Engineering Portfolio Project

## Application Preview

## System Architecture

```mermaid
flowchart TD
    A["NASA C-MAPSS Dataset<br/>FD001"] --> B["Data Preprocessing"]
    B --> C["Feature Engineering<br/>Rolling Mean, Std, Diff"]
    
    C --> D["RUL Prediction Model<br/>Random Forest"]
    C --> E["Sensor Reliability Analysis"]
    
    D --> F["Uncertainty Estimation<br/>Conformal Prediction"]
    D --> G["SHAP Explainability"]
    
    E --> H["Sensor Health Assessment"]
    F --> I["Maintenance Decision Support"]
    G --> I
    H --> I
    
    I --> J["FastAPI Backend"]
    J --> K["React Dashboard"]
    
    K --> L["Fleet Monitoring"]
    K --> M["Engine-Level Analysis"]
    K --> N["Maintenance Prioritization"]
```

### Fleet Command Center
![PredictX AI Overview](frontend/public/images/screenshots/overview.png)

### Fleet Intelligence
![Fleet Monitoring](frontend/public/images/screenshots/fleet.png)

### Engine-Level Analysis
![Engine Analysis](frontend/public/images/screenshots/engine-analysis.png)
 


## Robustness Evaluation

PredictX AI evaluates model performance under different sensor corruption scenarios to understand how unreliable inputs affect Remaining Useful Life (RUL) predictions.

### Experimental Scenarios

- Clean sensor data
- 10% missing sensor readings
- 20% sensor noise
- Dropout of two sensors

### Baseline vs Robust Model

Results below report mean MAE across five corruption seeds.

| Scenario | Baseline MAE | Robust MAE |
|---|---:|---:|
| Clean | 21.4164 | 21.7773 |
| Missing 10% | 21.8004 | 22.0201 |
| Noise 20% | 21.3706 | 21.2029 |
| Two-sensor dropout | 23.2965 | 22.0181 |

### Key Observations

- Robust training improved average MAE under two-sensor dropout by approximately 5.49%.
- It also improved performance under 20% sensor noise.
- The robust model showed slightly worse performance on clean data.
- Missing-data performance was also slightly worse.

These results indicate a trade-off between clean-data accuracy and robustness under certain sensor corruption scenarios.

### Evaluation Limitations

- Results are based on five corruption seeds.
- Experiments use the NASA C-MAPSS FD001 subset.
- Results should not be interpreted as proof of generalization to real industrial machinery.
- Further evaluation on independent datasets and operating conditions is required.

## Uncertainty Estimation

PredictX AI uses conformal prediction to estimate prediction intervals around Remaining Useful Life (RUL) predictions.

Instead of presenting only a point estimate, the system provides:

- Predicted Remaining Useful Life
- Lower prediction bound
- Upper prediction bound
- Prediction interval width
- Empirical coverage evaluation

### Exploratory Evaluation

| Metric | Result |
|---|---:|
| Target coverage | 90% |
| Observed coverage | 99% (99/100 engines) |
| Calibration quantile | 103.7137 cycles |
| Mean interval width | 174.79 cycles |
| MAE | 21.78 cycles |
| RMSE | 30.34 cycles |

### Interpretation

The prediction interval communicates a range of plausible RUL values rather than presenting the point prediction as perfectly certain.

Wider intervals indicate greater uncertainty, while narrower intervals indicate more concentrated predictions.

### Important Statistical Limitation

These results are exploratory. The calibration and model-selection process was not fully independent, so the observed 99% coverage must not be interpreted as a formal 90% conformal coverage guarantee.

A dedicated independent calibration and evaluation split is required to establish a defensible coverage guarantee under the appropriate assumptions.

## Explainability and Sensor Reliability

### SHAP-Based Explainability

PredictX AI uses SHAP (SHapley Additive exPlanations) to understand the contribution of input features to model predictions.

Global feature importance identifies features with the highest average absolute SHAP values.

Top contributing features in the saved analysis:

| Feature | Mean Absolute SHAP |
|---|---:|
| cycle | 42.62 |
| sensor_11_rolling_mean | 8.33 |
| sensor_4_rolling_mean | 5.50 |

The project also generates:
- SHAP summary plots
- SHAP feature importance plots
- SHAP waterfall visualizations
- Local feature contribution reports

Note: The saved local explanation is a sample explanation and is not linked to a specific engine ID.

### Sensor Reliability Assessment

The system evaluates sensor data quality using rule-based indicators:

- Missing reading rate
- Stuck reading rate
- Outlier rate
- Noise ratio
- Overall sensor health score

The analysis covers 21 sensors across 100 test engines.

Sensor health status is used as an additional signal in maintenance decision support.

### Limitations

- Sensor health scores are rule-based indicators, not calibrated probabilities of physical sensor failure.
- SHAP describes model feature contributions; it does not establish causality.
- Feature importance can vary across datasets and operating conditions.


## Maintenance Decision Support

PredictX AI combines RUL predictions, uncertainty estimates and sensor health indicators to generate prototype maintenance priorities.

### Decision Inputs

- Predicted Remaining Useful Life
- Prediction interval width
- Average sensor health score
- Number of unhealthy sensors
- Sensor health status

### Prototype Priority Rules

| Priority | Conditions |
|---|---|
| High | Predicted RUL <= 30 OR average sensor health < 65 |
| Medium | Predicted RUL <= 60 OR average sensor health < 85 OR interval width > 150 |
| Low | None of the above conditions |

### Recommended Actions

The system associates maintenance priorities with suggested actions, such as scheduling inspections and monitoring RUL trends.

### Evaluation Snapshot

The current evaluation report contains 100 engine records.

| Maintenance Priority | Engines |
|---|---:|
| High | 19 |
| Medium | 81 |
| Low | 0 |

### Prediction Trust

Prediction trust is presented alongside maintenance priority to communicate when a prediction should be interpreted cautiously.

### Important Limitations

- These thresholds are prototype rules, not validated industrial maintenance policies.
- Priority levels are decision-support indicators, not automatic maintenance orders.
- True RUL is used for offline evaluation only; it is not available as an operational input when predicting an engine's future RUL.
