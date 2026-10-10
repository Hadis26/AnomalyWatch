# AnomalyWatch

**An interactive machine-learning application for detecting anomalies in multivariate sensor data.**

AnomalyWatch provides an end-to-end unsupervised anomaly detection workflow using Isolation Forest, with an interactive Streamlit dashboard for dataset exploration, anomaly visualization, and exporting labeled observations.

## Features

- Upload CSV datasets and select numerical features.
- Detect anomalous observations using Isolation Forest.
- Adjust the expected anomaly proportion.
- Handle missing numerical observations using median imputation.
- Visualize detected anomalies using interactive Plotly charts.
- Rank observations by anomaly score.
- Export results as CSV.
- Evaluate model performance against synthetic ground-truth anomalies.

## Technology Stack

- **Python** — application and data processing
- **Scikit-learn** — Isolation Forest and preprocessing
- **Pandas / NumPy** — numerical data processing
- **Streamlit** — interactive interface
- **Plotly** — interactive visualizations
- **pytest** — automated testing

## Installation

Clone the repository:

```bash
git clone https://github.com/Hadis26/AnomalyWatch.git
cd AnomalyWatch
```

### Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Alternatively, using Conda:

```bash
conda create -n anomalywatch python=3.11 -y
conda activate anomalywatch
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start

Generate a reproducible synthetic sensor dataset:

```bash
python -m scripts.generate_data
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` and upload `data/sample_sensors.csv`.

Select the temperature, vibration, and pressure features to perform multivariate anomaly detection.

## Dashboard

![AnomalyWatch dashboard](assets/dashboard.png)

The interface displays the number of observations, detected anomalies, anomaly rate, interactive visualizations, and downloadable results.

## Methodology

AnomalyWatch uses **Isolation Forest**, an unsupervised anomaly detection algorithm that isolates observations through random partitions of the feature space.

The processing pipeline is:

1. Load a CSV dataset and select numerical features.
2. Replace infinite values and impute missing values using feature medians.
3. Standardize selected features.
4. Fit an Isolation Forest model.
5. Obtain anomaly predictions and anomaly scores.
6. Visualize and export the results.

Higher anomaly scores indicate observations considered more unusual by the fitted model.

The contamination setting controls the expected fraction of training observations classified as anomalous. It does not represent a calibrated probability that an individual observation is faulty.

## Experimental Evaluation

A synthetic dataset containing 500 sensor observations was generated with 20 deliberately injected anomalies affecting temperature, vibration, and pressure.

With contamination set to 0.04, the following results were obtained:

| Metric | Result |
|---|---:|
| Total observations | 500 |
| Injected anomalies | 20 |
| Detected anomalies | 20 |
| True positives | 20 |
| False positives | 0 |
| False negatives | 0 |
| True negatives | 480 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-score | 1.00 |

These results demonstrate successful recovery of the injected anomalies in this controlled experiment. They do not establish equivalent performance on unseen industrial datasets.

Reproduce the evaluation:

```bash
python -m scripts.evaluate
```

The script generates `results/evaluation.json` and `results/labeled_sensors.csv`.

## Testing

The project includes automated unit tests covering detection of extreme outliers, data preservation, missing values, invalid contamination settings, and reproducibility.

Run:

```bash
python -m pytest tests/ -v
```

**Current result: 5 tests passed.**

## Project Structure

```text
AnomalyWatch/
├── anomalywatch/
│   ├── __init__.py
│   └── detector.py
├── scripts/
│   ├── generate_data.py
│   └── evaluate.py
├── tests/
│   └── test_detector.py
├── data/
│   └── sample_sensors.csv
├── assets/
│   └── dashboard.png
├── results/
│   ├── evaluation.json
│   └── labeled_sensors.csv
├── app.py
├── requirements.txt
└── README.md
```

## Limitations

- The current model is trained and evaluated on each uploaded dataset.
- The contamination parameter must be selected appropriately.
- Anomalies are identified by statistical unusualness, not necessarily by physical fault severity.
- Temporal dependencies are not explicitly modeled.
- The synthetic evaluation includes deliberately distinguishable anomalies.
- Real-world deployment requires domain-specific validation, suitable data splits, and monitoring of false alarms.

## References

- Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). *Isolation Forest*. IEEE International Conference on Data Mining.
- [Scikit-learn: IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

AnomalyWatch is released under the MIT License.

See the [LICENSE](LICENSE) file for the full license terms.