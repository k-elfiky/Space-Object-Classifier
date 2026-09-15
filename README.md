<div align="center">

# Space Object Classification

**Distinguishing galaxies, stars, and quasars from SDSS photometry and redshift**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-blueviolet?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.2%25-brightgreen?style=flat-square)

**Overview** • [Dataset](#dataset) • [Approach](#approach) • [Getting Started](#getting-started) • [Results](#results)

</div>

A complete machine learning pipeline that classifies celestial objects into **GALAXY**, **STAR**, and **QSO** (quasi-stellar object) using 100,000 records from the Sloan Digital Sky Survey (SDSS) **Data Release 18**. The model reaches **99.2% test accuracy** end-to-end with an XGBoost classifier tuned via `GridSearchCV`.

The full workflow—from data loading through hyperparameter tuning to evaluation—runs in a single Jupyter notebook (`main.ipynb`), making it easy to explore, reproduce, and extend.

## How It Works

Astronomical surveys observe objects in multiple photometric bands. The SDSS `ugriz` system measures brightness in five filters (ultraviolet `u`, green `g`, red `r`, near-infrared `i` and `z`). Galaxies, stars, and quasars leave characteristic signatures in these colors:

- **Stars** are point sources in our own Galaxy with a well-defined stellar locus in color space.
- **Galaxies** are extended objects at cosmological distances whose emission is redshifted.
- **Quasars (QSOs)** are extremely distant, luminous active galactic nuclei with distinctive colors.

The classifier learns these signatures directly from the photometry and redshift, no imaging required.

## Features

- **Mean-shifted color features** — converts raw photometric magnitudes into distance-independent color indices (`u-g`, `g-r`, `r-i`, `i-z`)
- **Robust preprocessing** — missing-value and duplicate checks, label encoding, standardization
- **Automated hyperparameter tuning** — 3-fold `GridSearchCV` over tree depth, learning rate, and number of estimators
- **Full evaluation** — train/test accuracy, 5-fold cross-validation, per-class classification report, confusion matrix

## Dataset

| Property | Value |
| --- | --- |
| Survey | Sloan Digital Sky Survey, Data Release 18 |
| Source file | `SDSS_DR18.csv` |
| Records | 100,000 rows / 43 columns |
| Missing values | 0 |
| Duplicate rows | 0 |
| Used as target | `class` |

### Class distribution

The dataset is **imbalanced**; XGBoost handles this well out of the box:

| Class | Count | Share |
| --- | ---: | ---: |
| `GALAXY` | 52,343 | 52.3% |
| `STAR` | 37,232 | 37.2% |
| `QSO` | 10,425 | 10.4% |

### SDSS magnitude notation

Magnitudes are on the [astronomical magnitude scale](https://en.wikipedia.org/wiki/Apparent_magnitude) (lower = brighter). A `-9999` placeholder signals a non-detection in some bands.

## Approach

The notebook pipelines the following steps:

1. **Select features** — keep the five photometric magnitudes (`u`, `g`, `r`, `i`, `z`) and `redshift`.
2. **Explore** — summary statistics, null/duplicate checks, class distribution (`main.ipynb`).
3. **Encode labels** — map `GALAXY → 0`, `QSO → 1`, `STAR → 2`.
4. **Engineer colors** — compute color indices `u-g`, `g-r`, `r-i`, `i-z`. Color differences cancel out distance and extinction, so the model learns shape/spectral type, not brightness.
5. **Split** — 80/20 stratified train/test split (`random_state=42`).
6. **Scale** — `StandardScaler` to normalize the feature ranges.
7. **Tune** — XGBoost + `GridSearchCV` (`cv=3`, scoring `accuracy`) over 27 hyperparameter combinations.
8. **Evaluate** — hold-out accuracy, 5-fold CV, per-class report, confusion matrix.

### Best hyperparameters (from `GridSearchCV`)

```python
best_params_ = {
    "learning_rate": 0.2,
    "max_depth": 5,
    "n_estimators": 200,
}
```

## Repository Structure

```text
Space Object Recognetion/
├── main.ipynb          # End-to-end classification workflow (Jupyter)
├── SDSS_DR18.csv       # Dataset: 100,000 SDSS DR18 records
├── README.md           # This documentation
└── venv/               # Local Python virtual environment
```

## Getting Started

### Prerequisites

- Python 3.10+ (notebook developed on 3.14)
- Jupyter (Notebook or Lab)

### Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

python -m pip install -r requirements.txt  #installing the dependencies from requirements.txt file
```

> [!NOTE]
> If you already have the bundled `venv/`, just activate it and install any missing packages:
> `pip install -r requirements.txt` (if you generate one) or the individual packages above.

### Run

```bash
jupyter notebook main.ipynb
```

Execute all cells top to bottom. Runtime is roughly 1–2 minutes for grid search on a typical laptop (`n_jobs=-1` uses all CPU cores).

> [!TIP]
> The tuning step runs 27 models × 3 folds. Drop the `[100, 200, 300]`/`[3, 5, 7]` grids to fewer values for a faster first pass.

## Results

| Metric | Value |
| --- | ---: |
| Best cross-validation score (tuning) | **0.9922** |
| 5-fold cross-validation mean | **0.9925** |
| Hold-out test accuracy | **0.9921** |
| Train accuracy | 0.9981 |
| Test samples | 20,000 |

### Per-class classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `GALAXY` | 0.99 | 0.99 | 0.99 | 10,373 |
| `QSO` | 0.99 | 0.96 | 0.98 | 2,115 |
| `STAR` | 0.99 | 1.00 | 1.00 | 7,512 |

The weakest class is **QSO**, the smallest class with the highest redshift range—its 96% recall reflects the intrinsic overlap between high-redshift stars and quasars in photometric color space.

> [!IMPORTANT]
> Metrics were computed on data from a single sky region and might not transfer perfectly to other SDSS footprints. For a production system, add cross-region validation.

## Possible Next Steps

- Add extra SDSS features (`petroFlux*`, `psfMag*`, `expAB*`) and compare against the color-only baseline.
- Address class imbalance with `scale_pos_weight` or resampling to lift QSO recall.
- Compare against alternatives (Random Forest, LightGBM, k-NN) on the same folds.
- Package the trained model for inference (e.g. `joblib`/`xgboost` save + a small prediction API).

## References

- SDSS Data Release 18: <https://www.sdss.org/dr18/>
- XGBoost documentation: <https://xgboost.readthedocs.io/>
