# Tutorial for QuanEstimation

An educational web application demonstrating quantum parameter estimation techniques, built on top of the [QuanEstimation](https://github.com/QuanEstimation/QuanEstimation) library.

## Project Structure

```
.
├── src/
│   ├── main.py                  # FastAPI backend (API endpoints)
│   ├── quantum_estimation.py    # Core simulation and visualization routines
│   ├── index.html               # Frontend UI with interactive parameter controls
│   └── doc/                     # HTML documents explaining each module's principles
│       ├── 1.html               # Single Qubit Parameter Estimation
│       ├── 2.html               # Two Qubit Parameter Estimation
│       ├── 3.html               # Bayesian Quantum Parameter Estimation
│       ├── 4.html               # Bayesian vs. Maximum Likelihood Estimation
│       ├── 5.html               # Measurement Optimization
│       └── 6.html               # State Optimization of LMG Model
└── turorial/
    └── QuanEstimation_turorial_git.pdf   # Reference tutorial PDF
```

## Dependencies

- Python 3.8+
- [QuanEstimation](https://github.com/QuanEstimation/QuanEstimation)
- [QuTiP](https://qutip.org/)
- FastAPI + Uvicorn
- NumPy, SciPy, Matplotlib

## Getting Started

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

## Modules

The web interface provides 6 interactive tabs:

| Tab | Topic | Description |
|-----|-------|-------------|
| 1 | Single Qubit Parameter Estimation | CFI/QFI evolution under Lindblad dissipation |
| 2 | Two Qubit Parameter Estimation | CFIM, QFIM, HCRB, and NHB bounds for two-qubit systems |
| 3 | Bayesian Quantum Parameter Estimation | Prior distribution and Bloch vector Z-component analysis |
| 4 | Bayesian vs. MLE | Posterior distribution (MAP) and likelihood function (MLE) comparison |
| 5 | Measurement Optimization | Projection, LC input, and rotation input measurement optimization |
| 6 | State Optimization | QFI convergence and optimal state distribution for the LMG model |

## API Endpoints

| Endpoint | Module |
|----------|--------|
| `/createImg` | Tab 1: Single Qubit |
| `/createImgQuantum` | Tab 2: Two Qubit |
| `/createImgBayesian` | Tab 3: Bayesian |
| `/createImgQuantumParameter` | Tab 4: MAP vs MLE |
| `/createImgProjection` | Tab 5-1: Projection Measurement |
| `/createImgLCInput` | Tab 5-2: LC Input Measurement |
| `/createImgRotationInput` | Tab 5-3: Rotation Input Measurement |
| `/createImgStateOpt` | Tab 6: State Optimization |

## References

- [QuanEstimation Documentation](https://quanestimation.github.io/QuanEstimation/)
- [QuTiP Documentation](https://qutip.org/docs/latest/)
