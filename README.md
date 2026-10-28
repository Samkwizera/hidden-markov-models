# Modeling Human Activity States with Hidden Markov Models

**Author:** Samuel Kwizera Ihimbazwe

A Gaussian HMM written from scratch in numpy (Baum–Welch training, Viterbi decoding) that infers
**standing / walking / jumping / still** from smartphone accelerometer and gyroscope data.

## Repository layout
| Path | Contents |
|---|---|
| `data/train/<activity>/<activity>_NN.csv` | 56 labelled training clips (14 per activity, ~8–10 s each, 100 Hz) |
| `data/test/` | 4 unseen test files: 3 continuous multi-activity sessions + 1 standing clip from a different day |
| `prepare_data.py` | builds `data/` from the raw Sensor Logger exports (merge sensors, trim, resample to 100 Hz, label) |
| `hmm.py` | `GaussianHMM` class: supervised init, scaled forward–backward, Baum–Welch, Viterbi |
| `hmm_activity_recognition.ipynb` | full pipeline: features, training, visualisations, evaluation |
| `figures/` | figures and metrics table used in the report |
| `report.pdf` | project report (5 pages) |

CSV columns: `timestamp` (ns), `seconds_elapsed`, `acc_x/y/z` (m/s², gravity removed), `gyro_x/y/z` (rad/s), `label`.

## Run
```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
jupyter notebook hmm_activity_recognition.ipynb
```

## Results (unseen data, Viterbi-decoded)
| State | Windows | Sensitivity | Specificity |
|---|---|---|---|
| standing | 35 | 0.914 | 0.922 |
| walking | 29 | 0.966 | 0.927 |
| jumping | 39 | 0.821 | 0.988 |
| still | 22 | 0.818 | 1.000 |

Overall accuracy: **0.880** (Baum–Welch model), 0.912 (supervised-initialised model).
