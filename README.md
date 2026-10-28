# Modeling Human Activity States with Hidden Markov Models

Author: Samuel Kwizera Ihimbazwe

I built a Hidden Markov Model from scratch in numpy that works out what a person is doing (standing, walking, jumping, or the phone lying still) from the accelerometer and gyroscope on a smartphone. The full write-up is in [report.pdf](report.pdf).

## Why I built this

My grandmother lives in the Western Province of Rwanda, far from me, and she is hard to reach. My family often goes days without knowing how she is or whether she is moving around normally. The idea is a wearable monitor that runs on a phone at her waist, so we can follow her movements from a distance and get help to her if something looks unusual, like a sudden hard movement or a long time without any movement. In this project jumping is my stand-in for a sudden high-impact movement such as a stumble, and still means the phone has been put down and we can no longer see how she is.

## Data collection

I recorded everything myself with the Sensor Logger app (v1.68) on an iPhone X (iOS 16.7), holding the phone at waist level. The app logged the accelerometer (gravity removed, m/s²) and the gyroscope (rad/s) at 100 Hz.

| Activity | Training files | Total duration |
|---|---|---|
| Standing | 14 | 136.9 s |
| Walking | 14 | 115.9 s |
| Jumping | 14 | 116.7 s |
| Still (phone on a table) | 14 | 116.3 s |

Each training clip is 5 to 10 s of a single activity, recorded on 26 Oct 2026. Videos of me recording each activity:

| Activity | Video |
|---|---|
| Standing | [Google Drive](https://drive.google.com/file/d/1YgA6UkHaGOPdqqx5GI96xit061Aif5J6/view?usp=sharing) |
| Walking | [Google Drive](https://drive.google.com/file/d/1x27T9-VPxknB-DW4e_CIrS3akjHLhK4d/view?usp=sharing) |
| Jumping | [Google Drive](https://drive.google.com/file/d/1fWQBKvB6WefJN4QSezeShmdJOlVRR3uY/view?usp=sharing) |
| Still | [Google Drive](https://drive.google.com/file/d/1DGSEliRQ-BV2BMIkxrYEzLXstcKhLoSF/view?usp=sharing) |

I kept four more files aside as unseen test data. Three are continuous recordings of about 40 s, made after the training session, where I went standing → walking → jumping → still in one go. The fourth is a standing recording from 23 Oct 2026, three days before the training session.

## Repository layout

| Path | Contents |
|---|---|
| `data/train/<activity>/<activity>_NN.csv` | the 56 labelled training clips |
| `data/test/` | the 4 unseen test files |
| `prepare_data.py` | turns the raw Sensor Logger exports into the CSVs in `data/` |
| `hmm.py` | the `GaussianHMM` class (Baum-Welch training, Viterbi decoding) |
| `hmm_activity_recognition.ipynb` | the full pipeline: features, training, plots and evaluation |
| `figures/` | every figure in the report, plus the metrics table |
| `report.pdf` | the 5-page report |

Every CSV has the columns `timestamp` (ns), `seconds_elapsed`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z` and `label`.

## How it works

### Cleaning

`prepare_data.py` merges the accelerometer and gyroscope files on their shared timestamp and cuts 0.5 s off each end, where I tapped the screen to start and stop logging. iOS timestamps drift a little around 10 ms, so it also interpolates every recording onto an exact 100 Hz grid. That way every window has the same number of samples, and a phone recording at another rate would line up with mine.

### Windowing and features

I use 2 s windows (200 samples) with a 1 s hop. Walking and jumping both repeat at about 2 Hz, so each window holds 3 to 4 full cycles, and the FFT resolution is 0.5 Hz. From each window I compute 10 features on the magnitude of acceleration and rotation, so the phone's orientation doesn't matter:

| Feature | Domain |
|---|---|
| acceleration mean, standard deviation, RMS and signal magnitude area | time |
| gyroscope mean and standard deviation | time |
| correlation between the x and z acceleration axes | time |
| dominant frequency of acceleration and of rotation | frequency (FFT) |
| spectral energy of acceleration | frequency (FFT) |

The intensity features are log-scaled, then every feature is Z-scored using the training windows only.

### The model

The HMM has four hidden states (standing, walking, jumping, still) and a diagonal Gaussian emission for each. I start the parameters from the labelled training windows, then refine them with Baum-Welch, which stops when the average log-likelihood changes by less than 1e-4 (it took 5 iterations). The training clips each contain one activity, so I chained them in random order into 6 sequences to give the model transitions to learn. Viterbi then decodes the most likely activity sequence for each test file.

![Transition matrix and emission means](figures/fig4_transition_emission.png)

## Results on unseen data

| State | Windows | Sensitivity | Specificity |
|---|---|---|---|
| Standing | 35 | 0.914 | 0.922 |
| Walking | 29 | 0.966 | 0.927 |
| Jumping | 39 | 0.821 | 0.988 |
| Still | 22 | 0.818 | 1.000 |

Overall accuracy is 0.880 with the Baum-Welch model and 0.912 with the parameters taken straight from the labels. Per test file it was 0.892, 0.857 and 0.872 on the three continuous sessions, and 1.000 on the standing file from the other day.

![Decoded test sessions](figures/fig7_decoded_sequences.png)

Almost every error is at an activity boundary. At the end of each jumping bout the model goes jumping → walking → standing → still, which is what I actually did: I slowed down, walked to the table and put the phone down, while my labels called the whole stretch jumping.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
jupyter notebook hmm_activity_recognition.ipynb
```

The notebook reads the cleaned CSVs in `data/`, so it runs as is. `prepare_data.py` needs the raw Sensor Logger exports in `raw/`, which I left out of the repo because the cleaned data is already here.

## Limitations

All the data comes from me, on one phone, in one place. Because the transition matrix was learned from clips chained in random order, it doesn't yet reflect how a real day goes. Before this could watch over my grandmother, it would need recordings from her and other people her age, long natural recordings so the transitions are realistic, and labels synced to video at the activity boundaries.
