# Modeling Human Activity States with Hidden Markov Models

Author: Samuel Kwizera Ihimbazwe

I built a Hidden Markov Model from scratch in numpy that works out whether someone is standing, walking, jumping, or has put the phone down, using a smartphone's accelerometer and gyroscope. The full write-up is in [report.pdf](report.pdf).

I made it with my grandmother in mind. She lives in the Western Province of Rwanda and is hard to reach, so the idea is a wearable monitor that lets my family follow her movements from a distance and get help to her if something looks unusual.

## Data

I recorded everything myself with Sensor Logger on an iPhone X at 100 Hz, holding the phone at waist level: 14 clips of 5 to 10 s for each activity (standing, walking, jumping, still), so about 2 minutes per activity. For testing I kept 4 unseen files: three continuous recordings that go through all four activities in one go, and a standing clip from a different day.

Videos of the recordings: [standing](https://drive.google.com/file/d/1YgA6UkHaGOPdqqx5GI96xit061Aif5J6/view?usp=sharing), [walking](https://drive.google.com/file/d/1x27T9-VPxknB-DW4e_CIrS3akjHLhK4d/view?usp=sharing), [jumping](https://drive.google.com/file/d/1fWQBKvB6WefJN4QSezeShmdJOlVRR3uY/view?usp=sharing), [still](https://drive.google.com/file/d/1DGSEliRQ-BV2BMIkxrYEzLXstcKhLoSF/view?usp=sharing).

## How it works

The signals are cut into 2 s windows, and each window becomes 10 features (7 time-domain, 3 from the FFT), Z-scored on the training data. The HMM has one Gaussian emission per activity. I train it with Baum-Welch, which stops when the log-likelihood stops improving, and decode the test files with Viterbi.

## Results

On the unseen test data the model reaches 88 % accuracy:

| Activity | Sensitivity | Specificity |
|---|---|---|
| Standing | 0.914 | 0.922 |
| Walking | 0.966 | 0.927 |
| Jumping | 0.821 | 0.988 |
| Still | 0.818 | 1.000 |

Most errors happen right at the switch from one activity to the next.

## Files

| Path | Contents |
|---|---|
| `data/train/`, `data/test/` | cleaned and labelled CSVs |
| `prepare_data.py` | builds `data/` from the raw Sensor Logger exports |
| `hmm.py` | the HMM (Baum-Welch and Viterbi) |
| `hmm_activity_recognition.ipynb` | features, training, plots and evaluation |
| `figures/` | figures used in the report |
| `report.pdf` | the report |

## Running it

```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
jupyter notebook hmm_activity_recognition.ipynb
```
