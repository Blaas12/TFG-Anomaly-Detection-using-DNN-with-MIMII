# Audio Anomaly Detection

This project implements a system for detecting anomalies in industrial sounds using deep learning models such as autoencoders and U-Nets.

## Dataset

The data used in this project comes from the [MIMII Dataset](https://paperswithcode.com/paper/mimii-dataset-sound-dataset-for), a sound dataset for malfunctioning industrial machine investigation.

> **Note:** Due to the dataset's large size (~120 GB), the raw audio files are not included in this repository.  
To run the experiments, download the dataset from the official source and place the `.wav` files under the `raw/` folder:

the structure goes as follows:
project_root/

raw/

fan/

pump/

...


## Setup

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```


## Preprocessing
Before training, convert the raw .wav files into .h5 format features using the following script:

```bash
python utils/rawToProcessed.py --base_path data/raw --machines fan pump --noise_levels 6_dB --output_dir data/processed
```
This will generate a folder data/processed/ with the extracted features (e.g., spectrograms).
