import os
import h5py
import librosa
import numpy as np
from tqdm import tqdm
import argparse

# Default parameters
SAMPLE_RATE = 22050
N_MELS = 64
SEGMENT_DURATION = 2.21  # duration per segment in seconds
SEGMENT_HOP = 1.0  # seconds between segment starts
N_FFT = 2048
HOP_LENGTH = 512
LABELS = ['normal', 'abnormal']


def segment_audio(file_path, sample_rate, segment_duration, hop_duration):
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)

    segments = []
    for start in range(0, len(y) - segment_samples + 1, hop_samples):
        segment = y[start:start + segment_samples]
        segments.append(segment)
    return segments


def extract_mel_spec(audio_segment, sample_rate, n_fft, hop_length, n_mels):
    mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=sample_rate, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
    return mel_spec_db.astype(np.float32)


def create_hdf5_for_machine(machine_path, output_path, sample_rate, segment_duration, hop_duration, n_fft, hop_length, n_mels):
    with h5py.File(output_path, "w") as h5f:
        for id_folder in tqdm(os.listdir(machine_path), desc=f"Processing {machine_path}"):
            id_folder_path = os.path.join(machine_path, id_folder)
            if os.path.isdir(id_folder_path):
                id_group = h5f.create_group(id_folder)

                for label in LABELS:
                    label_folder_path = os.path.join(id_folder_path, label)
                    if os.path.isdir(label_folder_path):
                        label_group = id_group.create_group(label)

                        mel_specs = []
                        filenames = []
                        files = [f for f in os.listdir(label_folder_path) if f.endswith(".wav")]
                        for file in files:
                            file_path = os.path.join(label_folder_path, file)
                            segments = segment_audio(file_path, sample_rate, segment_duration, hop_duration)
                            for i, segment in enumerate(segments):
                                mel_spec = extract_mel_spec(segment, sample_rate, n_fft, hop_length, n_mels)
                                mel_specs.append(mel_spec)
                                relative_path = os.path.join(id_folder, label, file)
                                filenames.append(f"{relative_path}_seg{i}")

                        mel_specs = np.array(mel_specs)
                        label_group.create_dataset("mel_spectrograms", data=mel_specs, compression="gzip")
                        label_group.create_dataset("filenames", data=np.array(filenames, dtype=h5py.string_dtype()), compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and store Mel spectrograms from segmented MIMII audio.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to MIMII dataset")
    parser.add_argument("--machines", nargs='+', default=["slider", "valve"], help="List of machine types")
    parser.add_argument("--noise_levels", nargs='+', default=["-6_dB", "0_dB", "6_dB"], help="Noise level folders (e.g., 6dB, 0dB)")
    parser.add_argument("--output_dir", type=str, default="hdf5_datasets", help="Output directory for HDF5 files")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--n_mels", type=int, default=N_MELS)
    parser.add_argument("--segment_duration", type=float, default=SEGMENT_DURATION)
    parser.add_argument("--segment_hop", type=float, default=SEGMENT_HOP)
    parser.add_argument("--n_fft", type=int, default=N_FFT)
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for machine in args.machines:
        for noise in args.noise_levels:
            machine_path = os.path.join(args.base_path, f"{noise}_{machine}")
            output_path = os.path.join(args.output_dir, f"{noise}_{machine}.h5")
            if os.path.exists(machine_path):
                create_hdf5_for_machine(machine_path, output_path, args.sample_rate, args.segment_duration,
                                        args.segment_hop, args.n_fft, args.hop_length, args.n_mels)
            else:
                print(f"[Warning] Path does not exist: {machine_path}")
