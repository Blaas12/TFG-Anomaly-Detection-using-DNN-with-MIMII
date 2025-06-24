import os
import h5py
import librosa
import numpy as np
from tqdm import tqdm

# Configuración
DATASET_PATH = "6_db_fan"  # Ruta de los archivos WAV
HDF5_PATH = "dataset.h5"  # Archivo de salida
SAMPLE_RATE = 22050  # Frecuencia de muestreo
N_MELS = 64  # Número de bandas Mel
DURATION = 5  # Duración en segundos (ajustar según dataset)
N_FFT = 2048  # Tamaño de ventana FFT
HOP_LENGTH = 512  # Salto de ventana

# Cargar y procesar audios
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
    return mel_spec_db.astype(np.float32)

# Crear archivo HDF5
with h5py.File(HDF5_PATH, "w") as h5f:
    # Recorrer todas las subcarpetas id_XX
    for id_folder in tqdm(os.listdir(DATASET_PATH), desc="Procesando carpetas 'id'"):
        id_folder_path = os.path.join(DATASET_PATH, id_folder)
        if os.path.isdir(id_folder_path):
            # Crear grupo en HDF5 para cada id_XX
            id_group = h5f.create_group(id_folder)
            
            # Procesar las carpetas 'normal' y 'abnormal' dentro de cada id_XX
            for label in ['normal', 'abnormal']:
                label_folder_path = os.path.join(id_folder_path, label)
                if os.path.isdir(label_folder_path):
                    # Crear subgrupo en HDF5 para cada etiqueta (normal, abnormal)
                    label_group = id_group.create_group(label)
                    
                    # Procesar los archivos WAV dentro de la carpeta correspondiente
                    mel_specs = []
                    files = [f for f in os.listdir(label_folder_path) if f.endswith(".wav")]
                    for file in files:
                        file_path = os.path.join(label_folder_path, file)
                        mel_spec = process_audio(file_path)
                        mel_specs.append(mel_spec)
                    
                    # Convertir la lista de espectrogramas a un array numpy
                    mel_specs = np.array(mel_specs)
                    
                    # Guardar el mel-espectrograma y los nombres de los archivos en HDF5
                    label_group.create_dataset("mel_spectrograms", data=mel_specs, compression="gzip")
                    label_group.create_dataset("filenames", data=np.array(files, dtype=h5py.special_dtype(vlen=str)), compression="gzip")
    
print(f"Dataset guardado en {HDF5_PATH}")

