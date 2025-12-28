"""
Nama: Dandy Arya Akbar
NIM: 1313623028
Program Studi: S1 Ilmu Komputer
Kelas: 2023 A
"""

import numpy as numpy
import pandas as pandas
import sys
import os

# Saya memastikan modul src bisa diimport sebelum mengimportnya
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from mlpnn import MultiLayerPERCEPTRONNeuralNetwork

def run_autoencoder():
    print("Saya sedang menjalankan Autoencoder Demo...")
    
    # 1. Load Data
    csv_file = "data/dataNN_1.csv"
    if not os.path.exists(csv_file):
        # Fallback path fix if running from root
        csv_file = os.path.join("..", "data", "dataNN_1.csv")
        
    print(f"Saya membaca dataset dari: {csv_file}")
    
    try:
        if "dataNN_1.csv" in csv_file:
            df = pandas.read_csv(csv_file, sep=";", decimal=",")
        else:
            df = pandas.read_csv(csv_file)
    except Exception as e:
        # Saya mencoba menyesuaikan path relatif terhadap lokasi script
        try:
            csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "dataNN_1.csv")
            df = pandas.read_csv(csv_file, sep=";", decimal=",")
        except:
            print(f"Gagal membaca file: {e}")
            return

    # Normalisasi Data (Penting untuk Sigmoid/Neural Net)
    # Saya melakukan Minimum-Maximum scaling agar data berada di range 0-1 (cocok dengan output sigmoid)
    X_raw = df.to_numpy(dtype=float)
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_normalized = (X_raw - X_min) / (X_max - X_min)
    
    # Fitur Input = Fitur Output untuk Autoencoder
    input_dim = X_normalized.shape[1]
    
    # 2. Konfigurasi Autoencoder
    # Arsitektur: Input(4) -> Hidden(3) -> Output(4) (Kompresi)
    # Atau Input(4) -> Hidden(2) -> Output(4) (Kompresi lebih kuat)
    architecture = [input_dim, 3, input_dim] 
    
    print(f"Arsitektur Autoencoder Saya: {architecture}")
    
    # Saya menggunakan learning rate yang moderat
    autoencoder = MultiLayerPERCEPTRONNeuralNetwork(
        layers=architecture,
        type_of_activation_function="sigmoid",
        seed=100,
        learning_rate=0.2
    )
    
    # 3. Training (Unsupervised: Target = Input)
    print("\nSaya memulai training autoencoder (Input diperintahkan untuk merekonstruksi dirinya sendiri)...")
    epochs = 5000
    
    # Split data? Untuk demo ini saya train full data (fit)
    autoencoder.train(X_normalized, X_normalized, epochs=epochs)
    
    # 4. Evaluasi
    reconstructed = autoencoder.predict(X_normalized)
    
    # Denormalize untuk melihat nilai asli (opsional, tapi bagus untuk verifikasi)
    reconstructed_original_scale = reconstructed * (X_max - X_min) + X_min
    
    print("\n--- Evaluasi Hasil Autoencoder ---")
    mse = autoencoder.calculate_loss(X_normalized, reconstructed)
    print(f"Final Reconstruction MSE (Normalized): {mse:.6f}")
    
    print("\nPerbandingan Sampel (Asli vs Rekonstruksi):")
    for i in range(min(5, len(X_normalized))): # Saya menampilkan 5 sampel pertama
        orig = X_raw[i]
        recon = reconstructed_original_scale[i]
        print(f"Sampel {i+1}:")
        print(f"  Asli       : {numpy.round(orig, 4)}")
        print(f"  Rekonstruksi: {numpy.round(recon, 4)}")
        print(f"  Delta (Perbedaan)      : {numpy.round(numpy.abs(orig - recon), 4)}")

if __name__ == "__main__":
    run_autoencoder()
