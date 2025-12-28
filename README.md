# 🧠 Multi-Layer Perceptron Neural Network (MLPNN): Backpropagation & Autoencoder

## 📋 Identitas Saya

- **Nama:** Dandy Arya Akbar
- **NIM:** 1313623028
- **Jurusan:** S1 Ilmu Komputer, Universitas Negeri Jakarta (UNJ)
- **Kelas:** 2023 A
- **Mata Kuliah:** Deep Learning

---

## 📖 Deskripsi Project

Project ini adalah implementasi lanjutan dari **Multi-Layer Perceptron (MLP)** yang dibangun dari nol (from scratch) menggunakan **Python** dan **NumPy**. project ini memperluas fungsi dasar _feed-forward_ dengan menambahkan algoritma **Backpropagation** untuk pelatihan (training) neural network , serta demonstrasi aplikasi **Autoencoder**.

Fitur Utama:

1.  **Backpropagation Training**: Implementasi manual algoritma pelatihan menggunakan _Stochastic Gradient Descent (SGD)_ dan _Chain Rule_.
2.  **Autoencoder**: Arsitektur neural network yang belajar merekonstruksi inputnya sendiri (kompresi data).
3.  **Fleksibilitas**: Mendukung konfigurasi layer dinamis, fungsi aktivasi (Sigmoid/ReLU), dan pengaturan _learning rate_.

---

## 📂 Struktur Folder & File Utama

- **`src/mlpnn.py`**: Inti dari project. Berisi kelas `MultiLayerPERCEPTRONNeuralNetwork` dengan metode:
  - `feed_forward_pass(x)`: Menghitung output layer per layer.
  - `backpropagate_single_sample(x, y)`: Menghitung gradien dan memperbarui bobot (inti Backpropagation).
  - `train(X, y)`: Melatih neural network selama sejumlah _epochs_.
  - `predict(X)`: Melakukan prediksi setelah pelatihan.
  - `sigmoid` & `sigmoid_derivative`: Fungsi aktivasi dan turunannya.
- **`src/autoencoder_runner.py`**: Script driver untuk mendemonstrasikan Autoencoder. Melatih MLP untuk merekonstruksi data input (`dataNN_1.csv`).
- **`tests/test_xor.py`**: Script verifikasi menggunakan masalah logika XOR (bukti pembelajaran non-linear).
- **`tests/test_mlpnn.py`**: Unit tests untuk memastikan integritas struktur dasar.

---

## 🚀 Panduan Penggunaan (Step-by-Step)

Berikut adalah langkah-langkah untuk menjalankan dan memverifikasi pekerjaan saya:

### 1. Menjalankan Demonstrasi Autoencoder

Script ini akan memuat dataset, menormalisasi data, melatih Autoencoder untuk mengompresi data, dan menampilkan hasil rekonstruksi.

**Perintah:**

```bash
python src/autoencoder_runner.py
```

**Apa yang diharapkan:**

- Program akan menampilkan arsitektur `[4, 3, 4]` (Input 4 -> Hidden 3 -> Output 4).
- Proses training berjalan selama 5000 epoch.
- Pada akhirnya, program mencetak **MSE (Mean Squared Error)** yang sangat kecil (misal: ~0.006) dan membandingkan sampel asli dengan hasil rekonstruksi.

### 2. Memverifikasi Algoritma Backpropagation (XOR Test)

Masalah XOR adalah standar emas untuk menguji apakah non-linearitas dan backpropagation berfungsi.

**Perintah:**

```bash
python tests/test_xor.py
```

**Apa yang diharapkan:**

- Program melatih neural network kecil pada input logika `[[0,0], [0,1], [1,0], [1,1]]`.
- Jika berhasil, output prediksi akan mendekati `[0, 1, 1, 0]`.
- Akan muncul sebuah pesan **"SUCCESS: XOR Solved!"** jika loss akhir di bawah threshold.

### 3. Menjalankan Unit Tests (Quality Assurance)

Untuk memastikan komponen dasar (fungsi aktivasi, inisialisasi bobot) valid.

**Perintah:**

```bash
python tests/test_mlpnn.py
```

---

## 🧠 Penjelasan Teknis Singkat

### Algoritma Backpropagation

Saya mengimplementasikan Backpropagation menggunakan aturan rantai (_chain rule_):

1.  **Hitung Error Output ($\delta^L$)**: Error antara prediksi dan target dikalikan turunan fungsi aktivasi.
2.  **Propagasi Mundur Error ($\delta^l$)**: Error dari layer depan dikirim mundur ke layer sebelumnya: $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$.
3.  **Update Bobot**: Menggunakan _Gradient Descent_: $W^l \leftarrow W^l - \eta \cdot \delta^l (a^{l-1})^T$.

### Autoencoder

Autoencoder adalah MLP dimana **Target = Input**. Saya menggunakan arsitektur "bottleneck" (jumlah neuron hidden < input) untuk memaksa neural network mempelajari representasi data yang lebih ringkas (kompresi) sebelum mengembalikannya ke bentuk asli.

---

## ✅ Catatan Penting

- Kode menggunakan _seed_ acak tetap (`seed=42` atau `seed=100`) untuk hasil yang dapat direproduksi (deterministik).
- Pastikan file data `dataNN_1.csv` berada di folder `data/` agar script-script saya dapat menemukannya.

---

### ✍️ Disusun oleh:

**Dandy Arya Akbar**
S1 Ilmu Komputer — 2023 A
Universitas Negeri Jakarta
