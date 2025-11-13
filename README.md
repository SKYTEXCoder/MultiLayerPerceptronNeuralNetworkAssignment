# 🧠 Implementasi Forward-Only Multi-Layer Perceptron Neural Network dengan menggunakan bahasa Python dan library NumPy

## 📋 Identitas Saya

* **Nama:** Dandy Arya Akbar
* **NIM:** 1313623028
* **Jurusan:** S1 Ilmu Komputer, Universitas Negeri Jakarta (UNJ)
* **Mata Kuliah:** Deep Learning
* **Dosen Pengampu:** Med Irzal
* **Semester:** 5
* **Versi Python:** 3.11.x

---

## 📖 Deskripsi Project

Proyek ini mengimplementasikan versi basic atau versi simplenya dari **Multi-Layer Perceptron Neural Network (MLPNN)** dengan model **Forward-Only** menggunakan **library NumPy**, sebagai bagian dari tugas mata kuliah Deep Learning. Tujuan utama proyek ini adalah untuk mendemonstrasikan pemahaman tentang bagaimana jaringan saraf bekerja **secara matematis** — mulai dari input data, melalui lapisan-lapisan neuron dengan bobot (weights) dan bias, hingga menghasilkan output — **tanpa melakukan proses pelatihan (training) atau backpropagation**.

Program ini dapat memuat dataset dalam format CSV (`dataNN_1.csv` atau `dataNN_2.csv`), menginisialisasi bobot (weights) dan bias (biases) secara acak, menjalankan proses *feed-forward*, serta menampilkan output untuk setiap sampel input.

---

## 🧩 Arsitektur Inti dari Neural Network

Neural Network ini memiliki **3 lapisan utama**:

| Lapisan | Deskripsi    | Jumlah Neuron |
| ------- | ------------ | ------------- |
| 0       | Input Layer  | 4             |
| 1       | Hidden Layer | 3             |
| 2       | Output Layer | 2             |

**Fungsi Aktivasi:** Sigmoid Function

Bobot dan bias diinisialisasi secara acak menggunakan **inisialisasi uniform Glorot/Xavier**, yang menjaga kestabilan variansi antar lapisan.

---

## 🧠 Penjelasan Teori

### 🔹 Gambaran Umum Proses Forward Pass

Proses *forward pass* adalah langkah di mana neural network mengambil vektor input dan menghitung output-nya dengan mempropagasikan data melalui setiap lapisan. Setiap lapisan melakukan operasi sebagai berikut:

1. **Perhitungan Jumlah Tertimbang (Transformasi Linear)**

   $$
   z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
   $$

   Di mana:

   * (W^{(l)}) adalah matriks bobot pada lapisan *l*.
   * (b^{(l)}) adalah vektor bias.
   * (a^{(l-1)}) adalah aktivasi dari lapisan sebelumnya.

2. **Fungsi Aktivasi**

   $$
   a^{(l)} = f(z^{(l)})
   $$

   Di mana (f) adalah fungsi aktivasi yang digunakan (dalam proyek ini: Sigmoid Function).

3. **Output Akhir**
   Lapisan aktivasi terakhir menghasilkan prediksi akhir dari neural network.

### 🔹 Tanpa Proses Training (Forward-Only)

Implementasi Neural Network ini bersifat **forward-only**, yang berarti:

* Hanya melakukan propagasi maju (*forward propagation*).
* **Tidak** melakukan proses *backpropagation* atau pembaruan weights.
* Menghasilkan output langsung berdasarkan bobot acak yang telah diinisialisasi.

---

## 💻 Penjelasan Kode

### Definisi Module/Class Utama

```python
class MultiLayerPERCEPTRONNeuralNetwork:
    def __init__(self, layers, type_of_activation_function="sigmoid", seed=None):
        ...
```

Mendefinisikan arsitektur MLPNN dan menginisialisasi bobot serta bias secara acak menggunakan metode inisialisasi Glorot.

### Fungsi Aktivasi

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
```

Project ini mendukung penggunaan dari dan menyediakan dua tipe fungsi aktivasi: yaitu fungsi-fungsi **Sigmoid** dan **ReLU**, untuk keperluan eksperimen dan testing.

### Proses Forward Pass

```python
def feed_forward_pass(self, x):
    activations = [x.astype(float)]
    for W, b in zip(self.weights, self.biases):
        z = np.dot(activations[-1], W) + b
        activations.append(self.activation(z))
    return activations
```

Fungsi ini memproses satu vektor input melalui seluruh lapisan neural network dan mengembalikan hasil aktivasi dari setiap lapisan.

### Proses Batch Forward

```python
def forward_batch(self, X):
    activation = X.astype(float)
    for W, b in zip(self.weights, self.biases):
        activation = self.activation(np.dot(activation, W) + b)
    return activation
```

Digunakan untuk menjalankan *forward pass* pada banyak sampel input secara bersamaan.

---

## 🧾 File Dataset Input

Dua dataset contoh disediakan di folder `data/`:

| Nama File      | Simbol Desimal | Pemisah Kolom    |
| -------------- | -------------- | ---------------- |
| `dataNN_1.csv` | `,` (koma)     | `;` (titik koma) |
| `dataNN_2.csv` | `.` (titik)    | `,` (koma)       |

Program akan otomatis mendeteksi file yang digunakan dan menyesuaikan parameter pembacaan (*delimiter* dan *decimal symbol*).

---

## ⚙️ Petunjuk Step-By-Step (Langkah Demi Langkah) Untuk Menjalankan Program

### 📦 Prasyarat

* Python **3.11.x** atau versi yang lebih baru
* Library-Library Python Berikut Ini:

  * `numpy`
  * `pandas`

Instal library yang diperlukan (jika belum terinstall) dengan perintah sebagai berikut:

```bash
pip install numpy pandas
```

### ▶️ Langkah Menjalankan Program

1. **Kloning atau ekstrak** folder proyek ini.
2. Pastikan struktur folder kurang lebih seperti berikut ini:

   ```
   project_root/
   ├── src/
   │   └── mlpnn.py
   └── data/
       ├── dataNN_1.csv
       └── dataNN_2.csv
   ```
3. Buka terminal di direktori utama project ini (project root directory).
4. Jalankan program python dengan perintah:

   ```bash
   python src/mlpnn.py
   ```

### 🧠 Mengganti Dataset Input

Untuk mengganti dataset input, ubah baris berikut pada file `src/mlpnn.py`:

```python
csv_input_data_file_path = "data/dataNN_1.csv"
```

Menjadi:

```python
csv_input_data_file_path = "data/dataNN_2.csv"
```

Dan sebaliknya. Program akan menyesuaikan pembacaan secara otomatis.

### 🧩 Contoh Output

```
Successfully loaded input dataset from 'data/dataNN_1.csv'.
Multi-Layer PERCEPTRON Neural Network (MLPNN) Summary:
 Layer 0 -> 1 weights shape: (4, 3), biases shape: (3,)
 Layer 1 -> 2 weights shape: (3, 2), biases shape: (2,)

Running a feed forward pass for each sample...
Sample 1: input=[29.602563  1.332298  2.970915 -1.250228] --> output=[0.319896 0.85951 ]
...
Program finished successfully.
```

---

## 🧠 Ringkasan Teoretis

* Proyek ini menunjukkan proses **komputasi maju (forward computation)** dari sebuah neural network menggunakan library NumPy.
* Menjelaskan bagaimana **neuron, bobot, bias, dan aktivasi** berinteraksi secara matematis.
* Bukan merupakan sebuah arsitektur neural network yang dapat dilakukan proses training (tidak ada logika, proses, atau algoritma backpropagation di sini).

---

## 🧾 Referensi

* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
* Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.
* Dokumentasi NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)

---

## ✅ Catatan-Catatan Penting

* Output bersifat **deterministik** karena menggunakan *random seed* tetap (`seed=42`).
* Program berjalan secara independen tanpa bergantung kepada library-library machine learning seperti TensorFlow, PyTorch, atau Keras.
* Kedua jenis tipe dataset contoh (`dataNN_1.csv` dan `dataNN_2.csv`) didukung secara otomatis.

---

### ✍️ Disusun oleh:

**Dandy Arya Akbar**
S1 Ilmu Komputer — Universitas Negeri Jakarta (UNJ)
Mata Kuliah Deep Learning, Semester 5
