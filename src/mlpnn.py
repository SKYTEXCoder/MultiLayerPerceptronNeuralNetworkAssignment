"""
Nama: Dandy Arya Akbar
NIM: 1313623028
Program Studi: S1 Ilmu Komputer
Kelas: 2023 A
"""

import numpy as numpy
import pandas as pandas
from typing import List, Callable, Tuple, Optional

# Saya mendefinisikan fungsi aktivasi sigmoid dan turunannya
def sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    return 1.0 / (1.0 + numpy.exp(-x))

def sigmoid_derivative(x: numpy.ndarray) -> numpy.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

# Saya mendefinisikan fungsi aktivasi ReLU dan turunannya
def relu(x: numpy.ndarray) -> numpy.ndarray:
    return numpy.maximum(0, x)

def relu_derivative(x: numpy.ndarray) -> numpy.ndarray:
    return (x > 0).astype(float)

_TYPES_OF_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}

class MultiLayerPERCEPTRONNeuralNetwork:
    """
    Saya membuat kelas ini untuk merepresentasikan Neural Network Multi-Layer Perceptron.
    Kelas ini mencakup inisialisasi, forward pass, dan sekarang backpropagation training.
    """
    def __init__(
        self,
        layers: List[int],
        type_of_activation_function: str = "sigmoid",
        seed: Optional[int] = None,
        learning_rate: float = 0.01
    ):
        assert (
            len(layers) >= 2
        ), "Jaringan harus memiliki setidaknya input dan output layer."
        self.layers = layers
        if type_of_activation_function not in _TYPES_OF_ACTIVATIONS:
            raise ValueError(
                f"Tipe fungsi aktivasi tidak didukung: '{type_of_activation_function}'."
            )
        self.activation_name = type_of_activation_function
        self.activation, self.activation_derivative = _TYPES_OF_ACTIVATIONS[
            type_of_activation_function
        ]
        # Saya menggunakan learning rate untuk mengontrol seberapa besar update bobot
        self.learning_rate = learning_rate
        self.rng = numpy.random.default_rng(seed)
        self.weights: List[numpy.ndarray] = []
        self.biases: List[numpy.ndarray] = []
        self._initialize_weights_randomly()

    def _initialize_weights_randomly(self):
        """
        Saya menginisialisasi bobot dan bias secara acak menggunakan metode Xavier/Glorot
        agar konvergensi training lebih baik.
        """
        self.weights = []
        self.biases = []
        for index in range(1, len(self.layers)):
            input_dim = self.layers[index - 1]
            output_dim = self.layers[index]
            limit = numpy.sqrt(6.0 / (input_dim + output_dim))
            W = self.rng.uniform(-limit, limit, size=(input_dim, output_dim))
            b = self.rng.uniform(-limit, limit, size=(output_dim,))
            self.weights.append(W)
            self.biases.append(b)

    def define_architecture(self) -> List[int]:
        return self.layers.copy()

    def compute_neuron_input(
        self, layer_index: int, neuron_index: int, input_vector: numpy.ndarray
    ) -> float:
        """
        Saya menghitung total input (z) untuk satu neuron spesifik.
        """
        if layer_index < 1 or layer_index >= len(self.layers):
            raise IndexError("layer_index harus di antara 1 dan jumlah layer - 1.")
        W = self.weights[layer_index - 1]
        b = self.biases[layer_index - 1]
        weights_to_neuron = W[:, neuron_index]
        total_input = float(
            numpy.dot(input_vector, weights_to_neuron) + b[neuron_index]
        )
        return total_input

    def feed_forward_pass(
        self, x: numpy.ndarray
    ) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
        """
        Saya melakukan forward pass untuk satu sampel input 'x'.
        Mengembalikan pre-activations (z) dan activations (a) untuk setiap layer.
        Ini akan digunakan dalam backpropagation.
        """
        if x.ndim != 1:
            raise ValueError("Input x harus berupa array 1D.")
        
        activations = [x.astype(float)]
        pre_activations = []
        activation_previous = x
        
        for W, b in zip(self.weights, self.biases):
            z = numpy.dot(activation_previous, W) + b
            pre_activations.append(z)
            activation = self.activation(z)
            activations.append(activation)
            activation_previous = activation
            
        return pre_activations, activations

    def forward_batch(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Saya melakukan helper function untuk melakukan forward pass pada batch data.
        Hanya mengembalikan output akhir.
        """
        if X.ndim != 2:
            raise ValueError("Input X harus array 2D (n_samples, n_features).")
        activation = X.astype(float)
        for W, b in zip(self.weights, self.biases):
            activation = self.activation(numpy.dot(activation, W) + b)
        return activation
    
    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        """
        Saya membuat alias untuk forward_batch agar lebih intuitif saat prediksi.
        """
        return self.forward_batch(X)

    def calculate_loss(self, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
        """
        Saya menghitung Mean Squared Error (MSE) sebagai fungsi loss.
        """
        # Mean Squared Error
        return float(numpy.mean((y_true - y_pred) ** 2))

    def backpropagate_single_sample(self, x: numpy.ndarray, y: numpy.ndarray):
        """
        Saya mengimplementasikan algoritma Backpropagation untuk satu sampel data.
        1. Forward pass untuk mendapatkan nilai aktivasi.
        2. Hitung error di output layer.
        3. Propagasi error mundur ke hidden layers.
        4. Hitung gradien untuk weights dan biases.
        5. Update weights dan biases menggunakan Stochastic Gradient Descent (SGD).
        """
        # 1. Forward Pass
        # pre_activations (z), activations (a)
        result_pre, result_act = self.feed_forward_pass(x)
        
        # List activations termasuk input layer di index 0
        # List pre_activations mulai dari layer hidden pertama (index 0 corresponds to weights[0])
        
        # 2. Hitung Output Layer Error
        # Output prediksi adalah aktivasi terakhir
        final_output = result_act[-1]
        target = y
        
        # Error term (delta) untuk output layer
        # Asumsi Loss Function = 0.5 * (target - output)^2
        # Derivative Loss w.r.t output = -(target - output)
        # Delta = (output - target) * activation_derivative(z_final)
        # Note: Tanda negatif bisa disesuaikan dengan aturan update weight (w = w - lr * grad)
        
        # Saya menggunakan (output - target) sebagai turunan loss MSE (tanpa faktor 2/n untuk SGD single sample)
        output_error = (final_output - target) * self.activation_derivative(result_pre[-1])
        
        gradients_W = [numpy.zeros_like(W) for W in self.weights]
        gradients_b = [numpy.zeros_like(b) for b in self.biases]
        
        # Simpan delta untuk layer saat ini
        delta = output_error
        
        # Gradien untuk layer terakhir
        # dC/dW = a_prev.T * delta
        # dC/db = delta
        gradients_W[-1] = numpy.outer(result_act[-2], delta)
        gradients_b[-1] = delta
        
        # 3. Propagasi Mundur (Backpropagate) ke hidden layers
        # Loop mundur dari layer kedua terakhir sampai pertama
        # num_layers = len(self.weights)
        # range(num_layers - 2, -1, -1) -> misal 3 layer bobot (0,1,2), loop: 1, 0
        
        for i in range(len(self.weights) - 2, -1, -1):
            # z untuk layer saat ini (i). 
            # Perhatikan: pre_activations index i sesuai dengan weights[i]
            z_current = result_pre[i]
            
            # Bobot layer berikutnya (i+1) yang menghubungkan ke layer ini
            W_next = self.weights[i+1]
            
            # Error delta dari layer berikutnya dipropagasi ke layer ini
            # delta_hidden = (delta_next dot W_next.T) * activation_derivative(z_current)
            delta = numpy.dot(delta, W_next.T) * self.activation_derivative(z_current)
            
            # Hitung gradien
            gradients_W[i] = numpy.outer(result_act[i], delta)
            gradients_b[i] = delta
            
        # 4. Update Weights dan Biases (SGD step)
        # Saya memperbarui parameter jaringan: param = param - learning_rate * gradient
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_W[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X: numpy.ndarray, y: numpy.ndarray, epochs: int = 1000):
        """
        Saya melatih neural network menggunakan data training X dan target y.
        Menggunakan Stochastic Gradient Descent (update per sampel).
        """
        print(f"Saya memulai training selama {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            
            # Iterasi setiap sampel
            for i in range(len(X)):
                sample_x = X[i]
                sample_y = y[i]
                
                # Forward pass hanya untuk hitung loss visualisasi (opsional, bisa berat)
                # Prediction dilakukan di dalam backpropagate sebenarnya, tapi kita panggil terpisah atau ambil dari backprop
                # Untuk efisiensi, kita bisa hitung loss di akhir epoch atau akumulasi error saat backprop
                
                # Lakukan Backpropagation dan Update Bobot
                self.backpropagate_single_sample(sample_x, sample_y)
                
            # Hitung total loss dataset setiap beberapa epoch untuk monitoring
            if epoch % 100 == 0 or epoch == 1:
                predictions = self.forward_batch(X)
                loss = self.calculate_loss(y, predictions)
                print(f" Epoch {epoch}/{epochs}, Loss (MSE): {loss:.6f}")

    def create_summary(self) -> str:
        returnedString = (
            f"MultiLayerPERCEPTRONNeuralNetwork architecture: {self.layers}\n"
        )
        returnedString += (
            f"Type Of Activation Function Used: {self.activation_name.capitalize()}\n"
        )
        returnedString += f"Number Of Weight Matrices: {len(self.weights)}\n"
        returnedString += f"Learning Rate: {self.learning_rate}\n"
        for index, W in enumerate(self.weights, start=1):
            returnedString += f" Layer {index - 1} -> {index} weights shape: {W.shape}, biases shape: {self.biases[index - 1].shape}\n"
        return returnedString


# Contoh penggunaan simple (Main block ini bisa di-comment out jika diimport)
if __name__ == "__main__":
    print("Saya sedang menjalankan tes sederhana (XOR)...")
    
    # Dataset XOR
    X_xor = numpy.array([[0,0], [0,1], [1,0], [1,1]])
    y_xor = numpy.array([[0], [1], [1], [0]])
    
    # Inisialisasi MLP: 2 input, 5 hidden, 1 output
    mlp_xor = MultiLayerPERCEPTRONNeuralNetwork(
        layers=[2, 5, 1], 
        type_of_activation_function="sigmoid", 
        seed=42, 
        learning_rate=0.1
    )
    
    print("\nSebelum Training:")
    print(mlp_xor.predict(X_xor))
    
    # Training
    mlp_xor.train(X_xor, y_xor, epochs=5000)
    
    print("\nSetelah Training:")
    preds = mlp_xor.predict(X_xor)
    print(preds)
    print("\nTarget:")
    print(y_xor)
