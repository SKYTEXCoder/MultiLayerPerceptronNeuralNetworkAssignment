"""
Nama: Dandy Arya Akbar
NIM: 1313623028
Program Studi: S1 Ilmu Komputer
Kelas: 2023 A
"""

import unittest
import numpy as np
import sys
import os

# Pastikan bisa import dari src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlpnn import MultiLayerPERCEPTRONNeuralNetwork, sigmoid, relu

class TestMLPNN(unittest.TestCase):
    
    def test_sigmoid_activation(self):
        # Test sigmoid(0) = 0.5
        x = np.array([0])
        self.assertAlmostEqual(sigmoid(x)[0], 0.5)
        
        # Test sigmoid range [0, 1]
        x_large = np.array([100, -100])
        s = sigmoid(x_large)
        self.assertAlmostEqual(s[0], 1.0)
        self.assertAlmostEqual(s[1], 0.0)

    def test_mlp_initialization(self):
        layers = [4, 3, 2]
        mlp = MultiLayerPERCEPTRONNeuralNetwork(layers, seed=42)
        
        # Cek jumlah weights matrix (harus len(layers) - 1)
        self.assertEqual(len(mlp.weights), 2)
        
        # Cek dimensi weight matrix layer 1 (Input 4 -> Hidden 3)
        # Shape di numpy: (input_dim, output_dim)
        self.assertEqual(mlp.weights[0].shape, (4, 3))
        
        # Cek dimensi bias layer 1 (Hidden 3)
        self.assertEqual(mlp.biases[0].shape, (3,))

    def test_forward_pass_shape(self):
        # 4 Input features
        X = np.array([[0.1, 0.2, 0.3, 0.4]]) 
        layers = [4, 3, 2]
        mlp = MultiLayerPERCEPTRONNeuralNetwork(layers, seed=42)
        
        output = mlp.predict(X)
        
        # Output harus (1 sample, 2 output neuron)
        self.assertEqual(output.shape, (1, 2))

if __name__ == '__main__':
    print("Saya sedang menjalankan Unit Tests dasar untuk MLPNN...")
    unittest.main()
