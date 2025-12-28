
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlpnn import MultiLayerPERCEPTRONNeuralNetwork

def test_xor():
    print("Testing XOR with Backpropagation...")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Aggressive parameters
    mlp = MultiLayerPERCEPTRONNeuralNetwork(
        layers=[2, 4, 1],
        type_of_activation_function="sigmoid",
        seed=42,
        learning_rate=0.5 # Higher learning rate
    )
    
    print("Initial Loss:", mlp.calculate_loss(y, mlp.predict(X)))
    
    mlp.train(X, y, epochs=10000)
    
    final_preds = mlp.predict(X)
    print("Final Predictions:")
    print(final_preds)
    
    loss = mlp.calculate_loss(y, final_preds)
    print(f"Final Loss: {loss:.6f}")
    
    # Assertions
    if loss < 0.05:
        print("SUCCESS: XOR Solved!")
    else:
        print("FAILURE: XOR Not Solved (Loss too high)")
        sys.exit(1)

if __name__ == "__main__":
    test_xor()
