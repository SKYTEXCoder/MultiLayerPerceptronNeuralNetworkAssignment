import numpy as numpy
import pandas as pandas
from typing import List, Callable, Tuple, Optional


def sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_derivative(x: numpy.ndarray) -> numpy.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: numpy.ndarray) -> numpy.ndarray:
    return numpy.maximum(0, x)


def relu_derivative(x: numpy.ndarray) -> numpy.ndarray:
    return (x > 0).astype(float)


_TYPES_OF_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}


class MultiLayerPERCEPTRONNeuralNetwork:
    def __init__(
        self,
        layers: List[int],
        type_of_activation_function: str = "sigmoid",
        seed: Optional[int] = None,
    ):
        assert (
            len(layers) >= 2
        ), "Network must have at the very least input and output layers."
        self.layers = layers
        if type_of_activation_function not in _TYPES_OF_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation function type: '{type_of_activation_function}'. Supported types are the following: {list(_TYPES_OF_ACTIVATIONS.keys())}"
            )
        self.activation_name = type_of_activation_function
        self.activation, self.activation_derivative = _TYPES_OF_ACTIVATIONS[
            type_of_activation_function
        ]
        self.rng = numpy.random.default_rng(seed)
        self.weights: List[numpy.ndarray] = []
        self.biases: List[numpy.ndarray] = []
        self._initialize_weights_randomly()

    def _initialize_weights_randomly(self):
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
        if layer_index < 1 or layer_index >= len(self.layers):
            raise IndexError(
                "layer_index must be between 1 and number_of_layers - 1 (inclusive)."
            )
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
        if x.ndim != 1:
            raise ValueError(
                "Input x must be a 1-dimensional (1D) numpy array for single forward pass."
            )
        activations = [x.astype(float)]
        pre_activations = []
        activation_previous = x
        for index, (W, b) in enumerate(zip(self.weights, self.biases), start=1):
            z = numpy.dot(activation_previous, W) + b
            pre_activations.append(z)
            activation = self.activation(z)
            activations.append(activation)
            activation_previous = activation
        return pre_activations, activations

    def forward_batch(self, X: numpy.ndarray) -> numpy.ndarray:
        if X.ndim != 2:
            raise ValueError(
                "Input X must be a 2-dimensional (2D) numpy array (n_samples, n_features)"
            )
        activation = X.astype(float)
        for W, b in zip(self.weights, self.biases):
            activation = self.activation(numpy.dot(activation, W) + b)
        return activation

    def create_summary(self) -> str:
        returnedString = (
            f"MultiLayerPERCEPTRONNeuralNetwork architecture: {self.layers}\n"
        )
        returnedString += (
            f"Type Of Activation Function Used: {self.activation_name.capitalize()}\n"
        )
        returnedString += f"Number Of Weight Matrices: {len(self.weights)}\n"
        for index, W in enumerate(self.weights, start=1):
            returnedString += f" Layer {index - 1} -> {index} weights shape: {W.shape}, biases shape: {self.biases[index - 1].shape}\n"
        return returnedString


csv_input_data_file_path = "data/dataNN_1.csv"

try:
    if (
        not isinstance(csv_input_data_file_path, str)
        or len(csv_input_data_file_path.strip()) == 0
        or not csv_input_data_file_path.endswith(".csv")
    ):
        raise ValueError(
            "Error: Either the provided CSV input data file path is invalid or it does not point to a CSV file."
        )
    if csv_input_data_file_path == "data/dataNN_1.csv":
        dataframe = pandas.read_csv(csv_input_data_file_path, sep=";", decimal=",")
    elif csv_input_data_file_path == "data/dataNN_2.csv":
        dataframe = pandas.read_csv(csv_input_data_file_path, sep=",", decimal=".")
    else:
        dataframe = pandas.read_csv(csv_input_data_file_path)
except Exception as exception:
    raise RuntimeError(
        f"Failed to read the CSV file at path '{csv_input_data_file_path}': {exception}"
    )

print("\n")
print(f"Successfully loaded input dataset from '{csv_input_data_file_path}'.")
print("Preview of the currently-loaded input dataset (only showing the first 6 rows):")
display_dataframe = dataframe.head(6)
print(display_dataframe)

X_all = dataframe.to_numpy(dtype=float)
expected_input_neurons = 4
if X_all.shape[1] < expected_input_neurons:
    raise ValueError(
        f"CSV has {X_all.shape[1]} columns but expected at least {expected_input_neurons} input features (columns)."
    )
X_inputs = X_all[:, :expected_input_neurons]

layers = [4, 3, 2]
mlpnn = MultiLayerPERCEPTRONNeuralNetwork(
    layers=layers, type_of_activation_function="sigmoid", seed=42
)

print("\nMulti-Layer PERCEPTRON Neural Network (MLPNN) Summary:")
print(mlpnn.create_summary())

print(
    "\nRunning a feed forward pass for each specific sample in the currently loaded input CSV file dataset and printing corresponding outputs:"
)
outputs = mlpnn.forward_batch(X_inputs)
for index, (input, output) in enumerate(zip(X_inputs, outputs), start=1):
    print(
        f"Sample {index}: input={numpy.round(input, 6)} --> output={numpy.round(output, 6)}"
    )

sample0 = X_inputs[0]
neuron_total = mlpnn.compute_neuron_input(
    layer_index=1, neuron_index=0, input_vector=sample0
)
print(
    f"\nExample neuron total input: Layer 1 Neuron 0 (Sample 0) --> {neuron_total:.6f}"
)

pre_activations, activations = mlpnn.feed_forward_pass(sample0)

print("\nDetailed feed forward pass for Sample 0:")

for layer_index, (z, a) in enumerate(zip(pre_activations, activations[1:]), start=1):
    ## print(f" Layer {layer_index} z (pre-activation) = {numpy.round(z, 6)}, activation = {numpy.round(a, 6)}")
    print(
        " Layer {} z (pre-activation) = {}, activation = [{}]".format(
            layer_index, numpy.round(z, 6), ", ".join(f"{v:.6f}" for v in a)
        )
    )


print("\nProgram finished successfully.")

print("\n")

# End of src/mlpnn.py
