from _context import vugrad

import numpy as np

import vugrad as vg

from vugrad import Op

class AdjustableMLP(vg.Module):
    """
    A modular MLP that allows for flexible adjustments:
    - Number of layers
    - Hidden layer size
    - Momentum term
    - Residual connections
    - Initialization types
    """

    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, use_residual=False, init_type="glorot"):
        """
        :param input_size: Number of input features.
        :param output_size: Number of output classes.
        :param hidden_size: Number of neurons in each hidden layer.
        :param num_layers: Total number of layers (minimum = 2: input and output layers).
        :param use_residual: Whether to add residual connections.
        :param init_type: Initialization method: "glorot", "zero", or "normal".
        """
        super().__init__()
        assert num_layers >= 2, "The network must have at least an input and output layer."

        self.use_residual = use_residual
        self.layers = []

        # Input layer
        self.layers.append(self._create_layer(input_size, hidden_size, init_type))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(self._create_layer(hidden_size, hidden_size, init_type))

        # Output layer
        self.layers.append(self._create_layer(hidden_size, output_size, init_type))

    def _create_layer(self, input_size, output_size, init_type):
        """
        Creates a linear layer with specified initialization.
        """
        layer = vg.Linear(input_size, output_size)

        if init_type == "normal":
            layer.w.value = np.random.normal(size=(output_size, input_size))
            layer.b.value = np.random.normal(size=(1, output_size))

        return layer

    def forward(self, input):
        """
        Forward pass with optional residual connections.
        """
        assert len(input.size()) == 2
        x = input

        for i, layer in enumerate(self.layers[:-1]):  # Exclude the final output layer
            prev_x = x
            x = layer(x)  # Linear transformation
            x = vg.sigmoid(x)  # Apply non-linearity

            if self.use_residual and i > 0:  # Add residual connection
                x = x + prev_x

        # Final output layer (no non-linearity)
        x = self.layers[-1](x)
        return vg.logsoftmax(x)

    def parameters(self):
        """
        Returns all parameters (weights and biases).
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def train_and_evaluate(mlp, learning_rate, momentum_factor, xtrain, ytrain, xval, yval, batch_size, num_epochs):
    """
    Train and evaluate the AdjustableMLP on MNIST for one run.
    Records loss and accuracy for training and validation datasets.
    """
    num_instances, num_features = xtrain.shape
    b = batch_size

    # Initialize momentum dictionary
    momentum = {param: np.zeros_like(param.value) for param in mlp.parameters()}

    train_loss = []
    val_acc = []

    for epoch in range(num_epochs):
        print(f'epoch {epoch:03}')
        o = mlp(vg.TensorNode(xval))
        oval = o.value

        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]

        o.clear()  # gc the computation graph
        print(f'       accuracy: {acc:.4}')
        cl = 0.0  # Running training loss
        for fr in range(0, num_instances, b):
            to = min(fr + b, num_instances)
            batch, targets = xtrain[fr:to, :], ytrain[fr:to]

            # Wrap the inputs in a Node
            batch = vg.TensorNode(value=batch)

            # Forward pass
            outputs = mlp(batch)
            loss = vg.logceloss(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient descent with momentum
            for param in mlp.parameters():
                momentum[param] = momentum_factor * momentum[param] - learning_rate * param.grad
                param.value += momentum[param]

            # Record training loss
            cl += loss.value

            # Clear gradients and computation graph
            loss.zero_grad()
            loss.clear()

        # Validation performance
        val_output = mlp(vg.TensorNode(xval))
        val_predictions = np.argmax(val_output.value, axis=1)
        num_correct = (val_predictions == yval).sum()
        val_accuracy = num_correct / yval.shape[0]

        train_loss.append(cl / num_instances)
        val_acc.append(val_accuracy)

        val_output.clear()
        print(f'   running loss: {cl / num_instances:.4}')

    return train_loss, val_acc

def mnist():
    return vg.load_mnist(final=False, flatten=True)

if __name__ == '__main__':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
    num_instances, num_features = xtrain.shape

    mlp = AdjustableMLP(input_size=num_features, output_size=num_classes, hidden_size=128, num_layers=3)

    train_and_evaluate(
        mlp, 0.0001, 0.5, xtrain, ytrain, xval, yval, 256, 20
    )