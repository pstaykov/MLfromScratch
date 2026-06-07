import numpy as np

def reLu(x):
    return np.maximum(0, x)

def reLu_derivative(x):
    return (x > 0).astype(float)

# Initialize input and target data
inputs = np.array([0, 1, 2, 3, 4], dtype=np.float32)
targets = np.array([0, 1, 2, 3, 4], dtype=np.float32)

# Initialize weights
np.random.seed(42)
Wxh = np.random.randn(5, 5) * 0.01
Whh = np.random.randn(5, 5) * 0.01
Why = np.random.randn(5, 5) * 0.01
bh = np.zeros(5)
by = np.zeros(5)

learning_rate = 0.01
hidden_size = 5

def forward(x):
    """
    Forward pass through RNN, returns output and hidden states for BPTT
    """
    h_states = [np.zeros((hidden_size,))]  # h_0
    raw_h_states = []  # z values before activation

    for t in range(len(x)):
        x_t = x[t]  # scalar input
        z = np.dot(Wxh, np.ones(5) * x_t) + np.dot(Whh, h_states[-1]) + bh
        h = reLu(z)
        h_states.append(h)
        raw_h_states.append(z)

    y = np.dot(Why, h_states[-1]) + by
    return y, h_states, raw_h_states

def mse_loss(y_pred, y_true):
    """Mean Squared Error loss"""
    return np.mean((y_pred.flatten() - y_true) ** 2)

def bptt(x, y_targets, h_states, raw_h_states):
    """
    Backpropagation Through Time (BPTT)
    Computes gradients for all parameters
    """
    T = len(x)

    # Initialize gradients
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    # Forward pass to get final output
    y_pred, _, _ = forward(x)

    # Output layer gradient
    dy = (y_pred.flatten() - y_targets) * 2 / len(y_targets)  # MSE derivative
    dWhy += np.dot(dy.reshape(-1, 1), h_states[-1].reshape(1, -1))
    dby += dy

    # Backpropagate through time
    dh_next = np.zeros(hidden_size)

    for t in reversed(range(T)):
        x_t = np.ones(5) * x[t]

        # Gradient from output
        dh = np.dot(Why.T, dy) + dh_next

        # Gradient through ReLU activation
        dz = dh * reLu_derivative(raw_h_states[t])

        # Parameter gradients
        dWxh += np.dot(dz.reshape(-1, 1), x_t.reshape(1, -1))
        dWhh += np.dot(dz.reshape(-1, 1), h_states[t].reshape(1, -1))
        dbh += dz

        # Gradient for next time step
        dh_next = np.dot(Whh.T, dz)

    # Clip gradients to prevent exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return dWxh, dWhh, dWhy, dbh, dby

def train(num_epochs=10):
    """
    Train the RNN for specified number of epochs
    """
    global Wxh, Whh, Why, bh, by

    print("Starting training...")
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}\n")

    for epoch in range(num_epochs):
        # Forward pass
        y_pred, h_states, raw_h_states = forward(inputs)

        # Compute loss
        loss = mse_loss(y_pred, targets)

        # Backward pass (BPTT)
        dWxh, dWhh, dWhy, dbh, dby = bptt(inputs, targets, h_states, raw_h_states)

        # Update weights
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.6f}")

    print("\nTraining completed!")
    print("\nFinal predictions:")
    y_final, _, _ = forward(inputs)
    print(f"Predictions: {y_final.flatten()}")
    print(f"Targets:     {targets}")

train(num_epochs=10)

