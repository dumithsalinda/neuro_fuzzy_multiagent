"""
cnn_agent.py
Agent that uses a ConvolutionalNeuralNetwork for structured/spatial data.
"""
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import ConvolutionalNeuralNetwork
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin
import numpy as np

@register_plugin("agent")
class CNNAgent:
    """
    Agent using a ConvolutionalNeuralNetwork model.
    Supports both 1D and 2D input (e.g., images with shape (height, width, channels)).
    """
    def __init__(self, input_shape, num_filters, kernel_size, output_dim, activation=np.tanh, policy=None):
        print(f"[DEBUG] CNNAgent.__init__ input_shape={input_shape} (type={type(input_shape)}), num_filters={num_filters} (type={type(num_filters)}), kernel_size={kernel_size} (type={type(kernel_size)}), output_dim={output_dim} (type={type(output_dim)})")
        self.model = ConvolutionalNeuralNetwork(input_shape, num_filters, kernel_size, output_dim, activation)
        self.policy = policy if policy is not None else self.default_policy

    def act(self, obs):
        out = self.model.forward(obs)
        return self.policy(out)

    def default_policy(self, out):
        return int(out.argmax())

    def train_step(self, x, y, lr=0.01):
        """
        Perform a single training step (forward + backward) on input x and target y.
        Computes and returns average cross-entropy loss and accuracy for the batch.
        """
        import numpy as np
        
        # If x is 2D, try to reshape to (batch, h, w, c)
        if x.ndim == 2:
            batch = x.shape[0]
            try:
                h, w, c = self.model.input_shape
                x = x.reshape((batch, h, w, c))
                
            except Exception as e:
                raise ValueError(f"Cannot reshape x of shape {x.shape} to (batch, h, w, c) with input_shape={self.model.input_shape}: {e}")
        else:
            pass
        # Forward pass
        logits = self.model.forward(x)  # shape: (batch, num_classes)
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Cross-entropy loss
        batch_size = logits.shape[0]
        # Ensure y is integer class indices
        if y.ndim > 1:
            y = y.flatten()
        correct_logprobs = -np.log(probs[np.arange(batch_size), y] + 1e-9)
        loss = np.mean(correct_logprobs)
        # Accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = np.mean(preds == y)
        # Backward pass
        if hasattr(self.model, 'backward'):
            self.model.backward(x, y, lr=lr)
        else:
            raise NotImplementedError("Model does not implement backward method.")
        return float(loss), float(accuracy)

    def predict(self, x, return_proba=False):
        """
        Predict class labels (and optionally probabilities) for input x.
        """
        logits = self.model.forward(x)
        # Compute softmax probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        if return_proba:
            return probs  # Return only probabilities for top-k support
        return preds

    def validate(self, x, y, batch_size=32):
        """
        Evaluate the model on the given data and return average loss and accuracy.
        """
        import numpy as np
        n = x.shape[0]
        losses = []
        accuracies = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size]
            yb = y[i:i+batch_size]
            logits = self.model.forward(xb)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            if yb.ndim > 1:
                yb = yb.flatten()
            correct_logprobs = -np.log(probs[np.arange(len(yb)), yb] + 1e-9)
            loss = np.mean(correct_logprobs)
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == yb)
            losses.append(loss)
            accuracies.append(acc)
        return float(np.mean(losses)), float(np.mean(accuracies))

    def save_model(self, path):
        """
        Save model weights to a file using pickle.
        """
        import pickle
        weights = {
            'W_conv': self.model.W_conv,
            'b_conv': self.model.b_conv,
            'W_fc': self.model.W_fc,
            'b_fc': self.model.b_fc,
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        

    def load_model(self, path):
        """
        Load model weights from a file using pickle.
        """
        import pickle
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.model.W_conv = weights['W_conv']
        self.model.b_conv = weights['b_conv']
        self.model.W_fc = weights['W_fc']
        self.model.b_fc = weights['b_fc']
        
