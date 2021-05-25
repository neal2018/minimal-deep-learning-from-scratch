import numpy as np
from linear import Layer, Linear, Tanh, Sequential
from torch.nn import RNN


class RNN(Layer):
    def __init__(self, input_size, hidden_size):
        # num_layers = 1, batch_first = False
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = Tanh()
        self.params["w_ih"] = np.random.randn(input_size, hidden_size)
        self.params["w_hh"] = np.random.randn(hidden_size, hidden_size)
        self.params["b_ih"] = np.random.randn(hidden_size)
        self.params["b_hh"] = np.random.randn(hidden_size)

    def forward(self, inputs, hiddens):
        # inputs: l * n * m, hiddens: 1 * n * hidden_size
        # outputs: l * n * hidden_size, hiddens: 1 * n * hidden_size
        self.inputs = inputs
        self.hiddens = hiddens
        outputs = np.zeros((inputs.shape[0], inputs.shape[1], self.hidden_size))
        for i, x in enumerate(inputs):
            hiddens = x @ self.params["w_ih"] + self.params["b_ih"] + hiddens @ self.params["w_hh"] + self.params["b_hh"]
            hiddens = self.activation(hiddens)
            outputs[i] = hiddens
        self.outputs = outputs
        return outputs, hiddens

    def backward(self, grads):
        # TODO
        pass

    def init_hidden(self):
        return np.zeros(1, self.hidden_size)


if __name__ == "__main__":
    # TODO
    pass
