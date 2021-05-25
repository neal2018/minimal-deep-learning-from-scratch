import numpy as np


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def params_and_grads(self):
        for k in self.params.keys():
            yield k, self.params[k], self.grads[k]


class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)  # k*m
        self.params["b"] = np.random.randn(output_size)  # 1*m

    def forward(self, inputs):
        # x: n*k (i.e., batch_size*feature_number)
        # y = x @ w + b
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]  # n*k @ k*m + 1*m (boardcasting) -> n*m

    def backward(self, grads):
        # grads: f'(y) = df(y)/dy from upstream, k*m
        # return df(y)/dx = f'(y) @ w
        # df(y)/dw = x.T @ f'(y)
        # df(y)/db = f'(y)
        self.grads["w"] = self.inputs.T @ grads  # k*n @ n*m -> k*m
        self.grads["b"] = grads.sum(axis=0)  # n*m -> 1*m
        return grads @ self.params["w"].T  # n*m @ m*k -> n*k


class Tanh(Layer):
    # elementwise
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, grads):
        # dg(f(x))/dx = dg/df * df/dx
        return grads * (1 - np.tanh(self.inputs)**2)


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return np.mean((inputs - targets) ** 2)

    def backward(self, inputs, targets):
        return 2 * (inputs - targets) / inputs.shape[0]


class BatchIterator:
    def __init__(self, inputs, targets, batch_size):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size

    def __iter__(self, need_shuffle=True):
        starts = np.arange(0, self.inputs.shape[0], self.batch_size)
        if need_shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            yield self.inputs[start:end], self.targets[start:end]


class Optimizer(Layer):
    def step(self, net):
        raise NotImplementedError

    def zero_grad(self, net):
        for name, param, grad in net.params_and_grads():
            grad.fill(0)

    def clip_grad_norm(self, net, max_norm, norm_type=2):
        for name, param, grad in net.params_and_grads():
            batch_size = grad.shape[0]
            norm = np.linalg.norm(grad, ord=norm_type)
            if norm > batch_size * max_norm:
                grad /= norm / (batch_size * max_norm)


class SGD(Optimizer):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr

    def step(self, net):
        for name, param, grad in net.params_and_grads():
            param -= self.lr * grad


def train(net, num_epochs, data_iterator, optimizer, loss_fn):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_inputs, batch_targets in data_iterator:
            predicted = net(batch_inputs)
            epoch_loss += loss_fn(predicted, batch_targets)
            grad = loss_fn.backward(predicted, batch_targets)
            net.backward(grad)
            optimizer.clip_grad_norm(net, 1)
            optimizer.step(net)
        print(epoch, epoch_loss)


if __name__ == '__main__':
    # generate some fizzbuzz data
    def fizzbuzz(x):
        if x % 3 == 0 and x % 5 == 0:
            return "fizzbuzz"
        elif x % 3 == 0:
            return "fizz"
        elif x % 5 == 0:
            return "buzz"
        else:
            return x

    def target_encoder(target):
        res = [0, 0, 0, 0]
        encode = {"fizzbuzz": 0, "fizz": 1, "buzz": 2, target: 3}
        ans = fizzbuzz(target)
        res[encode[ans]] = 1
        return res

    ENCODE_SIZE = 16

    def input_encoder(input):
        return [input >> i & 1 for i in range(ENCODE_SIZE)]

    train_range = range(101, 2000)
    train_inputs = np.array([input_encoder(x) for x in train_range])
    train_targets = np.array([target_encoder(x) for x in train_range])

    test_range = range(101)
    test_inputs = np.array([input_encoder(x) for x in test_range])
    test_targets = np.array([target_encoder(x) for x in test_range])

    net = Sequential([
        Linear(ENCODE_SIZE, 128),
        Tanh(),
        Linear(128, 4)
    ])

    train(net, 2000, BatchIterator(train_inputs, train_targets, 32), SGD(0.05), MSELoss())

    # testing
    for i, (test_input, test_target) in enumerate(zip(test_inputs, test_targets)):
        test_input, test_target = np.expand_dims(test_input, 0), np.expand_dims(test_target, 0)
        predicted = net(test_input)
        labels = ["fizzbuzz", "fizz", "buzz", i]
        print(i, labels[np.argmax(predicted)], labels[np.argmax(test_target)])
