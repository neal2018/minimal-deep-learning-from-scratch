import numpy as np
from linear import Layer, Linear, Sequential, SGD, BatchIterator, train


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        # assume stride == 1 and not dilation/grouping for simplicity
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # denoted as k in following comments
        self.padding = padding
        # params["w"]: out_channels * in_channels * k * k
        self.params["w"] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.params["b"] = np.random.randn(out_channels)  # 1 * out_channels

    def forward(self, inputs):
        # inputs: n * in_channels * h * w
        # h_out = (1 + (h + 2*p - k) // stride), w_out = (1 + (w + 2*p - k) // stride)
        # return: n * out_channels * h_out * w_out
        self.inputs = inputs
        # add padding -> n * in_channels * (h + p) * (w + p)
        inputs = np.pad(inputs, ((0,), (0,), (self.padding,), (self.padding,)))
        # spilt to kernel size views on axes (2, 3) -> n * in_channels * h_out * w_out * k * k
        views = np.lib.stride_tricks.sliding_window_view(inputs, (self.kernel_size, self.kernel_size), (2, 3))
        # sum up on (1, 4, 5): n * in_channels * h_out * w_out * k * k -> n * h_out * w_out
        # sum up on (1, 2, 3): out_channels * in_channels * k * k -> out_channels
        # append together -> n * h_out * w_out * out_channels
        convolved_matrix = np.tensordot(views, self.params["w"], ((1, 4, 5), (1, 2, 3)))
        # transpose: n * h_out * w_out * out_channels -> n * out_channels * h_out *  w_out
        convolved_matrix = convolved_matrix.transpose(0, 3, 1, 2)
        return convolved_matrix

    def backward(self, grads):
        # grads: n * out_channels * h_out * w_out
        # return: n * in_channels * h * w
        h_out, w_out = grads.shape[2:]
        h, w = self.inputs.shape[2:]
        self.grads["b"] = grads.sum(axis=(0, 2, 3))  # 1 * out_channels
        # calculate grad w: out_channels * in_channels * k * k
        # add padding ->  n * in_channels * (h + p) * (w + p)
        inputs = np.pad(self.inputs, ((0,), (0,), (self.padding,), (self.padding,)))
        # spilt to grad size views on axes (2, 3) -> n * in_channels * k * k * h_out * w_out
        views = np.lib.stride_tricks.sliding_window_view(inputs, (h_out, w_out), (2, 3))
        # grads sum up on (0, 2, 3): n * out_channels * h_out * w_out -> out_channels
        # views sum up on (0, 4, 5): n * in_channels * k * k * h_out * w_out -> n * k * k
        # append together -> out_channels * in_channels * k * k
        self.grads["w"] = np.tensordot(grads, views, ((0, 2, 3), (0, 4, 5)))

        # calculate grad x:  n * in_channels * h * w
        weight_rotated = np.rot90(self.params["w"], 2, (2, 3))  # 180 rotation
        # add padding to make sure output size is (h + 2*p, w + 2*p) after following convolution
        # (k + 2*p_h - h_out)//stride + 1 = h + 2*p => p_h = ((h + 2*p -1) * stride + h_out - k)//2; p_w is similar
        padding_h = (h + 2 * self.padding - 1 + h_out - self.kernel_size) // 2
        padding_w = (w + 2 * self.padding - 1 + w_out - self.kernel_size) // 2
        weight_padded = np.pad(weight_rotated, ((0,), (0,), (padding_h,), (padding_w,)))
        # spilt to grad size views -> out_channels * in_channels * (h + 2*p) * (w + 2*p) * h_out * w_out
        wight_views = np.lib.stride_tricks.sliding_window_view(weight_padded, (h_out, w_out), (2, 3))
        # grads sum up on (1, 2, 3): n * out_channels * h_out * w_out -> n
        # wight_views sum up on (0, 4, 5): out_channels * in_channels * (h + 2*p) * (w + 2*p) * h_out * w_out
        #       -> in_channels * (h + 2*p) * (w + 2*p)
        # append together -> n * in_channels * (h + 2*p) * (w + 2*p)
        x_grads = np.tensordot(grads, wight_views, ((1, 2, 3), (0, 4, 5)))
        # remvoe padding -> n * in_channels * h * w
        x_grads = x_grads[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_grads


class MaxPool2d(Layer):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, inputs):
        # inputs: n * in_channels * h * w
        k = self.kernel_size
        self.h, self.w = inputs.shape[2:]
        # submatrices: n * in_channels * h_out * w_out * k * k
        submatrices = np.lib.stride_tricks.sliding_window_view(inputs, (k, k), (2, 3))[:, :, ::k, ::k]
        # pooled_input: n * in_channels * h_out * w_out
        pooled_input = np.max(submatrices, axis=(4, 5))
        # repeated_maxes: n * in_channels * h * w
        repeated_maxes = pooled_input.repeat(k, axis=2).repeat(k, axis=3)[:, :, :self.h, :self.w]
        self.max_mask = repeated_maxes == inputs
        return pooled_input

    def backward(self, grads):
        # grads: n * in_channels * h_out * w_out
        k = self.kernel_size
        # repeated_grads: n * in_channels * h * w
        repeated_grads = grads.repeat(k, axis=2).repeat(k, axis=3)[:, :, :self.h, :self.w]
        return repeated_grads * self.max_mask


class Flatten(Layer):
    def __init__(self, start_dim=1, end_dim=-1):
        # flatten start_dim to end_dim, inclusive
        # n * d1 * d2 * ... * dk -> n * (d1 * ... * dk)
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim + 1 if end_dim != -1 else None  # change to exclusive

    def forward(self, inputs):
        self.input_shape = inputs.shape
        output_shape = list(inputs.shape)
        output_shape[self.start_dim: self.end_dim] = [np.prod(inputs.shape[self.start_dim: self.end_dim])]
        return inputs.reshape(output_shape)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class LeakyReLU(Layer):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs):
        self.negative_mask = inputs < 0
        inputs_copy = inputs.copy()
        inputs_copy[self.negative_mask] *= self.negative_slope
        return inputs_copy

    def backward(self, grads):
        dx = np.ones_like(grads)
        dx[self.negative_mask] = self.negative_slope
        res = grads * dx
        return res


def softmax(x):
    # x: n * c
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, epsilon=1e-12):
        # inputs: n * c, targets: n * c
        self.inputs_sm = softmax(inputs)
        # prevent log(0)
        inputs_clipped = np.clip(self.inputs_sm, epsilon, 1 - epsilon)
        return -np.sum(targets * np.log(inputs_clipped)) / inputs.shape[0]

    def backward(self, inputs, targets):
        return (self.inputs_sm - targets) / inputs.shape[0]


if __name__ == '__main__':
    # fetch mnist data
    from urllib import request
    import gzip
    import os

    data_dir = "./data/"
    files = {"train_inputs": "train-images-idx3-ubyte.gz",
             "test_inputs": "t10k-images-idx3-ubyte.gz",
             "train_targets": "train-labels-idx1-ubyte.gz",
             "test_targets": "t10k-labels-idx1-ubyte.gz"}

    def download_mnist(data_dir, files):
        # download files to `data_dir` folder
        base_url = "http://yann.lecun.com/exdb/mnist/"
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        for name in files.values():
            if not os.path.exists(data_dir + name):
                print(f"Downloading {name} ...")
                request.urlretrieve(base_url + name, data_dir + name)

    def load_mnist(data_dir, files):
        with gzip.open(data_dir + files["train_inputs"], 'rb') as f:
            train_inputs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(data_dir + files["test_inputs"], 'rb') as f:
            test_inputs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(data_dir + files["train_targets"], 'rb') as f:
            train_targets = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open(data_dir + files["test_targets"], 'rb') as f:
            test_targets = np.frombuffer(f.read(), np.uint8, offset=8)
        return train_inputs, train_targets, test_inputs, test_targets

    download_mnist(data_dir, files)  # or manually download files to `data_dir`
    train_inputs, train_targets, test_inputs, test_targets = load_mnist(data_dir, files)
    # we need a smaller dataset, otherwise training is slow since no gpu is used
    train_inputs = train_inputs[:2000]
    train_targets = train_targets[:2000]
    test_inputs = test_inputs[:1000]
    test_targets = test_targets[:1000]
    # encode to one hot
    train_targets = np.eye(10)[train_targets]
    test_targets = np.eye(10)[test_targets]

    # model
    net = Sequential([
        Conv2d(1, 16, 3, 1),
        LeakyReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(16 * 14 * 14, 128),
        LeakyReLU(),
        Linear(128, 10)
    ])

    train(net, 500, BatchIterator(train_inputs, train_targets, 64), SGD(0.001), CrossEntropyLoss())
    # testing
    predicted = net(test_inputs)
    correct_nums = sum(np.argmax(predicted, axis=1) == np.argmax(test_targets, axis=1))
    print(f"testing accuracy is {correct_nums/test_inputs.shape[0]}")
