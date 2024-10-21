"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        middle = self.layer1.forward(x)
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size
        print(f"x: {self.bias.value.shape}")

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        print(
            f"multiplying x: {x.shape} ({x}) and weights: {self.weights.value.shape} ({self.weights.value})"
        )

        def conform_shape(data, weights):
            data_index = len(data) - 1
            weights_index = len(weights) - 1
            result = []
            while data_index >= 0 and weights_index >= 0:
                if data[data_index] == weights[weights_index]:
                    result.append(data[data_index])
                elif data[data_index] == 1:
                    result.append(weights[weights_index])
                elif weights[weights_index] == 1:
                    result.append(data[data_index])
                else:
                    result.append(1)
                data_index -= 1
                weights_index -= 1

            return tuple(result)

        x_view = x.view(*conform_shape(x.shape, self.weights.value.shape))
        out = x.backend.mul_zip(x_view, self.weights.value)
        out = out.backend.add_zip(out, self.bias.value)
        return out


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
