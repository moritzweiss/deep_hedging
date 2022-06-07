import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

m = 1  # dimension of price
d = 2  # number of layers in strategy
n = 32  # nodes in the first but last layers


class DenseLayer(nn.Module):

    def __init__(self, num_assets, nodes):
        super(DenseLayer, self).__init__()

        self.linear1 = torch.nn.Linear(in_features=num_assets, out_features=nodes)
        self.linear2 = torch.nn.Linear(in_features=nodes, out_features=num_assets)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class HedgeNetwork(nn.Module):
    def __init__(self, time_steps=10, num_assets=2, nodes=32):
        super(HedgeNetwork, self).__init__()
        self.dense_layers = \
            nn.ModuleList([DenseLayer(num_assets=num_assets, nodes=nodes) for _ in range(time_steps)])
        self.time_steps = time_steps

    def forward(self, x):
        # input a sample from the data set
        # x[0], x[1], ... , x[N], prices
        price = x[:, 0, :]
        hedge = torch.tensor(0.0)

        for j in range(self.time_steps):
            strategy = price
            strategy = self.dense_layers[j](strategy)
            pricenew = x[:, j + 1, :]  # dimensions = (batch, n_time_steps, n_assets)
            priceincr = torch.sub(pricenew, price)
            hedgenew = torch.mul(strategy, priceincr)
            hedgenew = torch.sum(hedgenew)  # mult = Lambda(lambda x : K.sum(x,axis=1))(mult)
            hedge = torch.add(hedge, hedgenew)  # building up the discretized stochastic integral
            price = pricenew
        return price, hedge


class Payoff(nn.Module):
    def __init__(self, strike=1.0):
        super(Payoff, self).__init__()
        self.strike = strike

    def forward(self, x, y):
        x = torch.max(x)
        x = torch.sub(x, self.strike)
        x = torch.relu(x)
        x = torch.sub(x, y)
        return x


class FullNetwork(nn.Module):
    def __init__(self, time_steps=10, num_assets=2, strike=1.0, nodes=32):
        super(FullNetwork, self).__init__()
        self.hn = HedgeNetwork(time_steps=time_steps, num_assets=num_assets, nodes=nodes)
        self.po = Payoff(strike=strike)

    def forward(self, x):
        x, y = self.hn(x)
        x = self.po(x, y)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

DL = HedgeNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(DL.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


Ktrain = 2 * 10 ** 4
Ktrain = 100
initialprice = 1.0
sigma = 0.2
N = 10
T = 1.0
m = 1

xtrain = ([initialprice * np.ones((Ktrain, m))] +
          [np.random.normal(-(sigma) ** 2 * T / (2 * N), sigma * np.sqrt(T) / np.sqrt(N), (Ktrain, m)) for i in
           range(N)])

ytrain = np.zeros((Ktrain, 1))

from torch.utils.data import Dataset


def GBMsimulator(So, mu, sigma, Cov, T, N):
    """
    Parameters
    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    dim = np.size(So)
    dt = T / N
    A = np.linalg.cholesky(Cov)
    S = []
    S.append(So)
    for _ in np.arange(start=1, stop=N + 1, dtype=int):
        drift = (mu - 0.5 * sigma ** 2) * dt
        Z = np.random.normal(0., 1., dim)
        diffusion = A.dot(Z) * np.sqrt(dt)
        S.append(S[-1] * np.exp(drift + diffusion))
    return np.vstack(S).astype(np.float32)


def generate_parameters(n_assets):
    correlation_matrix = np.diag(np.ones(n_assets))
    standard_deviations = np.ones(n_assets)*0.2
    covariance_matrix = np.diag(standard_deviations).dot(correlation_matrix).dot(np.diag(standard_deviations))
    s_0 = np.ones(shape=(n_assets,))
    drift = np.zeros(shape=(n_assets, ))
    return covariance_matrix, standard_deviations, drift, s_0


class GBMIncrementsDataset(Dataset):
    def __init__(self, n_samples=100, n_assets=1, T=1.0, time_steps=10):
        params = generate_parameters(n_assets=n_assets)
        self.data = [GBMsimulator(So=params[3], mu=params[2], sigma=params[1],
                                  Cov=params[0], T=T, N=time_steps) for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0.0


ds = GBMIncrementsDataset()
data_loader = DataLoader(ds, batch_size=20)
x, _ = next(iter(data_loader))

for n in range(data_loader.batch_size):
    plt.plot(x[n, :, 0])



FN = FullNetwork()
x = FN(x)
x = x.detach().numpy()

price, hedge = pred = DL(x)

x = x.numpy()


plt.show()

# plt.show()
print(1)
