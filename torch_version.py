import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.stats import norm


def BS(S0, strike, T, sigma):
    return S0*norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))


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
            hedgenew = torch.sum(hedgenew, dim=1, keepdim=True)  # mult = Lambda(lambda x : K.sum(x,axis=1))(mult)
            hedge = torch.add(hedge, hedgenew)  # building up the discretized stochastic integral
            price = pricenew
        return price, hedge


class Payoff(nn.Module):
    def __init__(self, bs_price, strike=1.0):
        super(Payoff, self).__init__()
        self.strike = strike
        self.bs_price = bs_price

    def forward(self, price, hedge):
        price, _ = torch.max(price, dim=1, keepdim=True)
        price = torch.sub(price, self.strike)
        price = torch.relu(price)
        price = torch.sub(price, hedge)
        price = torch.sub(price, self.bs_price)
        return price


class FullNetwork(nn.Module):
    def __init__(self, bs_price, time_steps=10, num_assets=2, strike=1.0, nodes=32):
        super(FullNetwork, self).__init__()
        self.hn = HedgeNetwork(time_steps=time_steps, num_assets=num_assets, nodes=nodes)
        self.po = Payoff(strike=strike, bs_price=bs_price)

    def forward(self, x):
        x, y = self.hn(x)
        x = self.po(x, y)
        return x


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        y = torch.zeros_like(pred)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(batch)
            print(loss)


def generate_gbm(So, mu, sigma, Cov, T, N):
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
    return {'cov': covariance_matrix, 'std': standard_deviations, 'drift': drift, 'initial_price': s_0}


class GBMDataSet(Dataset):
    def __init__(self, n_samples=100, n_assets=1, T=1.0, time_steps=10):
        params = generate_parameters(n_assets=n_assets)
        self.data = [generate_gbm(So=params['initial_price'], mu=params['drift'], sigma=params['std'],
                                  Cov=params['cov'], T=T, N=time_steps) for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.zeros(1)



n_assets = 1 
batch_size = 256 
ds = GBMDataSet(n_samples=int(1e5), n_assets=n_assets)
data_loader = DataLoader(ds, batch_size=batch_size)
x, y = next(iter(data_loader))

# for n in range(data_loader.batch_size):
#     plt.plot(x[n, :, 0])

priceBS=BS(1.0, 1.0, 1.0, 0.2)

FN = FullNetwork(num_assets=n_assets, bs_price=priceBS)
x = FN(x)
x = x.detach().numpy()



# training 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

DL = HedgeNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(DL.parameters(), lr=1e-3)

for epoch in range(10):
    FN.train(True)
    train(dataloader=data_loader, model=FN, loss_fn=loss_fn, optimizer=optimizer, device=device)


# plt.show()
print(1)
