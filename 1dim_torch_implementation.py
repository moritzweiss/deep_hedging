import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.stats import norm
# from torchinfo import summary 
from torchsummary import summary 
from scipy.stats import norm

# parameters 
N=10 # time disrectization
S0=1 # initial value of the asset
strike=1 # strike for the call option 
T=1.0 # maturity
sigma=0.2 # volatility in Black Scholes


# black scholes price 
def BS(S0, strike, T, sigma):
    return S0*norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))
priceBS=BS(S0,strike,T,sigma)
print('Price of a Call option in the Black scholes model with initial price', S0, 'strike', strike, 'maturity', T , 'and volatility' , sigma, 'is equal to', BS(S0,strike,T,sigma))


# neural network implementation

# dense layer 
class DenseLayer(nn.Module):

    def __init__(self, num_assets=1, nodes=32):
        super(DenseLayer, self).__init__()

        self.linear1 = torch.nn.Linear(in_features=num_assets, out_features=nodes, bias=True)
        self.linear2 = torch.nn.Linear(in_features=nodes, out_features=num_assets, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class HedgeNetwork(nn.Module):
    def __init__(self, time_steps=10, num_assets=1, nodes=32):
        super(HedgeNetwork, self).__init__()
        self.dense_layers = \
            nn.ModuleList([DenseLayer(num_assets=num_assets, nodes=nodes) for _ in range(time_steps)])
        self.time_steps = time_steps

    def forward(self, x):
        # input a sample from the data set
        # x[0], x[1], ... , x[N], prices
        price = x[:, 0, :] # (batch, 1)
        hedge = torch.zeros_like(price) # (batch, 1)

        for j in range(self.time_steps):
            strategy = price
            strategy = self.dense_layers[j](strategy)
            pricenew = x[:, j + 1, :]  # dimensions = (batch, n_time_steps, n_assets)
            priceincr = torch.sub(pricenew, price)
            hedgenew = torch.mul(strategy, priceincr)
            hedge = torch.add(hedge, hedgenew)  # building up the discretized stochastic integral
            price = pricenew
        return price, hedge


class PayOff(nn.Module):
    def __init__(self, bs_price, strike=1.0):
        super(PayOff, self).__init__()
        self.strike = strike
        self.bs_price = bs_price

    def forward(self, price, hedge):
        price = torch.sub(price, self.strike)
        price = torch.relu(price) 
        price = torch.sub(price, hedge) # subtract hedge from the price 
        price = torch.sub(price, self.bs_price) # broadcasted to common shape 
        return price


class Model(nn.Module):
    def __init__(self, bs_price, time_steps=10, strike=1.0, nodes=32, num_assets=1):
        super(Model, self).__init__()
        self.hn = HedgeNetwork(time_steps=time_steps, num_assets=num_assets, nodes=nodes)
        self.po = PayOff(strike=strike, bs_price=bs_price)

    def forward(self, x):
        x, y = self.hn(x)
        x = self.po(x, y)
        return x

## data set implementation

def generate_gbm(So, sigma, T, N):
    dt = T / N
    S = []
    S.append(So)
    for _ in np.arange(start=1, stop=N + 1, dtype=int):
        drift = (- 0.5 * sigma ** 2) * dt
        Z = np.random.normal(0., 1., 1)
        diffusion = sigma*np.sqrt(dt)*Z
        S.append(S[-1] * np.exp(drift + diffusion))
    return np.vstack(S).astype(np.float32)


class GBMDataSet(Dataset):
    def __init__(self, n_samples=100, T=1.0, time_steps=N, sigma=0.2):
        self.data = [generate_gbm(So=S0, sigma=0.2, T=T, N=time_steps) for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.zeros(1)


# test data loader 
n_assets = 1 
batch_size = 32 
N=20
sigma=0.2
n_samples = int(100)
ds = GBMDataSet(n_samples=n_samples, T=T, time_steps=N, sigma=sigma)
data_loader = DataLoader(ds, batch_size=batch_size)
x, y = next(iter(data_loader))

## plot 
# for n in range(data_loader.batch_size):
#     plt.plot(x[n, :, 0])

# test hedge network 
model = HedgeNetwork(time_steps=N)
price, hedge = model(x) 
print(summary(model, input_size=(21,1), batch_size=256))
print(x[:,-1,:] == price)

# tests payoff network  
model = PayOff(bs_price=priceBS, strike=strike)
output = model(price, hedge)

# test full model  
model = Model(bs_price=priceBS, time_steps=N, num_assets=1, strike=strike, nodes=32)
output = model(x)
print(summary(model, input_size=(21,1), batch_size=256))


# data set
n_samples = int(2e4)
# n_samples = int(1e2)
batch_size = 32
ds = GBMDataSet(n_samples=n_samples, T=T, time_steps=N, sigma=sigma)
data_loader = DataLoader(ds, batch_size=batch_size)

# device 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# training loop 
model = Model(bs_price=priceBS, time_steps=N, num_assets=1, strike=strike, nodes=32)
model.to(device)
model.train(True)
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'[{epoch + 1}] loss: {running_loss / n_samples:.4E}')
    running_loss = 0.0

print('Finished Training')
print(1)


# hedging strategy 
s_range = np.arange(start=0.5, stop=1.5, step=0.1)
hh = []
for s in s_range:
    x = model.hn.dense_layers[10](torch.ones(1)*s)
    hh.append(x.detach().numpy())

# 
k=10
z=norm.cdf((np.log(s_range/strike)+0.5*(T-k*T/N)*sigma**2)/(np.sqrt(T-k*T/N)*sigma))

plt.plot(s_range,hh,s_range,z)
plt.legend(['deep hedge', 'model hedge'])
plt.show()

print(1)







