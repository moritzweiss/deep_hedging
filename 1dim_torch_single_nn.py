import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm


# black scholes price 
def BS(S0, strike, T, sigma):
    return S0*norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))

# neural network implementation
class DenseLayer(nn.Module):

    def __init__(self, num_assets=1, nodes=32):
        super(DenseLayer, self).__init__()

        self.linear1 = torch.nn.Linear(in_features=num_assets+1, out_features=nodes, bias=True)
        self.linear2 = torch.nn.Linear(in_features=nodes, out_features=num_assets, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class ForwardPass(nn.Module):
    def __init__(self, strike, bs_price, time_steps=10, num_assets=1, nodes=32):
        super(ForwardPass, self).__init__()
        self.dense_layer = DenseLayer()
        self.time_steps = time_steps
        self.strike = strike
        self.bs_price = bs_price


    def forward(self, x):
        # input a sample from the data set
        # x[0], x[1], ... , x[N], prices
        price = x[:, 0, :] # (batch, 1)
        profit = torch.zeros_like(price) # (batch, 1)

        for j in range(self.time_steps):
            time = (j/self.time_steps)*torch.ones_like(price)
            state = torch.hstack([time, price])
            strategy = self.dense_layer(state)
            pricenew = x[:, j + 1]  # dimensions = (batch, n_time_steps, n_assets)
            priceincr = torch.sub(pricenew, price)
            profit = profit + torch.mul(strategy, priceincr)
            price = pricenew

        price = torch.sub(price, self.strike)
        price = torch.relu(price) 
        price = torch.sub(price, profit) # subtract hedge from the price 
        price = torch.sub(price, self.bs_price) # broadcasted to common shape         
        return price


def generate_gbm(So, sigma, T, N, batch_size=32):
    data = []
    dt = T / N
    for _ in range(batch_size):
        S = []
        S.append(So)    
        for _ in np.arange(start=1, stop=N + 1, dtype=int):
            drift = (- 0.5 * sigma ** 2) * dt
            Z = np.random.normal(0., 1., 1)
            diffusion = sigma*np.sqrt(dt)*Z
            S.append(S[-1] * np.exp(drift + diffusion))
        data.append(np.vstack(S).astype(np.float32).reshape(1,N+1,1))
    return np.vstack(data) 


# parameters 
S0=1 
strike=1 
T=1.0 
sigma=0.2 
n_assets = 1 
batch_size = 32 
N=20
sigma=0.2
bs_price=BS(S0,strike,T,sigma)

# test forward pass 
x = generate_gbm(sigma=sigma, T=1.0, N=N, So=S0)
model = ForwardPass(time_steps=N, strike=1.0, bs_price=bs_price)
x = torch.tensor(x)
output = model(x) 
loss = torch.sum(torch.pow(output, 2))/batch_size


# training loop 
model = ForwardPass(time_steps=N, strike=1.0, bs_price=bs_price)
optimizer = torch.optim.Adam(model.parameters())
running_loss = 0 
for epoch in range(int(1e4)):  
    optimizer.zero_grad()
    x = generate_gbm(sigma=sigma, T=1.0, N=N, So=1.0, batch_size=batch_size)
    x = torch.tensor(x)
    output = model(x) 
    loss = torch.sum(torch.pow(output, 2))/batch_size
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if epoch % 100 == 0: 
        print(f'[{epoch + 1}] loss: {running_loss / (epoch+1):.4E}')
        running_loss = 0.0

print('Finished Training')


# hedging strategy at time 10
batch_size=1
price_range = np.arange(start=0.5, stop=1.5, step=0.1)
hedge = []
for price in price_range:
    time = torch.ones(batch_size, 1)*0.5
    price = torch.ones_like(time)*price
    state = torch.hstack([time, price])
    x = model.dense_layer(state)
    hedge.append(x.detach().numpy().reshape(1))

k=10
z=norm.cdf((np.log(price_range/strike)+0.5*(T-k*T/N)*sigma**2)/(np.sqrt(T-k*T/N)*sigma))
plt.plot(price_range,hedge,price_range,z)
plt.legend(['deep hedge', 'model hedge'])
plt.show()








