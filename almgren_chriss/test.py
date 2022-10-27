# %%
import numpy as np 
from torch import nn 
import torch 
import matplotlib.pyplot as plt 
# create model for just one batch of data, no spread 

# choose paramters as in Cheridito's model, trade 3500 lots over 10 minutes
sigma = 3.3e-2 
N = 10
temp_impact = 5e-6
perm_impact = 2.5e-7
initial_price = 172
initial_inventory = 3500

# twap inventory 
twap_inventory = np.array([(N-n)/N for n in range(N+1)])

 
class Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=2, out_features=32)        
        self.linear2 = torch.nn.Linear(in_features=32, out_features=1)
        self.activation1 = torch.nn.Tanh()
        self.activation2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x


class ForwardPass(nn.Module):

    def __init__(self, N, initial_price, initial_inventory, policy, batch_size):
        super().__init__()
        self.N = N 
        self.initial_price = initial_price
        self.initial_inventory = initial_inventory
        self.inventories = []
        self.prices = []
        self.policy = policy  
        self.initial_price = torch.tensor(initial_price, dtype=torch.float32).expand((batch_size,1))
        self.initial_inventory = torch.tensor(initial_inventory, dtype=torch.float32).expand((batch_size, 1))

    def forward(self, noise):
        # noise has shape (batch_size, 5)
        inventory = self.initial_inventory
        price = self.initial_price
        profit = torch.zeros(size=inventory.shape) 
        for n in range(self.N+1):
            # 0,1,....,N, sell everything at N-1, such that inventory is zero at N-1 
            if n == N-1:
                profit = profit + inventory*(price - temp_impact*inventory)
                inventory = torch.zeros_like(inventory)
                self.inventories.append(inventory)
                break
            else: 
                time = torch.tensor(n/self.N).expand(self.initial_price.shape)
                state = torch.hstack([time, inventory/self.initial_inventory]) 
                action = self.policy(state)*inventory
                profit = profit + action*(price-temp_impact*action)
                # update states
                price = price + noise[:, n] - perm_impact*action            
                inventory = inventory - action 
                # log history 
                self.inventories.append(inventory)
                self.prices.append(price)
        with torch.no_grad():
            self.inventories = (torch.hstack(self.inventories)/self.initial_inventory).numpy()
            self.prices = torch.hstack(self.prices).numpy()
        # return implementation shortfall 
        return self.initial_inventory*self.initial_price-profit 
    
    def reset(self):
        self.inventories = [self.initial_inventory]
        self.prices= [self.initial_price]
        return None 


# test forward pass
batch_size = 32
policy = Policy()
forward_pass = ForwardPass(N=N, initial_price=initial_price, initial_inventory=initial_inventory, policy=policy, batch_size=batch_size) 
noise = np.random.normal(loc=0, scale=sigma, size=(batch_size, N, 1))
noise = torch.tensor(noise, dtype=torch.float32)
implementation_shortfall = forward_pass.forward(noise=noise)

# full training loop 
policy = Policy()
forward_pass = ForwardPass(N=N, initial_price=initial_price, initial_inventory=initial_inventory, policy=policy, batch_size=batch_size) 
optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
running_loss = 0 
for n in range(int(1e4)):      
    noise = np.random.normal(loc=0, scale=sigma, size=(batch_size, N, 1))
    noise = torch.tensor(noise, dtype=torch.float32)
    optimizer.zero_grad()
    outputs = forward_pass.forward(noise=noise)
    loss = torch.sum(outputs)/batch_size
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if n % 250 == 0:
        with torch.no_grad():
            print(policy.linear1.weight.grad[0, :])
            print(running_loss/(n+1))
    forward_pass.reset()


with torch.no_grad():
    noise = np.random.normal(loc=0, scale=sigma, size=(32, N, 1))
    noise = torch.tensor(noise, dtype=torch.float32)
    forward_pass.forward(noise=noise)
    inventories = forward_pass.inventories
    prices = forward_pass.prices
    plt.plot(range(N+1), inventories[0, :])
    plt.plot(range(N+1), twap_inventory)
    plt.legend(['deep learning solution', 'TWAP'])
    plt.title('learning twap in AC framework')
    plt.figure()
    for n in range(batch_size):
        plt.plot(range(N), prices[n, :])
    plt.show()
    print('done')

