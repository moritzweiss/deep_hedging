import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Concatenate, Dropout, Subtract, \
                        Flatten, MaxPooling2D, Multiply, Lambda, Add, Dot
from keras.backend import constant
from keras import optimizers

from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
from keras import initializers
from keras.constraints import max_norm
import keras.backend as K

import matplotlib.pyplot as plt


N=20 # time disrectization
S0=1 # initial value of the asset
strike=1 # strike for the call option
T=1.0 # maturity
sigma=0.2 # volatility in Black Scholes
R=10 # number of Trajectories

logS= np.zeros((N,R))
logS[0,]=np.log(S0)*np.ones((1,R))

for i in range(R):
    for j in range(N-1):
        increment = np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N))
        logS[j+1,i] =logS[j,i]+increment

S=np.exp(logS)

# for i in range(R):
#    plt.plot(S[:,i])
# plt.show()


import scipy.stats as scipy
from scipy.stats import norm

#Blackscholes price

def BS(S0, strike, T, sigma):
    return S0*scipy.norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*scipy.norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))

priceBS=BS(S0,strike,T,sigma)
print('Price of a Call option in the Black scholes model with initial price', S0, 'strike', strike, 'maturity', T , 'and volatility' , sigma, 'is equal to', BS(S0,strike,T,sigma))


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
    dt = T/N
    A = np.linalg.cholesky(Cov)
    S = []
    S.append(So)
    for _ in np.arange(start=1, stop=N+1, dtype=int):
        drift = (mu - 0.5 * sigma**2) * dt
        Z = np.random.normal(0., 1., dim)
        diffusion = A.dot(Z) * np.sqrt(dt)
        S.append(S[-1]*np.exp(drift + diffusion))
    return np.vstack(S)


correlation_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
standard_deviations = np.array([0.2, 0.2])
covariance_matrix = np.diag(standard_deviations).dot(correlation_matrix).dot(np.diag(standard_deviations))
s_0 = np.ones(shape=(2,))


# for _ in range(R):
#     s = GBMsimulator(So=s_0, mu=np.zeros(shape=(2,)), Cov=covariance_matrix, T=T, N=20, sigma=standard_deviations)
#     plt.plot(s[:, 0])
# plt.show()


m = 1
# dimension of price
d = 2 # number of layers in strategy
n = 32  # nodes in the first but last layers

# architecture is the same for all networks
layers = []
for j in range(N):
    for i in range(d):
        if i < d-1:
            nodes = n
            layer = Dense(nodes, activation='tanh',trainable=True,
                      kernel_initializer=initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                      bias_initializer='random_normal',
                      name=str(i)+str(j))
        else:
            nodes = m
            layer = Dense(nodes, activation='linear', trainable=True,
                          kernel_initializer=initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                          bias_initializer='random_normal',
                          name=str(i)+str(j))
        layers = layers + [layer]


class PayOff(tf.keras.layers.Layer):
    def __init__(self, strike=1.0, bs_price=1.0):
        super(PayOff, self).__init__()
        self.strike = 1.0
        self.bs_price = tf.constant(1.0)

    def call(self, inputs):
        # strike = tf.fill(tf.shape(inputs), self.strike)
        # outputs = tf.keras.layers.subtract([inputs, strike])
        outputs = tf.math.subtract(inputs, self.strike)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.math.reduce_max(outputs, keepdims=True)
        outputs = tf.math.subtract(outputs, self.bs_price)
        return outputs


### define model


price = Input(shape=(m,))
hedge = Input(shape=(m,))

inputs = [price]+[hedge]


for j in range(N):
    strategy = price
    for k in range(d):
        strategy= layers[k+(j)*d](strategy) # strategy at j is the hedging strategy at j , i.e. the neural network g_j
    incr = Input(shape=(m,))
    logprice= Lambda(lambda x : K.log(x))(price)
    logprice = Add()([logprice, incr])
    pricenew=Lambda(lambda x : K.exp(x))(logprice)# creating the price at time j+1
    priceincr=Subtract()([pricenew, price])
    hedgenew = Multiply()([strategy, priceincr])
    #mult = Lambda(lambda x : K.sum(x,axis=1))(mult) # this is only used for m > 1
    hedge = Add()([hedge,hedgenew]) # building up the discretized stochastic integral
    inputs = inputs + [incr]
    price = pricenew

# payoff= Lambda(lambda x : 0.5*(K.abs(x-strike)+x-strike) - priceBS)(price)
payoff = tf.math.subtract(price, strike)
payoff = tf.keras.activations.relu(payoff)
payoff = tf.math.reduce_max(payoff)
#PO = PayOff()
# payoff = PO(price)price
outputs = Subtract()([payoff, hedge]) # payoff minus priceBS minus hedge

inputs = inputs
outputs= outputs

model_hedge = Model(inputs=inputs, outputs=outputs)




Ktrain = 2*10**4
# Ktrain = 10
initialprice = S0

# xtrain consists of the price S0,
#the initial hedging being 0, and the increments of the log price process

# create training set
# initial hedge is zero
# x_train = [np.zeros((Ktrain,m))]

# x_train += [GBMsimulator(So=s_0, mu=)]

xtrain = ([initialprice*np.ones((Ktrain,m))] +
          [np.zeros((Ktrain,m))]+
          [np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktrain,m)) for i in range(N)])

ytrain = np.zeros((Ktrain, 1))

model_hedge.compile(optimizer='adam',loss='mean_squared_error')

for i in range(50):
    model_hedge.fit(x=xtrain, y=ytrain, epochs=1, verbose=True)
    # plt.hist(model_hedge.predict(xtrain))
    # plt.show()
    # print(np.mean(model_hedge.predict(xtrain)))


print(1)











