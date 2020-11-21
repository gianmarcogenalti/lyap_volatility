import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lyap import lyapunov_coeff, sliding_window
#%%%%%%%%%%% stock choice %%%%%%%%%%%%%%%%%%
stock = 'VIX'
df = pd.read_csv(stock+'.csv').loc[251:4265]
df.index = np.arange(len(df))
#%%%%%%%%%%% NN parameters %%%%%%%%%%%%%%%%%%%
tf.random.set_seed(17)
np.random.seed(17)

num_epochs = 500
lr = 0.01
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience=2),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
opt = tf.keras.optimizers.Adam(learning_rate = lr)
#%%%%%%%%%%% data scaling %%%%%%%%%%%%%%%%%%%
scaler = MinMaxScaler()
close = np.array(df['Close']).reshape(-1,1)
close = np.divide(close,100).T.reshape(-1)#scaler.fit_transform(close).T.reshape(-1)
#%%%%%%%%%%% metrics %%%%%%%%%%%%%%%%%%%%%%%%%
models_goodness = dict()
est_lyapunov = dict()
# Dimensionality parameters
L = 1  # step granularity
m = 4  # window size
q = 2   # number of neurons

print((L,m,q))
X, y = sliding_window(close, m, L)
model = Sequential([
  Dense(q, activation='tanh', input_shape=(m,)),
  Dense(1),
])

model.compile(
  optimizer=opt,
  loss='mean_squared_error',
  metrics=['mse']
)

model.fit(X, y, epochs=num_epochs, verbose = False, callbacks = my_callbacks)
models_goodness.update({(L,m,q) : model.evaluate(X, y)[1]})

alpha_0 = model.layers[1].get_weights()[1]
alpha_1 = model.layers[1].get_weights()[0]
beta_0 = model.layers[0].get_weights()[1]
beta_1 = np.asmatrix(model.layers[0].get_weights()[0])
est_lyapunov.update({(L,m,q) : lyapunov_coeff(q, m, L, close, alpha_0, alpha_1, beta_0, beta_1)})


print(est_lyapunov)
print(models_goodness)
