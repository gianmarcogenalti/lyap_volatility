
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from lyap import lyapunov_coeff, sliding_window

#%%%%%%%%%%% stock choice %%%%%%%%%%%%%%%%%%
stock = 'GC=F'
df = pd.read_csv(stock+'.csv').dropna()#.loc[251:4265]
df.index = np.arange(len(df))
with open('goodness_'+stock+'.pkl', 'rb') as input:
    losses= pickle.load(input)
with open('lyapunovs_'+stock+'.pkl', 'rb') as input:
    l= pickle.load(input)

x =  list(range(len(l)))
x_ = list(map(str,list(l.keys())))

fig, ax = plt.subplots()
plt.xticks(x, x_, rotation = 45)
ax.xaxis.set_major_locator(ticker.AutoLocator())
plt.plot(x, list(l.values()))
plt.grid()
plt.title('Lyapunov Exponent of {}'.format(stock))
ax.set_xlabel('(L, m, q) choice')
ax.set_ylabel('λ')
ax.axhline(0, color = 'red')
plt.savefig('plots/'+stock+'_1.png')

fig, ax = plt.subplots()
plt.xticks(x, x_, rotation = 45)
ax.xaxis.set_major_locator(ticker.AutoLocator())
plt.plot(x, list(losses.values()))
plt.grid()
plt.title('FFNN Loss of {}'.format(stock))
ax.set_xlabel('(L, m, q) choice')
ax.set_ylabel('loss')
plt.savefig('plots/'+stock+'_2.png')

tf.random.set_seed(1)
np.random.seed(1)

num_epochs = 500
lr = 0.001
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience=2),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
opt = tf.keras.optimizers.Adam()
#%%%%%%%%%%% data scaling %%%%%%%%%%%%%%%%%%%
scaler = MinMaxScaler()
close = np.array(df['Close']).reshape(-1,1)
close = scaler.fit_transform(close).T.reshape(-1)
#close = (close/1000).T.reshape(-1)
#%%%%%%%%%%% metrics %%%%%%%%%%%%%%%%%%%%%%%%%

L,m,q = 2,10,4
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
model.fit(X, y, epochs=num_epochs, callbacks = my_callbacks, verbose = False)

fig, ax = plt.subplots()
plt.xticks(x, x_, rotation = 45)
ax.xaxis.set_major_locator(ticker.AutoLocator())
plt.plot(close[range(0,len(close),L)], label = 'True data')
plt.plot(model.predict(X),'k--', color = 'orange',label='FFNN')
legend = ax.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.grid()
plt.title('True Data vs FFNN fit of {}'.format(stock))
ax.set_xlabel('Days')
ax.set_ylabel('Closing Price')
plt.savefig('plots/'+stock+'_3.png')
