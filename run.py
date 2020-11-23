import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from lyap import lyapunov_coeff, sliding_window

#%%%%%%%%%%% stock choice %%%%%%%%%%%%%%%%%%
stock = 'VIX'
df = pd.read_csv(stock+'.csv').dropna()#.loc[251:4265]
df.index = np.arange(len(df))
#%%%%%%%%%%% NN parameters %%%%%%%%%%%%%%%%%%%
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
#close = (close/100).T.reshape(-1)
#%%%%%%%%%%% metrics %%%%%%%%%%%%%%%%%%%%%%%%%
models_goodness = dict()
est_lyapunov = dict()
# Dimensionality parameters
L_ = [1,2]  # step granularity
m_ = [4,6,8,10]  # window size
q_ = [4,8,12]   # number of neurons
for L in L_:
    for q in q_:
        for m in m_:
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

            model.fit(X, y, epochs=num_epochs, callbacks = my_callbacks, verbose = False)
            models_goodness.update({(L,m,q) : model.evaluate(X, y)[1]})
            print('Loss: {}'.format(models_goodness[(L,m,q)]))
            alpha_0 = model.layers[1].get_weights()[1]
            alpha_1 = model.layers[1].get_weights()[0]
            beta_0 = model.layers[0].get_weights()[1]
            beta_1 = np.asmatrix(model.layers[0].get_weights()[0])
            l =  lyapunov_coeff(q, m, L, close, alpha_0, alpha_1, beta_0, beta_1)
            if isinstance(l, complex):
                l = l.real
            est_lyapunov.update({(L,m,q) : l})
            print('Lyapunov: {}'.format(est_lyapunov[(L,m,q)]))

with open('lyapunovs_'+stock+'.pkl', 'wb') as f:
    pickle.dump(est_lyapunov,f)
with open('goodness_'+stock+'.pkl', 'wb') as f:
    pickle.dump(models_goodness,f)
print(est_lyapunov)
print(models_goodness)

plt.figure()
plt.plot(close)
plt.plot(model.predict(X))
plt.title(stock)
plt.show()
