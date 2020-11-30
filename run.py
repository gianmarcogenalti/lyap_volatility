import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from lyap import lyapunov_coeff, sliding_window

#%%%%%%%%%%% stock choice %%%%%%%%%%%%%%%%%%%%
stock = 'VIX' # make sure that stock.csv is in the folder stocks_dfs
df = pd.read_csv('stocks_dfs/'+stock+'.csv').dropna()
df.index = np.arange(len(df))
#%%%%%%%%%%% NN parameters %%%%%%%%%%%%%%%%%%%
tf.random.set_seed(1) # for experiment reproducibility
np.random.seed(1)

num_epochs = 500 # number of epochs
lr = 0.001 # learning rate
my_callbacks = [ # callbacks EarlyStopping and logs
    tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience=2),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
opt = tf.keras.optimizers.Adam() # optimizer Adam
#%%%%%%%%%%% data scaling %%%%%%%%%%%%%%%%%%%
scaler = MinMaxScaler() # we scale the data from 0 to 1 with minmax scaler
close = np.array(df['Close']).reshape(-1,1) # we consider the closing price of the stock
close = scaler.fit_transform(close).T.reshape(-1)
#close = (close/100).T.reshape(-1)
#%%%%%%%%%%% metrics %%%%%%%%%%%%%%%%%%%%%%%%%
models_goodness = dict() # for every combination (L,m,q) we will store here the FFNN loss
est_lyapunov = dict() # for every combination (L,m,q) we will store here estimated lyapunov coefficient

#%%%%%%%% Dimensionality parameters %%%%%%%%%%
L_ = [1,2]  # step granularity
m_ = [4,6,8,10]  # window size
q_ = [1,2,4,8,12]   # number of neurons

#%%%%%%%%% FFNN tuning and lyapunov coeff estimation %%%%%%%%
for L in L_:
    for q in q_:
        for m in m_:
            print((L,m,q))
            X, y = sliding_window(close, m, L) # we parse the time series in sliding windows format
            model = Sequential([ # single hidden-layer FFNN
              Dense(q, activation='tanh', input_shape=(m,)),
              Dense(1),
            ])

            model.compile(  # we compile the model using as metric the mean squared error
              optimizer=opt,
              loss='mean_squared_error',
              metrics=['mse']
            )

            model.fit(X, y, epochs=num_epochs, callbacks = my_callbacks, verbose = False) # model is fitted on the data
            models_goodness.update({(L,m,q) : model.evaluate(X, y)[1]}) # loss is stored
            print('Loss: {}'.format(models_goodness[(L,m,q)]))
            alpha_0 = model.layers[1].get_weights()[1] # we extract the FFNN weights
            alpha_1 = model.layers[1].get_weights()[0]
            beta_0 = model.layers[0].get_weights()[1]
            beta_1 = np.asmatrix(model.layers[0].get_weights()[0])
            l =  lyapunov_coeff(q, m, L, close, alpha_0, alpha_1, beta_0, beta_1) # compute the lyapunov coefficient of the series
            if isinstance(l, complex):
                l = l.real
            est_lyapunov.update({(L,m,q) : l})
            print('Lyapunov: {}'.format(est_lyapunov[(L,m,q)]))

#%%%%%%%%%%%%% Output storage %%%%%%%%%%%%%%%%%%%

with open('lyapunov_coefficients/lyapunovs_'+stock+'.pkl', 'wb') as f: # we store the lyapunov coefficients in pickles
    pickle.dump(est_lyapunov,f)
with open('models_losses/goodness_'+stock+'.pkl', 'wb') as f: # we store the model losses in pickles
    pickle.dump(models_goodness,f)
print(est_lyapunov)
print(models_goodness)
