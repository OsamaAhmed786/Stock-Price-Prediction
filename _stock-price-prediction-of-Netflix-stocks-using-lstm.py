import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import L
np.random.seed(7)

# data is of netflix from date-(1-aug-2003)_to_(28-aug-2020) from yahoo finance
df = pd.read_csv(r'C:\Users\admin\Desktop\NSE-TATAGLOBAL11.csv', header=0)
df = df.sort_index(ascending=True, axis=0)
df.head()


df.shape

df.describe()


df.isnull().any()


# # Time series graph of data

# In[7]:


fig = px.line(df, x='Date', y='Close')
fig.show()


# In[8]:


# Taking diff indicators for prediction
# ohlc_avg is the average of open, high, low, close values
# hlc_avgs is the average of high, low, close value
# we will take only ohlc_avg data only in whole nb 
ohlc_data = df.iloc[:, 1:5]
ohlc_avg = ohlc_data.mean(axis=1)
hlc_avg = df[['High', 'Low', 'Close']].mean(axis=1)
close = df.Close


# In[9]:


fig1 = go.Figure()

fig1.add_trace(go.Scatter(x = df.index, y = ohlc_avg,
                  name='OHLC avg'))
fig1.add_trace(go.Scatter(x = df.index, y = hlc_avg,
                  name='HLC avg'))
fig1.add_trace(go.Scatter(x = df.index, y = close,
                  name='close column data'))
fig1.show()


# In[10]:


if not os.path.exists("images"):
    os.mkdir("images")


# In[12]:


# we will create a new df which has only 2 column which is useful to predict data
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'ohlc_avg'])
for i in range(0, len(df)):
  new_data['Date'][i] = df['Date'][i]
  new_data['ohlc_avg'][i] = ohlc_avg[i]


# In[13]:


new_data.head()


# In[14]:


# setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)


# In[15]:


print(len(new_data))


# In[16]:


ds = new_data.values


# In[17]:


train = int(len(new_data)*0.8)
test = len(new_data) - train
train, test = new_data.iloc[0:train,:], new_data.iloc[train:len(new_data),:]


# In[18]:


train.shape


# In[19]:


test.shape


# # Normalizing data

# In[20]:


# we have normalize the data cuz data is like 149...., 488..something like that
# so we have to normalize betwwen 0 and 1
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(ds)


# 
# # splitting the data into x_train, y_train

# In[21]:


# splitting the data to x_train, y_train
# we will first train upto 60 and then predict on 61 and then 
# we will train from 61 to 120 then predict on 121 likewise we will go
x_train, y_train = [], []
for i in range(60, len(train)):
  x_train.append(scaled_data[i-60:i,0])
  y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)


# In[22]:


# now we have reshape the array to 3-d to pass the data into lstm [number of samples, time steps/batch_size, features] 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[23]:


x_train.shape


# # Modelling

# In[24]:


# create and fit the lstm network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.25))
model.add(LSTM(units=50))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[25]:


model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=1)


# # Prediction

# In[26]:


# predicting 920 values, using past 60 from the train data
inputs = new_data[len(new_data)-len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scalar.transform(inputs)


# In[27]:


inputs.shape


# In[28]:


x_test = []
for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)


# In[29]:


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[30]:


predicted_price = model.predict(x_test)
# inverse transform for getting back all normal values from scaled values
predicted_price = scalar.inverse_transform(predicted_price)


# In[31]:


rms=np.sqrt(np.mean(np.power((test-predicted_price),2)))
rms


# In[32]:


# create a new column of predicted values
test['Prediction'] = predicted_price
test.head()


# In[33]:


# Graph for comparing the results of model predicted and original value
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x = train.index, y = train.ohlc_avg,
                  name='train'))
fig2.add_trace(go.Scatter(x = test.index, y = test.ohlc_avg,
                  name='test_ohlc_avg'))
fig2.add_trace(go.Scatter(x = test.index, y = test.Prediction,
                  name='test'))
fig2.show()


# In[34]:


fig3 = px.line(df, x='Date', y='Close')
fig3.show()


# In[ ]:




