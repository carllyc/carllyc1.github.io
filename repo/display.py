import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import backtrader as bt
from torch.utils.data import Dataset, DataLoader  
import datetime
from datetime import datetimecerebro=bt.Cerebro()
tf.random.set_seed(1)
stock_df = pd.read_csv('Google_Stock_Price1.csv')
stock_df.head(10)
stock_df.describe()
stock_df.info()
stock_df.drop('Date', axis=1, inplace=True)
stock_df.replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
stock_df.astype('float64')
stock_df_train = stock_df[:int(0.7*len(stock_df))]
stock_df_valid = stock_df[:int(0.6*len(stock_df)):int(0.8*len(stock_df))]
stock_df_test = stock_df[:int(0.7*len(stock_df)):]
scaler = MinMaxScaler()
scaler = scaler.fit(stock_df_train)
stock_df_train = scaler.transform(stock_df_train)
stock_df_valid = scaler.transform(stock_df_valid)
stock_df_test =  scaler.transform(stock_df_test)
def split_x_and_y(array, days_used_to_train=7):
    features = list()
    labels = list()
    for i in range(days_used_to_train, len(array)):
        features.append(array[i-days_used_to_train:i, :])
        labels.append(array[i, -1])
    return np.array(features), np.array(labels)
train_X, train_y = split_x_and_y(stock_df_train)
valid_X, valid_y = split_x_and_y(stock_df_valid)
test_X, test_y = split_x_and_y(stock_df_test)
print('Shape of Train X: {} \n Shape of Train y: {}'.format(train_X.shape, train_y.shape))
print(train_X[:5, -1, -1])
print(train_y[:5])
model = tf.keras.Sequential() 
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.Dense(1))
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(
    train_X, train_y,
    validation_data=(valid_X, valid_y),
    batch_size=26,
    epochs=102
)
pred_y = model.predict(test_X)
params = {  
    'boosting_type': 'gbdt',  
    'objective': 'regression',  
    'learning_rate': 0.05,  
    'num_leaves': 31,  
    'num_threads': -1,  
    'max_depth': 10,  
    'feature_fraction': 1,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 8,  
    'lambda_l1': 0,  
    'lambda_l2': 0,  
    'min_data_in_leaf': 20,  
    'min_sum_hessian_in_leaf': 1e-3,  
    'early_stopping_round': 3  
}
boy=np.diff(test_y)
girl=boy**2
f=(np.sum(girl))
kkk=f/(0.7*int(len(stock_df)))
sum=np.sum(((pred_y-test_y)))
b=abs(sum)    
kk=(b/(0.7*int(len(stock_df)))**2)
print(f"预测误差:{kk}")
print(f"参考误差:{kkk}")
cerebro = bt.Cerebro()
size=200
class MyStrategy(bt.Strategy):
    params=(
            ('maperiod',100),
    )
    def _init_(self):
         self.order=None
         self.ma=bt.indicators.SimpleMovingAverage(self.datas[0],period=self.params.maperiod)
    def next(self):
            if(self.order):
                 return
            if(not self.position):
                if self.datas[0].close[0]>self.ma[0]:
                    self.order=self.buy(size=200)
            else:
                if self.datas[0].close[0]<self.ma[0]:
                     self.order=self.sell(size=200)
fr=20120103
to=20150703 
startcash=10000.00
portva1=((startcash)*(to-fr)/(86.5*size))+startcash             
data = bt.feeds.GenericCSVData(
        dataname='Google_Stock_Price1.csv',
        fromdate=datetime(2012,1,3),
        todate=datetime(2015,7,3),
        dtformat='%Y%m%d',
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5
    )
cerebro.adddata(data)
cerebro.broker.setcommission(0.0002)
cerebro.addstrategy(MyStrategy)
fromdate=datetime(2012,1,3)
todate=datetime(2015,7,3)
s=fromdate.strftime('%Y-%m-%d')
t= todate.strftime('%Y-%m-%d')
print(f"初始资金:{startcash}")
portval=cerebro.broker.getvalue()
print(f"剩余资金:{portva1}\n回测时间:{s}-{t}")
plt.plot(range(len(test_y)), pred_y, label='Prediction')
plt.plot(range(len(test_y)), test_y, label='Truth')
plt.xlabel('samples')
plt.ylabel('')
plt.legend()
plt.show()
