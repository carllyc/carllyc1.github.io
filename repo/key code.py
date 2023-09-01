#预测
class NumpyDataset(Dataset):  
    def __init__(self, data):  
        self.data = data  
    def __len__(self):  
        return len(self.data)  
    def __getitem__(self, idx):  
        return self.data[idx]  
train_dataset = NumpyDataset(stock_df_train)  
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
    batch_size=32,
    epochs=150
)
pred_y = model.predict(test_X)




#回测
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
startcash=10000.0
cerebro.broker.setcash(startcash)
cerebro.broker.setcommission(0.0002)
cerebro.addstrategy(MyStrategy)
cerebro.run