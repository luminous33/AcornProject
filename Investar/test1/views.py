from django.shortcuts import render
import io
import urllib, base64
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import Analyzer

def home(request):
    name=request.GET['name']
    mk = Analyzer.MarketDB()
    raw_df = mk.get_daily_price(name, '2018-05-04', '2020-09-01')

    window_size = 10 
    data_size = 5

    def MinMaxScaler(data):
        """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
        return numerator / (denominator + 1e-7)

    dfx = raw_df[['open','high','low','volume', 'close']]
    dfx = MinMaxScaler(dfx)   # 삼성전자 OHLVC정보를  MinMaxScaler()를 이용하여 0~1사이 값으로 변환
    dfy = dfx[['close']]      # dfx는 OHLVC가격정보.  dfy는 종가정보.

    x = dfx.values.tolist()
    y = dfy.values.tolist()



    # 데이터셋 준비하기
    data_x = []
    data_y = []
    for i in range(len(y) - window_size):
        _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
        _y = y[i + window_size]     # 다음 날 종가
        data_x.append(_x)
        data_y.append(_y)
    print(_x, "->", _y)



    # 훈련용데이터셋
    train_size = int(len(data_y) * 0.7)
    train_x = np.array(data_x[0 : train_size])
    train_y = np.array(data_y[0 : train_size])



    #테스트용 데이터셋
    test_size = len(data_y) - train_size
    test_x = np.array(data_x[train_size : len(data_x)])
    test_y = np.array(data_y[train_size : len(data_y)])




    # 모델 생성
    model = Sequential()
    model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, data_size)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_x, train_y, epochs=60, batch_size=30)
    pred_y = model.predict(test_x)

    # Visualising the results
    plt.figure()
    plt.plot(test_y, color='red', label='real SEC stock price')
    plt.plot(pred_y, color='blue', label='predicted SEC stock price')
    plt.title('SEC stock price prediction')
    plt.xlabel('time')
    plt.ylabel('stock price')
    plt.legend()
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return render(request,'home.html',{'data':uri})
    
def bol(request):
    name=request.GET['name']
    mk = Analyzer.MarketDB()
    df = mk.get_daily_price(name, '2018-11-01')
      
    df['MA20'] = df['close'].rolling(window=20).mean() 
    df['stddev'] = df['close'].rolling(window=20).std() 
    df['upper'] = df['MA20'] + (df['stddev'] * 2)
    df['lower'] = df['MA20'] - (df['stddev'] * 2)
    df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])

    df['II'] = (2*df['close']-df['high']-df['low'])/(df['high']-df['low'])*df['volume']  # ①
    df['IIP21'] = df['II'].rolling(window=21).sum()/df['volume'].rolling(window=21).sum()*100  # ②
    df = df.dropna()

    plt.figure(figsize=(9, 9))
    plt.subplot(3, 1, 1)
    plt.title('SK Hynix Bollinger Band(20 day, 2 std) - Reversals')
    plt.plot(df.index, df['close'], 'b', label='Close')
    plt.plot(df.index, df['upper'], 'r--', label ='Upper band')
    plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
    plt.plot(df.index, df['lower'], 'c--', label ='Lower band')
    plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')

    plt.legend(loc='best')
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['PB'], 'b', label='%b')
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)  # ③
    plt.bar(df.index, df['IIP21'], color='g', label='II% 21day')  # ④
    plt.grid(True)
    plt.legend(loc='best')

    
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return render(request,'bol.html',{'data':uri})