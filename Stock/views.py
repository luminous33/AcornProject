from django.shortcuts import render
import io
import urllib, base64
import datetime
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import Analyzer
from .models import CompanyInfo
from .models import DailyPrice
from bs4 import BeautifulSoup
from urllib.request import urlopen
import DualMomentum 
import json

def homepage(request):
    url = 'https://finance.naver.com/sise/'
    with urlopen(url) as doc:
        html = BeautifulSoup(doc, 'lxml') 
        kospi = html.find('span',id='KOSPI_now').text
        kospiChange = html.find('span',id='KOSPI_change').text
        kosdak = html.find('span',id='KOSDAQ_now').text
        kosdakChange = html.find('span',id='KOSDAQ_change').text
        kospi200 = html.find('span',id='KPI200_now').text
        kospi200Change = html.find('span',id='KPI200_change').text
    color="blue"
    
    
#듀얼 부분
    dm = DualMomentum.DualMomentum()
    rm = dm.get_rltv_momentum('2020-02-01','2020-4-31',10)
    json_records = rm.reset_index().to_json(orient ='records') 
    datarm = [] 
    datarm = json.loads(json_records) 
    
    am = dm.get_abs_momentum(rm, '2020-05-01','2020-8-31')
    json_records = am.reset_index().to_json(orient ='records') 
    dataam = [] 
    dataam = json.loads(json_records)
    
#최종 context 부분
    context={
        'color':color,
        'kospi':kospi,
        'kospiChange':kospiChange,
        'kosdak':kosdak,
        'kosdakChange':kosdakChange,
        'kospi200':kospi200,
        'kospi200Change':kospi200Change,
        'd': datarm,
        'd2': dataam
    
    }
    
    return render(request, 'homepage.html',context)
    
def empty(request):
    return render(request, 'empty.html')
    
def chart(request):
    return render(request, 'chart.html')
    
def introduce(request):
    return render(request, 'introduce.html')
    
def trading(request):
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
    return render(request,'trading.html',{'data':uri})
    
def bol2(request):
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
    return render(request,'bol2.html',{'data':uri,'name':name})
    
def triple(request):
    name=request.GET['name']
    mk = Analyzer.MarketDB()
    df = mk.get_daily_price('name', '2018-01-01')

    ema60 = df.close.ewm(span=60).mean()
    ema130 = df.close.ewm(span=130).mean() 
    macd = ema60 - ema130
    signal = macd.ewm(span=45).mean() 
    macdhist = macd - signal

    df = df.assign(ema130=ema130, ema60=ema60, macd=macd, signal=signal,
        macdhist=macdhist).dropna()
    df['number'] = df.index.map(mdates.date2num)
    ohlc = df[['number','open','high','low','close']]

    ndays_high = df.high.rolling(window=14, min_periods=1).max()      # ①
    ndays_low = df.low.rolling(window=14, min_periods=1).min()        # ②
    fast_k = (df.close - ndays_low) / (ndays_high - ndays_low) * 100  # ③
    slow_d= fast_k.rolling(window=3).mean()                           # ④
    df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()             # ⑤

    plt.figure(figsize=(9, 7))
    p1 = plt.subplot(2, 1, 1)
    plt.title('Triple Screen Trading - Second Screen')
    plt.grid(True)
    candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
    p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot(df.number, df['ema130'], color='c', label='EMA130')
    plt.legend(loc='best')
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    return render(request,'triple.html',{'data':uri,'name':name})

#주식검색    
def search(request):
    prices = DailyPrice.objects.filter(date="2020-09-01")
    context = {'prices':prices}

    return render(request, 'search.html', context)
    
