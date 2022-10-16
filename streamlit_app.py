import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from itertools import product
import warnings
from binance.client import Client
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
from ast import literal_eval
import streamlit.components.v1 as components
import time
warnings.filterwarnings('ignore')
plt.style.use("seaborn")

# st.beta+set_page_config(layout="wide")
st.write("""# Pannoz Bot Services""")

image = Image.open('Pannoz Design Files copy 8.png')
st.image(image, use_column_width=True)


st.write("Welcome to ***Pannoz*** automed bot trade !""")

st.sidebar.header('Pair your coin:')


st.markdown('''# **Binance Price App**
A simple cryptocurrency price app pulling price data from *Binance API*.
''')

st.header('**Selected Price**')

# Load market data from Binance API
df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')

# Custom function for rounding values
def round_value(input_value):
    if input_value.values > 1:
        a = float(round(input_value, 2))
    else:
        a = float(round(input_value, 8))
    return a

col1, col2, col3 = st.columns(3)

# Widget (Cryptocurrency selection box)
col1_selection = st.sidebar.selectbox('Price 1', df.symbol, list(df.symbol).index('BTCBUSD') )
col2_selection = st.sidebar.selectbox('Price 2', df.symbol, list(df.symbol).index('ETHBUSD') )
col3_selection = st.sidebar.selectbox('Price 3', df.symbol, list(df.symbol).index('BNBBUSD') )
col4_selection = st.sidebar.selectbox('Price 4', df.symbol, list(df.symbol).index('XRPBUSD') )
col5_selection = st.sidebar.selectbox('Price 5', df.symbol, list(df.symbol).index('ADABUSD') )
col6_selection = st.sidebar.selectbox('Price 6', df.symbol, list(df.symbol).index('DOGEBUSD') )
col7_selection = st.sidebar.selectbox('Price 7', df.symbol, list(df.symbol).index('SHIBBUSD') )
col8_selection = st.sidebar.selectbox('Price 8', df.symbol, list(df.symbol).index('DOTBUSD') )
col9_selection = st.sidebar.selectbox('Price 9', df.symbol, list(df.symbol).index('MATICBUSD') )

# DataFrame of selected Cryptocurrency
col1_df = df[df.symbol == col1_selection]
col2_df = df[df.symbol == col2_selection]
col3_df = df[df.symbol == col3_selection]
col4_df = df[df.symbol == col4_selection]
col5_df = df[df.symbol == col5_selection]
col6_df = df[df.symbol == col6_selection]
col7_df = df[df.symbol == col7_selection]
col8_df = df[df.symbol == col8_selection]
col9_df = df[df.symbol == col9_selection]

# Apply a custom function to conditionally round values
col1_price = round_value(col1_df.weightedAvgPrice)
col2_price = round_value(col2_df.weightedAvgPrice)
col3_price = round_value(col3_df.weightedAvgPrice)
col4_price = round_value(col4_df.weightedAvgPrice)
col5_price = round_value(col5_df.weightedAvgPrice)
col6_price = round_value(col6_df.weightedAvgPrice)
col7_price = round_value(col7_df.weightedAvgPrice)
col8_price = round_value(col8_df.weightedAvgPrice)
col9_price = round_value(col9_df.weightedAvgPrice)

# Select the priceChangePercent column
col1_percent = f'{float(col1_df.priceChangePercent)}%'
col2_percent = f'{float(col2_df.priceChangePercent)}%'
col3_percent = f'{float(col3_df.priceChangePercent)}%'
col4_percent = f'{float(col4_df.priceChangePercent)}%'
col5_percent = f'{float(col5_df.priceChangePercent)}%'
col6_percent = f'{float(col6_df.priceChangePercent)}%'
col7_percent = f'{float(col7_df.priceChangePercent)}%'
col8_percent = f'{float(col8_df.priceChangePercent)}%'
col9_percent = f'{float(col9_df.priceChangePercent)}%'

# Create a metrics price box
col1.metric(col1_selection, col1_price, col1_percent)
col2.metric(col2_selection, col2_price, col2_percent)
col3.metric(col3_selection, col3_price, col3_percent)
col1.metric(col4_selection, col4_price, col4_percent)
col2.metric(col5_selection, col5_price, col5_percent)
col3.metric(col6_selection, col6_price, col6_percent)
col1.metric(col7_selection, col7_price, col7_percent)
col2.metric(col8_selection, col8_price, col8_percent)
col3.metric(col9_selection, col9_price, col9_percent)

st.header('**All Price**')
st.dataframe(df)

st.info('Credit: Created by Chanin Nantasenamat (aka [Data Professor](https://youtube.com/dataprofessor/))')

st.markdown("""
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)
#Chart
st.title("Chart")

st.header('Tradingview')
components.html(
            """
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
            <div id="technical-analysis"></div>
            <div class="tradingview-widget-copyright"><a href="https://in.tradingview.com/symbols/AAPL/" rel="noopener" target="_blank"><span class="blue-text">AAPL Chart</span></a> by TradingView</div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget(
            {
            "container_id": "technical-analysis",
            "width": 998,
            "height": 610,
            "symbol": "AAPL",
            "interval": "D",
            "timezone": "exchange",
            "theme": "light",
            "style": "1",
            "toolbar_bg": "#f1f3f6",
            "withdateranges": true,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "save_image": false,
            "studies": [
                "ROC@tv-basicstudies",
                "StochasticRSI@tv-basicstudies",
                "MASimple@tv-basicstudies"
            ],
            "show_popup_button": true,
            "popup_width": "1000",
            "popup_height": "650",
            "locale": "in"
            }
            );
            </script>
            </div>
            <!-- TradingView Widget END -->
            """,
            height=700, width=1000
        )


class Long_Only_Backtesterpl():
    def __init__(self,filepath ,symbol, start, end, tc):
        
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.get_data()
        self.results = None
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        
        
    
    
    def __repr__ (self):
        return "Long_Only_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol,
                                                                                self.start,self.end)
    def get_data(self):
        raw  = pd.read_csv(self.filepath, parse_dates = ["Date"], index_col = "Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw
   
    
    def test_strategy(self, percentiles = None, thresh = None):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).
        
        Parameters
        ===========
        percentiles: tuple (return_perc, vol_low_perc, vol_high_perc)
            return and volume percentiles to be considered for the strategy.
            
        thresh: tuple (return_thresh, vol_low_thresh, vol_high_thesh)
            return and volume thresholds to be considered for the strategy
      '''
        
        self.prepare_data(percentiles = percentiles, thresh = thresh)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        self.print_performance()
  
    def prepare_data(self, percentiles, thresh):
        ''' Prepares the Data for Backtesting.
        '''
        
        data = self.data[["Close","Volume","returns"]].copy()
        data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
        data.loc[data.vol_ch < -3, "vol_ch"] = np.nan
        
        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), percentiles[0])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[1], percentiles[2]])
        elif thresh:
            self.return_thresh = thresh[0]
            self.volume_thresh = [thresh[1],thresh[2]]
         
        #condition
        cond1 = data.returns >= self.return_thresh
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        
        data["position"] = 1
        data.loc[cond1 & cond2, "position"] = 0
        
        self.results = data
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
    
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
    
        self.results = data
   
    def plot_results(self):
        ''' Plot the cumulative performance of the trading strategy compared to buy-and hold.
        '''
        if self.results is None:
            st.write("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            st.line_chart(self.results[["creturns", "cstrategy"]])
    def optimize_strategy(self, return_range, vol_low_range, vol_high_range, metric = "Multiple"):
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).
        
        Parameters
        ===========
        return_range: tuple
            tuples of the form (start, end, step size).
        
        vol_low_range: tuple
            tuples of the form (start, end, step size).
        
        vol_high_range: tuple
            tuples of the from (start, end, step size).
            
        metric: str
            performance metric to be optimized (can bee "Multiple" or "Sharpe")
            
        '''
        
        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
            [1, 2, 3] >> 1, 2, 3
        return_range = range(*return_range)
        vol_low_range = range(*vol_low_range)
        vol_high_range = range(*vol_high_range)
        
        combinations = list(product(return_range, vol_low_range, vol_high_range))
        
        performance = []
        for comb in combinations:
            self.prepare_data(percentiles = comb, thresh = None)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
            
        self.results_overview = pd.DataFrame(data = np.array(combinations), columns = ["returns", "vol_low","vol_high"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()
        
    def find_best_strategy(self):
        ''' Find the optimal strategy (global maximum)
        '''
        
        best = self.results_overview.nlargest(1, "performance")
        return_perc = best.returns.iloc[0]
        vol_perc = [best.vol_low.iloc[0], best.vol_high.iloc[0]]
        perf = best.performance.iloc[0]
        st.write("Return_Perc: {} | Volume_Perc: | {}: {}:{}".format(return_perc,vol_perc,self.metric,round(perf,6))) 
        self.test_strategy(percentiles = (return_perc, vol_perc[0], vol_perc[1]))

    def print_performance(self):
        ''' Calculates and prints various Performance Metrics.
        '''
        
        data = self.results.copy()
        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        cagr =              round(self.calculate_cagr(data.strategy), 6)
        ann_mean =          round(self.calculate_annualized_mean(data.strategy),6)
        ann_std =           round(self.calculate_annualized_std(data.strategy), 6)
        sharpe =            round(self.calculate_sharpe(data.strategy), 6)
        
        st.write(88 * "=")
        st.write("SIMPLE PRICE & VOLUME STRATEGY | INSTRUMENT = {} | THRESHOLDS = {:4f}, {}".format(self.symbol,self.return_thresh,
                                                                                               self.volume_thresh))
        st.write(88 * "-")
        st.write("PERFORMANCE MEASURES:")
        st.write("\n")
        st.write("Multiple (Strategy)          {}".format(strategy_multiple))
        st.write("Multiple (Buy-and-Hold)      {}".format(bh_multiple))
        st.write(26 * "-")
        st.write("Out-/Underperformance:       {}".format(outperf))
        st.write("\n")
        st.write("CARG)                        {}".format(cagr))
        st.write("Annualized Mean:             {}".format(ann_mean))
        st.write("Annualized Std:              {}".format(ann_std))
        st.write("Sharpe Ration:               {}".format(sharpe))
        
        st.write(88 * "=")
    
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
   
    def calculate_annualized_mean(self, series): 
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series): 
        return series.std()  * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series): 
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
        
        
class Long_Short_Backtestersma():
    def __init__(self,filepath ,symbol, start, end, tc):
        
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.get_data()
        self.results = None
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        
        
    
    
    def __repr__ (self):
        return "Long_Only_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol,
                                                                                self.start,self.end)
    def get_data(self):
        raw  = pd.read_csv(self.filepath, parse_dates = ["Date"], index_col = "Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw
   
    
    def test_strategy(self, smas):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).
        
        Parameters
        ===========
       smas : tuple (SMA_S, SMA_M, SMA_L)
           Simple Moving Averages to be consirdered for the strategy
      '''
        self.SMA_S = smas[0]
        self.SMA_M = smas[1]
        self.SMA_L = smas[2]

        
        self.prepare_data(smas = smas)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        self.print_performance()
  
    def prepare_data(self, smas):
        ''' Prepares the Data for Backtesting.
        '''
        ############################## Strategy-Specific #######################
        
        data = self.data[["Close","Volume","returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window = smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window = smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window = smas[2]).mean()
        
        data.dropna(inplace = True)
        
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
       
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1
        
        ########################################################################
        self.results = data
   
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
    
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
    
        self.results = data
   
    def plot_results(self):
        ''' Plot the cumulative performance of the trading strategy compared to buy-and hold.
        '''
        if self.results is None:
            st.write("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            st.line_chart(self.results[["creturns", "cstrategy"]])
    
    def optimize_strategy(self, SMA_S_range, SMA_M_range, SMA_L_range, metric = "Multiple"):
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).
        
        Parameters
        ===========
        return_range: tuple
            tuples of the form (start, end, step size).
        
        vol_low_range: tuple
            tuples of the form (start, end, step size).
        
        vol_high_range: tuple
            tuples of the from (start, end, step size).
            
        metric: str
            performance metric to be optimized (can bee "Multiple" or "Sharpe")
            
        '''
        
        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
            
        SMA_S_range = range(*SMA_S_range)
        SMA_M_range = range(*SMA_M_range)
        SMA_L_range = range(*SMA_L_range)

        combinations = list(product(SMA_S_range, SMA_M_range, SMA_L_range))
        
        performance = []
        for comb in combinations:
            self.prepare_data(smas = comb)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
            
        self.results_overview = pd.DataFrame(data = np.array(combinations), columns = ["SMA_S", "SMA_M","SMA_L"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()
        
    def find_best_strategy(self):
        ''' Find the optimal strategy (global maximum)
        '''
        
        best = self.results_overview.nlargest(1, "performance")
        SMA_S = best.SMA_S.iloc[0]
        SMA_M = best.SMA_M.iloc[0]
        SMA_L = best.SMA_L.iloc[0]

        perf = best.performance.iloc[0]
        st.write("SMA_S: {} | SMA_M: {} | SMA_L: {} | {}: {}".format(SMA_S, SMA_M, SMA_L, self.metric,round(perf,6))) 
        self.test_strategy(smas = (SMA_S, SMA_M, SMA_L))
    ################################### Performance ########################################
    def print_performance(self):
        ''' Calculates and prints various Performance Metrics.
        '''
        
        data = self.results.copy()
        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        cagr =              round(self.calculate_cagr(data.strategy), 6)
        ann_mean =          round(self.calculate_annualized_mean(data.strategy),6)
        ann_std =           round(self.calculate_annualized_std(data.strategy), 6)
        sharpe =            round(self.calculate_sharpe(data.strategy), 6)
        
        st.write(88 * "=")
        st.write("TRIPLE SMA STRATEGY | INSTRUMENT = {} | SMAs = {}".format(self.symbol,[self.SMA_S, self.SMA_M, self.SMA_L]))
        st.write(88 * "-")
        st.write("PERFORMANCE MEASURES:")
        st.write("\n")
        st.write("Multiple (Strategy)          {}".format(strategy_multiple))
        st.write("Multiple (Buy-and-Hold)      {}".format(bh_multiple))
        st.write(26 * "-")
        st.write("Out-/Underperformance:       {}".format(outperf))
        st.write("\n")
        st.write("CARG)                        {}".format(cagr))
        st.write("Annualized Mean:             {}".format(ann_mean))
        st.write("Annualized Std:              {}".format(ann_std))
        st.write("Sharpe Ration:               {}".format(sharpe))
        
        st.write(88 * "=")
    
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
   
    def calculate_annualized_mean(self, series): 
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series): 
        return series.std()  * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series): 
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
        
                
        
        
        
filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2021-10-07"
tc = -0.00085


filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2021-10-07"
tc = -0.00085
sma_s = 5
sma_m = 50
sma_l = 200

st.title('Back Testing')
st.header('Selectbox')
bot_testingoption = ["Price & Volume", "SMA"]
bot_testingselect = st.selectbox("Which one would you like to compare?",options = bot_testingoption)
                                 
# testerpl = Long_Only_Backtesterpl( filepath = filepath, symbol = symbol,
#                                  start = start ,end = end, tc = tc)
# testersma = Long_Only_Backtestersma( filepath = filepath, symbol = symbol,
#                                  start = start ,end = end, tc = tc)                                 

# tester.test_strategypl(percentiles = (90,5,20))
# tester.plot_results()

if bot_testingselect == "Price & Volume":
    testerpl = Long_Only_Backtesterpl( filepath = filepath, symbol = symbol,
                                 start = start ,end = end, tc = tc)
    testerpl.test_strategy(percentiles = (90,5,20))
    testerpl.plot_results()
elif bot_testingselect == "SMA":
    testersma = Long_Short_Backtestersma(filepath = filepath, symbol = symbol,
                               start = start , end = end, tc = tc)  
    testersma.test_strategy(smas = (sma_s, sma_m, sma_l))
    testersma.plot_results()
else:
    st.write("Please Choose the selectbox")

st.header('**Demo Trading**')
st.write('Input your Api and Secret Api here')
apikey = st.text_input('apikey')
secretkey = st.text_input('secretkey')
  
st.button('Check balance demo')
  
    

st.header('**Spot Trading**')
st.write('Input your Api and Secret Api here')
api_key = st.text_input('Api_key',"BxDrXykHj6KU0pw9Kcwce7pw2dfT9yprOrBWWoxvC2hc1Lr50pzWv3y8XisNTE1S")
secret_key = st.text_input('Secret_key',"In5oYNadkct4tBWiekGByQ5XXo9qXY9uWNaiWDZmTlLXxXfVB6FwJ9x8sEKujqSc")

client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True)
client
def input_key(api_key,secret_key):
    
    account = client.get_account()
    assets = pd.json_normalize(account, 'balances')
    st.write(assets)        
    return client
# account = client.get_account()
# assets = pd.json_normalize(account, 'balances')
# st.write(assets)        
 
client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True)
client
def input_key(api_key,secret_key):
    
    account = client.get_account()
    assets = pd.json_normalize(account, 'balances')
    st.write(assets)        
    return client
# account = client.get_account()
# assets = pd.json_normalize(account, 'balances')
# st.write(assets)        
 
if st.button('Check balance'):
    st.header('Here your account balance:')
    input_key(api_key,secret_key)
# api_key = 'BxDrXykHj6KU0pw9Kcwce7pw2dfT9yprOrBWWoxvC2hc1Lr50pzWv3y8XisNTE1S'
# secret_key = 'In5oYNadkct4tBWiekGByQ5XXo9qXY9uWNaiWDZmTlLXxXfVB6FwJ9x8sEKujqSc'



class LongOnlyTrader_pv():
    def __init__ (self, symbol, bar_length, return_thresh, volume_thresh, units, position = 0):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units 
        self.position = position 
        self.trades = 0 
        self.trade_values = [] 
        
        #********************add strategy-specific attributes here ********************
        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh
        #******************************************************************************
        
    # async def main():
    #         client = await AsyncClient.create()
    #         bm = BinanceSocketManager(client)
    #         # start any sockets here, i.e a trade socket
    #         ts = bm.trade_socket(symbol)
    #         # then start receiving messages
    #         async with ts as tscm:
    #             while True:
    #                 res = await tscm.recv()
    #                 print(res)

    #         await client.close_connection()

    # if __name__ == "__main__":

    #         loop = asyncio.get_event_loop()
    #         loop.run_until_complete(main())

    def start_trading(self, historical_days):
         
           
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
          
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
           
     # "else" to be added later in the couse   
    
    def get_most_recent(self, symbol, interval, days): 
    
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))

        bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000)

        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open" , "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades", 
                      "Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]

        self.data = df # Create self.data
  
    def stream_candles(self, msg):
        
        # extract the required items from msg
        
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first    = float(msg["k"]["o"])
        high     = float(msg["k"]["h"])
        low      = float(msg["k"]["l"])
        close    = float(msg["k"]["c"])
        volume   = float(msg["k"]["v"])
        complete =       msg["k"]["x"]
        
        # Stop trading session # NEW
        if self.trades >= 5:#event_time >= datetime(2022, 9, 26, 10, 45): # year month day hours minute
            self.twm.stop()
            if self.position != 0:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units) 
                self.report_trade(order, "GOIN NEUTRAL AND STOP")
                self.position = 0
            else:
                st.write("STOP")
                
        #more stop examples:
        # if self.trades >= xyz
        # if self.cum_profits <> xyz
        
        #print out
        #st.write(".", end = "", flush = True) # just print something to get a feedback (everything OK)

        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low , close, volume, complete]
        
        # perpare featurs and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.define_strategy()
            self.execute_trades()
       
            
    def  define_strategy(self): # " strategy-specific"
        
        df = self.data.copy()
        
        #*************************** define your strategy here *****************************
        df = df[["Close", "Volume"]].copy()
        df["returns"] = np.log(df.Close / df.Close.shift())
        df["vol_ch"] = np.log(df.Volume.div(df.Volume.shift(1)))
        df.loc[df.vol_ch > 3, "vol_ch"] = np.nan
        df.loc[df.vol_ch < -3, "vol_ch"] = np.nan
        
        cond1 = df.returns >= self.return_thresh
        cond2 = df.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        
        df["position"] = 1
        df.loc[cond1 & cond2, "position"] = 0
        #************************************************************************************
        
        self.prepared_data = df.copy()

    def execute_trades(self):
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG") #NEW
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") #NEW
            self.position = 0
    
    def report_trade(self, order, going): 
        
        #extrat data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units)
            
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]),3)
            self.cum_profits = round(np.sum(self.trade_values),3)
        else:
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]),3)
            
        # print trade report
        st.write(2 * "\n" + 100* "-")
        st.write("{} | {}".format(time, going))
        st.write("{} | Base_Units = {} | Quote_Units = {} | Price = {}".format(time, base_units, quote_units, price))
        st.write("{} | Profit = {} | CumProfits = {}".format(time, real_profit, self.cum_profits))
        st.write(100 * "-" + "\n")
        
class LongOnlyTrader_sma():
    def __init__ (self, symbol, bar_length, sma_s, sma_m, sma_l, units, position = 0):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units 
        self.position = position 
        self.trades = 0 
        self.trade_values = [] 
        
        #********************add strategy-specific attributes here ********************
        self.SMA_S = sma_s
        self.SMA_M = sma_m
        self.SMA_L = sma_l
        #******************************************************************************
        
    def start_trading(self, historical_days):
       
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
          
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
            
     # "else" to be added later in the couse   
    
    def get_most_recent(self, symbol, interval, days): 
    
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))

        bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000)

        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open" , "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades", 
                      "Taker Buy Base Asset Volume","Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]

        self.data = df # Create self.data
  
    def stream_candles(self, msg):
        
        # extract the required items from msg
        
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first    = float(msg["k"]["o"])
        high     = float(msg["k"]["h"])
        low      = float(msg["k"]["l"])
        close    = float(msg["k"]["c"])
        volume   = float(msg["k"]["v"])
        complete =       msg["k"]["x"]
        
        # Stop trading session # NEW
        if self.trades >= 5: # stop stream after 5 trades # event_time >= datetime(2022, 8, 2, 10, 45): # year month day hours minute
            self.twm.stop()
            if self.position == 1:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units) 
                self.report_trade(order, "GOIN NEUTRAL AND STOP")
                self.position = 0
            elif self.position == -1:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units) 
                self.report_trade(order, "GOIN NEUTRAL AND STOP")
                self.position = 0
            else:
                st.write("STOP")
                
        #more stop examples:
        # if self.trades >= xyz
        # if self.cum_profits <> xyz
        
        #print out
        st.write(".", end = "", flush = True) # just print something to get a feedback (everything OK)

        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low , close, volume, complete]
        
        # perpare featurs and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.define_strategy()
            self.execute_trades() 
            
    def  define_strategy(self): # " strategy-specific"
        
        data = self.data.copy()
        
        #*************************** define your strategy here *****************************
        data = data[["Close"]].copy()
        
        data["SMA_S"] = data.Close.rolling(window = self.SMA_S).mean()
        data["SMA_M"] = data.Close.rolling(window = self.SMA_M).mean()
        data["SMA_L"] = data.Close.rolling(window = self.SMA_L).mean()
        
        data.dropna(inplace = True)
        
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)

        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1
        #************************************************************************************
        
        self.prepared_data = data.copy()

    def execute_trades(self):
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG") #NEW
            elif self.position == -1:
                order =  client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
                time.sleep(1)
                order =  client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") #NEW
            elif self.position == 1:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") #NEW
            self.position = 0
        if self.prepared_data["position"].iloc[-1] == 1: # if position is short -> go/stay long
                if self.position == 0:
                    order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                    self.report_trade(order, "GOING SHORT") #NEW
                elif self.position == 1:
                    order =  client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                    self.report_trade(order, "GOING NEUTRAL")
                    time.sleep(1)
                    order =  client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                    self.report_trade(order, "GOING SHORT")
                self.position = -1
    
    def report_trade(self, order, going): 
        
        #extrat data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit = "ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units)
            
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]),3)
            self.cum_profits = round(np.sum(self.trade_values),3)
        else:
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]),3)
            
        # print trade report
        st.write(2 * "\n" + 100* "-")
        st.write("{} | {}".format(time, going))
        st.write("{} | Base_Units = {} | Quote_Units = {} | Price = {}".format(time, base_units, quote_units, price))
        st.write("{} | Profit = {} | CumProfits = {}".format(time, real_profit, self.cum_profits))
        st.write(100 * "-" + "\n")
        
        


# api_key = 'BxDrXykHj6KU0pw9Kcwce7pw2dfT9yprOrBWWoxvC2hc1Lr50pzWv3y8XisNTE1S'
# secret_key = 'In5oYNadkct4tBWiekGByQ5XXo9qXY9uWNaiWDZmTlLXxXfVB6FwJ9x8sEKujqSc'


st.header('Bot name(s)')
bot_tradeoption = ["-","Price & Volume Bot", "SMA Bot"]
bot_tradeselect = st.selectbox("Select your bot trade",
                                 options = bot_tradeoption)

if bot_tradeselect == "Price & Volume Bot":
    symbol = "BTCUSDT"
    bar_length = "1m"
    return_thresh = 0
    volume_thresh = [-3,3]
    units = 0.001
    position = 0 
    trader_pv = LongOnlyTrader_pv(symbol = symbol, bar_length = bar_length, return_thresh = return_thresh,
                            volume_thresh = volume_thresh, units = units, position = position)  
    trader_pv.start_trading(historical_days = 1/24) 
elif bot_tradeselect == "SMA Bot":
    symbol = "BTCUSDT"
    bar_length = "1m"
    sma_s = 10
    sma_m = 20
    sma_l = 50
    units = 0.001
    position = 0
    trader_sma = LongOnlyTrader_sma(symbol = symbol, bar_length = bar_length, sma_s = sma_s, sma_m = sma_m, sma_l = sma_l,
                             units = units, position = position)
    trader_sma.start_trading(historical_days = 1/24)
else:
    st.write("Pls Choose the selectbox")
    
# account = client.get_account()
# assets = pd.json_normalize(account, 'balances')
# st.write(assets)        
 
if st.button('run'):
    st.write('Running your bot . . .')
    




                                 
# testerpl = Long_Only_Backtesterpl( filepath = filepath, symbol = symbol,
#                                  start = start ,end = end, tc = tc)
# testersma = Long_Only_Backtestersma( filepath = filepath, symbol = symbol,
#                                  start = start ,end = end, tc = tc)                                 

# tester.test_strategypl(percentiles = (90,5,20))
# tester.plot_results()


    


      
        
        




