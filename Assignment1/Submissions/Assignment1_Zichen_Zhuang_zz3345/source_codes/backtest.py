import warnings
warnings.filterwarnings("ignore")

import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import yfinance as yf

daily_filename = "../../assets/WRDS-Security Daily.csv" ### Replace with your own daily stock information file name
selected_stock_filename = "./results/stock_selected.csv" ### Replace with your own selected stocks file name
weights_filename = "./results/portfolio_weights.csv" ### Replace with your own stock weights file name
enddate = "2025-10-05" ### Replace with today's date

fee_rate = 1e-3

class Stock:
    def __init__(self, name) -> None:
        self.name: str = name
        self.market_value: float = 0
        self.price: float = 0
        self.quantity: int = 0

    def calc_market_value(self, price: float) -> float:
        if price is not None and math.isnan(price) is False:
            self.price = price
        self.market_value = self.price * self.quantity
        return self.market_value

    def purchase(self, price: float, quantity: int) -> float:
        if quantity == 0:
            return 0
        self.price = price
        self.quantity += quantity
        self.calc_market_value(price)
        return price * quantity
    
    def sell(self, price: float, quantity: int) -> float:
        if price == 0 and quantity == self.quantity:
            self.quantity -= quantity
            return self.price * quantity
        
        if quantity == self.quantity:
            self.price = 0
        else:
            self.price = price

        self.quantity -= quantity
        self.calc_market_value(price)
        return price * quantity
    
    def info(self, price: float = None) -> None:
        print(f"Stock's name: {self.name}")
        print(f"Stock's quantity: {self.quantity}")
        print(f"Stock's market value: {self.calc_market_value(price)}")

class Portfolio:
    def __init__(self, capital) -> None:
        self.capital: float = capital
        self.total_market_value: float = 0
        self.fee: float = 0
        self.stocklist: list[Stock] = []

    def calc_total_market_value(self) -> float:
        total_market_value = 0
        for stock in self.stocklist:
            total_market_value += stock.market_value
        self.total_market_value = total_market_value + self.capital
        return self.total_market_value
    
    def purchase(self, stock: Stock, price: float, quantity: int) -> None:
        if price * quantity * (1 + fee_rate) > self.capital:
            return
        capital_change = stock.purchase(price, quantity)
        self.fee += capital_change * fee_rate
        self.capital -= capital_change * (1 + fee_rate)
        return 
    
    def sell(self, stock: Stock, price: float, quantity: int) -> None:
        capital_change = stock.sell(price, quantity)
        self.fee += capital_change * fee_rate
        self.capital += capital_change * (1 - fee_rate)
        return
    
    def quarterly_rebalancing(self, weightsdf: pd.DataFrame, dailyinfo: pd.DataFrame) -> None:
        ### First, use open price to calculate today's total market value
        for stock in self.stocklist:
            infodf = dailyinfo[dailyinfo.index == stock.name]
            if infodf.empty:
                openprice = stock.price
            else:
                openprice = infodf.iloc[0]["open"]
            stock.market_value = stock.calc_market_value(openprice)

        market_value = self.calc_total_market_value()
        current_names = weightsdf.index

        ### Check if holding stocks should be sold
        for stock in self.stocklist:
            name = stock.name
            if name not in dailyinfo.index:
                openprice = stock.price
            else:
                infodf = dailyinfo[dailyinfo.index == name]
                openprice = infodf.iloc[0]["open"]
            if math.isnan(openprice) or openprice < 1e-3:
                openprice = stock.price
            
            if name not in current_names:
                self.sell(stock, openprice, stock.quantity)
            else:
                ### Check the difference between holding quantity and needed quantity
                quantity = math.floor(weightsdf.loc[name].values[0] * market_value / openprice)
                diff = quantity - stock.quantity
                if diff > 0:
                    self.purchase(stock, openprice, diff)
                else:
                    self.sell(stock, openprice, -diff)

        portfolio_names = [stock.name for stock in self.stocklist]
        ### Check if hold all needed stocks
        for name in current_names:
            if name not in portfolio_names:
                infodf = dailyinfo[dailyinfo.index == name]
                if infodf.empty:
                    continue
                openprice = infodf.iloc[0]["open"]
                if math.isnan(openprice) or openprice < 1e-3:
                    continue
                quantity = math.floor(weightsdf.loc[name].values[0] * market_value / openprice)
                stock = Stock(name)
                self.purchase(stock, openprice, quantity)
                self.stocklist.append(stock)

        ### Update stock list to delete all useless stocks
        self.stocklist = [stock for stock in self.stocklist if stock.quantity > 0]

        ### Update market value using latest stock list
        for stock in self.stocklist:
            infodf = dailyinfo[dailyinfo.index == stock.name]
            if infodf.empty:
                closeprice = stock.price
            else:
                closeprice = infodf.iloc[0]["close"]
            stock.market_value = stock.calc_market_value(closeprice)
        return
    
    def daily_trading(self, date: datetime, weightsdf: pd.DataFrame, dailyinfo: pd.DataFrame) -> None:
        deleteidx = []
        for idx, stock in enumerate(self.stocklist):
            name = stock.name
            if name not in dailyinfo.index:
                openprice = stock.price
                closeprice = stock.price
            else:
                infodf = dailyinfo[dailyinfo.index == name]
                openprice = infodf["open"].iloc[-1]
                closeprice = infodf["close"].iloc[-1]
            
            if name not in weightsdf.index:
                # print(date, stock.name, stock.quantity)
                self.sell(stock, openprice, stock.quantity)
                deleteidx.append(idx)
                continue
            stock.market_value = stock.calc_market_value(closeprice)
            continue

        self.stocklist = [stock for i, stock in enumerate(self.stocklist) if i not in deleteidx]
        return

    def daily_update(self, date: datetime, weightsdf: pd.DataFrame, dailyinfo: pd.DataFrame = None, rebalancing: bool = False) -> None:
        if rebalancing:
            self.quarterly_rebalancing(weightsdf, dailyinfo)
        else:
            self.daily_trading(date, weightsdf, dailyinfo)
        return
    
def evaluate_strategy(portfolio_values, benchmark_values=None, risk_free_rate=0.0):
    portfolio_values = np.array(portfolio_values)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1

    annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1

    annual_volatility = returns.std() * np.sqrt(252)

    excess_return = returns.mean() - risk_free_rate / 252
    sharpe_ratio = excess_return / returns.std() * np.sqrt(252)

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).sum() / (len(returns) - (returns == 0).sum())

    if benchmark_values is not None:
        benchmark_values = np.array(benchmark_values)
        benchmark_returns = pd.Series(benchmark_values).pct_change().dropna()
        excess_returns = returns.values - benchmark_returns.values
        tracking_error = np.std(excess_returns)
        information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(252) if tracking_error != 0 else np.nan
    else:
        information_ratio = np.nan

    print("Evaluation metrics of your portfolio:")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Annual Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Information Ratio: {information_ratio:.4f}")

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Max Drawdown": max_drawdown,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Win Rate": win_rate,
        "Information Ratio": information_ratio
    }
    
if __name__ == "__main__":
    ### Load some useful data
    print("Loading portfolio weights...")
    weights = pd.read_csv(weights_filename)
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])

    print("Loading stock daily price...")
    usefulcolumns = ["datadate", "prcod", "prccd", "ajexdi", "tic"]
    dailyinfo = pd.read_csv(daily_filename, usecols=usefulcolumns)
    dailyinfo["datadate"] = pd.to_datetime(dailyinfo["datadate"])
    dailyinfo["ajexdi"] = dailyinfo["ajexdi"].replace(np.nan, 1)

    dailyinfo["open"] = (dailyinfo["prcod"] / dailyinfo["ajexdi"]).astype(float)
    dailyinfo["close"] = (dailyinfo["prccd"] / dailyinfo["ajexdi"]).astype(float)

    dailyinfo.dropna(inplace=True)

    # Set start date 2018-01-01
    backtest_start_date = pd.Timestamp("2018-01-01")
    dailyinfo = dailyinfo[(dailyinfo["datadate"] >= backtest_start_date) & (dailyinfo["datadate"] <= weights["trade_date"].max())]
    dates = sorted(dailyinfo["datadate"].unique())

    print("Loading benchmark data...")
    try:
        benchmark_df = pd.read_csv('./results/benchmark_data.csv', parse_dates=['date'])
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], utc=True).dt.tz_localize(None)
        benchmark_df = benchmark_df.set_index('date')
        
        # Align with portfolio dates
        SP500 = []
        QQQ = []
        for date in dates:
            date_pd = pd.to_datetime(date)
            if date_pd in benchmark_df.index:
                SP500.append(benchmark_df.loc[date_pd, 'SP500'])
                QQQ.append(benchmark_df.loc[date_pd, 'QQQ'])
            else:
                # Forward fill
                valid_dates = benchmark_df.index[benchmark_df.index <= date_pd]
                if len(valid_dates) > 0:
                    SP500.append(benchmark_df.loc[valid_dates[-1], 'SP500'])
                    QQQ.append(benchmark_df.loc[valid_dates[-1], 'QQQ'])
                else:
                    SP500.append(benchmark_df['SP500'].iloc[0])
                    QQQ.append(benchmark_df['QQQ'].iloc[0])
        
        print(f"  ✓ Loaded {len(SP500)} days of benchmark data")
    except Exception as e:
        print(f"  ✗ Failed to load benchmark data: {e}")
        SP500 = [100] * len(dates)
        QQQ = [100] * len(dates)

    ### Strategy Backtest
    print("Backtesting using historical data...")
    stock_selected = pd.read_csv(selected_stock_filename)
    stock_selected["trade_date"] = pd.to_datetime(stock_selected["trade_date"])
    stock_selected["hold_date"] = stock_selected["trade_date"] - pd.DateOffset(months=3)
    tradedates = stock_selected["hold_date"].unique()

    capital = 1e7
    port = Portfolio(capital)
    port_mvs_quart = []
    
    # mark first day rebalancing
    first_rebalancing_done = False

    for date in tqdm(dates, desc="Backtesting"):
        cur_weights = weights[weights["trade_date"] == date][["tic", "weights"]].set_index("tic")
        cur_info = dailyinfo[dailyinfo["datadate"] == date].set_index("tic")

        if not first_rebalancing_done or date in tradedates:
            port.daily_update(date, cur_weights, cur_info, rebalancing = True)
            first_rebalancing_done = True
        else:
            port.daily_update(date, cur_weights, cur_info, rebalancing = False)

        port_mvs_quart.append(port.calc_total_market_value())
    
    ### Results Visualization
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(dates, [data/port_mvs_quart[0] for data in port_mvs_quart], label='Portfolio Value with Quarterly Adjustment', color='blue', linewidth=2)
    ax.plot(dates, [data/SP500[0] for data in SP500], label='SP500', color='green', linewidth=2)
    ax.plot(dates, [data/QQQ[0] for data in QQQ], label='QQQ', color='red', linewidth=2)

    ax.set_title("Portfolio Values", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig("./results/portfolio_performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved chart: ./results/portfolio_performance.png")

    quart_metrics = evaluate_strategy(port_mvs_quart, benchmark_values=SP500)

    with open("./results/performance_metrics.json", "w") as f:
        json.dump({k: str(v) for k, v in quart_metrics.items()}, f, indent=4)
    
    # Save daily portfolio values for chart generation
    daily_values_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': port_mvs_quart
    })
    daily_values_df.to_csv('./results/portfolio_daily_values.csv', index=False)
    print(f"\n✓ Saved daily portfolio values: {len(port_mvs_quart)} days")
    
    plt.savefig("./results/portfolio_performance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved chart: ./results/portfolio_performance.png")
