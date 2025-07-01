import os
import time
import logging
from datetime import datetime, timedelta, time as dtime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests

# Alpaca imports
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, OrderType, AssetClass

# ─── CONFIGURATION ──────────────────────────────────────────────
load_dotenv()
API_KEY    = "PKTZNXVE54MF1WERXH7O"
API_SECRET = "mXX6EyyqOfACLPwlNDniOuCUX2hG2vCtcZpVNDx3"
PAPER_MODE = True

REQUEST_SYMBOL = "BTC/USD"
SHORT_EMA     = 9
LONG_EMA      = 21
RSI_PERIOD    = 14
ATR_PERIOD    = 14
TRADE_RISK    = 0.01  # 1% of cash per trade
FEE_RATE      = 0.0005
INITIAL_CASH  = 100000.0
MAX_DAILY_LOSS = 1000
MAX_TRADES_PER_DAY = 10
TRADING_START = dtime(8, 0)   # 8:00 UTC
TRADING_END   = dtime(23, 0)  # 18:00 UTC
SLEEP_SECONDS = 60
MAX_ALLOC_PCT = 0.10  # 10% of cash per trade


# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ─── LOGGING SETUP ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

# ─── Alpaca Clients ─────────────────────────────────────────────
def init_alpaca_clients():
    try:
        hist_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)
        trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER_MODE)
        logging.info("Alpaca clients initialized successfully.")
        return hist_client, trading_client
    except Exception as e:
        logging.error(f"Failed to initialize Alpaca clients: {e}")
        exit(1)

hist_client, trading_client = init_alpaca_clients()

# ─── Helper: Fetch Crypto Bars ─────────────────────────────────
def fetch_crypto_bars(request_symbol: str, start, end, timeframe: TimeFrame) -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if isinstance(start, datetime):
                start = start.isoformat()
            if isinstance(end, datetime):
                end = end.isoformat()
            req = CryptoBarsRequest(
                symbol_or_symbols=[request_symbol],
                timeframe=timeframe,
                start=start,
                end=end,
            )
            raw = hist_client.get_crypto_bars(req)
            df_all = raw.df
            if df_all.empty:
                logging.warning(f"No data returned for {request_symbol} between {start} and {end}.")
                return pd.DataFrame()
            if isinstance(df_all.columns, pd.MultiIndex):
                symbol_returned = df_all.columns.levels[0][0]
                data = df_all[symbol_returned].copy()
            else:
                data = df_all.copy()
            data = data.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index()
                data = data.set_index("timestamp")
                data.index = pd.to_datetime(data.index)
            else:
                if "timestamp" in data.columns:
                    data = data.set_index("timestamp")
                    data.index = pd.to_datetime(data.index)
                    data.index.name = "ts"
            return data
        except Exception as e:
            logging.error(f"Error fetching crypto bars (attempt {attempt+1}): {e}")
            time.sleep(5)
    return pd.DataFrame()

# ─── Indicators ────────────────────────────────────────────────
def add_indicators(df):
    df['EMA_short'] = df['Close'].ewm(span=SHORT_EMA, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=LONG_EMA, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=ATR_PERIOD).mean()
    return df

# ─── Position Sizing ───────────────────────────────────────────
def get_position_size(cash, atr, risk_per_trade=TRADE_RISK):
    if atr == 0 or np.isnan(atr):
        return 0.0
    dollar_risk = cash * risk_per_trade
    qty = round(dollar_risk / atr, 6)
    min_qty = 0.0001
    return max(qty, min_qty)

# ─── Trading Hours ─────────────────────────────────────────────
def is_trading_time():
    now = datetime.utcnow().time()
    return TRADING_START <= now <= TRADING_END

# ─── Risk Management ───────────────────────────────────────────
def check_risk_limits(account, trades_today, start_equity):
    if float(account.equity) < start_equity - MAX_DAILY_LOSS:
        logging.warning("Max daily loss reached. Stopping trading.")
        send_telegram_alert("Max daily loss reached. Stopping trading.")
        return False
    if trades_today >= MAX_TRADES_PER_DAY:
        logging.warning("Max trades per day reached. Stopping trading.")
        send_telegram_alert("Max trades per day reached. Stopping trading.")
        return False
    return True

# ─── Bracket Order ─────────────────────────────────────────────
def submit_bracket_order(trading_client, symbol, qty, side, take_profit_pct=0.01, stop_loss_pct=0.005):
    try:
        price = float(get_latest_price(symbol))
        take_profit = price * (1 + take_profit_pct if side == OrderSide.BUY else 1 - take_profit_pct)
        stop_loss = price * (1 - stop_loss_pct if side == OrderSide.BUY else 1 + stop_loss_pct)
        order = OrderRequest(
            symbol=symbol,
            qty=str(qty),
            side=side,
            type=OrderType.MARKET,
            time_in_force="gtc",
            take_profit=TakeProfitRequest(limit_price=str(round(take_profit, 2))),
            stop_loss=StopLossRequest(stop_price=str(round(stop_loss, 2))),
            asset_class=AssetClass.CRYPTO,
        )
        resp = trading_client.submit_order(order)
        logging.info(f"Bracket order submitted: {side.name} {qty} {symbol} @ {price:.2f}")
        send_telegram_alert(f"Bracket order: {side.name} {qty} {symbol} @ {price:.2f}")
        return True
    except Exception as e:
        logging.error(f"Bracket order failed: {e}")
        send_telegram_alert(f"Bracket order failed: {e}")
        return False

def get_latest_price(symbol):
    # Fetch the latest close price (1 bar 30 mins)
    now_utc = datetime.utcnow()
    end_dt = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_dt = (now_utc - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = fetch_crypto_bars(symbol, start=start_dt, end=end_dt, timeframe=TimeFrame(30, TimeFrameUnit.Minute))
    if not df.empty:
        return df['Close'].iloc[-1]
    else:
        raise Exception("No price data available.")

# ─── End-of-Day Position Closing ───────────────────────────────
def close_all_positions(trading_client):
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            if float(pos.qty) > 0:
                order = OrderRequest(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force="gtc",
                    asset_class=AssetClass.CRYPTO,
                )
                trading_client.submit_order(order)
                logging.info(f"Closed position: {pos.symbol} qty={pos.qty}")
                send_telegram_alert(f"Closed position: {pos.symbol} qty={pos.qty}")
    except Exception as e:
        logging.error(f"Failed to close positions: {e}")
        send_telegram_alert(f"Failed to close positions: {e}")

# ─── Main Trading Loop ─────────────────────────────────────────
def run_daytrader():
    trades_today = 0
    start_equity = float(trading_client.get_account().equity)
    logging.info("🚀 Starting daytrading loop. Press Ctrl+C to exit.")

    try:
        while True:
            if not is_trading_time():
                logging.info("Outside trading hours.")
                time.sleep(SLEEP_SECONDS)
                continue

            account = trading_client.get_account()
            if not check_risk_limits(account, trades_today, start_equity):
                break

            now_utc = datetime.utcnow()
            end_dt = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            start_dt = (now_utc - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")  # 30 days of 30-min bars
            df = fetch_crypto_bars(REQUEST_SYMBOL, start=start_dt, end=end_dt, timeframe=TimeFrame(30, TimeFrameUnit.Minute))  # 30-min bars
            if df.empty or len(df) < LONG_EMA + 2:
                logging.warning("Not enough data to compute indicators. Skipping this cycle.")
                time.sleep(SLEEP_SECONDS)
                continue

            df = add_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
             # Log indicator values
            logging.info(f"Indicators: EMA_short={latest['EMA_short']:.2f}, EMA_long={latest['EMA_long']:.2f}, RSI={latest['RSI']:.2f}, ATR={latest['ATR']:.2f}")

            # Signal logic: EMA crossover + RSI filter
            signal = None
            if (prev['EMA_short'] < prev['EMA_long']) and (latest['EMA_short'] > latest['EMA_long']) and (latest['RSI'] < 70):
                signal = "BUY"
            elif (prev['EMA_short'] > prev['EMA_long']) and (latest['EMA_short'] < latest['EMA_long']) and (latest['RSI'] > 30):
                signal = "SELL"

            # Check current position
            try:
                positions = trading_client.get_all_positions()
                btc_position = next((p for p in positions if p.symbol.replace("USDC", "USD") == REQUEST_SYMBOL.replace("/", "")), None)
                currently_long = btc_position is not None and float(btc_position.qty) > 0
            except Exception as e:
                logging.error(f"Could not fetch positions: {e}")
                currently_long = False

            # Execute trade
  
            if signal == "BUY" and not currently_long:
                MAX_NOTIONAL = 200000
                NOTIONAL_BUFFER = 0.97  # 97% of max to avoid all issues
                CASH_BUFFER = 0.97      # 97% of cash to be safe

                cash = float(account.cash)
                atr = latest['ATR']
                price = float(latest['Close'])
                safe_price = price * 1.01  # Assume price could move up 1% before fill

                # ATR-based position sizing
                qty_atr = get_position_size(cash, atr)
                # Notional cap
                qty_notional = round((MAX_NOTIONAL * NOTIONAL_BUFFER) / safe_price, 6)
                # Cash cap
                qty_cash = round((cash * CASH_BUFFER) / safe_price, 6)
                # Use the smallest qty
                qty = min(qty_atr, qty_notional, qty_cash)
                # Max allocation cap
                qty_max_alloc = round((cash * MAX_ALLOC_PCT) / safe_price, 6)
                qty = min(qty, qty_max_alloc)
                logging.info(f"Max allocation cap: qty_max_alloc={qty_max_alloc}, applied final_qty={qty}")
                notional = qty * safe_price

                # Final check
                if notional > cash * CASH_BUFFER:
                    qty = round((cash * CASH_BUFFER) / safe_price, 6)
                    notional = qty * safe_price
                    logging.info(f"Final adjustment: qty set to {qty} to ensure notional {notional:.2f} < available cash {cash:.2f}")

                logging.info(
                    f"Order sizing: qty_atr={qty_atr}, qty_notional={qty_notional}, qty_cash={qty_cash}, final_qty={qty}, notional={notional:.2f}, cash={cash:.2f}"
                )

                if qty > 0:
                    success = submit_bracket_order(trading_client, REQUEST_SYMBOL, qty, OrderSide.BUY)
                    if success:
                        trades_today += 1
                else:
                    logging.warning("Calculated qty is zero or negative. No order submitted.")
                
                if qty > 0:
                    submit_bracket_order(trading_client, REQUEST_SYMBOL, qty, OrderSide.BUY)
                    trades_today += 1
            elif signal == "SELL" and currently_long:
                qty_to_sell = btc_position.qty if btc_position else "0.0001"
                submit_bracket_order(trading_client, REQUEST_SYMBOL, qty_to_sell, OrderSide.SELL)
                trades_today += 1
            else:
                logging.info(f"No trade. Signal = {signal}, Currently long = {currently_long}")

            # End-of-day close
            if datetime.utcnow().time() > TRADING_END:
                logging.info("End of trading session. Closing all positions.")
                close_all_positions(trading_client)
                break

            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        logging.info("🛑 Bot terminated by user. Closing all positions.")
        close_all_positions(trading_client)

# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("=== Alpaca BTC Daytrader Bot ===")
    logging.info(f"Paper mode: {PAPER_MODE}")
    logging.info(f"Using API_KEY = {API_KEY[:4]}****, SECRET = {API_SECRET[:4]}****")
    run_daytrader() 
