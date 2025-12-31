"""
Market Prediction Trading Game - Streamlit Version
===================================================
A Streamlit game where users predict market behavior using 15-minute intraday charts.

Features:
- View 15-minute candle charts for NIFTY 500 stocks
- Take LONG/SHORT positions
- Track PnL in real-time
- Uses yfinance for real-time data (last 60 days)

Author: Trading Strategy Game
Date: 2025-12-31
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="📈 Market Prediction Game",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GAME CONFIGURATION
# ============================================================================
INITIAL_CAPITAL = 100000
MARKET_OPEN = "09:30:00"
MARKET_CLOSE = "15:15:00"
NIFTY_500_PATH = pathlib.Path(__file__).parent / "ind_nifty500list.csv"

# Market session times (15-minute intervals for NSE)
MARKET_TIMES = [
    "09:15:00", "09:30:00", "09:45:00", "10:00:00", "10:15:00", "10:30:00", "10:45:00",
    "11:00:00", "11:15:00", "11:30:00", "11:45:00", "12:00:00", "12:15:00",
    "12:30:00", "12:45:00", "13:00:00", "13:15:00", "13:30:00", "13:45:00",
    "14:00:00", "14:15:00", "14:30:00", "14:45:00", "15:00:00", "15:15:00"
]

# Colors - Modern dark theme
COLORS = {
    'profit': '#00ff88',
    'loss': '#ff4757',
    'neutral': '#ffd93d',
    'long': '#00d4aa',
    'short': '#ff6b6b',
    'candle_up': '#00ff88',
    'candle_down': '#ff4757',
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Increase top padding to prevent cutoff */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 0rem !important;
    }
    .main-header {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 0.3rem 0.5rem;
        background: linear-gradient(90deg, #1e3a5f, #2d4a6f);
        border-radius: 6px;
        margin-bottom: 0.3rem;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        border: 1px solid #3d5a80;
    }
    /* Metric label styling */
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 2px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: #333;
    }
    .metric-delta {
        font-size: 12px;
        color: #888;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #30363d;
    }
    .profit { color: #00ff88 !important; }
    .loss { color: #ff4757 !important; }
    .neutral { color: #ffd93d !important; }
    .position-long {
        background: linear-gradient(135deg, #00d4aa33, #00d4aa11);
        border: 2px solid #00d4aa;
        border-radius: 10px;
        padding: 1rem;
    }
    .position-short {
        background: linear-gradient(135deg, #ff6b6b33, #ff6b6b11);
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    /* Green LONG button */
    [data-testid="stButton"]:has(button:contains("LONG")) button {
        background-color: #00c853 !important;
        border-color: #00c853 !important;
    }
    /* Red SHORT button */
    [data-testid="stButton"]:has(button:contains("SHORT")) button {
        background-color: #ff1744 !important;
        border-color: #ff1744 !important;
    }
    .trade-history {
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data
def load_nifty_500_tickers():
    """Load list of NIFTY 500 tickers"""
    try:
        df = pd.read_csv(NIFTY_500_PATH)
        return df['Symbol'].tolist(), df[['Symbol', 'Company Name']].set_index('Symbol').to_dict()['Company Name']
    except Exception as e:
        st.error(f"Error loading NIFTY 500 list: {e}")
        return [], {}


def get_trading_days_last_60():
    """Get list of trading days in the last 60 days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    # Generate business days (excluding weekends)
    dates = pd.bdate_range(start=start_date, end=end_date - timedelta(days=1))
    return [d.date() for d in dates]


import os

# Local cache directory
CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_cached_file_path(ticker):
    """Get the cache file path for a ticker"""
    return CACHE_DIR / f"{ticker}_15m.parquet"


def download_stock_data(ticker):
    """Download stock data from yfinance with local file caching"""
    cache_file = get_cached_file_path(ticker)
    
    # Check if we have cached data that's recent (less than 6 hours old)
    if cache_file.exists():
        file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if file_age < 6 * 3600:  # 6 hours
            try:
                df = pd.read_parquet(cache_file)
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                return df
            except Exception:
                pass  # Fall through to download
    
    try:
        # Add .NS suffix for NSE stocks
        yf_ticker = f"{ticker}.NS"
        
        # Download maximum 15-minute data (last 60 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59)  # yfinance allows ~60 days for 15m
        
        df = yf.download(
            yf_ticker,
            start=start_date,
            end=end_date,
            interval="15m",
            progress=False
        )
        
        if df.empty:
            return None
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Remove timezone info for easier handling
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)
        
        # Calculate returns within each day
        df['trade_date'] = df.index.date
        df['returns'] = df.groupby('trade_date')['close'].pct_change()
        df = df.drop(columns=['trade_date'])
        
        # Save to cache
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass  # Continue even if caching fails
        
        return df
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return None


def get_available_dates_with_data(df, min_candles=10):
    """Get list of available trading dates that have sufficient data"""
    if df is None or df.empty:
        return []
    
    # Group by date and count candles
    df_copy = df.copy()
    df_copy['trade_date'] = df_copy.index.date
    candle_counts = df_copy.groupby('trade_date').size()
    
    # Filter dates with minimum candles
    valid_dates = candle_counts[candle_counts >= min_candles].index.tolist()
    return sorted(valid_dates)


def get_available_dates(df):
    """Get list of available trading dates from downloaded data"""
    if df is None or df.empty:
        return []
    dates = df.index.date
    unique_dates = sorted(list(set(dates)))
    return unique_dates


def get_day_data(df, date):
    """Get data for a specific date"""
    if df is None:
        return None
    date_str = str(date)
    try:
        day_data = df.loc[date_str].copy()
        if isinstance(day_data, pd.Series):
            return None  # Only one row, not enough data
        return day_data
    except KeyError:
        return None


def filter_market_hours(day_data):
    """Filter data to market hours only (9:15 AM to 3:30 PM for NSE)"""
    if day_data is None or day_data.empty:
        return None
    filtered = day_data[
        (day_data.index.time >= datetime.strptime("09:15:00", "%H:%M:%S").time()) &
        (day_data.index.time <= datetime.strptime("15:30:00", "%H:%M:%S").time())
    ]
    return filtered if not filtered.empty else None

# ============================================================================
# GAME STATE MANAGEMENT
# ============================================================================
def init_game_state():
    """Initialize or reset game state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.capital = INITIAL_CAPITAL
        st.session_state.initial_capital = INITIAL_CAPITAL
        st.session_state.position = None  # None, 'LONG', or 'SHORT'
        st.session_state.entry_price = 0
        st.session_state.entry_time_index = 0
        st.session_state.position_size = 0
        st.session_state.realized_pnl = 0
        st.session_state.total_trades = 0
        st.session_state.winning_trades = 0
        st.session_state.trade_history = []
        st.session_state.day_pnl = 0
        st.session_state.current_ticker = None
        st.session_state.stock_data = None
        st.session_state.day_data = None
        st.session_state.current_date = None
        st.session_state.current_time_index = 3  # Show first 3 candles
        st.session_state.game_started = False
        st.session_state.day_ended = False


def reset_day():
    """Reset day-specific state"""
    st.session_state.position = None
    st.session_state.entry_price = 0
    st.session_state.entry_time_index = 0
    st.session_state.position_size = 0
    st.session_state.day_pnl = 0
    st.session_state.current_time_index = 3
    st.session_state.day_ended = False


def reset_game():
    """Full game reset"""
    st.session_state.capital = INITIAL_CAPITAL
    st.session_state.realized_pnl = 0
    st.session_state.total_trades = 0
    st.session_state.winning_trades = 0
    st.session_state.trade_history = []
    reset_day()
    st.session_state.game_started = False
    st.session_state.current_ticker = None
    st.session_state.stock_data = None
    st.session_state.day_data = None
    st.session_state.current_date = None


# ============================================================================
# TRADING FUNCTIONS
# ============================================================================
def get_current_price():
    """Get current market price"""
    if st.session_state.day_data is None:
        return 0
    if len(st.session_state.day_data) == 0:
        return 0
    if st.session_state.current_time_index <= 0:
        return 0
    idx = min(st.session_state.current_time_index - 1, len(st.session_state.day_data) - 1)
    try:
        price = float(st.session_state.day_data.iloc[idx]['close'])
        return price
    except Exception:
        return 0


def calculate_current_pnl():
    """Calculate unrealized PnL"""
    if st.session_state.position is None:
        return 0
    
    current_price = get_current_price()
    if st.session_state.position == 'LONG':
        return (current_price - st.session_state.entry_price) * st.session_state.position_size
    else:  # SHORT
        return (st.session_state.entry_price - current_price) * st.session_state.position_size


def open_position(position_type):
    """Open a LONG or SHORT position"""
    if st.session_state.position is not None:
        st.warning("Already have an open position!")
        return False
    
    if st.session_state.day_ended:
        st.warning("Market day has ended!")
        return False
    
    if st.session_state.day_data is None or len(st.session_state.day_data) == 0:
        st.error("No data available. Please load a stock first.")
        return False
    
    current_price = get_current_price()
    if current_price <= 0:
        st.error(f"Cannot open position: Invalid price data (price={current_price})")
        return False
    
    st.session_state.position = position_type
    st.session_state.entry_price = current_price
    st.session_state.entry_time_index = max(0, st.session_state.current_time_index - 1)
    # Use 90% of capital for position
    st.session_state.position_size = max(1, int((st.session_state.capital * 0.9) / current_price))
    return True


def close_position():
    """Close the current position"""
    if st.session_state.position is None:
        return
    
    exit_price = get_current_price()
    pnl = calculate_current_pnl()
    
    # Record trade - safely get times with bounds checking
    data_len = len(st.session_state.day_data) if st.session_state.day_data is not None else 0
    
    entry_time = ""
    if data_len > 0 and 0 <= st.session_state.entry_time_index < data_len:
        entry_time = st.session_state.day_data.index[st.session_state.entry_time_index].strftime('%H:%M')
    
    exit_idx = min(st.session_state.current_time_index - 1, data_len - 1)
    exit_time = ""
    if data_len > 0 and exit_idx >= 0:
        exit_time = st.session_state.day_data.index[exit_idx].strftime('%H:%M')
    
    st.session_state.trade_history.append({
        'ticker': st.session_state.current_ticker,
        'date': st.session_state.current_date,
        'type': st.session_state.position,
        'entry': st.session_state.entry_price,
        'exit': exit_price,
        'qty': st.session_state.position_size,
        'pnl': pnl,
        'entry_time': entry_time,
        'exit_time': exit_time
    })
    
    st.session_state.realized_pnl += pnl
    st.session_state.day_pnl += pnl
    st.session_state.capital += pnl
    st.session_state.total_trades += 1
    if pnl > 0:
        st.session_state.winning_trades += 1
    
    # Reset position
    st.session_state.position = None
    st.session_state.entry_price = 0
    st.session_state.entry_time_index = 0
    st.session_state.position_size = 0


def advance_candle():
    """Move to next 15-minute candle"""
    if st.session_state.day_ended:
        return
    
    if st.session_state.day_data is None:
        return
    
    data_len = len(st.session_state.day_data)
    
    # Check if we can advance
    if st.session_state.current_time_index >= data_len:
        st.session_state.day_ended = True
        if st.session_state.position is not None:
            close_position()  # Auto-close at market end
        return
    
    st.session_state.current_time_index += 1
    
    # Check if market close or near end of day
    if st.session_state.current_time_index >= data_len:
        st.session_state.day_ended = True
        if st.session_state.position is not None:
            close_position()  # Auto-close at market end


# ============================================================================
# CHART FUNCTIONS
# ============================================================================
def create_chart():
    """Create the candlestick chart with Plotly"""
    if st.session_state.day_data is None:
        return None
    
    # Get visible data
    visible_data = st.session_state.day_data.iloc[:st.session_state.current_time_index]
    
    if len(visible_data) == 0:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.22,
        row_heights=[0.8, 0.5],
        subplot_titles=('Price Chart', 'Returns Chart')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=visible_data.index,
            open=visible_data['open'],
            high=visible_data['high'],
            low=visible_data['low'],
            close=visible_data['close'],
            name='Price',
            increasing_line_color=COLORS['candle_up'],
            decreasing_line_color=COLORS['candle_down'],
            increasing_fillcolor=COLORS['candle_up'],
            decreasing_fillcolor=COLORS['candle_down'],
        ),
        row=1, col=1
    )
    
    # Add entry marker if in position
    if st.session_state.position is not None and st.session_state.entry_time_index < len(visible_data):
        entry_time = visible_data.index[st.session_state.entry_time_index]
        marker_color = COLORS['long'] if st.session_state.position == 'LONG' else COLORS['short']
        marker_symbol = 'triangle-up' if st.session_state.position == 'LONG' else 'triangle-down'
        
        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[st.session_state.entry_price],
                mode='markers',
                marker=dict(size=15, color=marker_color, symbol=marker_symbol),
                name=f'{st.session_state.position} Entry',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add horizontal line for entry price
        fig.add_hline(
            y=st.session_state.entry_price,
            line_dash="dash",
            line_color=marker_color,
            annotation_text=f"Entry: ₹{st.session_state.entry_price:.2f}",
            row=1, col=1
        )
    
    # Returns bar chart
    if 'returns' in visible_data.columns:
        returns = visible_data['returns'].fillna(0) * 100
        colors = [COLORS['profit'] if r >= 0 else COLORS['loss'] for r in returns]
        
        fig.add_trace(
            go.Bar(
                x=visible_data.index,
                y=returns,
                marker_color=colors,
                name='Returns %',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    current_price = get_current_price()
    current_time = visible_data.index[-1].strftime('%H:%M') if len(visible_data) > 0 else ""
    
    fig.update_layout(
        title=f"📊 {st.session_state.current_ticker} | 📅 {st.session_state.current_date} | 🕐 {current_time} | ₹{current_price:,.2f}",
        template='plotly_dark',
        height=400,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Returns %", row=2, col=1)
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    init_game_state()
    
    # Sidebar
    with st.sidebar:
        # Game title banner in sidebar
        st.markdown('<div class="main-header">🎮 Market Prediction Game 📈</div>', unsafe_allow_html=True)
        st.markdown("")
        st.header("🎯 Game Controls")
        
        # Load tickers
        tickers, company_names = load_nifty_500_tickers()
        
        st.subheader("📊 Select Stock & Date")
        
        # Selection method
        selection_method = st.radio(
            "Selection Method:",
            ["🎲 Random", "📝 Manual Selection"],
            key="selection_method"
        )
        
        if selection_method == "🎲 Random":
            if st.button("🎲 Generate Random Stock & Date", type="primary", width='stretch'):
                # Pick random ticker
                random_ticker = random.choice(tickers)
                
                with st.spinner(f"Loading {random_ticker} data (this may take a moment on first load)..."):
                    # Download data (uses local file caching)
                    stock_data = download_stock_data(random_ticker)
                    
                    if stock_data is not None and not stock_data.empty:
                        # Get dates with sufficient data (at least 10 candles)
                        available_dates = get_available_dates_with_data(stock_data, min_candles=10)
                        
                        if not available_dates:
                            st.error(f"No dates with sufficient data for {random_ticker}. Try again.")
                            return
                        
                        # Pick a random date from valid dates
                        selected_date = random.choice(available_dates)
                        
                        day_data = get_day_data(stock_data, selected_date)
                        day_data = filter_market_hours(day_data)
                        
                        if day_data is not None and len(day_data) >= 5:
                            reset_day()
                            st.session_state.current_ticker = random_ticker
                            st.session_state.stock_data = stock_data
                            st.session_state.day_data = day_data
                            st.session_state.current_date = selected_date
                            st.session_state.game_started = True
                            st.success(f"Loaded {random_ticker} for {selected_date} ({len(day_data)} candles)")
                            st.rerun()
                        else:
                            candle_count = len(day_data) if day_data is not None else 0
                            st.error(f"Not enough data for this date (only {candle_count} candles). Try again.")
                    else:
                        st.error(f"Could not load data for {random_ticker}")
        
        else:  # Manual Selection
            # Stock selection
            selected_ticker = st.selectbox(
                "Select Stock:",
                options=tickers,
                format_func=lambda x: f"{x} - {company_names.get(x, '')[:30]}",
                key="manual_ticker"
            )
            
            # Load stock data to get available dates
            if st.button("📥 Load Stock Data", width='stretch', key="load_stock_btn"):
                with st.spinner(f"Loading {selected_ticker} data..."):
                    stock_data = download_stock_data(selected_ticker)
                    if stock_data is not None:
                        st.session_state.temp_stock_data = stock_data
                        st.session_state.temp_ticker = selected_ticker
                        st.rerun()
            
            # Show date selection if stock data is loaded
            if 'temp_stock_data' in st.session_state and st.session_state.temp_stock_data is not None:
                stock_data = st.session_state.temp_stock_data
                available_dates = get_available_dates_with_data(stock_data, min_candles=10)
                
                if available_dates:
                    st.success(f"Found {len(available_dates)} dates with sufficient data")
                    
                    selected_date = st.selectbox(
                        "Select Date:",
                        options=available_dates[::-1],  # Most recent first
                        format_func=lambda x: x.strftime("%Y-%m-%d (%A)"),
                        key="manual_date"
                    )
                    
                    if st.button("📈 Start Trading", type="primary", width='stretch'):
                        day_data = get_day_data(stock_data, selected_date)
                        day_data = filter_market_hours(day_data)
                        
                        if day_data is not None and len(day_data) >= 5:
                            reset_day()
                            st.session_state.current_ticker = st.session_state.temp_ticker
                            st.session_state.stock_data = stock_data
                            st.session_state.day_data = day_data
                            st.session_state.current_date = selected_date
                            st.session_state.game_started = True
                            # Clear temp data
                            del st.session_state.temp_stock_data
                            del st.session_state.temp_ticker
                            st.rerun()
                        else:
                            candle_count = len(day_data) if day_data is not None else 0
                            st.error(f"Not enough data for this date (only {candle_count} candles).")
                else:
                    st.error("No dates with sufficient data found for this stock.")
        
        st.divider()
        
        # Reset buttons
        if st.button("🔄 New Day (Keep Capital)", width='stretch'):
            reset_day()
            st.session_state.game_started = False
            st.rerun()
        
        if st.button("🔃 Reset Game", width='stretch'):
            reset_game()
            st.rerun()
        
        st.divider()
        
        # Game Rules
        with st.expander("📜 Game Rules"):
            st.markdown("""
            **Objective:** Predict market direction and profit!
            
            **Rules:**
            - Start with ₹1,00,000 capital
            - View 15-minute candles
            - Take LONG or SHORT positions
            - Close before market end (auto-close)
            
            **Controls:**
            - 📈 **LONG**: Buy expecting price UP
            - 📉 **SHORT**: Sell expecting price DOWN
            - ⏭️ **NEXT**: Skip to next candle
            - 💰 **CLOSE**: Close your position
            """)
    
    # Main content area
    if not st.session_state.game_started:
        # Welcome screen
        st.markdown("""### 🎯 Welcome to the Market Prediction Game!
        
Test your trading skills against the market using real 15-minute intraday data.

**How to Play:**
1. **Select a stock and date** from the sidebar (random or manual)
2. **Analyze the chart** - first 3 candles are shown
3. **Take a position** - LONG if you think price will go up, SHORT if down
4. **Advance time** - Click NEXT to reveal the next candle
5. **Close your position** - Lock in your profit or loss
6. **Repeat!** - Try to beat the market!

**💡 Tips:**
- Look for patterns in the first few candles
- Consider volume and momentum
- Don't let emotions drive your decisions
- Practice risk management

---

👈 **Use the sidebar to start the game!**
""")
        
        # Show overall stats if there are any trades
        if st.session_state.total_trades > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 Your Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            total_return = ((st.session_state.capital - st.session_state.initial_capital) / st.session_state.initial_capital) * 100
            win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
            
            with col1:
                st.metric("Capital", f"₹{st.session_state.capital:,.2f}")
            with col2:
                st.metric("Total P&L", f"₹{st.session_state.realized_pnl:,.2f}",
                         delta=f"{total_return:.2f}%")
            with col3:
                st.metric("Total Trades", st.session_state.total_trades)
            with col4:
                st.metric("Win Rate", f"{win_rate:.1f}%")
    
    else:
        # Game screen
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        current_price = get_current_price()
        unrealized_pnl = calculate_current_pnl()
        total_value = st.session_state.capital + unrealized_pnl
        total_return = ((total_value - st.session_state.initial_capital) / st.session_state.initial_capital) * 100
        win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
        # add a blank row for spacing
        
        with col1:
            st.markdown("<p class='metric-label'>💼 Capital</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>₹{st.session_state.capital:,.0f}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p class='metric-label'>📊 Unrealized P&L</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>₹{unrealized_pnl:,.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-delta'>Day: ₹{st.session_state.day_pnl:,.0f}</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<p class='metric-label'>📈 Total Return</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{total_return:+.1f}%</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<p class='metric-label'>🎯 Win Rate</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{win_rate:.0f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-delta'>{st.session_state.winning_trades}/{st.session_state.total_trades} trades</p>", unsafe_allow_html=True)
        with col5:
            candles_remaining = len(st.session_state.day_data) - st.session_state.current_time_index if st.session_state.day_data is not None else 0
            st.markdown("<p class='metric-label'>⏱️ Candles Left</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{candles_remaining}</p>", unsafe_allow_html=True)
        
        # Chart and controls side by side
        chart_col, controls_col = st.columns([3, 1])
        
        with chart_col:
            chart = create_chart()
            if chart:
                st.plotly_chart(chart, width='stretch', key="main_chart")
        
        with controls_col:
            # Trading buttons - vertical layout with custom colors
            # Apply button styling: LONG=Green, SHORT=Red, others=Grey
            st.markdown("""
                <style>
                /* LONG button - Green */
                button[kind="secondary"]:has(p:contains("LONG")),
                div[data-testid="stButton"] button:contains("LONG") {
                    background-color: #00c853 !important;
                    border-color: #00c853 !important;
                    color: white !important;
                }
                /* SHORT button - Red */
                button[kind="secondary"]:has(p:contains("SHORT")),
                div[data-testid="stButton"] button:contains("SHORT") {
                    background-color: #ff1744 !important;
                    border-color: #ff1744 !important;
                    color: white !important;
                }
                /* NEXT and CLOSE buttons - Grey */
                button[kind="secondary"]:has(p:contains("NEXT")),
                div[data-testid="stButton"] button:contains("NEXT"),
                button[kind="secondary"]:has(p:contains("CLOSE")),
                div[data-testid="stButton"] button:contains("CLOSE") {
                    background-color: #6c757d !important;
                    border-color: #6c757d !important;
                    color: white !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            if st.session_state.day_ended:
                st.warning("📅 Day ended!")
            else:
                if st.session_state.position is None:
                    if st.button("📈 LONG", width='stretch', key="btn_long"):
                        if open_position('LONG'):
                            st.rerun()
                    if st.button("📉 SHORT", width='stretch', key="btn_short"):
                        if open_position('SHORT'):
                            st.rerun()
                else:
                    pnl = calculate_current_pnl()
                    pnl_color = "🟢" if pnl >= 0 else "🔴"
                    st.markdown(f"**{st.session_state.position}** @ ₹{st.session_state.entry_price:,.0f}")
                    st.markdown(f"{pnl_color} P&L: **₹{pnl:+,.0f}**")
                    if st.button(f"💰 CLOSE", width='stretch', key="btn_close"):
                        close_position()
                        st.rerun()
                
                st.divider()
                if st.button("⏭️ NEXT", width='stretch', key="btn_next"):
                    advance_candle()
                    st.rerun()
        
        # Compact trade history in expander
        with st.expander("📜 Trade History", expanded=False):
            day_trades = [t for t in st.session_state.trade_history 
                         if t['date'] == st.session_state.current_date]
            
            if day_trades:
                trade_df = pd.DataFrame(day_trades)
                trade_df['pnl_formatted'] = trade_df['pnl'].apply(lambda x: f"₹{x:+,.0f}")
                display_df = trade_df[['type', 'entry_time', 'exit_time', 'pnl_formatted']]
                display_df.columns = ['Type', 'Entry', 'Exit', 'P&L']
                st.dataframe(display_df, width='stretch', hide_index=True, height=150)
            else:
                st.info("No trades yet")
            
            if len(st.session_state.trade_history) > len(day_trades):
                st.markdown("**All-Time Trades:**")
                all_trades_df = pd.DataFrame(st.session_state.trade_history)
                all_trades_df['pnl_formatted'] = all_trades_df['pnl'].apply(lambda x: f"₹{x:+,.0f}")
                all_trades_df['date'] = all_trades_df['date'].astype(str)
                display_all = all_trades_df[['ticker', 'date', 'type', 'pnl_formatted']]
                display_all.columns = ['Ticker', 'Date', 'Type', 'P&L']
                st.dataframe(display_all, width='stretch', hide_index=True, height=150)


if __name__ == "__main__":
    main()
