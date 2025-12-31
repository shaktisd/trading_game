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
    layout="centered",
    initial_sidebar_state="collapsed"
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
    
    # Create subplots with better separation
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
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
    
    # Update layout - prominent and professional
    fig.update_layout(
        title="Predict the Market!",
        template='plotly_dark',
        height=400,  # Taller for better visibility
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=15, t=15, b=30),
        paper_bgcolor='rgba(26, 32, 44, 0.8)',
        plot_bgcolor='rgba(26, 32, 44, 0.5)',
        font=dict(size=11),
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Returns %", row=2, col=1)
    
    return fig


# ============================================================================
# MAIN APP - Mobile Friendly Layout
# ============================================================================
def main():
    init_game_state()
    
    # Load tickers
    tickers, company_names = load_nifty_500_tickers()
    
    # Main content area
    if not st.session_state.game_started:
        # Welcome screen
        st.title('Predict the Market!')
        
        # Show overall stats if there are any trades
        if st.session_state.total_trades > 0:
            total_return = ((st.session_state.capital - st.session_state.initial_capital) / st.session_state.initial_capital) * 100
            win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💼 Capital", f"₹{st.session_state.capital:,.0f}")
            with col2:
                st.metric("📊 P&L", f"₹{st.session_state.realized_pnl:+,.0f}", 
                         delta_color="normal" if st.session_state.realized_pnl >= 0 else "inverse")
            with col3:
                st.metric("🎯 Trades", st.session_state.total_trades)
            with col4:
                st.metric("✅ Win Rate", f"{win_rate:.0f}%", f"{st.session_state.winning_trades}/{st.session_state.total_trades}")
        
        # Stock Selection Section
        st.markdown("### 🎯 Start Trading")
        
        # Selection method tabs
        tab1, tab2 = st.tabs(["🎲 Random", "📝 Manual"])
        
        with tab1:
            st.markdown("Get a random stock and date to trade:")
            if st.button("🎲 Generate Random Stock", type="primary", width='stretch'):
                random_ticker = random.choice(tickers)
                
                with st.spinner(f"Loading {random_ticker}..."):
                    stock_data = download_stock_data(random_ticker)
                    
                    if stock_data is not None and not stock_data.empty:
                        available_dates = get_available_dates_with_data(stock_data, min_candles=10)
                        
                        if not available_dates:
                            st.error(f"No data for {random_ticker}. Try again.")
                        else:
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
                                st.rerun()
                            else:
                                st.error("Not enough data. Try again.")
                    else:
                        st.error(f"Could not load {random_ticker}")
        
        with tab2:
            selected_ticker = st.selectbox(
                "Select Stock:",
                options=tickers,
                format_func=lambda x: f"{x} - {company_names.get(x, '')[:20]}",
                key="manual_ticker"
            )
            
            if st.button("📥 Load Stock", width='stretch', key="load_stock_btn"):
                with st.spinner(f"Loading {selected_ticker}..."):
                    stock_data = download_stock_data(selected_ticker)
                    if stock_data is not None:
                        st.session_state.temp_stock_data = stock_data
                        st.session_state.temp_ticker = selected_ticker
                        st.rerun()
            
            if 'temp_stock_data' in st.session_state and st.session_state.temp_stock_data is not None:
                stock_data = st.session_state.temp_stock_data
                available_dates = get_available_dates_with_data(stock_data, min_candles=10)
                
                if available_dates:
                    selected_date = st.selectbox(
                        "Select Date:",
                        options=available_dates[::-1],
                        format_func=lambda x: x.strftime("%Y-%m-%d"),
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
                            del st.session_state.temp_stock_data
                            del st.session_state.temp_ticker
                            st.rerun()
                        else:
                            st.error("Not enough data for this date.")
                else:
                    st.error("No valid dates found.")
        
        # Game Rules in expander
        with st.expander("📜 How to Play"):
            st.markdown("""
            **🎯 Objective:** Predict market direction!
            
            **📈 LONG** = Price going UP  
            **📉 SHORT** = Price going DOWN  
            **⏭️ NEXT** = Reveal next candle  
            **💰 CLOSE** = Lock in profit/loss
            
            Start with ₹1,00,000. Beat the market!
            """)
        
        # Reset button if there are trades
        if st.session_state.total_trades > 0:
            if st.button("🔃 Reset All Progress", width='stretch'):
                reset_game()
                st.rerun()
    
    else:
        # Game screen - Stock info header at the very top
        current_price = get_current_price()
        candles_remaining = len(st.session_state.day_data) - st.session_state.current_time_index if st.session_state.day_data is not None else 0
        current_time = ""
        if st.session_state.day_data is not None and len(st.session_state.day_data) > 0:
            idx = min(st.session_state.current_time_index - 1, len(st.session_state.day_data) - 1)
            if idx >= 0:
                current_time = st.session_state.day_data.index[idx].strftime('%H:%M')
        
        st.markdown(f"**{st.session_state.current_ticker}** | {st.session_state.current_date} | {current_time} | ₹{current_price:,.2f} | ⏱️ {candles_remaining} left")
        
        # Calculate metrics for later use
        unrealized_pnl = calculate_current_pnl()
        total_value = st.session_state.capital + unrealized_pnl
        total_return = ((total_value - st.session_state.initial_capital) / st.session_state.initial_capital) * 100
        win_rate = (st.session_state.winning_trades / st.session_state.total_trades * 100) if st.session_state.total_trades > 0 else 0
        
        # Chart - full width
        chart = create_chart()
        # Define the configuration dictionary
        config = {
            'displayModeBar': False
        }
        if chart:
            st.plotly_chart(chart, width='stretch', key="main_chart", config=config)
        
        # Position status and trading controls
        if st.session_state.position is not None:
            pnl = calculate_current_pnl()
            if pnl >= 0:
                st.success(f"**{st.session_state.position} Position** @ ₹{st.session_state.entry_price:,.0f} | Qty: {st.session_state.position_size} | P&L: **₹{pnl:+,.0f}**")
            else:
                st.error(f"**{st.session_state.position} Position** @ ₹{st.session_state.entry_price:,.0f} | Qty: {st.session_state.position_size} | P&L: **₹{pnl:+,.0f}**")
        
        # Trading buttons - single row layout for visibility
        if st.session_state.day_ended:
            st.warning("📅 Market day ended!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Day", width='stretch'):
                    reset_day()
                    st.session_state.game_started = False
                    st.rerun()
            with col2:
                if st.button("🔃 Reset All", width='stretch'):
                    reset_game()
                    st.rerun()
        else:
            if st.session_state.position is None:
                # No position - show LONG, SHORT, NEXT in a row
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("📈 LONG", width='stretch', key="btn_long", type="primary"):
                        if open_position('LONG'):
                            st.rerun()
                with col2:
                    if st.button("📉 SHORT", width='stretch', key="btn_short"):
                        if open_position('SHORT'):
                            st.rerun()
                with col3:
                    if st.button("⏭️ NEXT", width='stretch', key="btn_next"):
                        advance_candle()
                        st.rerun()
            else:
                # In position - show CLOSE and NEXT in a row
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("💰 CLOSE POSITION", width='stretch', key="btn_close", type="primary"):
                        close_position()
                        st.rerun()
                with col2:
                    if st.button("⏭️ NEXT CANDLE", width='stretch', key="btn_next"):
                        advance_candle()
                        st.rerun()
        
        # Compact controls row
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 New Day", width='stretch', key="new_day_btn"):
                reset_day()
                st.session_state.game_started = False
                st.rerun()
        with col2:
            if st.button("🔃 Reset", width='stretch', key="reset_btn"):
                reset_game()
                st.rerun()
        with col3:
            pass  # Empty for balance
        
        # Compact trade history in expander
        with st.expander("📜 Trade History"):
            day_trades = [t for t in st.session_state.trade_history 
                         if t['date'] == st.session_state.current_date]
            
            if day_trades:
                trade_df = pd.DataFrame(day_trades)
                trade_df['pnl_formatted'] = trade_df['pnl'].apply(lambda x: f"₹{x:+,.0f}")
                display_df = trade_df[['type', 'entry_time', 'exit_time', 'pnl_formatted']]
                display_df.columns = ['Type', 'Entry', 'Exit', 'P&L']
                st.dataframe(display_df, width='stretch', hide_index=True, height=120)
            else:
                st.info("No trades yet")
            
            if len(st.session_state.trade_history) > len(day_trades):
                st.markdown("**All Trades:**")
                all_trades_df = pd.DataFrame(st.session_state.trade_history)
                all_trades_df['pnl_formatted'] = all_trades_df['pnl'].apply(lambda x: f"₹{x:+,.0f}")
                all_trades_df['date'] = all_trades_df['date'].astype(str)
                display_all = all_trades_df[['ticker', 'date', 'type', 'pnl_formatted']]
                display_all.columns = ['Ticker', 'Date', 'Type', 'P&L']
                st.dataframe(display_all, width='stretch', hide_index=True, height=120)
        
        # Metrics row - moved to bottom
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💼 Capital", f"₹{st.session_state.capital:,.0f}")
        with col2:
            st.metric("📊 Open P&L", f"₹{unrealized_pnl:+,.0f}", 
                     delta_color="normal" if unrealized_pnl >= 0 else "inverse")
        with col3:
            st.metric("📈 Return", f"{total_return:+.1f}%",
                     delta_color="normal" if total_return >= 0 else "inverse")
        with col4:
            st.metric("🎯 Win Rate", f"{win_rate:.0f}%", f"{st.session_state.winning_trades}/{st.session_state.total_trades}")


if __name__ == "__main__":
    main()
