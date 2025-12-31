# Trading Game Copilot Instructions

## Project Overview
A Streamlit-based market prediction game where users predict NIFTY 500 stock price movements using real 15-minute intraday charts from yfinance. Single-file architecture (`app.py`, ~900 lines) with session state management for game logic.

## Architecture & Design Patterns

### Single-File Structure
All code lives in [app.py](../app.py) organized in clear comment blocks:
- **Page Configuration** (lines 28-34): Streamlit setup with wide layout
- **Game Configuration** (lines 38-74): Constants for capital, market hours, colors
- **Custom CSS** (lines 78-159): Dark theme styling with metric cards and position highlights
- **Data Loading** (lines 163-295): yfinance integration with local parquet caching
- **Game State** (lines 299-342): Session state initialization and reset functions
- **Trading Logic** (lines 346-462): Position management and PnL calculations
- **Chart Rendering** (lines 466-588): Plotly candlestick charts with returns subplot
- **Main UI** (lines 592-896): Sidebar controls and main game interface

### State Management Pattern
**All game state lives in `st.session_state`** (not database). Key pattern:
```python
st.session_state.position  # None, 'LONG', or 'SHORT'
st.session_state.capital   # Updated on position close
st.session_state.day_data  # Current day's DataFrame (filtered to market hours)
st.session_state.current_time_index  # Which candle to reveal (starts at 3)
```

**Critical**: Always call `st.rerun()` after state mutations to refresh UI. Never directly modify capital without closing position.

### Data Caching Strategy
- **File-based cache**: `cache/*.parquet` for downloaded stock data
- **Cache TTL**: 6 hours (see [app.py](../app.py#L193-198))
- **Cache key**: `{ticker}_15m.parquet` format
- **yfinance limits**: 60 days of 15-minute data maximum
- Data filtered to NSE market hours (09:15-15:30) after download

## Development Workflow

### Environment Setup with uv
This project uses **uv** for fast Python package management:

```powershell
# Install dependencies (first time setup)
uv pip install -r requirements.txt

# Or if using uv's sync feature
uv sync
```

**Why uv**: 10-100x faster than pip, handles virtual environments automatically, Rust-powered resolver.

### Running the App
```powershell
# With uv (recommended - manages environment automatically)
uv run streamlit run app.py

# Or traditional approach
streamlit run app.py
```
**Always use Streamlit**, not `python app.py`. The app runs in watch mode - file changes auto-reload.

### Testing Changes
1. Make edits to [app.py](../app.py)
2. Streamlit auto-reloads (watch terminal for errors)
3. If app crashes, check terminal for stack trace
4. Common issues: session state key errors, DataFrame indexing, missing data validation

### Debugging Session State
Add temporary debug expander in sidebar:
```python
with st.expander("🐛 Debug"):
    st.write(st.session_state)
```
Remove before committing. Never log sensitive state in production.

## Project-Specific Conventions

### Data Flow
1. User selects stock/date → `download_stock_data()` (checks cache first)
2. `get_day_data()` extracts single date → `filter_market_hours()` removes pre/post hours
3. `st.session_state.day_data` stores filtered DataFrame
4. `st.session_state.current_time_index` controls visible candles (incremented by "NEXT" button)
5. Chart shows `day_data.iloc[:current_time_index]` (progressive reveal pattern)

### Position Lifecycle
```python
open_position('LONG')  # Sets entry_price, position_size (90% of capital)
advance_candle()       # Reveals next candle, auto-closes if day_ended
close_position()       # Calculates PnL, updates capital, appends to trade_history
```

**Critical**: Positions auto-close when `current_time_index >= len(day_data)`. Always validate data exists before opening positions (see [app.py](../app.py#L390-398)).

### UI Layout Pattern
- **Sidebar**: Stock selection, game controls, rules expander
- **Main area**: Metrics row (5 columns) → Chart + Controls (3:1 ratio) → Trade history expander
- **Color coding**: Green (#00ff88) for profit/LONG, Red (#ff4757) for loss/SHORT

### Error Handling
- Always check `if df is None or df.empty` before operations
- Bounds-check `current_time_index` against `len(day_data)`
- Graceful degradation: Show error message, don't crash app
- Example: [app.py](../app.py#L391-394) validates price before opening position

## Package Management

### Using uv Commands
```powershell
uv pip install <package>        # Add new dependency
uv pip list                     # List installed packages
uv pip freeze > requirements.txt # Update requirements
uv run python -c "import sys; print(sys.executable)"  # Check Python path
```

**Note**: `.venv/` is git-ignored. Use `uv venv` to recreate virtual environment if needed.

## External Dependencies

### yfinance Integration
- **Ticker format**: NSE stocks require `.NS` suffix (e.g., `RELIANCE.NS`)
- **Interval**: `interval="15m"` for 15-minute candles
- **Date range**: Last 60 days only (yfinance limitation)
- **Multi-level columns**: Always flatten with `df.columns.get_level_values(0)`
- **Timezone**: Convert `Asia/Kolkata` then localize to None for easier handling

### Data Files
- **ind_nifty500list.csv**: NSE stock universe (503 rows: Company Name, Industry, Symbol, Series, ISIN)
- **cache/**: Auto-created parquet files, safe to delete (will re-download)

## Common Modifications

### Adding New Indicators
1. Calculate in `get_day_data()` after returns calculation
2. Add subplot to `create_chart()` (update `row_heights` array)
3. Update `rows=` parameter in `make_subplots()`

### Changing Capital/Position Sizing
- Initial capital: `INITIAL_CAPITAL` constant (line 39)
- Position sizing: 90% of capital (line 405), change multiplier in `open_position()`

### Modifying Market Hours
- Edit `MARKET_TIMES` list (lines 43-49) for different intervals
- Update `filter_market_hours()` time range (lines 292-295)

## Known Issues
- **Empty README.md**: Intentionally empty (document if needed)
- **pyproject.toml missing deps**: Use `requirements.txt` instead (Streamlit project pattern)
- **Cache grows unbounded**: Manually clear `cache/` directory periodically

## Quick Reference
- **Reset game state**: Click "Reset Game" in sidebar or delete session state keys
- **Force data refresh**: Delete `cache/{ticker}_15m.parquet` and restart
- **Check data quality**: Look for `get_available_dates_with_data(..., min_candles=10)` filtering
