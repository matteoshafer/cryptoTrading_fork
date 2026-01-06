"""
Cryptocurrency Trading Strategy Dashboard
Interactive dashboard for viewing trading signals, backtest results, and model performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from main import run_trading_strategy, backtest_strategy, load_data
from functions import fullDataPath
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Cryptocurrency Trading Strategy Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Configuration")
    
    # Coin selection
    coin = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=["BTC", "ETH"],
        index=0
    )
    
    # Ensemble parameters
    st.sidebar.subheader("🎯 Trading Parameters")
    
    buy_min_bull = st.sidebar.slider(
        "Min Bullish Models for Buy",
        min_value=1,
        max_value=10,
        value=1,
        help="Minimum number of models that must be bullish to trigger a buy signal"
    )
    
    buy_threshold = st.sidebar.slider(
        "Buy Return Threshold (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Minimum predicted return percentage to trigger buy"
    ) / 100.0
    
    sell_max_bull = st.sidebar.slider(
        "Max Bullish Models for Sell",
        min_value=0,
        max_value=5,
        value=0,
        help="Maximum number of bullish models allowed before sell signal"
    )
    
    sell_threshold = st.sidebar.slider(
        "Sell Return Threshold (%)",
        min_value=-2.0,
        max_value=0.0,
        value=0.0,
        step=0.1,
        help="Maximum predicted return percentage to trigger sell"
    ) / 100.0
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    
    # Get available data range
    try:
        data = load_data(coin)
        if not data.empty:
            min_date = data.index.min().date() if hasattr(data.index.min(), 'date') else pd.to_datetime(data.index.min()).date()
            max_date = data.index.max().date() if hasattr(data.index.max(), 'date') else pd.to_datetime(data.index.max()).date()
        else:
            min_date = datetime(2024, 1, 1).date()
            max_date = datetime.now().date()
    except:
        min_date = datetime(2024, 1, 1).date()
        max_date = datetime.now().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
        return
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running trading strategy analysis..."):
            try:
                # Run trading strategy
                result_df = run_trading_strategy(
                    coin=coin,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    verbose=False,
                    buy_min_bull_count=buy_min_bull,
                    buy_threshold_return=buy_threshold,
                    sell_max_bull_count=sell_max_bull,
                    sell_threshold_return=sell_threshold
                )
                
                if not result_df.empty:
                    # Run backtest
                    backtest_results = backtest_strategy(result_df, initial_capital=10000.0)
                    
                    # Store in session state
                    st.session_state['result_df'] = result_df
                    st.session_state['backtest_results'] = backtest_results
                    st.session_state['coin'] = coin
                    st.success("✅ Analysis completed successfully!")
                else:
                    st.error("No data available for the selected date range")
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if 'result_df' in st.session_state and 'backtest_results' in st.session_state:
        result_df = st.session_state['result_df']
        backtest_results = st.session_state['backtest_results']
        coin = st.session_state.get('coin', coin)
        
        # Prepare data for visualization
        result_df_display = result_df.copy()
        if 'time' in result_df_display.columns:
            result_df_display['date'] = pd.to_datetime(result_df_display['time'])
            date_col = 'date'
        elif result_df_display.index.name == 'time' or isinstance(result_df_display.index, pd.DatetimeIndex):
            result_df_display['date'] = pd.to_datetime(result_df_display.index)
            date_col = 'date'
        else:
            result_df_display['date'] = pd.to_datetime(result_df_display.index)
            date_col = 'date'
        
        # Main metrics
        st.header("📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{backtest_results['total_return']*100:.2f}%",
                delta=f"{backtest_results['final_capital'] - backtest_results['initial_capital']:,.2f}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Final Capital",
                f"${backtest_results['final_capital']:,.2f}",
                delta=f"{backtest_results['final_capital'] - backtest_results['initial_capital']:,.2f}",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                backtest_results['total_trades'],
                delta=f"Win Rate: {backtest_results['win_rate']*100:.1f}%"
            )
        
        with col4:
            avg_return = backtest_results['avg_return_per_trade'] * 100 if backtest_results['total_trades'] > 0 else 0
            st.metric(
                "Avg Return/Trade",
                f"{avg_return:.2f}%",
                delta=f"Max: {backtest_results['max_return']*100:.2f}%"
            )
        
        # Price chart with signals
        st.header("📈 Price Chart with Trading Signals")
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=result_df_display[date_col],
            y=result_df_display['P_t'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Buy signals
        buy_signals = result_df_display[result_df_display['ensemble_buy'] == 1]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals[date_col],
                y=buy_signals['P_t'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=15, color='green')
            ))
        
        # Sell signals
        sell_signals = result_df_display[result_df_display['ensemble_sell'] == 1]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals[date_col],
                y=sell_signals['P_t'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=15, color='red')
            ))
        
        # SMA20 if available
        if 'SMA_20' in result_df_display.columns:
            fig.add_trace(go.Scatter(
                x=result_df_display[date_col],
                y=result_df_display['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{coin} Price with Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Two columns for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Signal Summary")
            # Count signals by model
            buy_cols = [col for col in result_df.columns if col.endswith('_buy')]
            signal_counts = []
            for col in buy_cols:
                model_name = col.replace('_buy', '').replace('_', ' ')
                signal_counts.append({
                    'Model': model_name,
                    'Buy Signals': int(result_df_display[col].sum())
                })
            
            signal_df = pd.DataFrame(signal_counts)
            signal_df = signal_df.sort_values('Buy Signals', ascending=False)
            
            fig_bar = px.bar(
                signal_df,
                x='Model',
                y='Buy Signals',
                title='Buy Signals by Model',
                color='Buy Signals',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("📈 Predicted Returns Distribution")
            if 'predicted_return' in result_df_display.columns:
                fig_hist = px.histogram(
                    result_df_display,
                    x='predicted_return',
                    nbins=30,
                    title='Distribution of Predicted Returns',
                    labels={'predicted_return': 'Predicted Return', 'count': 'Frequency'}
                )
                fig_hist.update_layout(height=400, template='plotly_white')
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Return")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Predicted returns data not available")
        
        # Trade log
        trades = backtest_results.get('trades', [])
        if trades:
            st.header("📋 Trade Log")
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            # Trade returns visualization
            if 'return_pct' in trades_df.columns:
                trade_returns = trades_df[trades_df['action'] == 'SELL']['return_pct'] * 100
                if len(trade_returns) > 0:
                    fig_returns = px.bar(
                        x=range(1, len(trade_returns) + 1),
                        y=trade_returns,
                        title='Trade Returns (%)',
                        labels={'x': 'Trade Number', 'y': 'Return (%)'},
                        color=trade_returns,
                        color_continuous_scale=['red', 'gray', 'green']
                    )
                    fig_returns.add_hline(y=0, line_dash="dash", line_color="black")
                    fig_returns.update_layout(height=300, template='plotly_white')
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.info("No trades were executed during this period")
        
        # Data table
        with st.expander("📄 View Raw Data"):
            st.dataframe(result_df_display, use_container_width=True, height=400)
        
        # Download button
        csv = result_df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"{coin}_trading_signals_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome message
        st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis' to get started!")
        
        # Show available data info
        st.sidebar.subheader("📊 Data Information")
        try:
            data = load_data(coin)
            if not data.empty:
                st.sidebar.success(f"✅ Data available")
                st.sidebar.write(f"**Date Range:**")
                st.sidebar.write(f"{data.index.min().date()} to {data.index.max().date()}")
                st.sidebar.write(f"**Total Days:** {len(data)}")
            else:
                st.sidebar.warning("⚠️ No data available")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
