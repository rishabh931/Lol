import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import atexit
import warnings

warnings.filterwarnings('ignore')

# === FIX FOR YFINANCE DATA LEAK (UNCLOSED SOCKET) ===
def create_requests_session():
    """Create a requests session with proper retry and cleanup"""
    session = requests.Session()
    
    # Add retry strategy
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Patch close to ensure cleanup
    original_close = session.close

    def safe_close():
        try:
            session.adapters.clear()
            original_close()
        except:
            pass

    session.close = safe_close
    return session

# Create global session for yfinance
SESSION = create_requests_session()
atexit.register(SESSION.close)  # Ensure session closes on exit

# === END DATA LEAK FIX ===

# Configure page
st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
    }
    .warning {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Main header
st.markdown('<h1 class="main-header">📊 Indian Stock Financial Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# API Key input (sidebar)
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password", key="api_input")
    
    if api_key:
        st.session_state.api_key_set = True
        try:
            genai.configure(api_key=api_key)
            st.success("✅ API Key configured successfully!")
        except Exception as e:
            st.error(f"❌ Error configuring API: {str(e)}")
            st.session_state.api_key_set = False
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This tool analyzes Indian stocks by fetching:
    - Last 10 quarters of financial data
    - Sales, Operating Profit, OPM%, Net Profit, EPS
    - Visual charts and AI-powered analysis
    
    Enter any Indian stock name to get started!
    """)

# Main content
if not st.session_state.api_key_set:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to use this tool.")
    st.stop()

# Stock input
col1, col2 = st.columns([3, 1])
with col1:
    stock_name = st.text_input("Enter Indian Stock Name:", placeholder="e.g., Reliance, TCS, HDFC Bank")
with col2:
    analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if not stock_name:
    st.info("👆 Please enter an Indian stock name to begin analysis.")
    st.stop()

if not analyze_button:
    st.info("👆 Click 'Analyze Stock' to fetch financial data.")
    st.stop()

# Stock symbol mapping
def get_stock_symbol(stock_name: str) -> str:
    stock_mappings = {
        'reliance': 'RELIANCE.NS',
        'tcs': 'TCS.NS',
        'hdfc bank': 'HDFCBANK.NS',
        'infosys': 'INFY.NS',
        'icici bank': 'ICICIBANK.NS',
        'hdfc': 'HDFC.NS',
        'sbi': 'SBIN.NS',
        'kotak mahindra': 'KOTAKBANK.NS',
        'bajaj finance': 'BAJFINANCE.NS',
        'hindustan unilever': 'HINDUNILVR.NS',
        'itc': 'ITC.NS',
        'wipro': 'WIPRO.NS',
        'axis bank': 'AXISBANK.NS',
        'l&t': 'LT.NS',
        'maruti suzuki': 'MARUTI.NS',
        'nestle': 'NESTLEIND.NS',
        'tatamotors': 'TATAMOTORS.NS',
        'mahindra & mahindra': 'M&M.NS',
        'adani enterprises': 'ADANIENT.NS',
        'adani green': 'ADANIGREEN.NS',
        'adani ports': 'ADANIPORTS.NS',
        'asian paints': 'ASIANPAINT.NS',
        'bharti airtel': 'BHARTIARTL.NS',
        'ultratech cement': 'ULTRACEMCO.NS',
        'sun pharmaceutical': 'SUNPHARMA.NS',
        'hcl technologies': 'HCLTECH.NS',
        'tech mahindra': 'TECHM.NS',
        'power grid': 'POWERGRID.NS',
        'ntpc': 'NTPC.NS',
        'ongc': 'ONGC.NS',
        'ioc': 'IOC.NS',
        'bpcl': 'BPCL.NS',
        'hpcl': 'HINDPETRO.NS',
        'coal india': 'COALINDIA.NS',
        'vedanta': 'VEDL.NS',
        'hindalco': 'HINDALCO.NS',
        'grasim': 'GRASIM.NS',
        'bajaj finserv': 'BAJAJFINSV.NS',
        'bajaj auto': 'BAJAJ-AUTO.NS',
        'hero motocorp': 'HEROMOTOCO.NS',
        'eicher motors': 'EICHERMOT.NS',
        'tata steel': 'TATASTEEL.NS',
        'lupin': 'LUPIN.NS',
        'cipla': 'CIPLA.NS',
        'dr reddy': 'DRREDDY.NS',
        'apollo hospitals': 'APOLLOHOSP.NS',
        'divis labs': 'DIVISLAB.NS'
    }
    
    stock_name_lower = stock_name.lower().strip()
    return stock_mappings.get(stock_name_lower, stock_name.upper().replace(' ', '') + '.NS')

# Fetch and process data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_quarterly_financials(symbol: str):
    try:
        with st.spinner(f"📊 Fetching financial data for {symbol}..."):
            # Use the patched session
            stock = yf.Ticker(symbol, session=SESSION)
            
            quarterly_income = stock.quarterly_financials
            if quarterly_income.empty:
                raise ValueError("No quarterly financial data available.")
            
            quarters = quarterly_income.columns[:10]
            data = []
            
            for quarter in quarters:
                try:
                    # Revenue
                    sales = None
                    for col in ['Total Revenue', 'Revenue']:
                        if col in quarterly_income.index:
                            sales = quarterly_income.loc[col, quarter]
                            break
                    if sales is None:
                        sales = quarterly_income.iloc[0, quarterly_income.columns.get_loc(quarter)]
                    
                    # Operating Profit
                    op_profit = None
                    for col in ['Operating Income', 'Operating Profit', 'EBIT']:
                        if col in quarterly_income.index:
                            op_profit = quarterly_income.loc[col, quarter]
                            break
                    
                    # Net Profit
                    net_profit = None
                    for col in ['Net Income', 'Net Income Common Stockholders']:
                        if col in quarterly_income.index:
                            net_profit = quarterly_income.loc[col, quarter]
                            break
                    
                    # OPM %
                    opm = ((op_profit / sales) * 100) if sales and op_profit and sales != 0 else None
                    
                    # EPS (simplified)
                    eps = None
                    if net_profit:
                        try:
                            shares = stock.info.get('sharesOutstanding', 1e6)
                            eps = net_profit / shares
                        except:
                            pass
                    
                    data.append({
                        'Quarter': quarter.strftime('%Y-%m'),
                        'Sales': sales,
                        'Operating_Profit': op_profit,
                        'OPM_Percent': opm,
                        'Net_Profit': net_profit,
                        'EPS': eps
                    })
                except:
                    continue
            
            df = pd.DataFrame(data)
            numeric_cols = ['Sales', 'Operating_Profit', 'OPM_Percent', 'Net_Profit', 'EPS']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(how='all')  # Remove any all-NaN rows
    
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {str(e)}")
        return pd.DataFrame()

# Format numbers
def format_numbers(df: pd.DataFrame) -> pd.DataFrame:
    formatted_df = df.copy()
    for col in ['Sales', 'Operating_Profit', 'Net_Profit']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"₹{x/1e7:.2f} Cr" if pd.notnull(x) and x != 0 else "N/A"
            )
    formatted_df['OPM_Percent'] = formatted_df['OPM_Percent'].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
    )
    formatted_df['EPS'] = formatted_df['EPS'].apply(
        lambda x: f"₹{x:.2f}" if pd.notnull(x) else "N/A"
    )
    return formatted_df

# Create visualizations
def create_visualizations(df: pd.DataFrame):
    plot_df = df.copy()
    for col in ['Sales', 'Operating_Profit', 'Net_Profit']:
        plot_df[col] = plot_df[col].apply(
            lambda x: float(str(x).replace('₹', '').replace(' Cr', '')) * 1e7 if isinstance(x, str) and 'Cr' in str(x) else (x if pd.notnull(x) else 0)
        )
    plot_df['OPM_Percent'] = plot_df['OPM_Percent'].apply(
        lambda x: float(str(x).replace('%', '')) if isinstance(x, str) and '%' in str(x) else (x if pd.notnull(x) else 0)
    )
    plot_df['EPS'] = plot_df['EPS'].apply(
        lambda x: float(str(x).replace('₹', '')) if isinstance(x, str) and '₹' in str(x) else (x if pd.notnull(x) else 0)
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Financial Trends - {stock_name.title()}', fontsize=16, fontweight='bold')
    x_pos = range(len(plot_df))

    axes[0,0].plot(x_pos, plot_df['Sales']/1e7, 'o-', label='Sales (Cr)', color='#1f77b4')
    axes[0,0].plot(x_pos, plot_df['Operating_Profit']/1e7, 's-', label='Op Profit (Cr)', color='#ff7f0e')
    axes[0,0].plot(x_pos, plot_df['Net_Profit']/1e7, '^-', label='Net Profit (Cr)', color='#2ca02c')
    axes[0,0].set_title('Revenue & Profit (₹ Cr)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(plot_df['Quarter'], rotation=45)

    axes[0,1].bar(x_pos, plot_df['OPM_Percent'], color='#2ca02c', alpha=0.7)
    axes[0,1].set_title('OPM %')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(plot_df['Quarter'], rotation=45)

    axes[1,0].plot(x_pos, plot_df['EPS'], 'D-', color='#9467bd')
    axes[1,0].set_title('EPS (₹)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(plot_df['Quarter'], rotation=45)

    sns.heatmap(plot_df[['OPM_Percent']].T, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1,1], cbar_kws={'label': 'OPM %'})
    axes[1,1].set_title('OPM Heatmap')

    plt.tight_layout()
    return fig

# Prepare AI data
def prepare_ai_analysis_data(df: pd.DataFrame) -> str:
    analysis_df = df.copy()
    for col in ['Sales', 'Operating_Profit', 'Net_Profit']:
        analysis_df[col] = analysis_df[col].apply(
            lambda x: float(str(x).replace('₹', '').replace(' Cr', '')) * 1e7 if isinstance(x, str) else (x if pd.notnull(x) else 0)
        )
    analysis_df['OPM_Percent'] = analysis_df['OPM_Percent'].apply(
        lambda x: float(str(x).replace('%', '')) if isinstance(x, str) else (x if pd.notnull(x) else 0)
    )
    analysis_df['EPS'] = analysis_df['EPS'].apply(
        lambda x: float(str(x).replace('₹', '')) if isinstance(x, str) else (x if pd.notnull(x) else 0)
    )

    latest = analysis_df.iloc[0]
    return f"""
    Stock: {stock_name.upper()}
    Latest Sales: ₹{latest['Sales']/1e7:.2f} Cr
    OPM: {latest['OPM_Percent']:.2f}%
    EPS: ₹{latest['EPS']:.2f}
    Net Profit: ₹{latest['Net_Profit']/1e7:.2f} Cr
    """

# AI analysis
@st.cache_data(ttl=3600)
def get_ai_analysis(data_str: str, api_key: str):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze this Indian stock's quarterly performance:
        {data_str}
        
        Provide concise insights on: Sales trend, Profitability, EPS growth, Risks, and Investment view.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis failed: {str(e)}"

# === MAIN EXECUTION ===
try:
    symbol = get_stock_symbol(stock_name)
    st.info(f"🔍 Analyzing: **{stock_name.title()}** ({symbol})")

    raw_data = fetch_quarterly_financials(symbol)
    if raw_data.empty or len(raw_data) == 0:
        st.error("❌ No financial data found. Please check the stock name.")
    else:
        display_data = format_numbers(raw_data)

        # Metrics
        st.markdown('<h2 class="sub-header">📈 Key Metrics</h2>', unsafe_allow_html=True)
        cols = st.columns(5)
        latest = raw_data.iloc[0]
        metrics = [
            ("Sales", f"₹{latest['Sales']/1e7:.2f} Cr" if pd.notnull(latest['Sales']) else "N/A"),
            ("Op Profit", f"₹{latest['Operating_Profit']/1e7:.2f} Cr" if pd.notnull(latest['Operating_Profit']) else "N/A"),
            ("OPM %", f"{latest['OPM_Percent']:.2f}%" if pd.notnull(latest['OPM_Percent']) else "N/A"),
            ("Net Profit", f"₹{latest['Net_Profit']/1e7:.2f} Cr" if pd.notnull(latest['Net_Profit']) else "N/A"),
            ("EPS", f"₹{latest['EPS']:.2f}" if pd.notnull(latest['EPS']) else "N/A")
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown(f'<div class="metric-card"><b>{label}</b><br>{value}</div>', unsafe_allow_html=True)

        # Table
        st.markdown('<h2 class="sub-header">📋 Quarterly Data</h2>', unsafe_allow_html=True)
        st.dataframe(display_data, use_container_width=True)

        # Charts
        st.markdown('<h2 class="sub-header">📊 Financial Trends</h2>', unsafe_allow_html=True)
        fig = create_visualizations(raw_data)
        st.pyplot(fig)

        # AI Analysis
        st.markdown('<h2 class="sub-header">🤖 AI Analysis</h2>', unsafe_allow_html=True)
        with st.spinner("Generating AI insights..."):
            ai_input = prepare_ai_analysis_data(raw_data)
            ai_result = get_ai_analysis(ai_input, api_key)
            st.markdown(ai_result)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown('<div class="footer">📊 Indian Stock Analyzer | Data: Yahoo Finance | AI: Gemini</div>', unsafe_allow_html=True)
