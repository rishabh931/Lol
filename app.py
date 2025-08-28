import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
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
        'tata motors': 'TATAMOTORS.NS',
        'tcs': 'TCS.NS',
        'lupin': 'LUPIN.NS',
        'cipla': 'CIPLA.NS',
        'dr reddy': 'DRREDDY.NS',
        'apollo hospitals': 'APOLLOHOSP.NS',
        'divis labs': 'DIVISLAB.NS'
    }
    
    stock_name_lower = stock_name.lower().strip()
    if stock_name_lower in stock_mappings:
        return stock_mappings[stock_name_lower]
    else:
        # Try to construct symbol
        symbol = stock_name.upper().replace(' ', '') + '.NS'
        return symbol

# Fetch and process data
@st.cache_data(show_spinner=False)
def fetch_quarterly_financials(symbol: str) -> pd.DataFrame:
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            stock = yf.Ticker(symbol)
            
            # Get quarterly financial statements
            quarterly_income = stock.quarterly_financials
            
            if quarterly_income.empty:
                raise ValueError(f"No financial data found for {symbol}")
            
            # Extract required metrics
            quarters = quarterly_income.columns[:10]  # Last 10 quarters
            
            data = []
            for quarter in quarters:
                try:
                    # Handle different column name variations
                    sales_cols = ['Total Revenue', 'Revenue', 'Total Revenues', 'Net Sales']
                    operating_income_cols = ['Operating Income', 'Operating Profit', 'EBIT Operating Income']
                    net_income_cols = ['Net Income', 'Net Income Common Stockholders', 'Net Income To Common Shareholders']
                    
                    # Get Sales/Revenue
                    sales = None
                    for col in sales_cols:
                        if col in quarterly_income.index:
                            sales = quarterly_income.loc[col, quarter]
                            break
                    if sales is None:
                        sales = quarterly_income.loc[quarterly_income.index[0], quarter]
                    
                    # Get Operating Profit
                    operating_profit = None
                    for col in operating_income_cols:
                        if col in quarterly_income.index:
                            operating_profit = quarterly_income.loc[col, quarter]
                            break
                    if operating_profit is None:
                        # Calculate from EBIT if available
                        ebit_cols = ['EBIT', 'Earnings Before Interest and Taxes']
                        for col in ebit_cols:
                            if col in quarterly_income.index:
                                operating_profit = quarterly_income.loc[col, quarter]
                                break
                    
                    # Get Net Profit
                    net_profit = None
                    for col in net_income_cols:
                        if col in quarterly_income.index:
                            net_profit = quarterly_income.loc[col, quarter]
                            break
                    
                    # Calculate metrics
                    if sales and operating_profit:
                        opm_percent = (operating_profit / sales) * 100
                    else:
                        opm_percent = None
                    
                    # Get shares outstanding for EPS calculation
                    shares_outstanding = None
                    try:
                        shares_data = stock.info.get('sharesOutstanding', None)
                        if shares_outstanding:
                            shares_outstanding = shares_data
                        else:
                            shares_hist = stock.get_shares_full()
                            if shares_hist is not None and len(shares_hist) > 0:
                                shares_outstanding = shares_hist.iloc[-1]
                    except:
                        pass
                    
                    eps = None
                    if net_profit and shares_outstanding:
                        eps = net_profit / shares_outstanding
                    
                    data.append({
                        'Quarter': quarter.strftime('%Y-%m'),
                        'Sales': sales,
                        'Operating_Profit': operating_profit,
                        'OPM_Percent': opm_percent,
                        'Net_Profit': net_profit,
                        'EPS': eps
                    })
                except Exception as e:
                    continue
            
            df = pd.DataFrame(data)
            
            # Clean and format the data
            numeric_columns = ['Sales', 'Operating_Profit', 'OPM_Percent', 'Net_Profit', 'EPS']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
    except Exception as e:
        raise Exception(f"Error fetching data for {symbol}: {str(e)}")

# Format numbers for display
def format_numbers(df: pd.DataFrame) -> pd.DataFrame:
    formatted_df = df.copy()
    
    # Format large numbers in crores
    for col in ['Sales', 'Operating_Profit', 'Net_Profit']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"₹{x/10000000:.2f} Cr" if pd.notnull(x) and x != 0 else "N/A"
            )
    
    # Format percentages
    if 'OPM_Percent' in formatted_df.columns:
        formatted_df['OPM_Percent'] = formatted_df['OPM_Percent'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
        )
    
    # Format EPS
    if 'EPS' in formatted_df.columns:
        formatted_df['EPS'] = formatted_df['EPS'].apply(
            lambda x: f"₹{x:.2f}" if pd.notnull(x) else "N/A"
        )
    
    return formatted_df

# Create visualizations
def create_visualizations(df: pd.DataFrame):
    # Convert back to numeric for plotting
    plot_df = df.copy()
    for col in ['Sales', 'Operating_Profit', 'Net_Profit']:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].apply(
                lambda x: float(x.replace('₹', '').replace(' Cr', '')) * 10000000 if isinstance(x, str) and x != 'N/A' else (x if pd.notnull(x) else 0)
            )
    
    if 'OPM_Percent' in plot_df.columns:
        plot_df['OPM_Percent'] = plot_df['OPM_Percent'].apply(
            lambda x: float(x.replace('%', '')) if isinstance(x, str) and x != 'N/A' else (x if pd.notnull(x) else 0)
        )
    
    if 'EPS' in plot_df.columns:
        plot_df['EPS'] = plot_df['EPS'].apply(
            lambda x: float(x.replace('₹', '')) if isinstance(x, str) and x != 'N/A' else (x if pd.notnull(x) else 0)
        )
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Quarterly Financial Analysis - {stock_name.upper()}', fontsize=16, fontweight='bold')
    
    # 1. Sales and Profits trend
    ax1 = axes[0, 0]
    x_pos = range(len(plot_df))
    ax1.plot(x_pos, plot_df['Sales'], marker='o', linewidth=2, label='Sales (₹Cr)', color='#1f77b4')
    ax1.plot(x_pos, plot_df['Operating_Profit'], marker='s', linewidth=2, label='Operating Profit (₹Cr)', color='#ff7f0e')
    ax1.plot(x_pos, plot_df['Net_Profit'], marker='^', linewidth=2, label='Net Profit (₹Cr)', color='#2ca02c')
    ax1.set_title('Revenue and Profit Trends')
    ax1.set_xlabel('Quarters')
    ax1.set_ylabel('Amount (₹Cr)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(plot_df['Quarter'], rotation=45)
    
    # 2. OPM Percentage trend
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, plot_df['OPM_Percent'], color='#2ca02c', alpha=0.7)
    ax2.set_title('Operating Profit Margin (OPM %)')
    ax2.set_xlabel('Quarters')
    ax2.set_ylabel('OPM (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_df['Quarter'], rotation=45)
    
    # Color bars based on value
    for i, bar in enumerate(bars):
        if i < len(plot_df['OPM_Percent']):
            value = plot_df['OPM_Percent'].iloc[i]
            bar.set_color('green' if value > 0 else 'red')
    
    # 3. EPS trend
    ax3 = axes[1, 0]
    ax3.plot(x_pos, plot_df['EPS'], marker='D', linewidth=2, color='#9467bd')
    ax3.set_title('Earnings Per Share (EPS)')
    ax3.set_xlabel('Quarters')
    ax3.set_ylabel('EPS (₹)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(plot_df['Quarter'], rotation=45)
    
    # 4. Profitability heatmap
    ax4 = axes[1, 1]
    heatmap_data = plot_df[['OPM_Percent']].copy()
    heatmap_data = heatmap_data.set_index(plot_df['Quarter'])
    sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'OPM %'})
    ax4.set_title('OPM Trend Heatmap')
    
    plt.tight_layout()
    return fig

# Prepare data for AI analysis
def prepare_ai_analysis_data(df: pd.DataFrame) -> str:
    # Convert to clean numeric format for analysis
    analysis_df = df.copy()
    
    # Remove formatting for AI analysis
    for col in ['Sales', 'Operating_Profit', 'Net_Profit']:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].apply(
                lambda x: float(x.replace('₹', '').replace(' Cr', '')) * 10000000 if isinstance(x, str) and x != 'N/A' else (x if pd.notnull(x) else 0)
            )
    
    if 'OPM_Percent' in analysis_df.columns:
        analysis_df['OPM_Percent'] = analysis_df['OPM_Percent'].apply(
            lambda x: float(x.replace('%', '')) if isinstance(x, str) and x != 'N/A' else (x if pd.notnull(x) else 0)
        )
    
    if 'EPS' in analysis_df.columns:
        analysis_df['EPS'] = analysis_df['EPS'].apply(
            lambda x: float(x.replace('₹', '')) if isinstance(x, str) and x != 'N/A' else (x if pd.notnull(x) else 0)
        )
    
    # Create summary statistics
    summary_stats = {
        'average_sales': analysis_df['Sales'].mean(),
        'average_opm': analysis_df['OPM_Percent'].mean(),
        'average_eps': analysis_df['EPS'].mean(),
        'sales_growth': ((analysis_df['Sales'].iloc[0] - analysis_df['Sales'].iloc[-1]) / analysis_df['Sales'].iloc[-1] * 100) if len(analysis_df) > 1 and analysis_df['Sales'].iloc[-1] != 0 else 0,
        'profit_growth': ((analysis_df['Net_Profit'].iloc[0] - analysis_df['Net_Profit'].iloc[-1]) / analysis_df['Net_Profit'].iloc[-1] * 100) if len(analysis_df) > 1 and analysis_df['Net_Profit'].iloc[-1] != 0 else 0
    }
    
    # Prepare data string for AI
    data_string = f"""
    Financial Analysis for Indian Stock: {stock_name.upper()}
    ==================================
    
    Last 10 Quarters Data:
    {analysis_df.to_string(index=False)}
    
    Summary Statistics:
    - Average Quarterly Sales: ₹{summary_stats['average_sales']/10000000:.2f} Cr
    - Average OPM: {summary_stats['average_opm']:.2f}%
    - Average EPS: ₹{summary_stats['average_eps']:.2f}
    - Sales Growth (latest vs oldest quarter): {summary_stats['sales_growth']:.2f}%
    - Profit Growth (latest vs oldest quarter): {summary_stats['profit_growth']:.2f}%
    """
    
    return data_string

# AI analysis
@st.cache_data(show_spinner=False)
def ai_analysis(data_string: str, _api_key: str) -> str:
    try:
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        As a financial analyst, analyze the following quarterly financial data for an Indian company. 
        Provide insights on:
        
        1. Sales Performance: Trend analysis, growth patterns, seasonal variations
        2. Profitability Analysis: Operating profit margin trends, net profit performance
        3. EPS Analysis: Earnings per share trends and implications
        4. Overall Financial Health: Company's financial position based on the data
        5. Key Concerns: Any red flags or areas of concern
        6. Future Outlook: Based on trends, what to expect going forward
        7. Investment Perspective: Brief view on investment potential
        
        {data_string}
        
        Provide a comprehensive but concise analysis in clear sections.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis failed: {str(e)}"

# Main execution
try:
    symbol = get_stock_symbol(stock_name)
    st.info(f"🔍 Analyzing: {stock_name.upper()} ({symbol})")
    
    # Fetch data
    raw_data = fetch_quarterly_financials(symbol)
    
    if raw_data.empty:
        st.error("❌ No financial data available for this stock. Please check the stock name.")
        st.stop()
    
    # Format for display
    display_data = format_numbers(raw_data)
    
    # Display key metrics
    st.markdown('<h2 class="sub-header">📈 Key Financial Metrics</h2>', unsafe_allow_html=True)
    
    latest_data = raw_data.iloc[0]  # Most recent quarter
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Latest Sales", 
                  f"₹{latest_data['Sales']/10000000:.2f} Cr" if pd.notnull(latest_data['Sales']) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Operating Profit", 
                  f"₹{latest_data['Operating_Profit']/10000000:.2f} Cr" if pd.notnull(latest_data['Operating_Profit']) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("OPM %", 
                  f"{latest_data['OPM_Percent']:.2f}%" if pd.notnull(latest_data['OPM_Percent']) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Net Profit", 
                  f"₹{latest_data['Net_Profit']/10000000:.2f} Cr" if pd.notnull(latest_data['Net_Profit']) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("EPS", 
                  f"₹{latest_data['EPS']:.2f}" if pd.notnull(latest_data['EPS']) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display data table
    st.markdown('<h2 class="sub-header">📋 Quarterly Financial Data</h2>', unsafe_allow_html=True)
    st.dataframe(display_data, use_container_width=True)
    
    # Display visualizations
    st.markdown('<h2 class="sub-header">📊 Financial Trends</h2>', unsafe_allow_html=True)
    fig = create_visualizations(raw_data)
    st.pyplot(fig)
    
    # AI Analysis
    st.markdown('<h2 class="sub-header">🤖 AI-Powered Analysis</h2>', unsafe_allow_html=True)
    with st.spinner("Generating AI analysis..."):
        ai_input_data = prepare_ai_analysis_data(raw_data)
        ai_result = ai_analysis(ai_input_data, api_key)
        st.markdown(ai_result)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown('<div class="footer">📊 Indian Stock Financial Analyzer | Powered by Yahoo Finance & Gemini AI</div>', unsafe_allow_html=True)
