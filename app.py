import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration & Constants
# ============================================================================
PAGE_CONFIG = {
    "page_title": "Sales Analysis",
    "page_icon": "🛒",
    "layout": "wide"
}

REQUIRED_COLUMNS = {
    'Date', 'Net Amount', 'Quantity', 'Transaction No_', 
    'Store No_', 'Item No_', 'Item Category', 'Subgroup Desc',
    'Department Desc', 'Search Description'
}

APRIORI_CONFIG = {
    "min_support_range": (0.005, 0.05),
    "min_support_default": 0.01,
    "min_lift_range": (1.0, 15.0),
    "min_lift_default": 3.0,
}

COLOR_SCHEME = {
    "weekend": "#E07B54",
    "weekday": "#5B8DB8",
    "positive": "#5B8DB8",
    "negative": "#E07B54",
}

# ============================================================================
# Helper Functions
# ============================================================================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load and preprocess sales data with error handling."""
    try:
        sales = pd.read_excel(file)
        
        # Validate required columns
        missing_cols = REQUIRED_COLUMNS - set(sales.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data type conversions and feature engineering
        sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
        sales = sales.dropna(subset=['Date'])  # Remove rows with invalid dates
        
        sales['Month'] = sales['Date'].dt.month
        sales['Weekday'] = sales['Date'].dt.day_name()
        sales['YearMonth'] = sales['Date'].dt.to_period('M').astype(str)
        sales['Weekend'] = sales['Weekday'].isin(['Saturday', 'Sunday'])
        sales = sales.drop_duplicates()
        
        logger.info(f"Data loaded successfully: {len(sales)} rows")
        return sales
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def validate_data_quality(sales: pd.DataFrame) -> dict:
    """Check data quality and return warnings."""
    warnings = {}
    
    if len(sales) == 0:
        warnings['empty'] = "Dataset is empty after loading."
    
    if (sales['Net Amount'] > 0).sum() == 0:
        warnings['no_positive'] = "No positive sales found in data."
    
    null_counts = sales[list(REQUIRED_COLUMNS)].isnull().sum()
    if (null_counts > 0).any():
        warnings['nulls'] = f"Nulls found: {null_counts[null_counts > 0].to_dict()}"
    
    return warnings


def calculate_support_from_transaction_count(transaction_count: int, total_transactions: int) -> float:
    """Convert transaction count to support percentage."""
    if total_transactions == 0:
        return 0.0
    return transaction_count / total_transactions


@st.cache_data
def run_association_analysis(sales: pd.DataFrame, min_support: float, min_lift: float) -> Tuple[pd.DataFrame, int]:
    """Run analysis"""
    try:
        sc = sales[(sales['Net Amount'] > 0) & (sales['Item Category'] != 'Service')].copy()
        
        if len(sc) == 0:
            raise ValueError("No valid transactions found after filtering.")
        
        basket = (sc.groupby(['Transaction No_', 'Subgroup Desc'])['Quantity']
                 .sum().unstack(fill_value=0).gt(0).astype('bool'))
        
        freq = apriori(basket, min_support=min_support, use_colnames=True)
        
        if len(freq) == 0:
            return pd.DataFrame(), 0
        
        rules = association_rules(freq, metric='lift', min_threshold=min_lift)
        
        if len(rules) > 0:
            rules['IF'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['THEN'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            rules['rule'] = rules['IF'] + '  →  ' + rules['THEN']
            rules = rules.sort_values('lift', ascending=False)
        
        return rules, len(rules)
    
    except Exception as e:
        logger.error(f"Error in association analysis: {e}")
        raise


@st.cache_data
def prepare_pareto_data(sales: pd.DataFrame) -> pd.DataFrame:
    """Prepare Pareto analysis data."""
    pareto = (sales.groupby('Search Description')['Net Amount']
             .sum().sort_values(ascending=False).reset_index())
    pareto['Cumulative %'] = (pareto['Net Amount'].cumsum() / 
                             pareto['Net Amount'].sum() * 100)
    return pareto


@st.cache_data
def prepare_correlation_matrix(sales: pd.DataFrame) -> pd.DataFrame:
    """Prepare department correlation matrix."""
    dept = (sales.groupby(['Transaction No_', 'Department Desc'])['Quantity']
           .sum().unstack().fillna(0))
    return dept.corr()


# ============================================================================
# Chart Helper Functions
# ============================================================================
def create_lift_chart(rules: pd.DataFrame) -> go.Figure:
    """Create bar chart for top 10 association rules by lift."""
    top10 = rules.head(10)
    fig = px.bar(
        top10, 
        x='lift', 
        y='rule', 
        orientation='h', 
        color='lift', 
        color_continuous_scale='Reds',
        title='Top 10 Rules by Lift Score',
        labels={'lift': 'Lift Score', 'rule': ''}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
    return fig


def create_support_confidence_chart(rules: pd.DataFrame) -> go.Figure:
    """Create scatter plot for support vs confidence."""
    fig = px.scatter(
        rules, 
        x='support', 
        y='confidence', 
        size='lift', 
        color='lift',
        color_continuous_scale='YlOrRd',
        hover_data=['IF', 'THEN'],
        title='All Association Rules — Support vs Confidence'
    )
    return fig


def create_monthly_trend_chart(sales: pd.DataFrame) -> go.Figure:
    """Create line chart for monthly revenue trend."""
    monthly = sales.groupby('YearMonth')['Net Amount'].sum().reset_index()
    fig = px.line(
        monthly, 
        x='YearMonth', 
        y='Net Amount', 
        markers=True,
        title='Monthly Revenue Trend',
        labels={'Net Amount': 'Revenue (AED)', 'YearMonth': 'Month'}
    )
    fig.update_traces(line_color=COLOR_SCHEME['negative'], 
                     marker=dict(color='black', size=8))
    return fig


def create_weekend_vs_weekday_chart(sales: pd.DataFrame) -> go.Figure:
    """Create bar chart comparing weekend vs weekday sales."""
    week = sales.copy()
    week['Day Type'] = week['Weekend'].map({True: 'Weekend', False: 'Weekday'})
    week_avg = week.groupby('Day Type')['Net Amount'].mean().reset_index()
    
    fig = px.bar(
        week_avg, 
        x='Day Type', 
        y='Net Amount', 
        color='Day Type',
        color_discrete_map={
            'Weekend': COLOR_SCHEME['weekend'],
            'Weekday': COLOR_SCHEME['weekday']
        },
        title='Average Sale — Weekend vs Weekday'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_daily_revenue_chart(sales: pd.DataFrame) -> go.Figure:
    """Create bar chart for daily revenue."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_sales = sales.groupby('Weekday')['Net Amount'].sum().reindex(day_order).reset_index()
    
    fig = px.bar(
        day_sales, 
        x='Weekday', 
        y='Net Amount',
        color='Net Amount',
        color_continuous_scale='Blues',
        title='Total Revenue by Day of Week'
    )
    return fig


def create_pareto_chart(pareto: pd.DataFrame) -> go.Figure:
    """Create Pareto chart with cumulative percentage."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(pareto))),
        y=pareto['Net Amount'],
        name='Revenue',
        marker_color=COLOR_SCHEME['positive'],
        opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(pareto))),
        y=pareto['Cumulative %'],
        name='Cumulative %',
        yaxis='y2',
        line=dict(color=COLOR_SCHEME['negative'], width=2)
    ))
    fig.add_hline(
        y=80, 
        line_dash='dash', 
        line_color='red',
        annotation_text='80% line',
        yref='y2'
    )
    fig.update_layout(
        title='Pareto Chart — 80/20 Analysis',
        yaxis2=dict(overlaying='y', side='right', range=[0, 105]),
        height=450
    )
    return fig


def create_heatmap_chart(corr: pd.DataFrame) -> go.Figure:
    """Create department affinity heatmap."""
    fig = px.imshow(
        corr,
        color_continuous_scale='YlOrRd',
        title='Department Co-Purchase Heatmap',
        aspect='auto'
    )
    fig.update_layout(height=600)
    return fig


def create_store_performance_chart(sales: pd.DataFrame) -> go.Figure:
    """Create store performance comparison chart."""
    store_perf = sales.groupby('Store No_').agg({
        'Net Amount': 'sum',
        'Transaction No_': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
    store_perf.columns = ['Store No_', 'Revenue', 'Transactions', 'Quantity']
    store_perf = store_perf.sort_values('Revenue', ascending=False)
    
    fig = px.bar(
        store_perf,
        x='Store No_',
        y='Revenue',
        color='Revenue',
        color_continuous_scale='Viridis',
        title='Revenue by Store Location',
        labels={'Store No_': 'Store Location', 'Revenue': 'Total Revenue (AED)'},
        hover_data=['Transactions', 'Quantity']
    )
    fig.update_layout(height=400)
    return fig


# ============================================================================
# Main App
# ============================================================================
st.set_page_config(**PAGE_CONFIG)
st.title("🛒 Sales Pattern Analysis Dashboard")
st.markdown("Upload your sales data and discover patterns ")
st.divider()

# File upload
uploaded_file = st.file_uploader("📂 Upload your Sales Excel file", type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload your Sales_Data.xlsx file to begin")
    st.stop()

# Load data
try:
    sales = load_data(uploaded_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

# Validate data quality
quality_warnings = validate_data_quality(sales)
if quality_warnings:
    for key, warning in quality_warnings.items():
        st.warning(f"⚠️ Data Quality: {warning}")

sales_pos = sales[sales['Net Amount'] > 0]

if len(sales_pos) == 0:
    st.error("No positive sales data found. Please check your file.")
    st.stop()

# ============================================================================
# Sidebar: Store Location Filter
# ============================================================================
st.sidebar.markdown("### 🏪 Store Location Filter")
st.sidebar.markdown("Select one or more store locations to analyze:")

# Get unique stores
all_stores = sorted(sales['Store No_'].unique().astype(str))
all_stores_list = ['All Stores'] + all_stores

# Create attractive filter UI
filter_style = """
<style>
    .store-filter-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 10px;
    }
</style>
"""

st.sidebar.markdown(filter_style, unsafe_allow_html=True)

# Multi-select for stores
selected_stores = st.sidebar.multiselect(
    label="📍 Choose Store(s)",
    options=all_stores,
    default=all_stores[:1] if len(all_stores) > 0 else all_stores,
    help="Select one or multiple store locations. Leave empty to view all stores."
)

# If no stores selected, use all stores
if len(selected_stores) == 0:
    selected_stores = all_stores
    st.sidebar.info("ℹ️ Showing all stores")

# Filter data by selected stores
filtered_sales = sales[sales['Store No_'].astype(str).isin(selected_stores)]
filtered_sales_pos = filtered_sales[filtered_sales['Net Amount'] > 0]

if len(filtered_sales_pos) == 0:
    st.error("❌ No data available for selected store location(s). Please select different stores.")
    st.stop()

# Display store filter summary
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Filter Summary")
    col_a, col_b = st.columns(2)
    col_a.metric("🏢 Stores Selected", len(selected_stores))
    col_b.metric("📦 Transactions", f"{len(filtered_sales_pos):,}")
    
    col_c, col_d = st.columns(2)
    col_c.metric("💰 Total Revenue", f"AED {filtered_sales_pos['Net Amount'].sum():,.0f}")
    col_d.metric("🔄 Total Returns", f"AED {abs(filtered_sales[filtered_sales['Net Amount'] < 0]['Net Amount'].sum()):,.0f}")

# Key metrics
store_display = ", ".join(selected_stores) if len(selected_stores) <= 3 else f"{', '.join(selected_stores[:3])}, +{len(selected_stores) - 3} more"
st.success(f"✅ Filtered Data — {len(filtered_sales):,} rows | Stores: {store_display} | {filtered_sales['Item No_'].nunique():,} products")

st.subheader("📊 Key Numbers")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"AED {filtered_sales_pos['Net Amount'].sum():,.0f}")
k2.metric("Transactions", f"{filtered_sales_pos['Transaction No_'].nunique():,}")
k3.metric("Unique Products", f"{filtered_sales['Item No_'].nunique():,}")
k4.metric("Selected Stores", f"{len(selected_stores)}")
k5.metric("Total Returns", f"AED {abs(filtered_sales[filtered_sales['Net Amount'] < 0]['Net Amount'].sum()):,.0f}")

st.divider()

# Store Performance Overview
st.subheader("🏪 Store Performance Overview")
st.plotly_chart(create_store_performance_chart(filtered_sales_pos), use_container_width=True)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔗 Association Rules",
    "📈 Revenue Trends",
    "📦 Pareto Analysis",
    "🏠 Department Heatmap"
])

# ============================================================================
# Tab 1: Association Rules
# ============================================================================
with tab1:
    st.subheader("🔗 Market Basket Analysis — What is Bought Together?")
    st.markdown("Apriori algorithm finds products frequently purchased together")
    st.markdown("### ⚙️ Parameters")
    
    # Calculate total unique transactions for this filtered dataset
    total_transactions = filtered_sales_pos['Transaction No_'].nunique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Minimum Support (by Transaction Count)**")
        st.info(f"📊 Total transactions in filtered data: **{total_transactions:,}**")
        
        # Create two sub-columns for slider and number input
        slider_col, input_col = st.columns([3, 1])
        
        with slider_col:
            # Calculate max transaction count (10% of total)
            max_transaction_count = max(50, int(total_transactions * 0.1))
            
            support_transactions = st.slider(
                "Slider: Select transaction count",
                min_value=1,
                max_value=max_transaction_count,
                value=min(50, total_transactions // 2),
                step=50,
                key="support_transactions_slider"
            )
        
        with input_col:
            # Number input for precise selection
            support_transactions_input = st.number_input(
                "Or type count",
                min_value=1,
                max_value=max_transaction_count,
                value=support_transactions,
                step=1,
                key="support_transactions_input"
            )
            support_transactions = int(support_transactions_input)
        
        # Convert transaction count to support percentage
        min_support = calculate_support_from_transaction_count(support_transactions, total_transactions)
        
        # Display the converted percentage
        support_percentage = min_support * 100
        st.markdown(f"**📈 Equivalent to {support_percentage:.2f}% support**")
    
    with col2:
        st.markdown("**Minimum Lift**")
        min_lift = st.slider(
            "Lift Score (how much more likely items are bought together)",
            min_value=APRIORI_CONFIG["min_lift_range"][0],
            max_value=APRIORI_CONFIG["min_lift_range"][1],
            value=APRIORI_CONFIG["min_lift_default"],
            step=0.5,
            help="Higher = stronger, more meaningful rules only",
            key="min_lift"
        )
        st.caption("💡 Lift > 1 means items are bought together more often than by chance")

    if st.button("🔍 Find Patterns", type="primary"):
        with st.spinner("Running Apriori algorithm... please wait"):
            try:
                rules, rule_count = run_association_analysis(filtered_sales_pos, min_support, min_lift)
                
                if rule_count == 0:
                    st.warning(f"❌ No rules found with these parameters. Try lowering the transaction count or lift threshold.")
                else:
                    st.success(f"✅ Found {rule_count} rules (min support: {support_transactions:,} transactions / {support_percentage:.2f}%, lift ≥ {min_lift})")
                    
                    st.plotly_chart(create_lift_chart(rules), use_container_width=True)
                    st.plotly_chart(create_support_confidence_chart(rules), use_container_width=True)
                    
                    st.dataframe(
                        rules[['IF', 'THEN', 'support', 'confidence', 'lift']]
                        .round(4).reset_index(drop=True),
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Error: {e}. Try lowering the transaction count value.")

# ============================================================================
# Tab 2: Revenue Trends
# ============================================================================
with tab2:
    st.subheader("📈 Revenue Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_monthly_trend_chart(filtered_sales_pos), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_weekend_vs_weekday_chart(filtered_sales_pos), use_container_width=True)
    
    st.plotly_chart(create_daily_revenue_chart(filtered_sales_pos), use_container_width=True)

# ============================================================================
# Tab 3: Pareto Analysis
# ============================================================================
with tab3:
    st.subheader("📦 Pareto Analysis — 80/20 Rule")
    
    try:
        pareto = prepare_pareto_data(filtered_sales_pos)
        top80 = pareto[pareto['Cumulative %'] <= 80]
        
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Products", f"{len(pareto):,}")
        p2.metric("Products for 80% Revenue", f"{len(top80):,}")
        p3.metric("That is only", f"{len(top80) / len(pareto) * 100:.1f}% of products")
        
        st.plotly_chart(create_pareto_chart(pareto), use_container_width=True)
        st.dataframe(pareto.head(10).round(2), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ============================================================================
# Tab 4: Department Heatmap
# ============================================================================
with tab4:
    st.subheader("🏠 Department Affinity Heatmap")
    
    try:
        with st.spinner("Building heatmap..."):
            corr = prepare_correlation_matrix(filtered_sales_pos)
            st.plotly_chart(create_heatmap_chart(corr), use_container_width=True)
        
        st.markdown("🔴 Dark red = strongly bought together | ⬜ White = no relationship")
    except Exception as e:
        st.error(f"❌ Error building heatmap: {e}")
