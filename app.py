import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

CLUSTERING_CONFIG = {
    "n_clusters_range": (2, 5),
    "n_clusters_default": 3,
    "random_state": 42,
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


@st.cache_data
def run_association_analysis(sales: pd.DataFrame, min_support: float, min_lift: float) -> Tuple[pd.DataFrame, int]:
    """Run Apriori algorithm for market basket analysis."""
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
def run_clustering_analysis(sales: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Run KMeans clustering on store data."""
    try:
        store_data = (sales.groupby('Store No_')
                     .agg(
                         Revenue=('Net Amount', 'sum'),
                         Quantity=('Quantity', 'sum'),
                         Transactions=('Transaction No_', 'nunique')
                     )
                     .reset_index())
        
        if len(store_data) < n_clusters:
            raise ValueError(f"Not enough stores ({len(store_data)}) for {n_clusters} clusters.")
        
        store_data.columns = ['Store', 'Revenue', 'Quantity', 'Transactions']
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(store_data[['Revenue', 'Quantity', 'Transactions']])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=CLUSTERING_CONFIG['random_state'], n_init=10)
        store_data['Cluster'] = kmeans.fit_predict(scaled).astype(str)
        
        return store_data
    
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
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


def create_store_clustering_chart(store_data: pd.DataFrame, n_clusters: int) -> go.Figure:
    """Create scatter plot for store clustering."""
    fig = px.scatter(
        store_data, 
        x='Revenue', 
        y='Transactions', 
        color='Cluster',
        size='Quantity', 
        text='Store',
        title=f'Store Segmentation — {n_clusters} Groups',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
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


# ============================================================================
# Main App
# ============================================================================
st.set_page_config(**PAGE_CONFIG)
st.title("🛒 Sales Hidden Pattern Analysis")
st.markdown("Upload your sales data and discover hidden patterns using Machine Learning")
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

# Key metrics
st.success(f"✅ Data loaded — {len(sales):,} rows | {sales['Store No_'].nunique()} stores | {sales['Item No_'].nunique():,} products")

st.subheader("📊 Key Numbers")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"AED {sales_pos['Net Amount'].sum():,.0f}")
k2.metric("Transactions", f"{sales_pos['Transaction No_'].nunique():,}")
k3.metric("Unique Products", f"{sales['Item No_'].nunique():,}")
k4.metric("Stores", f"{sales['Store No_'].nunique()}")
k5.metric("Total Returns", f"AED {abs(sales[sales['Net Amount'] < 0]['Net Amount'].sum()):,.0f}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 Association Rules",
    "🏪 Store Clustering",
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
    
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=APRIORI_CONFIG["min_support_range"][0],
            max_value=APRIORI_CONFIG["min_support_range"][1],
            value=APRIORI_CONFIG["min_support_default"],
            step=0.005,
            help="0.01 = combo must appear in 1% of transactions"
        )
    with col2:
        min_lift = st.slider(
            "Minimum Lift",
            min_value=APRIORI_CONFIG["min_lift_range"][0],
            max_value=APRIORI_CONFIG["min_lift_range"][1],
            value=APRIORI_CONFIG["min_lift_default"],
            step=0.5,
            help="Higher = stronger rules only"
        )

    if st.button("🔍 Find Patterns", type="primary"):
        with st.spinner("Running Apriori algorithm... please wait"):
            try:
                rules, rule_count = run_association_analysis(sales_pos, min_support, min_lift)
                
                if rule_count == 0:
                    st.warning(f"No rules found with these parameters. Try lowering minimum support.")
                else:
                    st.success(f"✅ Found {rule_count} rules with lift ≥ {min_lift}")
                    
                    st.plotly_chart(create_lift_chart(rules), use_container_width=True)
                    st.plotly_chart(create_support_confidence_chart(rules), use_container_width=True)
                    
                    st.dataframe(
                        rules[['IF', 'THEN', 'support', 'confidence', 'lift']]
                        .round(4).reset_index(drop=True),
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Error: {e}. Try lowering the minimum support value.")

# ============================================================================
# Tab 2: Store Clustering
# ============================================================================
with tab2:
    st.subheader("🏪 Store Segmentation — KMeans Clustering")
    st.markdown("Groups stores by similar behaviour — Revenue, Quantity and Transactions")
    
    n_clusters = st.slider(
        "Number of store groups",
        min_value=CLUSTERING_CONFIG["n_clusters_range"][0],
        max_value=CLUSTERING_CONFIG["n_clusters_range"][1],
        value=CLUSTERING_CONFIG["n_clusters_default"],
        step=1,
        help="3 = divide stores into 3 groups"
    )
    
    try:
        store_data = run_clustering_analysis(sales_pos, n_clusters)
        st.plotly_chart(create_store_clustering_chart(store_data, n_clusters), use_container_width=True)
        st.dataframe(
            store_data.sort_values('Revenue', ascending=False).reset_index(drop=True),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ============================================================================
# Tab 3: Revenue Trends
# ============================================================================
with tab3:
    st.subheader("📈 Revenue Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_monthly_trend_chart(sales_pos), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_weekend_vs_weekday_chart(sales_pos), use_container_width=True)
    
    st.plotly_chart(create_daily_revenue_chart(sales_pos), use_container_width=True)

# ============================================================================
# Tab 4: Pareto Analysis
# ============================================================================
with tab4:
    st.subheader("📦 Pareto Analysis — 80/20 Rule")
    
    try:
        pareto = prepare_pareto_data(sales_pos)
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
# Tab 5: Department Heatmap
# ============================================================================
with tab5:
    st.subheader("🏠 Department Affinity Heatmap")
    
    try:
        with st.spinner("Building heatmap..."):
            corr = prepare_correlation_matrix(sales_pos)
            st.plotly_chart(create_heatmap_chart(corr), use_container_width=True)
        
        st.markdown("🔴 Dark red = strongly bought together | ⬜ White = no relationship")
    except Exception as e:
        st.error(f"❌ Error building heatmap: {e}")

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption("Built with Python · Streamlit · Plotly · mlxtend · scikit-learn")
