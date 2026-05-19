import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Analysis", page_icon="🛒", layout="wide")
st.title(" Sales Hidden Pattern Analysis")
st.markdown("Upload your sales data and discover hidden patterns using Machine Learning")
st.divider()

uploaded_file = st.file_uploader("📂 Upload your Sales Excel file", type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload your Sales_Data.xlsx file to begin")
    st.stop()

@st.cache_data
def load_data(file):
    sales = pd.read_excel(file)
    sales['Date']      = pd.to_datetime(sales['Date'])
    sales['Month']     = sales['Date'].dt.month
    sales['Weekday']   = sales['Date'].dt.day_name()
    sales['YearMonth'] = sales['Date'].dt.to_period('M').astype(str)
    sales['Weekend']   = sales['Weekday'].isin(['Saturday', 'Sunday'])
    sales              = sales.drop_duplicates()
    return sales

sales     = load_data(uploaded_file)
sales_pos = sales[sales['Net Amount'] > 0]

st.success(f"Data loaded — {len(sales):,} rows | {sales['Store No_'].nunique()} stores | {sales['Item No_'].nunique():,} products")

st.subheader(" Key Numbers")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",    f"AED {sales_pos['Net Amount'].sum():,.0f}")
k2.metric("Transactions",     f"{sales_pos['Transaction No_'].nunique():,}")
k3.metric("Unique Products",  f"{sales['Item No_'].nunique():,}")
k4.metric("Stores",           f"{sales['Store No_'].nunique()}")
k5.metric("Total Returns",    f"AED {abs(sales[sales['Net Amount']<0]['Net Amount'].sum()):,.0f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 Association Rules",
    "🏪 Store Clustering",
    "📈 Revenue Trends",
    "📦 Pareto Analysis",
    "🏠 Department Heatmap"
])

with tab1:
    st.subheader("🔗 Market Basket Analysis — What is Bought Together?")
    st.markdown("Apriori algorithm finds products frequently purchased together")
    st.markdown("### ⚙️ Parameters")
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", min_value=0.005, max_value=0.05, value=0.01, step=0.005, help="0.01 = combo must appear in 1% of transactions")
    with col2:
        min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=15.0, value=3.0, step=0.5, help="Higher = stronger rules only")

    if st.button("🔍 Find Patterns", type="primary"):
        with st.spinner("Running Apriori algorithm... please wait"):
            try:
                sc = sales[(sales['Net Amount'] > 0) & (sales['Item Category'] != 'Service')].copy()
                basket = (sc.groupby(['Transaction No_', 'Subgroup Desc'])['Quantity'].sum().unstack(fill_value=0).gt(0).astype('bool'))
                freq  = apriori(basket, min_support=min_support, use_colnames=True)
                rules = association_rules(freq, metric='lift', min_threshold=min_lift)
                rules['IF']   = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['THEN'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                rules['rule'] = rules['IF'] + '  →  ' + rules['THEN']
                rules = rules.sort_values('lift', ascending=False)
                st.success(f" Found {len(rules)} rules with lift ≥ {min_lift}")
                top10 = rules.head(10)
                fig1 = px.bar(top10, x='lift', y='rule', orientation='h', color='lift', color_continuous_scale='Reds', title='Top 10 Rules by Lift Score', labels={'lift': 'Lift Score', 'rule': ''})
                fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
                st.plotly_chart(fig1, use_container_width=True)
                fig2 = px.scatter(rules, x='support', y='confidence', size='lift', color='lift', color_continuous_scale='YlOrRd', hover_data=['IF', 'THEN'], title='All Rules — Support vs Confidence (bubble = lift strength)')
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(rules[['IF', 'THEN', 'support', 'confidence', 'lift']].round(4).reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}. Try lowering the minimum support value.")

with tab2:
    st.subheader("🏪 Store Segmentation — KMeans Clustering")
    st.markdown("Groups stores by similar behaviour — Revenue, Quantity and Transactions")
    n_clusters = st.slider("Number of store groups", min_value=2, max_value=5, value=3, step=1, help="3 = divide stores into 3 groups")
    store_data = sales_pos.groupby('Store No_').agg(Revenue=('Net Amount','sum'), Quantity=('Quantity','sum'), Transactions=('Transaction No_','nunique')).reset_index()
    store_data.columns = ['Store', 'Revenue', 'Quantity', 'Transactions']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(store_data[['Revenue', 'Quantity', 'Transactions']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    store_data['Cluster'] = kmeans.fit_predict(scaled).astype(str)
    fig3 = px.scatter(store_data, x='Revenue', y='Transactions', color='Cluster', size='Quantity', text='Store', title=f'Store Segmentation — {n_clusters} Groups', color_discrete_sequence=px.colors.qualitative.Set1)
    fig3.update_traces(textposition='top center')
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(store_data.sort_values('Revenue', ascending=False).reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("📈 Revenue Trends")
    col1, col2 = st.columns(2)
    with col1:
        monthly = sales_pos.groupby('YearMonth')['Net Amount'].sum().reset_index()
        fig4 = px.line(monthly, x='YearMonth', y='Net Amount', markers=True, title='Monthly Revenue Trend', labels={'Net Amount': 'Revenue (AED)', 'YearMonth': 'Month'})
        fig4.update_traces(line_color='red', marker=dict(color='black', size=8))
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        week = sales_pos.copy()
        week['Day Type'] = week['Weekend'].map({True: 'Weekend', False: 'Weekday'})
        week_avg = week.groupby('Day Type')['Net Amount'].mean().reset_index()
        fig5 = px.bar(week_avg, x='Day Type', y='Net Amount', color='Day Type', color_discrete_map={'Weekend': '#E07B54', 'Weekday': '#5B8DB8'}, title='Average Sale — Weekend vs Weekday')
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_sales = sales_pos.groupby('Weekday')['Net Amount'].sum().reindex(day_order).reset_index()
    fig6 = px.bar(day_sales, x='Weekday', y='Net Amount', color='Net Amount', color_continuous_scale='Blues', title='Total Revenue by Day of Week')
    st.plotly_chart(fig6, use_container_width=True)

with tab4:
    st.subheader("📦 Pareto Analysis — 80/20 Rule")
    pareto = sales_pos.groupby('Search Description')['Net Amount'].sum().sort_values(ascending=False).reset_index()
    pareto['Cumulative %'] = pareto['Net Amount'].cumsum() / pareto['Net Amount'].sum() * 100
    top80 = pareto[pareto['Cumulative %'] <= 80]
    p1, p2, p3 = st.columns(3)
    p1.metric("Total Products", f"{len(pareto):,}")
    p2.metric("Products for 80% Revenue", f"{len(top80):,}")
    p3.metric("That is only", f"{len(top80)/len(pareto)*100:.1f}% of products")
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(x=list(range(len(pareto))), y=pareto['Net Amount'], name='Revenue', marker_color='#5B8DB8', opacity=0.7))
    fig7.add_trace(go.Scatter(x=list(range(len(pareto))), y=pareto['Cumulative %'], name='Cumulative %', yaxis='y2', line=dict(color='#E07B54', width=2)))
    fig7.add_hline(y=80, line_dash='dash', line_color='red', annotation_text='80% line', yref='y2')
    fig7.update_layout(title='Pareto Chart', yaxis2=dict(overlaying='y', side='right', range=[0, 105]), height=450)
    st.plotly_chart(fig7, use_container_width=True)
    st.dataframe(pareto.head(10).round(2), use_container_width=True)

with tab5:
    st.subheader("🏠 Department Affinity Heatmap")
    with st.spinner("Building heatmap..."):
        dept = sales_pos.groupby(['Transaction No_', 'Department Desc'])['Quantity'].sum().unstack().fillna(0)
        corr = dept.corr()
        fig8 = px.imshow(corr, color_continuous_scale='YlOrRd', title='Department Co-Purchase Heatmap', aspect='auto')
        fig8.update_layout(height=600)
        st.plotly_chart(fig8, use_container_width=True)
    st.markdown("🔴 Dark red = strongly bought together | ⬜ White = no relationship")

st.divider()
st.caption("Built with Python · Streamlit · Plotly · mlxtend · scikit-learn")