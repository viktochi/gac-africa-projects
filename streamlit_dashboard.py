import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*fillna.*inplace.*')


# Page configuration
st.set_page_config(
    page_title="Canada-Africa Projects Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data
# @st.cache_data(ttl=3600)  # Cache for 1 hour, then refresh
def load_data():
    """Load all CSV files and perform initial processing"""
    
    # Read the files
    main_df = pd.read_csv('canada_africa_projects_main.csv')
    country_df = pd.read_csv('canada_africa_country_breakdown.csv')
    sector_df = pd.read_csv('canada_africa_sector_breakdown.csv')
    
    # Clean and process main dataframe
    main_df['Start Date'] = pd.to_datetime(main_df['Start Date'], errors='coerce')
    main_df['End Date'] = pd.to_datetime(main_df['End Date'], errors='coerce')
    main_df['Start_Year'] = main_df['Start Date'].dt.year
    main_df['Maximum Contribution'] = pd.to_numeric(main_df['Maximum Contribution'], errors='coerce')
    country_df['Weighted_Contribution'] = pd.to_numeric(country_df['Weighted_Contribution'], errors='coerce')
    
    # Handle missing values
    main_df = main_df.fillna({
        'Entrepreneurship_Focus_Level': 'None',
        'Results_Entrepreneurship_Level': 'None',
        'Expected_Entrepreneurship_Level': 'None'
    })
    
    # Process country and sector data
    country_df['Weighted_Contribution'] = pd.to_numeric(country_df['Weighted_Contribution'], errors='coerce')
    sector_df['Weighted_Contribution'] = pd.to_numeric(sector_df['Weighted_Contribution'], errors='coerce')
    
    return main_df, country_df, sector_df

# Load data
main_df, country_df, sector_df = load_data()



# Dashboard 1: Portfolio Overview Functions
def create_portfolio_kpis(main_df, filtered_df=None, filtered_country_df=None):
    """Create KPI metrics for portfolio overview"""
    
    df = filtered_df if filtered_df is not None else main_df
    country_data = filtered_country_df if filtered_country_df is not None else country_df
    
    total_projects = len(df)
    total_funding = country_data['Weighted_Contribution'].sum()
    avg_duration = df['Project_Duration_Years'].mean()
    entrepreneurship_projects = len(df[df['Entrepreneurship_Focus_Level'] != 'None'])
    
    # Create KPI subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Total Projects', 'Total Funding (Billions)', 
                       'Avg Duration (Years)', 'Entrepreneurship Projects'],
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Total Projects
    fig.add_trace(go.Indicator(
        mode="number",
        value=total_projects,
        title={"text": "Total Projects"},
        number={'font': {'size': 48, 'color': '#1f4e79'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=1)
    
    # Total Funding
    fig.add_trace(go.Indicator(
        mode="number",
        value=total_funding/1e9,
        title={"text": "Total Funding (Billions)"},
        number={'font': {'size': 48, 'color': '#70ad47'}, 'prefix': '$', 'suffix': 'B'},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=2)
    
    # Average Duration
    fig.add_trace(go.Indicator(
        mode="number",
        value=avg_duration,
        title={"text": "Avg Duration (Years)"},
        number={'font': {'size': 48, 'color': '#ffa500'}, 'suffix': ' yrs'},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=1)
    
    # Entrepreneurship Projects
    fig.add_trace(go.Indicator(
        mode="number",
        value=entrepreneurship_projects,
        title={"text": "Entrepreneurship Projects"},
        number={'font': {'size': 48, 'color': '#7030a0'}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(248,249,250,0.8)',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_country_funding_chart(country_df, top_n=100):
    """Create horizontal bar chart of funding by country"""
    
    # Aggregate by country
    country_summary = country_df.groupby(['Country', 'Entrepreneurship_Level']).agg({
        'Weighted_Contribution': 'sum',
        'Project_Number': 'nunique'
    }).reset_index()
    
    # Get top countries by funding
    top_countries = country_summary.groupby('Country')['Weighted_Contribution'].sum().nlargest(top_n)
    country_summary = country_summary[country_summary['Country'].isin(top_countries.index)]
    
    # Color mapping for entrepreneurship levels
    color_map = {
        'None': '#d3d3d3',
        'Low': '#ffd966', 
        'Medium': '#ff9900',
        'High': '#e65100'
    }
    
    fig = px.bar(
        country_summary, 
        y='Country', 
        x='Weighted_Contribution',
        color='Entrepreneurship_Level',
        color_discrete_map=color_map,
        title='Funding Distribution by Country (Top 100)',
        labels={'Weighted_Contribution': 'Total Funding ($)', 'Country': 'Country'},
        orientation='h'
    )
    
    fig.update_layout(
        height=600,
        xaxis_title='Total Funding ($)',
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        showlegend=True,
        legend_title_text='Entrepreneurship Level',
        xaxis=dict(tickformat=',.0f')
    )
    
    return fig

def create_timeline_trend(main_df):
    """Create dual-axis timeline showing project count and funding trends"""
    
    # Aggregate by year
    yearly_data = main_df.groupby(['Start_Year', 'Entrepreneurship_Focus_Level']).agg({
        'Project Number': 'count',
        'Maximum Contribution': 'sum'
    }).reset_index()
    
    yearly_total = main_df.groupby('Start_Year').agg({
        'Project Number': 'count',
        'Maximum Contribution': 'sum'
    }).reset_index()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add project count line
    fig.add_trace(
        go.Scatter(
            x=yearly_total['Start_Year'],
            y=yearly_total['Project Number'],
            mode='lines+markers',
            name='Project Count',
            line=dict(color='#4472c4', width=3),
            fill='tonexty',
            fillcolor='rgba(68,114,196,0.3)'
        ),
        secondary_y=False
    )
    
    # Add funding line
    fig.add_trace(
        go.Scatter(
            x=yearly_total['Start_Year'],
            y=yearly_total['Maximum Contribution']/1e6,  # Convert to millions
            mode='lines+markers',
            name='Total Funding ($M)',
            line=dict(color='#70ad47', width=3),
            fill='tonexty',
            fillcolor='rgba(112,173,71,0.2)'
        ),
        secondary_y=True
    )
    
    # Add reference line for Africa Strategy launch
    fig.add_vline(
        x=2025, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Canada-Africa Strategy Launch"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Number of Projects", secondary_y=False)
    fig.update_yaxes(title_text="Total Funding ($ Millions)", secondary_y=True)
    
    fig.update_layout(
        title='Project Timeline and Funding Trends',
        height=500,
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig

def create_sector_treemap(sector_df):
    """Create treemap of sector portfolio"""
    
    # Aggregate by sector category
    sector_summary = sector_df.groupby('Sector_Category').agg({
        'Weighted_Contribution': 'sum',
        'Project_Number': 'nunique',
        'Entrepreneurship_Level': lambda x: (x != 'None').mean()  # Entrepreneurship rate
    }).reset_index()
    
    sector_summary.columns = ['Sector_Category', 'Total_Funding', 'Project_Count', 'Entrepreneurship_Rate']
    
    # Create treemap using Graph Objects instead of Plotly Express
    fig = go.Figure(go.Treemap(
        labels=sector_summary['Sector_Category'],
        parents=[''] * len(sector_summary),  # All sectors are at the same level
        values=sector_summary['Total_Funding'],
        textinfo="label+percent parent+value",
        texttemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percentParent}",
        hovertemplate='<b>%{label}</b><br>Total Funding: $%{value:,.0f}<br>Projects: %{customdata}<br>Entrepreneurship Rate: %{color:.1%}<extra></extra>',
        customdata=sector_summary['Project_Count'],
        marker=dict(
            colors=sector_summary['Entrepreneurship_Rate'],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Entrepreneurship Rate")
        )
    ))
    
    fig.update_layout(
        title='Sector Portfolio Analysis',
        height=600
    )
    
    return fig

# Dashboard 2: Entrepreneurship Analysis Functions
def create_entrepreneurship_donut(main_df):
    """Create donut chart for entrepreneurship focus distribution"""
    
    focus_counts = main_df['Entrepreneurship_Focus_Level'].value_counts()
    
    # Custom colors
    colors = ['#ff4444', '#ffaa00', '#ffdd00', '#00aa44']  # Red, Orange, Yellow, Green
    
    fig = go.Figure(data=[go.Pie(
        labels=focus_counts.index,
        values=focus_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        textposition='outside'
    )])
    
    # Add center text
    total_projects = len(main_df)
    entrepreneurship_rate = len(main_df[main_df['Entrepreneurship_Focus_Level'] != 'None']) / total_projects
    
    fig.add_annotation(
        text=f"<b>Total Projects</b><br>{total_projects:,}<br><br><b>Entrepreneurship Rate</b><br>{entrepreneurship_rate:.1%}",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        title='Entrepreneurship Focus Distribution',
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    
    return fig

def create_funding_correlation(main_df):
    """Create scatter plot of entrepreneurship score vs funding"""
    
    # Filter out zero scores and contributions for better visualization
    plot_df = main_df[(main_df['Entrepreneurship_Score'] > 0) & (main_df['Maximum Contribution'] > 0)].copy()
    plot_df['Log_Contribution'] = np.log10(plot_df['Maximum Contribution'])
    
    # Color mapping
    color_map = {
        'Small (<1M)': '#add8e6',
        'Medium (1-10M)': '#4169e1', 
        'Large (>10M)': '#191970'
    }
    
    # Shape mapping
    plot_df['Shape'] = plot_df['Is_Multi_Country'].map({'Yes': 'diamond', 'No': 'circle'})
    
    fig = px.scatter(
        plot_df,
        x='Entrepreneurship_Score',
        y='Log_Contribution',
        size='Project_Duration_Years',
        color='Funding_Tier',
        symbol='Is_Multi_Country',
        color_discrete_map=color_map,
        title='Entrepreneurship Score vs. Funding Correlation',
        labels={
            'Entrepreneurship_Score': 'Entrepreneurship Score',
            'Log_Contribution': 'Log10(Maximum Contribution)',
            'Project_Duration_Years': 'Duration (Years)'
        },
        hover_data=['Primary_Country', 'Primary_Sector', 'Status']
    )
    
    # Add trend line
    if len(plot_df) > 1:
        z = np.polyfit(plot_df['Entrepreneurship_Score'], plot_df['Log_Contribution'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(plot_df['Entrepreneurship_Score'].min(), plot_df['Entrepreneurship_Score'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Calculate R-squared
        correlation = np.corrcoef(plot_df['Entrepreneurship_Score'], plot_df['Log_Contribution'])[0,1]
        r_squared = correlation ** 2
        
        fig.add_annotation(
            text=f"R² = {r_squared:.3f}",
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    fig.update_layout(height=600, plot_bgcolor='white')
    
    return fig

def create_policy_heatmap(main_df):
    """Create heatmap of policy markers by country"""
    
    # Get top 10 countries by project count
    top_countries = main_df['Primary_Country'].value_counts().head(10).index
    
    # Filter and aggregate
    heatmap_df = main_df[main_df['Primary_Country'].isin(top_countries)]
    
    policy_cols = ['Trade_Dev_Level', 'Governance_Level', 'Gender_Level', 'Youth_Level', 'ICT_Level']
    policy_names = ['Trade Dev', 'Governance', 'Gender', 'Youth', 'ICT']
    
    # Calculate average policy scores by country
    policy_matrix = heatmap_df.groupby('Primary_Country')[policy_cols].mean()
    
    fig = go.Figure(data=go.Heatmap(
        z=policy_matrix.values,
        x=policy_names,
        y=policy_matrix.index,
        colorscale='YlOrRd',
        zmin=0,
        zmax=2,
        text=np.round(policy_matrix.values, 1),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Policy Level (0-2)")
    ))
    
    fig.update_layout(
        title='Policy Markers Heatmap by Country',
        height=600,
        xaxis_title='Policy Dimension',
        yaxis_title='Country'
    )
    
    return fig

def create_expected_vs_achieved(main_df):
    """Create scatter plot comparing expected vs achieved results"""
    
    # Create numeric encoding for levels
    level_encoding = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
    
    plot_df = main_df.copy()
    plot_df['Expected_Numeric'] = plot_df['Expected_Entrepreneurship_Level'].map(level_encoding)
    plot_df['Results_Numeric'] = plot_df['Results_Entrepreneurship_Level'].map(level_encoding)
    
    # Filter out missing data
    plot_df = plot_df.dropna(subset=['Expected_Numeric', 'Results_Numeric'])
    
    if len(plot_df) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available for comparison", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.scatter(
        plot_df,
        x='Expected_Numeric',
        y='Results_Numeric',
        size='Maximum Contribution',
        color='Status',
        title='Expected vs. Achieved Entrepreneurship Results',
        labels={
            'Expected_Numeric': 'Expected Level',
            'Results_Numeric': 'Achieved Level'
        },
        hover_data=['Primary_Country', 'Title']
    )
    
    # Add perfect performance line (y=x)
    max_val = max(plot_df['Expected_Numeric'].max(), plot_df['Results_Numeric'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Alignment',
        line=dict(color='black', dash='dash', width=2)
    ))
    
    # Update axis labels
    fig.update_xaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=['None', 'Low', 'Medium', 'High']
    )
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=['None', 'Low', 'Medium', 'High']
    )
    
    fig.update_layout(height=600, plot_bgcolor='white')
    
    return fig

# Dashboard 3: Project Management Insights Functions
def create_complexity_bubble(main_df):
    """Create bubble chart for project complexity analysis"""
    
    fig = px.scatter(
        main_df,
        x='Country_Count',
        y='Sector_Count',
        size='Maximum Contribution',
        color='Entrepreneurship_Focus_Level',
        title='Project Complexity Analysis',
        labels={
            'Country_Count': 'Number of Countries',
            'Sector_Count': 'Number of Sectors',
            'Maximum Contribution': 'Funding ($)'
        },
        hover_data=['Primary_Country', 'Title', 'Status'],
        color_discrete_map={'None': '#d3d3d3', 'Low': '#ffd966', 'Medium': '#ff9900', 'High': '#e65100'}
    )
    
    # Add quadrant lines
    fig.add_vline(x=1.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=1.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(text="Simple<br>Projects", x=0.8, y=0.8, showarrow=False, bgcolor="white", opacity=0.8)
    fig.add_annotation(text="Country<br>Complex", x=3, y=0.8, showarrow=False, bgcolor="white", opacity=0.8)
    fig.add_annotation(text="Sector<br>Complex", x=0.8, y=4, showarrow=False, bgcolor="white", opacity=0.8)
    fig.add_annotation(text="Highly<br>Complex", x=3, y=4, showarrow=False, bgcolor="white", opacity=0.8)
    
    fig.update_layout(height=600, plot_bgcolor='white')
    
    return fig

def create_duration_boxplot(main_df):
    """Create box plot of project duration by entrepreneurship level"""
    
    # Filter out extreme outliers for better visualization
    q99 = main_df['Project_Duration_Years'].quantile(0.99)
    plot_df = main_df[main_df['Project_Duration_Years'] <= q99]
    
    fig = px.box(
        plot_df,
        x='Entrepreneurship_Focus_Level',
        y='Project_Duration_Years',
        color='Results_Entrepreneurship_Level',
        title='Project Duration vs. Entrepreneurship Success Patterns',
        labels={
            'Entrepreneurship_Focus_Level': 'Entrepreneurship Focus Level',
            'Project_Duration_Years': 'Project Duration (Years)',
            'Results_Entrepreneurship_Level': 'Results Level'
        }
    )
    
    # Add average duration line
    avg_duration = main_df['Project_Duration_Years'].mean()
    fig.add_hline(
        y=avg_duration,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average Duration: {avg_duration:.1f} years"
    )
    
    fig.update_layout(height=600, plot_bgcolor='white')
    
    return fig

def create_multistakeholder_performance(main_df):
    """Create grouped bar chart for multi-stakeholder project performance"""
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = main_df.copy()
    
    # Create grouping variables
    df_copy['Stakeholder_Type'] = df_copy['Is_Multi_Country'] + ' Country, ' + df_copy['Is_Multi_Sector'] + ' Sector'
    
    # Aggregate performance by stakeholder type and funding tier
    performance_df = df_copy.groupby(['Stakeholder_Type', 'Funding_Tier']).agg({
        'Entrepreneurship_Score': ['mean', 'count', 'std']
    }).round(2)
    
    performance_df.columns = ['Avg_Score', 'Count', 'Std_Dev']
    performance_df = performance_df.reset_index()
    
    fig = px.bar(
        performance_df,
        x='Stakeholder_Type',
        y='Avg_Score',
        color='Funding_Tier',
        title='Multi-stakeholder Project Performance',
        labels={
            'Stakeholder_Type': 'Project Configuration',
            'Avg_Score': 'Average Entrepreneurship Score'
        },
        text='Count',
        error_y='Std_Dev'
    )
    
    fig.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig.update_layout(height=600, plot_bgcolor='white')
    
    return fig

def create_geographic_network(country_df):
    """Create geographic visualization (simplified version)"""
    
    # Aggregate funding by country
    country_summary = country_df.groupby('Country').agg({
        'Weighted_Contribution': 'sum',
        'Project_Number': 'nunique',
        'Entrepreneurship_Level': lambda x: (x != 'None').mean()
    }).reset_index()
    
    country_summary.columns = ['Country', 'Total_Funding', 'Project_Count', 'Entrepreneurship_Rate']
    
    # Create horizontal bar chart as proxy for map
    fig = px.bar(
        country_summary.nlargest(50, 'Total_Funding'),
        x='Total_Funding',
        y='Country',
        color='Entrepreneurship_Rate',
        title='Geographic Distribution of Canada-Africa Projects',
        labels={
            'Total_Funding': 'Total Funding ($)',
            'Entrepreneurship_Rate': 'Entrepreneurship Rate',
            'Project_Count': 'Number of Projects'
        },
        color_continuous_scale='Viridis',
        hover_data=['Project_Count']
    )
    
    fig.update_layout(
        height=700,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white'
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("🌍 Canada-Africa Projects Analysis Dashboard")
    st.markdown("---")
    

    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Year range filter
    min_year = int(main_df['Start_Year'].min())
    max_year = int(main_df['Start_Year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(1996, max_year),
        step=1
    )
    
    # Entrepreneurship level filter
    entrepreneurship_levels = st.sidebar.multiselect(
        "Entrepreneurship Level",
        options=main_df['Entrepreneurship_Focus_Level'].unique(),
        default=list(main_df['Entrepreneurship_Focus_Level'].unique())
    )
    
    # Filter data based on selections
    filtered_main = main_df[
        (main_df['Start_Year'] >= year_range[0]) & 
        (main_df['Start_Year'] <= year_range[1]) &
        (main_df['Entrepreneurship_Focus_Level'].isin(entrepreneurship_levels))
    ]
    
    filtered_country = country_df[
        (country_df['Start_Year'] >= year_range[0]) & 
        (country_df['Start_Year'] <= year_range[1]) &
        (country_df['Entrepreneurship_Level'].isin(entrepreneurship_levels))
    ]
    
    filtered_sector = sector_df[
        (sector_df['Start_Year'] >= year_range[0]) & 
        (sector_df['Start_Year'] <= year_range[1]) &
        (sector_df['Entrepreneurship_Level'].isin(entrepreneurship_levels))
    ]
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Entrepreneurship Analysis", "Project Management Insights"])
    
    with tab1:
        st.header("Portfolio Overview")
        
        # Row 1: KPIs and Country Funding
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_portfolio_kpis(main_df, filtered_main, filtered_country), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_country_funding_chart(filtered_country), use_container_width=True)
        
        # Row 2: Timeline and Sector Analysis
        col3, col4 = st.columns(2)
        
        with col3:
            st.plotly_chart(create_timeline_trend(filtered_main), use_container_width=True)
        
        with col4:
            st.plotly_chart(create_sector_treemap(filtered_sector), use_container_width=True)
    
    with tab2:
        st.header("Entrepreneurship Analysis")
        
        # Row 1: Donut Chart and Policy Heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_entrepreneurship_donut(filtered_main), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_policy_heatmap(filtered_main), use_container_width=True)
        
        # Row 2: Correlation and Expected vs Achieved
        col3, col4 = st.columns(2)
        
        with col3:
            st.plotly_chart(create_funding_correlation(filtered_main), use_container_width=True)
        
        with col4:
            st.plotly_chart(create_expected_vs_achieved(filtered_main), use_container_width=True)
    
    with tab3:
        st.header("Project Management Insights")
        
        # Row 1: Complexity and Duration Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_complexity_bubble(filtered_main), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_duration_boxplot(filtered_main), use_container_width=True)
        
        # Row 2: Multi-stakeholder Performance and Geographic
        col3, col4 = st.columns(2)
        
        with col3:
            st.plotly_chart(create_multistakeholder_performance(filtered_main), use_container_width=True)
        
        with col4:
            st.plotly_chart(create_geographic_network(filtered_country), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("📊 **Data Source**: Canada-Africa Projects Database")
    st.markdown("🔧 **Built with**: Streamlit + Plotly")

if __name__ == "__main__":
    main() 