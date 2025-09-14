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
    page_title="Canada-Africa Entrepreneurial Orientation Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data
@st.cache_data(ttl=3600)
def load_data():
    """Load all CSV files and perform initial processing"""
    
    # Read the files (check both current directory and parent directory)
    import os
    
    # Try current directory first, then parent directory
    if os.path.exists('canada_africa_projects_main.csv'):
        main_df = pd.read_csv('canada_africa_projects_main.csv')
        country_df = pd.read_csv('canada_africa_country_breakdown.csv')
        sector_df = pd.read_csv('canada_africa_sector_breakdown.csv')
    elif os.path.exists('../canada_africa_projects_main.csv'):
        main_df = pd.read_csv('../canada_africa_projects_main.csv')
        country_df = pd.read_csv('../canada_africa_country_breakdown.csv')
        sector_df = pd.read_csv('../canada_africa_sector_breakdown.csv')
    else:
        raise FileNotFoundError("CSV files not found. Please run analysis_main.py first.")
    
    # Clean and process main dataframe
    main_df['Start Date'] = pd.to_datetime(main_df['Start Date'], errors='coerce')
    main_df['End Date'] = pd.to_datetime(main_df['End Date'], errors='coerce')
    main_df['Start_Year'] = main_df['Start Date'].dt.year
    main_df['Maximum Contribution'] = pd.to_numeric(main_df['Maximum Contribution'], errors='coerce')
    
    # Handle missing values for research metrics
    research_cols = ['Entrepreneurial_Orientation_Score', 'Capacity_Building_Score', 'Wealth_Creation_Score']
    for col in research_cols:
        if col in main_df.columns:
            main_df[col] = main_df[col].fillna(0)
        else:
            # Create missing columns with default values
            main_df[col] = 0
    
    # Ensure RQ_Alignment_Level exists
    if 'RQ_Alignment_Level' not in main_df.columns:
        main_df['RQ_Alignment_Level'] = 'Not Aligned'
    
    # Ensure RQ_Alignment_Score exists
    if 'RQ_Alignment_Score' not in main_df.columns:
        main_df['RQ_Alignment_Score'] = 0
    
    # Process country and sector data
    country_df['Weighted_Contribution'] = pd.to_numeric(country_df['Weighted_Contribution'], errors='coerce')
    sector_df['Weighted_Contribution'] = pd.to_numeric(sector_df['Weighted_Contribution'], errors='coerce')
    
    # Ensure research columns exist in country and sector data
    research_cols_country = ['Entrepreneurial_Orientation', 'Capacity_Building', 'Wealth_Creation', 'RQ_Alignment_Score']
    for col in research_cols_country:
        if col not in country_df.columns:
            country_df[col] = 0
    
    research_cols_sector = ['Entrepreneurial_Orientation', 'Capacity_Building', 'Wealth_Creation', 'RQ_Alignment_Score']
    for col in research_cols_sector:
        if col not in sector_df.columns:
            sector_df[col] = 0
    
    return main_df, country_df, sector_df

# Load data
main_df, country_df, sector_df = load_data()

# Research Question Focused Functions
def create_research_overview_kpis(main_df, filtered_df=None):
    """Create KPI metrics focused on research question"""
    
    df = filtered_df if filtered_df is not None else main_df
    
    total_projects = len(df)
    total_funding = df['Maximum Contribution'].sum()
    
    # Research-specific metrics
    rq_aligned_projects = len(df[df['RQ_Alignment_Level'].isin(['Medium Alignment', 'High Alignment'])])
    high_alignment_projects = len(df[df['RQ_Alignment_Level'] == 'High Alignment'])
    avg_rq_score = df['RQ_Alignment_Score'].mean()
    
    # Create KPI subplot
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Total Projects', 'Total Funding ($B)', 'RQ Aligned Projects',
            'High Alignment Projects', 'Average RQ Score', 'Alignment Rate (%)'
        ],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Row 1
    fig.add_trace(go.Indicator(
        mode="number", value=total_projects,
        title={"text": "Total Projects"},
        number={'font': {'size': 36, 'color': '#1f4e79'}}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number", value=total_funding/1e9,
        title={"text": "Total Funding ($B)"},
        number={'font': {'size': 36, 'color': '#70ad47'}, 'prefix': '$', 'suffix': 'B'}
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="number", value=rq_aligned_projects,
        title={"text": "RQ Aligned Projects"},
        number={'font': {'size': 36, 'color': '#ff6b35'}}
    ), row=1, col=3)
    
    # Row 2
    fig.add_trace(go.Indicator(
        mode="number", value=high_alignment_projects,
        title={"text": "High Alignment Projects"},
        number={'font': {'size': 36, 'color': '#e74c3c'}}
    ), row=2, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number", value=avg_rq_score,
        title={"text": "Average RQ Score"},
        number={'font': {'size': 36, 'color': '#9b59b6'}}
    ), row=2, col=2)
    
    alignment_rate = (rq_aligned_projects / total_projects * 100) if total_projects > 0 else 0
    fig.add_trace(go.Indicator(
        mode="number", value=alignment_rate,
        title={"text": "Alignment Rate (%)"},
        number={'font': {'size': 36, 'color': '#3498db'}, 'suffix': '%'}
    ), row=2, col=3)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(248,249,250,0.8)',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_research_framework_sunburst(main_df):
    """Create sunburst chart showing the research framework hierarchy"""
    
    # Prepare data for sunburst
    framework_data = []
    
    for _, row in main_df.iterrows():
        if row['RQ_Alignment_Score'] > 0:
            # Level 1: Overall alignment
            alignment_level = row['RQ_Alignment_Level']
            
            # Level 2: Component scores
            eo_level = 'High EO' if row['Entrepreneurial_Orientation_Score'] >= 2 else 'Low EO' if row['Entrepreneurial_Orientation_Score'] >= 1 else 'No EO'
            cb_level = 'High CB' if row['Capacity_Building_Score'] >= 2 else 'Low CB' if row['Capacity_Building_Score'] >= 1 else 'No CB'
            wc_level = 'High WC' if row['Wealth_Creation_Score'] >= 2 else 'Low WC' if row['Wealth_Creation_Score'] >= 1 else 'No WC'
            
            framework_data.extend([
                {'ids': f"{alignment_level}", 'labels': alignment_level, 'parents': '', 'values': 1},
                {'ids': f"{alignment_level}-{eo_level}", 'labels': eo_level, 'parents': alignment_level, 'values': row['Entrepreneurial_Orientation_Score']},
                {'ids': f"{alignment_level}-{cb_level}", 'labels': cb_level, 'parents': alignment_level, 'values': row['Capacity_Building_Score']},
                {'ids': f"{alignment_level}-{wc_level}", 'labels': wc_level, 'parents': alignment_level, 'values': row['Wealth_Creation_Score']}
            ])
    
    # Aggregate the data
    df_sunburst = pd.DataFrame(framework_data)
    if len(df_sunburst) > 0:
        df_sunburst = df_sunburst.groupby(['ids', 'labels', 'parents'])['values'].sum().reset_index()
        
        fig = go.Figure(go.Sunburst(
            ids=df_sunburst['ids'],
            labels=df_sunburst['labels'],
            parents=df_sunburst['parents'],
            values=df_sunburst['values'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Score: %{value}<extra></extra>',
            maxdepth=2
        ))
        
        fig.update_layout(
            title="Research Framework: Entrepreneurial Orientation → Capacity Building → Wealth Creation",
            height=600,
            font_size=12
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="No data available for sunburst chart", x=0.5, y=0.5, showarrow=False)
    
    return fig

def create_three_pillar_correlation(main_df):
    """Create 3D scatter plot showing correlation between the three pillars"""
    
    # Filter projects with scores in all three dimensions
    plot_df = main_df[
        (main_df['Entrepreneurial_Orientation_Score'] > 0) |
        (main_df['Capacity_Building_Score'] > 0) |
        (main_df['Wealth_Creation_Score'] > 0)
    ].copy()
    
    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available for correlation analysis", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure(data=go.Scatter3d(
        x=plot_df['Entrepreneurial_Orientation_Score'],
        y=plot_df['Capacity_Building_Score'],
        z=plot_df['Wealth_Creation_Score'],
        mode='markers',
        marker=dict(
            size=np.sqrt(plot_df['Maximum Contribution'] / 1e6),  # Size by funding
            color=plot_df['RQ_Alignment_Score'],
            colorscale='Viridis',
            colorbar=dict(title="RQ Alignment Score"),
            opacity=0.7
        ),
        text=plot_df['Primary_Country'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Entrepreneurial Orientation: %{x}<br>' +
                      'Capacity Building: %{y}<br>' +
                      'Wealth Creation: %{z}<br>' +
                      'Funding: $%{marker.size}M<extra></extra>'
    ))
    
    fig.update_layout(
        title='Three-Pillar Analysis: Entrepreneurial Orientation, Capacity Building & Wealth Creation',
        scene=dict(
            xaxis_title='Entrepreneurial Orientation Score',
            yaxis_title='Capacity Building Score',
            zaxis_title='Wealth Creation Score'
        ),
        height=700
    )
    
    return fig

def create_effectiveness_analysis(main_df):
    """Create analysis of project effectiveness in achieving research goals"""
    
    # Filter projects with both expected and results data
    effectiveness_df = main_df[
        (main_df['Expected_Entrepreneurial_Orientation'] > 0) |
        (main_df['Expected_Capacity_Building'] > 0) |
        (main_df['Expected_Wealth_Creation'] > 0)
    ].copy()
    
    if len(effectiveness_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No effectiveness data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create effectiveness metrics
    effectiveness_metrics = []
    
    for dimension in ['Entrepreneurial', 'Capacity_Building', 'Wealth_Creation']:
        expected_col = f'Expected_{dimension}_Orientation' if dimension == 'Entrepreneurial' else f'Expected_{dimension}'
        results_col = f'Results_{dimension}_Orientation' if dimension == 'Entrepreneurial' else f'Results_{dimension}'
        
        if expected_col in effectiveness_df.columns and results_col in effectiveness_df.columns:
            effectiveness = effectiveness_df[effectiveness_df[expected_col] > 0][results_col].sum() / \
                          effectiveness_df[effectiveness_df[expected_col] > 0][expected_col].sum()
            
            effectiveness_metrics.append({
                'Dimension': dimension.replace('_', ' '),
                'Effectiveness_Rate': effectiveness,
                'Projects_Count': len(effectiveness_df[effectiveness_df[expected_col] > 0])
            })
    
    if effectiveness_metrics:
        effectiveness_df_plot = pd.DataFrame(effectiveness_metrics)
        
        fig = px.bar(
            effectiveness_df_plot,
            x='Dimension',
            y='Effectiveness_Rate',
            title='Project Effectiveness: Expected vs. Achieved Results',
            labels={'Effectiveness_Rate': 'Achievement Rate', 'Dimension': 'Research Dimension'},
            text='Projects_Count'
        )
        
        fig.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="100% Achievement")
        fig.update_layout(height=500, yaxis_title="Achievement Rate (Results/Expected)")
    else:
        fig = go.Figure()
        fig.add_annotation(text="No effectiveness data available", x=0.5, y=0.5, showarrow=False)
    
    return fig

def create_temporal_research_trends(main_df):
    """Create timeline showing research framework evolution over time"""
    
    # Aggregate by year and research alignment
    yearly_research = main_df.groupby(['Start_Year', 'RQ_Alignment_Level']).agg({
        'Project Number': 'count',
        'Maximum Contribution': 'sum',
        'RQ_Alignment_Score': 'mean'
    }).reset_index()
    
    # Create stacked area chart
    fig = px.area(
        yearly_research,
        x='Start_Year',
        y='Project Number',
        color='RQ_Alignment_Level',
        title='Evolution of Research Question Alignment Over Time',
        labels={'Project Number': 'Number of Projects', 'Start_Year': 'Year'},
        color_discrete_map={
            'Not Aligned': '#95a5a6',
            'Low Alignment': '#f39c12',
            'Medium Alignment': '#e67e22',
            'High Alignment': '#e74c3c'
        }
    )
    
    fig.add_vline(x=2025, line_dash="dash", line_color="blue", annotation_text="Canada-Africa Strategy")
    fig.update_layout(height=500, plot_bgcolor='white')
    
    return fig

def create_country_research_heatmap(country_df):
    """Create heatmap showing research alignment by country"""
    
    # Aggregate by country
    country_research = country_df.groupby('Country').agg({
        'RQ_Alignment_Score': 'mean',
        'Entrepreneurial_Orientation': 'mean',
        'Capacity_Building': 'mean',
        'Wealth_Creation': 'mean',
        'Weighted_Contribution': 'sum',
        'Project_Number': 'nunique'
    }).reset_index()
    
    # Get top 20 countries by funding
    top_countries = country_research.nlargest(20, 'Weighted_Contribution')
    
    # Prepare heatmap data
    heatmap_data = top_countries[['Country', 'Entrepreneurial_Orientation', 'Capacity_Building', 'Wealth_Creation']].set_index('Country')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Entrepreneurial Orientation', 'Capacity Building', 'Wealth Creation'],
        y=heatmap_data.index,
        colorscale='YlOrRd',
        text=np.round(heatmap_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Average Score")
    ))
    
    fig.update_layout(
        title='Research Framework Performance by Country (Top 20 by Funding)',
        height=600,
        xaxis_title='Research Dimensions',
        yaxis_title='Country'
    )
    
    return fig

def create_sector_research_analysis(sector_df):
    """Create sector analysis focused on research question"""
    
    # Aggregate by sector category
    sector_research = sector_df.groupby('Sector_Category').agg({
        'RQ_Alignment_Score': 'mean',
        'Entrepreneurial_Orientation': 'mean',
        'Capacity_Building': 'mean',
        'Wealth_Creation': 'mean',
        'Weighted_Contribution': 'sum',
        'Project_Number': 'nunique'
    }).reset_index()
    
    # Create bubble chart
    fig = px.scatter(
        sector_research,
        x='Capacity_Building',
        y='Wealth_Creation',
        size='Weighted_Contribution',
        color='Entrepreneurial_Orientation',
        hover_name='Sector_Category',
        title='Sector Analysis: Capacity Building vs. Wealth Creation',
        labels={
            'Capacity_Building': 'Average Capacity Building Score',
            'Wealth_Creation': 'Average Wealth Creation Score',
            'Entrepreneurial_Orientation': 'Entrepreneurial Orientation Score'
        },
        color_continuous_scale='Viridis',
        size_max=60
    )
    
    fig.update_layout(height=600, plot_bgcolor='white')
    
    return fig

# Main Streamlit App
def main():
    st.title("🌍 Canada-Africa Entrepreneurial Orientation Analysis")
    st.markdown("### Research Question: *To what extent do Canada-Africa projects demonstrate entrepreneurial orientation that fosters capacity building and drives civic wealth creation?*")
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("📊 Research Filters")
    
    # Year range filter
    if 'Start_Year' in main_df.columns:
        min_year = int(main_df['Start_Year'].min())
        max_year = int(main_df['Start_Year'].max())
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(2010, max_year),
            step=1
        )
    else:
        year_range = (2010, 2025)
    
    # Research alignment filter
    if 'RQ_Alignment_Level' in main_df.columns:
        alignment_levels = st.sidebar.multiselect(
            "Research Question Alignment",
            options=main_df['RQ_Alignment_Level'].unique(),
            default=list(main_df['RQ_Alignment_Level'].unique())
        )
    else:
        alignment_levels = []
    
    # Funding tier filter
    if 'Funding_Tier' in main_df.columns:
        funding_tiers = st.sidebar.multiselect(
            "Funding Tier",
            options=main_df['Funding_Tier'].unique(),
            default=list(main_df['Funding_Tier'].unique())
        )
    else:
        funding_tiers = []
    
    # Filter data based on selections
    filtered_main = main_df.copy()
    if 'Start_Year' in main_df.columns:
        filtered_main = filtered_main[
            (filtered_main['Start_Year'] >= year_range[0]) & 
            (filtered_main['Start_Year'] <= year_range[1])
        ]
    
    if alignment_levels and 'RQ_Alignment_Level' in main_df.columns:
        filtered_main = filtered_main[filtered_main['RQ_Alignment_Level'].isin(alignment_levels)]
    
    if funding_tiers and 'Funding_Tier' in main_df.columns:
        filtered_main = filtered_main[filtered_main['Funding_Tier'].isin(funding_tiers)]
    
    filtered_country = country_df.copy()
    filtered_sector = sector_df.copy()
    
    # Apply same filters to country and sector data if possible
    if 'Start_Year' in country_df.columns:
        filtered_country = filtered_country[
            (filtered_country['Start_Year'] >= year_range[0]) & 
            (filtered_country['Start_Year'] <= year_range[1])
        ]
    
    if 'Start_Year' in sector_df.columns:
        filtered_sector = filtered_sector[
            (filtered_sector['Start_Year'] >= year_range[0]) & 
            (filtered_sector['Start_Year'] <= year_range[1])
        ]
    
    # Dashboard tabs focused on research question
    tab1, tab2, tab3, tab4 = st.tabs([
        "Research Overview", 
        "Three-Pillar Analysis", 
        "Effectiveness & Trends",
        "Geographic & Sector Insights"
    ])
    
    with tab1:
        st.header("Research Question Overview")
        st.markdown("*Analysis of entrepreneurial orientation, capacity building, and civic wealth creation in Canada-Africa projects*")
        
        # Row 1: KPIs
        st.plotly_chart(create_research_overview_kpis(main_df, filtered_main), use_container_width=True)
        
        # Row 2: Framework visualization
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RQ_Alignment_Level' in filtered_main.columns:
                alignment_dist = filtered_main['RQ_Alignment_Level'].value_counts()
                fig_donut = px.pie(
                    values=alignment_dist.values, 
                    names=alignment_dist.index,
                    title="Research Question Alignment Distribution",
                    hole=0.4
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Research alignment data not available")
        
        with col2:
            st.plotly_chart(create_research_framework_sunburst(filtered_main), use_container_width=True)
    
    with tab2:
        st.header("Three-Pillar Analysis")
        st.markdown("*Exploring the relationship between Entrepreneurial Orientation, Capacity Building, and Wealth Creation*")
        
        # 3D correlation plot
        st.plotly_chart(create_three_pillar_correlation(filtered_main), use_container_width=True)
        
        # Correlation matrix
        col1, col2 = st.columns(2)
        
        with col1:
            # Create correlation matrix for research dimensions
            research_cols = ['Entrepreneurial_Orientation_Score', 'Capacity_Building_Score', 'Wealth_Creation_Score']
            available_cols = [col for col in research_cols if col in filtered_main.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = filtered_main[available_cols].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=[col.replace('_Score', '').replace('_', ' ') for col in corr_matrix.columns],
                    y=[col.replace('_Score', '').replace('_', ' ') for col in corr_matrix.index],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 3),
                    texttemplate="%{text}",
                    textfont={"size": 14}
                ))
                
                fig_corr.update_layout(
                    title="Research Dimensions Correlation Matrix",
                    height=400
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Insufficient data for correlation analysis")
        
        with col2:
            # Research framework scores by funding tier
            if 'Funding_Tier' in filtered_main.columns and len(available_cols) > 0:
                funding_research = filtered_main.groupby('Funding_Tier')[available_cols].mean().reset_index()
                funding_research_melted = funding_research.melt(
                    id_vars=['Funding_Tier'], 
                    value_vars=available_cols,
                    var_name='Research_Dimension',
                    value_name='Average_Score'
                )
                funding_research_melted['Research_Dimension'] = funding_research_melted['Research_Dimension'].str.replace('_Score', '').str.replace('_', ' ')
                
                fig_funding = px.bar(
                    funding_research_melted,
                    x='Funding_Tier',
                    y='Average_Score',
                    color='Research_Dimension',
                    title="Research Framework Scores by Funding Tier",
                    barmode='group'
                )
                
                st.plotly_chart(fig_funding, use_container_width=True)
            else:
                st.info("Funding tier analysis not available")
    
    with tab3:
        st.header("Effectiveness & Temporal Trends")
        st.markdown("*Analyzing project effectiveness and evolution over time*")
        
        # Row 1: Effectiveness analysis
        st.plotly_chart(create_effectiveness_analysis(filtered_main), use_container_width=True)
        
        # Row 2: Temporal trends
        st.plotly_chart(create_temporal_research_trends(filtered_main), use_container_width=True)
        
        # Research insights
        st.subheader("Key Research Insights")
        
        if 'RQ_Alignment_Score' in filtered_main.columns:
            high_alignment = len(filtered_main[filtered_main['RQ_Alignment_Level'] == 'High Alignment'])
            total_projects = len(filtered_main)
            alignment_rate = (high_alignment / total_projects * 100) if total_projects > 0 else 0
            
            avg_funding_high = filtered_main[filtered_main['RQ_Alignment_Level'] == 'High Alignment']['Maximum Contribution'].mean()
            avg_funding_low = filtered_main[filtered_main['RQ_Alignment_Level'] == 'Not Aligned']['Maximum Contribution'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("High Alignment Rate", f"{alignment_rate:.1f}%")
            
            with col2:
                if not pd.isna(avg_funding_high):
                    st.metric("Avg Funding (High Alignment)", f"${avg_funding_high/1e6:.1f}M")
                else:
                    st.metric("Avg Funding (High Alignment)", "N/A")
            
            with col3:
                if not pd.isna(avg_funding_low):
                    funding_diff = ((avg_funding_high - avg_funding_low) / avg_funding_low * 100) if avg_funding_low > 0 else 0
                    st.metric("Funding Premium", f"{funding_diff:+.1f}%")
                else:
                    st.metric("Funding Premium", "N/A")
    
    with tab4:
        st.header("Geographic & Sector Insights")
        st.markdown("*Country and sector analysis of research framework implementation*")
        
        # Row 1: Country heatmap
        st.plotly_chart(create_country_research_heatmap(filtered_country), use_container_width=True)
        
        # Row 2: Sector analysis
        st.plotly_chart(create_sector_research_analysis(filtered_sector), use_container_width=True)
    
    # Footer with research methodology
    st.markdown("---")
    st.markdown("### 📋 Research Methodology")
    
    with st.expander("View Research Framework Details"):
        st.markdown("""
        **Research Question:** *To what extent do Canada-Africa projects demonstrate entrepreneurial orientation that fosters capacity building and drives civic wealth creation?*
        
        **Three-Pillar Framework:**
        
        1. **Entrepreneurial Orientation (40% weight)**
           - Innovation & Innovative approaches
           - Risk-taking & Experimental methods  
           - Proactiveness & Market-driven initiatives
        
        2. **Capacity Building (35% weight)**
           - Skills development & Training
           - Institutional strengthening
           - Knowledge transfer & Technical assistance
        
        3. **Civic Wealth Creation (25% weight)**
           - Economic impact & Job creation
           - Social capital & Community building
           - Sustainable development & Empowerment
        
        **Alignment Scoring:**
        - **High Alignment:** All three pillars present (≥2 keywords each)
        - **Medium Alignment:** All three pillars present (≥1 keyword each)
        - **Low Alignment:** Some keywords present (≥2 total)
        - **Not Aligned:** Minimal or no relevant keywords
        """)
    
    st.markdown("📊 **Data Source**: Canada-Africa Projects Database | 🔧 **Built with**: Streamlit + Plotly")

if __name__ == "__main__":
    main()
