#!/usr/bin/env python3
"""
Test script to verify data loading for the research question dashboard
"""

import pandas as pd
import os
import sys

def test_data_loading():
    """Test if the CSV files can be loaded correctly"""
    
    print("Testing data loading for research question dashboard...")
    print("="*60)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check for CSV files
    csv_files = [
        'canada_africa_projects_main.csv',
        'canada_africa_country_breakdown.csv', 
        'canada_africa_sector_breakdown.csv'
    ]
    
    print("\nChecking for CSV files:")
    for file in csv_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024 / 1024
            print(f"  ✅ {file} ({size:.1f} MB)")
        elif os.path.exists(f'../{file}'):
            size = os.path.getsize(f'../{file}') / 1024 / 1024
            print(f"  ✅ ../{file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file} - Not found")
    
    # Try loading the main dataset
    print("\nTesting main dataset loading:")
    try:
        if os.path.exists('canada_africa_projects_main.csv'):
            df = pd.read_csv('canada_africa_projects_main.csv')
        elif os.path.exists('../canada_africa_projects_main.csv'):
            df = pd.read_csv('../canada_africa_projects_main.csv')
        else:
            print("  ❌ Main dataset not found")
            return False
        
        print(f"  ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check for research-specific columns
        research_cols = [
            'RQ_Alignment_Level',
            'RQ_Alignment_Score', 
            'Entrepreneurial_Orientation_Score',
            'Capacity_Building_Score',
            'Wealth_Creation_Score'
        ]
        
        print("\nChecking research columns:")
        for col in research_cols:
            if col in df.columns:
                unique_vals = df[col].nunique()
                print(f"  ✅ {col} ({unique_vals} unique values)")
            else:
                print(f"  ❌ {col} - Missing")
        
        # Show sample data
        print(f"\nSample RQ_Alignment_Level distribution:")
        if 'RQ_Alignment_Level' in df.columns:
            print(df['RQ_Alignment_Level'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    
    if success:
        print("\n✅ Data loading test passed! You can now run the Streamlit dashboard.")
        print("\nTo run the dashboard:")
        print("  streamlit run research_question_dashboard.py")
    else:
        print("\n❌ Data loading test failed. Please run analysis_main.py first.")
        print("\nTo fix this:")
        print("  cd ..")
        print("  python analysis_main.py")
        print("  cd streamlit")
        print("  streamlit run research_question_dashboard.py")
