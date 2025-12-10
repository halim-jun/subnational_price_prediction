"""
Validate and visualize the prepared modeling dataset.

Quick validation script to check data quality and visualize key patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def load_data(data_dir='data/processed/modeling'):
    """Load all datasets."""
    data_dir = Path(data_dir)
    
    temporal_train = pd.read_parquet(data_dir / 'temporal_train.parquet')
    temporal_test = pd.read_parquet(data_dir / 'temporal_test.parquet')
    spatial_test = pd.read_parquet(data_dir / 'spatial_test.parquet')
    full_train = pd.read_parquet(data_dir / 'full_train.parquet')
    
    with open(data_dir / 'dataset_summary.json', 'r') as f:
        summary = json.load(f)
    
    return {
        'temporal_train': temporal_train,
        'temporal_test': temporal_test,
        'spatial_test': spatial_test,
        'full_train': full_train,
        'summary': summary
    }


def validate_no_leakage(data):
    """Validate that there's no spatial or temporal leakage."""
    print("\n" + "="*60)
    print("LEAKAGE VALIDATION")
    print("="*60)
    
    temporal_train = data['temporal_train']
    temporal_test = data['temporal_test']
    spatial_test = data['spatial_test']
    
    # Temporal validation
    max_train_date = temporal_train['date'].max()
    min_test_date = temporal_test['date'].min()
    min_spatial_test_date = spatial_test['date'].min()
    
    print(f"\n✓ Temporal Check:")
    print(f"  - Max train date: {max_train_date}")
    print(f"  - Min temporal test date: {min_test_date}")
    print(f"  - Min spatial test date: {min_spatial_test_date}")
    
    assert max_train_date < min_test_date, "TEMPORAL LEAKAGE DETECTED!"
    assert max_train_date < min_spatial_test_date, "TEMPORAL LEAKAGE DETECTED!"
    print(f"  → No temporal leakage ✓")
    
    # Spatial validation
    train_h3 = set(temporal_train['h3_index'].unique())
    test_h3 = set(temporal_test['h3_index'].unique())
    spatial_test_h3 = set(spatial_test['h3_index'].unique())
    
    overlap_train_spatial = train_h3 & spatial_test_h3
    
    print(f"\n✓ Spatial Check:")
    print(f"  - Train H3 cells: {len(train_h3)}")
    print(f"  - Temporal test H3 cells: {len(test_h3)}")
    print(f"  - Spatial test H3 cells: {len(spatial_test_h3)}")
    print(f"  - Overlap (train & spatial_test): {len(overlap_train_spatial)}")
    
    assert len(overlap_train_spatial) == 0, "SPATIAL LEAKAGE DETECTED!"
    print(f"  → No spatial leakage ✓")
    
    print("\n" + "="*60)
    print("✅ ALL LEAKAGE CHECKS PASSED")
    print("="*60)


def plot_data_overview(data, save_path='data/processed/modeling/validation_plots'):
    """Create overview visualizations."""
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    temporal_train = data['temporal_train']
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Target variable distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MoM distribution
    axes[0, 0].hist(temporal_train['inflation_mom'].dropna(), bins=50, 
                    edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('MoM Inflation Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('MoM Inflation (log returns)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 0].legend()
    
    # YoY distribution
    axes[0, 1].hist(temporal_train['inflation_yoy'].dropna(), bins=50,
                    edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_title('YoY Inflation Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('YoY Inflation (log returns)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 1].legend()
    
    # MoM over time
    monthly_mom = temporal_train.groupby('date')['inflation_mom'].agg(['mean', 'std'])
    axes[1, 0].plot(monthly_mom.index, monthly_mom['mean'], marker='o', 
                    linewidth=2, markersize=4, label='Mean')
    axes[1, 0].fill_between(monthly_mom.index, 
                            monthly_mom['mean'] - monthly_mom['std'],
                            monthly_mom['mean'] + monthly_mom['std'],
                            alpha=0.3, label='±1 Std')
    axes[1, 0].set_title('MoM Inflation Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('MoM Inflation')
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # YoY over time
    monthly_yoy = temporal_train.groupby('date')['inflation_yoy'].agg(['mean', 'std'])
    axes[1, 1].plot(monthly_yoy.index, monthly_yoy['mean'], marker='o',
                    linewidth=2, markersize=4, label='Mean', color='coral')
    axes[1, 1].fill_between(monthly_yoy.index,
                            monthly_yoy['mean'] - monthly_yoy['std'],
                            monthly_yoy['mean'] + monthly_yoy['std'],
                            alpha=0.3, label='±1 Std', color='coral')
    axes[1, 1].set_title('YoY Inflation Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('YoY Inflation')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'target_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path / 'target_distributions.png'}")
    plt.close()
    
    # 2. Extreme events
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (event, color, label) in enumerate([
        ('flood_3m', 'blue', 'Flood (3m)'),
        ('flood_6m', 'navy', 'Flood (6m)'),
        ('drought_3m', 'orange', 'Drought (3m)'),
        ('drought_6m', 'red', 'Drought (6m)')
    ]):
        ax = axes[idx // 2, idx % 2]
        counts = temporal_train[event].value_counts().sort_index()
        
        bars = ax.bar(['No', 'Yes'], counts, color=[color, 'lightcoral'], alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        ax.set_title(f'{label} Frequency', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add percentages
        total = counts.sum()
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'extreme_events.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'extreme_events.png'}")
    plt.close()
    
    # 3. Feature correlations
    numeric_cols = temporal_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['inflation_mom', 'inflation_yoy']]
    corr_data = temporal_train[feature_cols + ['inflation_mom', 'inflation_yoy']].corr()
    
    # Focus on target correlations
    target_corr = corr_data[['inflation_mom', 'inflation_yoy']].drop(['inflation_mom', 'inflation_yoy'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(target_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=False, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation with Target Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'feature_correlations.png'}")
    plt.close()
    
    # 4. Country comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    country_stats = temporal_train.groupby('countryiso3').agg({
        'inflation_mom': ['mean', 'std'],
        'inflation_yoy': ['mean', 'std']
    })
    
    # MoM by country
    ax = axes[0]
    countries = country_stats.index
    x_pos = np.arange(len(countries))
    
    means = country_stats['inflation_mom']['mean']
    stds = country_stats['inflation_mom']['std']
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', 
           edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(countries)
    ax.set_title('Average MoM Inflation by Country', fontsize=14, fontweight='bold')
    ax.set_ylabel('MoM Inflation')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # YoY by country
    ax = axes[1]
    means = country_stats['inflation_yoy']['mean']
    stds = country_stats['inflation_yoy']['std']
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='coral',
           edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(countries)
    ax.set_title('Average YoY Inflation by Country', fontsize=14, fontweight='bold')
    ax.set_ylabel('YoY Inflation')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / 'country_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'country_comparison.png'}")
    plt.close()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("="*60)


def print_summary_stats(data):
    """Print detailed summary statistics."""
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    temporal_train = data['temporal_train']
    temporal_test = data['temporal_test']
    spatial_test = data['spatial_test']
    
    datasets = [
        ('Temporal Train', temporal_train),
        ('Temporal Test', temporal_test),
        ('Spatial Test', spatial_test)
    ]
    
    for name, df in datasets:
        print(f"\n{name}:")
        print("-" * 60)
        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  H3 cells: {df['h3_index'].nunique()}")
        print(f"  Countries: {sorted(df['countryiso3'].unique())}")
        
        print(f"\n  Inflation Statistics:")
        print(f"    MoM - Mean: {df['inflation_mom'].mean():.4f}, "
              f"Std: {df['inflation_mom'].std():.4f}, "
              f"Range: [{df['inflation_mom'].min():.4f}, {df['inflation_mom'].max():.4f}]")
        print(f"    YoY - Mean: {df['inflation_yoy'].mean():.4f}, "
              f"Std: {df['inflation_yoy'].std():.4f}, "
              f"Range: [{df['inflation_yoy'].min():.4f}, {df['inflation_yoy'].max():.4f}]")
        
        print(f"\n  Extreme Events:")
        print(f"    Flood (3m): {df['flood_3m'].sum()} ({df['flood_3m'].mean()*100:.1f}%)")
        print(f"    Flood (6m): {df['flood_6m'].sum()} ({df['flood_6m'].mean()*100:.1f}%)")
        print(f"    Drought (3m): {df['drought_3m'].sum()} ({df['drought_3m'].mean()*100:.1f}%)")
        print(f"    Drought (6m): {df['drought_6m'].sum()} ({df['drought_6m'].mean()*100:.1f}%)")
        
        print(f"\n  Missing Values:")
        null_counts = df.isnull().sum()
        null_counts = null_counts[null_counts > 0]
        if len(null_counts) > 0:
            for col, count in null_counts.items():
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"    None")


def main():
    """Main validation routine."""
    print("="*60)
    print("MODELING DATASET VALIDATION")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    data = load_data()
    print("✓ Datasets loaded successfully")
    
    # Validate no leakage
    validate_no_leakage(data)
    
    # Print summary stats
    print_summary_stats(data)
    
    # Create visualizations
    plot_data_overview(data)
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check validation plots in data/processed/modeling/validation_plots/")
    print("2. Review README.md for modeling guidelines")
    print("3. Start with baseline models (Linear Regression, Random Forest, GBM)")
    print("4. Consider spatial autocorrelation with advanced models")
    print("5. Evaluate on both temporal_test and spatial_test")


if __name__ == '__main__':
    main()

