"""
Compare all trained models and generate summary report.

Retrieves results from MLflow and creates comparison tables and plots.

Usage:
    python src/models/compare_models.py
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def get_mlflow_results():
    """Retrieve all experiment results from MLflow."""
    
    mlflow.set_experiment("food_price_forecasting")
    
    # Get all runs
    runs = mlflow.search_runs(order_by=["start_time DESC"])
    
    if len(runs) == 0:
        print("No runs found in MLflow!")
        return None
    
    # Select relevant columns
    relevant_cols = [
        'run_id', 'start_time', 'tags.mlflow.runName',
        'params.model_name', 'params.target',
        'metrics.train_rmse', 'metrics.test_rmse', 'metrics.spatial_rmse',
        'metrics.train_mae', 'metrics.test_mae', 'metrics.spatial_mae',
        'metrics.train_r2', 'metrics.test_r2', 'metrics.spatial_r2'
    ]
    
    # Filter columns that exist
    available_cols = [col for col in relevant_cols if col in runs.columns]
    results = runs[available_cols].copy()
    
    # Rename columns for clarity
    results.columns = results.columns.str.replace('tags.mlflow.runName', 'run_name')
    results.columns = results.columns.str.replace('params.', '')
    results.columns = results.columns.str.replace('metrics.', '')
    
    return results


def create_comparison_tables(results):
    """Create comparison tables for both targets."""
    
    output_dir = Path('data/processed/modeling/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    targets = ['inflation_mom', 'inflation_yoy']
    summaries = {}
    
    for target in targets:
        target_results = results[results['target'] == target].copy()
        
        if len(target_results) == 0:
            print(f"No results for {target}")
            continue
        
        # Create summary DataFrame
        summary = target_results[[
            'model_name', 
            'test_rmse', 'test_mae', 'test_r2',
            'spatial_rmse', 'spatial_mae', 'spatial_r2'
        ]].copy()
        
        # Sort by test_rmse
        summary = summary.sort_values('test_rmse')
        
        # Save to CSV
        output_file = output_dir / f'{target}_comparison.csv'
        summary.to_csv(output_file, index=False)
        
        summaries[target] = summary
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"{target.upper()} - Model Comparison")
        print(f"{'='*80}")
        print(summary.to_string(index=False))
        print(f"\nBest model (by Test RMSE): {summary.iloc[0]['model_name']}")
        print(f"Best Test RMSE: {summary.iloc[0]['test_rmse']:.4f}")
    
    return summaries


def plot_model_comparison(summaries):
    """Create visualization comparing models."""
    
    output_dir = Path('data/processed/modeling/results')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (target, summary) in enumerate(summaries.items()):
        if summary is None or len(summary) == 0:
            continue
        
        row = idx
        
        # Test RMSE
        ax = axes[row, 0]
        bars = ax.barh(summary['model_name'], summary['test_rmse'], 
                       color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Test RMSE', fontweight='bold')
        ax.set_title(f'{target.replace("_", " ").title()} - Test RMSE', 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=9)
        
        # Test MAE
        ax = axes[row, 1]
        bars = ax.barh(summary['model_name'], summary['test_mae'],
                       color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Test MAE', fontweight='bold')
        ax.set_title(f'{target.replace("_", " ").title()} - Test MAE',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=9)
        
        # Test R²
        ax = axes[row, 2]
        bars = ax.barh(summary['model_name'], summary['test_r2'],
                       color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Test R²', fontweight='bold')
        ax.set_title(f'{target.replace("_", " ").title()} - Test R²',
                     fontsize=12, fontweight='bold')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}',
                   ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir / 'model_comparison.png'}")
    plt.close()
    
    # Spatial test comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (target, summary) in enumerate(summaries.items()):
        if summary is None or len(summary) == 0:
            continue
        
        ax = axes[idx]
        
        x = np.arange(len(summary))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, summary['test_rmse'], width, 
                       label='Temporal Test', color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, summary['spatial_rmse'], width,
                       label='Spatial Test', color='coral', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('RMSE', fontweight='bold')
        ax.set_title(f'{target.replace("_", " ").title()} - Temporal vs Spatial Test',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(summary['model_name'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_vs_spatial.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved temporal vs spatial plot: {output_dir / 'temporal_vs_spatial.png'}")
    plt.close()


def print_key_insights(summaries):
    """Print key insights from the comparison."""
    
    print(f"\n\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    for target, summary in summaries.items():
        if summary is None or len(summary) == 0:
            continue
        
        print(f"\n{target.upper()}:")
        print("-" * 80)
        
        # Best model
        best_model = summary.iloc[0]
        print(f"  🏆 Best Model: {best_model['model_name']}")
        print(f"     - Test RMSE: {best_model['test_rmse']:.4f}")
        print(f"     - Test MAE: {best_model['test_mae']:.4f}")
        print(f"     - Test R²: {best_model['test_r2']:.4f}")
        print(f"     - Spatial RMSE: {best_model['spatial_rmse']:.4f}")
        
        # Worst model
        worst_model = summary.iloc[-1]
        print(f"\n  📉 Worst Model: {worst_model['model_name']}")
        print(f"     - Test RMSE: {worst_model['test_rmse']:.4f}")
        
        # Spatial generalization
        print(f"\n  🌍 Spatial Generalization:")
        for _, row in summary.iterrows():
            spatial_diff = row['spatial_rmse'] - row['test_rmse']
            direction = "better" if spatial_diff < 0 else "worse"
            print(f"     {row['model_name']}: Spatial RMSE {direction} by {abs(spatial_diff):.4f}")
        
        # R² analysis
        positive_r2 = summary[summary['test_r2'] > 0]
        if len(positive_r2) > 0:
            print(f"\n  ✅ Models with positive R²: {len(positive_r2)}/{len(summary)}")
            for _, row in positive_r2.iterrows():
                print(f"     {row['model_name']}: R² = {row['test_r2']:.4f}")
        else:
            print(f"\n  ⚠️  No models achieved positive R² (all worse than mean baseline)")


def main():
    """Main comparison routine."""
    
    print("="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)
    
    # Retrieve results
    print("\nRetrieving results from MLflow...")
    results = get_mlflow_results()
    
    if results is None:
        return
    
    print(f"Found {len(results)} runs")
    
    # Create comparison tables
    summaries = create_comparison_tables(results)
    
    # Create plots
    print(f"\n\nCreating visualizations...")
    plot_model_comparison(summaries)
    
    # Print insights
    print_key_insights(summaries)
    
    print(f"\n\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print("\nResults saved to:")
    print("  - data/processed/modeling/results/inflation_mom_comparison.csv")
    print("  - data/processed/modeling/results/inflation_yoy_comparison.csv")
    print("  - data/processed/modeling/results/model_comparison.png")
    print("  - data/processed/modeling/results/temporal_vs_spatial.png")
    print("\nTo view MLflow UI:")
    print("  mlflow ui")
    print("  Open: http://localhost:5000")


if __name__ == '__main__':
    main()

