# H3 Resolution Comparison

## Generated Datasets

### H3 Level 4 (Coarser)
- **File**: `data/processed/sorghum_price_with_precipitation_h3_4.csv`
- **Hexagon Size**: ~1,770 km² per hexagon
- **Total Records**: 6,909
- **Unique H3 Cells**: 236
- **Markets per H3-month**: 1.6
- **Null Values**: 
  - Price: 225 (3.3%)
  - Inflation: 460 (6.7%)

### H3 Level 5 (Finer)
- **File**: `data/processed/sorghum_price_with_precipitation_h3_5.csv`
- **Hexagon Size**: ~253 km² per hexagon
- **Total Records**: 7,271
- **Unique H3 Cells**: 272
- **Markets per H3-month**: 1.5
- **Null Values**:
  - Price: 225 (3.1%)
  - Inflation: 496 (6.8%)

## Key Differences

### Spatial Resolution
- **H3-4**: Each hexagon covers ~1,770 km² (roughly 42 km × 42 km)
- **H3-5**: Each hexagon covers ~253 km² (roughly 16 km × 16 km)
- **Ratio**: H3-4 cells are ~7x larger than H3-5 cells

### Data Characteristics

| Metric | H3-4 | H3-5 | Difference |
|--------|------|------|------------|
| Total rows | 6,909 | 7,271 | +362 (+5.2%) |
| Unique cells | 236 | 272 | +36 (+15.3%) |
| Markets/cell-month | 1.6 | 1.5 | -0.1 |
| Data with inflation | 6,449 | 6,775 | +326 (+5.1%) |

## Modeling Considerations

### H3 Level 4 Advantages
✅ **Fewer cells** → Less spatial autocorrelation issues  
✅ **More aggregation** → Smoother data, fewer outliers  
✅ **Faster training** → Less data complexity  
✅ **Better for regional analysis** → Captures larger-scale patterns  

### H3 Level 5 Advantages
✅ **Finer resolution** → Better captures local variation  
✅ **More cells** → More spatial diversity in training data  
✅ **Less aggregation** → Preserves local market dynamics  
✅ **Better for micro-level** → Captures smaller-scale climate effects  

## Recommendation

**Test both levels** with the LSTM model and compare:

1. **Predictive Performance**:
   - RMSE, MAE, R² on test sets
   - Compare temporal test vs spatial test performance

2. **Spatial Autocorrelation**:
   - Check if H3-4's larger cells reduce spatial leakage
   - Measure Moran's I for residuals

3. **Feature Importance**:
   - See if spatial lag features are more/less important at different resolutions

4. **Training Efficiency**:
   - Compare training time and convergence

## Next Steps

```bash
# Run LSTM for H3-4
python src/preprocessing/prepare_modeling_dataset.py --h3-data data/processed/sorghum_price_with_precipitation_h3_4.csv

# Run LSTM for H3-5
python src/preprocessing/prepare_modeling_dataset.py --h3-data data/processed/sorghum_price_with_precipitation_h3_5.csv

# Compare results
python src/models/compare_h3_results.py
```

---

**Created**: 2025-10-22  
**Status**: Ready for modeling comparison


