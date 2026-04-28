export type Target = "c_food_price_index" | "c_maize_fao" | "c_sorghum";
export type Horizon = 1 | 2 | 3;

export interface PredictionRow {
  admin2_name: string;
  country_name: string;
  date: string;
  actual: number;
  predicted: number;
  error: number;
}

export interface MetricSummary {
  target: string;
  horizon: number;
  r2: number;
  mape: number;
  rmse: number;
  mae: number;
  n: number;
  KEN_mape?: number;
  KEN_r2?: number;
  SOM_mape?: number;
  SOM_r2?: number;
}

export interface TimeSeriesPoint {
  date: string;
  actual: number;
  predicted: number;
  error: number;
}

export interface FeatureImportance {
  group: string;
  mean: number;
  std: number;
}

export interface PerAdminRow {
  country_name: string;
  admin2_name: string;
  mae: number;
  mape: number;
  mean_actual: number;
  n: number;
}
