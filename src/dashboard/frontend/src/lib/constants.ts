import type { Target, Horizon } from "@/types";

export const TARGETS: Target[] = [
  "c_food_price_index",
  "c_maize_fao",
  "c_sorghum",
];

export const TARGET_DISPLAY: Record<Target, string> = {
  c_food_price_index: "Food Price Index",
  c_maize_fao: "Maize (FAO)",
  c_sorghum: "Sorghum",
};

export const HORIZONS: Horizon[] = [1, 2, 3];

export const GROUP_COLORS: Record<string, string> = {
  Autoregressive: "#E91E63",
  FLDAS: "#2196F3",
  Vegetation: "#4CAF50",
  "Climate Index": "#FF9800",
  Static: "#9C27B0",
  Conflict: "#F44336",
  "Spatial ID": "#607D8B",
  Temporal: "#795548",
};
