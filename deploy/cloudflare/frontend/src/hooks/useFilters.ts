import { create } from "zustand";
import type { Target, Horizon } from "@/types";

interface FilterState {
  target: Target;
  horizon: Horizon;
  setTarget: (t: Target) => void;
  setHorizon: (h: Horizon) => void;
}

export const useFilters = create<FilterState>((set) => ({
  target: "c_food_price_index",
  horizon: 1,
  setTarget: (target) => set({ target }),
  setHorizon: (horizon) => set({ horizon }),
}));
