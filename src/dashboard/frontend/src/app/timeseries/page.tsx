"use client";

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
  ScatterChart,
  Scatter,
  ZAxis,
  Label,
} from "recharts";
import { useFilters } from "@/hooks/useFilters";
import { getTimeSeries, getRegions, getAllPredictions } from "@/lib/api";
import { TARGET_DISPLAY } from "@/lib/constants";
import type { TimeSeriesPoint, PredictionRow } from "@/types";

export default function TimeSeriesPage() {
  const { target, horizon } = useFilters();
  const [regions, setRegions] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>("");
  const [data, setData] = useState<TimeSeriesPoint[]>([]);
  const [loading, setLoading] = useState(false);

  // All predictions for scatter
  const [allPreds, setAllPreds] = useState<PredictionRow[]>([]);
  const [scatterDateIdx, setScatterDateIdx] = useState(0);

  useEffect(() => {
    getRegions(target, horizon).then((r) => {
      setRegions(r);
      if (r.length > 0 && !r.includes(selectedRegion)) {
        setSelectedRegion(r[0]);
      }
    });
    getAllPredictions(target, horizon).then((preds) => {
      setAllPreds(preds);
      const dates = [...new Set(preds.map((p) => p.date))].sort();
      setScatterDateIdx(0);
    });
  }, [target, horizon]);

  useEffect(() => {
    if (!selectedRegion) return;
    setLoading(true);
    getTimeSeries(target, horizon, selectedRegion).then((d) => {
      setData(d);
      setLoading(false);
    });
  }, [target, horizon, selectedRegion]);

  // Scatter data grouped by date
  const scatterDates = useMemo(() => {
    return [...new Set(allPreds.map((p) => p.date))].sort();
  }, [allPreds]);

  const currentScatterDate = scatterDates[scatterDateIdx] || "";

  const scatterData = useMemo(() => {
    return allPreds
      .filter((p) => p.date === currentScatterDate)
      .map((p) => ({
        admin2_name: p.admin2_name,
        country_name: p.country_name,
        actual: p.actual,
        predicted: p.predicted,
        error: p.error,
      }));
  }, [allPreds, currentScatterDate]);

  // Compute axis range from all data for consistent scale
  const scatterRange = useMemo(() => {
    if (allPreds.length === 0) return { min: 0, max: 1 };
    const allVals = allPreds.flatMap((p) => [p.actual, p.predicted]);
    return {
      min: Math.min(...allVals) * 0.95,
      max: Math.max(...allVals) * 1.05,
    };
  }, [allPreds]);

  // Per-month summary stats
  const monthlyStats = useMemo(() => {
    if (scatterData.length === 0) return null;
    const errors = scatterData.map((d) => d.predicted - d.actual);
    const absErrors = errors.map(Math.abs);
    const pctErrors = scatterData.map(
      (d) => Math.abs(d.predicted - d.actual) / Math.max(d.actual, 1e-6)
    );
    const ss_res = errors.reduce((s, e) => s + e * e, 0);
    const meanActual =
      scatterData.reduce((s, d) => s + d.actual, 0) / scatterData.length;
    const ss_tot = scatterData.reduce(
      (s, d) => s + (d.actual - meanActual) ** 2,
      0
    );
    const r2 = ss_tot === 0 ? 0 : 1 - ss_res / ss_tot;

    return {
      n: scatterData.length,
      r2,
      mae: absErrors.reduce((s, v) => s + v, 0) / absErrors.length,
      mape: pctErrors.reduce((s, v) => s + v, 0) / pctErrors.length,
      rmse: Math.sqrt(ss_res / errors.length),
    };
  }, [scatterData]);

  // Split scatter data by country for coloring
  const kenData = scatterData.filter((d) => d.country_name === "KEN");
  const somData = scatterData.filter((d) => d.country_name === "SOM");

  // ── MoM Change Rate ──────────────────────────────────────────
  // Build admin -> date -> row lookup
  const adminDateLookup = useMemo(() => {
    const lookup: Record<string, Record<string, PredictionRow>> = {};
    for (const p of allPreds) {
      if (!lookup[p.admin2_name]) lookup[p.admin2_name] = {};
      lookup[p.admin2_name][p.date] = p;
    }
    return lookup;
  }, [allPreds]);

  const momData = useMemo(() => {
    const idx = scatterDateIdx;
    if (idx <= 0 || scatterDates.length < 2) return [];

    const currDate = scatterDates[idx];
    const prevDate = scatterDates[idx - 1];

    const result: {
      admin2_name: string;
      country_name: string;
      mom_actual: number;
      mom_predicted: number;
    }[] = [];

    for (const admin of Object.keys(adminDateLookup)) {
      const curr = adminDateLookup[admin]?.[currDate];
      const prev = adminDateLookup[admin]?.[prevDate];
      if (!curr || !prev) continue;
      if (prev.actual === 0 || prev.predicted === 0) continue;

      result.push({
        admin2_name: admin,
        country_name: curr.country_name,
        mom_actual:
          ((curr.actual - prev.actual) / Math.abs(prev.actual)) * 100,
        mom_predicted:
          ((curr.predicted - prev.predicted) / Math.abs(prev.predicted)) * 100,
      });
    }
    return result;
  }, [adminDateLookup, scatterDates, scatterDateIdx]);

  const momRange = useMemo(() => {
    if (momData.length === 0) return { min: -10, max: 10 };
    const allVals = momData.flatMap((d) => [d.mom_actual, d.mom_predicted]);
    const absMax = Math.max(...allVals.map(Math.abs), 1);
    return { min: -absMax * 1.05, max: absMax * 1.05 };
  }, [momData]);

  const momKen = momData.filter((d) => d.country_name === "KEN");
  const momSom = momData.filter((d) => d.country_name === "SOM");

  // MoM correlation
  const momCorr = useMemo(() => {
    if (momData.length < 3) return null;
    const n = momData.length;
    const xArr = momData.map((d) => d.mom_predicted);
    const yArr = momData.map((d) => d.mom_actual);
    const xMean = xArr.reduce((s, v) => s + v, 0) / n;
    const yMean = yArr.reduce((s, v) => s + v, 0) / n;
    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = xArr[i] - xMean;
      const dy = yArr[i] - yMean;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const r = denX === 0 || denY === 0 ? 0 : num / Math.sqrt(denX * denY);
    return r;
  }, [momData]);

  return (
    <div className="p-8 space-y-6 max-w-7xl">
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-[11px] font-semibold tracking-[0.18em] text-blue-600 uppercase">
            Time Series
          </p>
          <h2 className="text-2xl font-semibold text-slate-900 mt-1">
            {TARGET_DISPLAY[target]} · h={horizon}
          </h2>
        </div>
        <select
          value={selectedRegion}
          onChange={(e) => setSelectedRegion(e.target.value)}
          className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500"
        >
          {regions.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {loading ? (
        <p className="text-slate-500">Loading...</p>
      ) : (
        <>
          {/* Actual vs Predicted line chart */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <h3 className="text-sm font-semibold mb-3 text-slate-700">
              Actual vs Predicted — {selectedRegion}
            </h3>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10 }}
                  interval={Math.max(0, Math.floor(data.length / 8) - 1)}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#2196F3"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  name="Actual"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#F44336"
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  dot={{ r: 2 }}
                  name="Predicted"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Error bar chart */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <h3 className="text-sm font-semibold mb-3 text-slate-700">
              Prediction Error — {selectedRegion}
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10 }}
                  interval={Math.max(0, Math.floor(data.length / 8) - 1)}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis />
                <Tooltip />
                <ReferenceLine y={0} stroke="#666" />
                <Bar dataKey="error" name="Error">
                  {data.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.error >= 0 ? "#ef4444" : "#3b82f6"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* ── Per-Month Analytics ─────────────────────────────────── */}
      <div className="border-t border-slate-200 pt-8">
        <h2 className="text-xl font-semibold text-slate-900 mb-4">
          Per-Month Analytics: All Regions
        </h2>

        {/* Date slider */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 mb-4">
          <div className="flex items-center gap-4">
            <span className="text-lg font-semibold text-slate-900 w-28">
              {currentScatterDate}
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(0, scatterDates.length - 1)}
              value={scatterDateIdx}
              onChange={(e) => setScatterDateIdx(Number(e.target.value))}
              className="flex-1 accent-blue-600"
            />
            <span className="text-xs text-slate-400">
              {scatterDates[0] || "..."} ~{" "}
              {scatterDates[scatterDates.length - 1] || "..."}
            </span>
          </div>

          {/* Monthly stats */}
          {monthlyStats && (
            <div className="flex gap-6 mt-3 text-sm text-slate-600">
              <span>
                <b className="text-slate-900">N:</b> {monthlyStats.n}
              </span>
              <span>
                <b className="text-slate-900">R²:</b>{" "}
                {monthlyStats.r2.toFixed(4)}
              </span>
              <span>
                <b className="text-slate-900">MAPE:</b>{" "}
                {(monthlyStats.mape * 100).toFixed(1)}%
              </span>
              <span>
                <b className="text-slate-900">RMSE:</b>{" "}
                {monthlyStats.rmse.toFixed(2)}
              </span>
              <span>
                <b className="text-slate-900">MAE:</b>{" "}
                {monthlyStats.mae.toFixed(2)}
              </span>
            </div>
          )}
        </div>

        {/* Scatter plot: Actual (Y) vs Predicted (X), per region */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-sm font-semibold mb-3 text-slate-700">
            Predicted vs Actual — {currentScatterDate} (each dot = 1 admin2)
          </h3>
          <ResponsiveContainer width="100%" height={500}>
            <ScatterChart
              margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="predicted"
                domain={[scatterRange.min, scatterRange.max]}
                tick={{ fontSize: 11 }}
              >
                <Label value="Predicted" position="bottom" offset={35} />
              </XAxis>
              <YAxis
                type="number"
                dataKey="actual"
                domain={[scatterRange.min, scatterRange.max]}
                tick={{ fontSize: 11 }}
              >
                <Label
                  value="Actual"
                  angle={-90}
                  position="left"
                  offset={40}
                />
              </YAxis>
              <ZAxis range={[60, 60]} />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.[0]) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-white border rounded shadow p-2 text-xs">
                      <b>{d.admin2_name}</b> ({d.country_name})
                      <br />
                      Actual: {d.actual.toFixed(3)}
                      <br />
                      Predicted: {d.predicted.toFixed(3)}
                      <br />
                      Error: {d.error.toFixed(4)}
                    </div>
                  );
                }}
              />
              <Legend />
              {/* Perfect prediction line */}
              <ReferenceLine
                segment={[
                  { x: scatterRange.min, y: scatterRange.min },
                  { x: scatterRange.max, y: scatterRange.max },
                ]}
                stroke="#999"
                strokeDasharray="6 3"
                label=""
              />
              <Scatter
                name="Kenya"
                data={kenData}
                fill="#2196F3"
                opacity={0.7}
              />
              <Scatter
                name="Somalia"
                data={somData}
                fill="#F44336"
                opacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* MoM Change Rate scatter */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 mt-4">
          <h3 className="text-sm font-semibold mb-3 text-slate-700">
            MoM Change Rate — {currentScatterDate} vs previous month
            (each dot = 1 admin2)
            {momCorr !== null && (
              <span className="ml-3 font-normal text-slate-400">
                Correlation: {momCorr.toFixed(3)}
              </span>
            )}
          </h3>
          {scatterDateIdx === 0 ? (
            <p className="text-slate-400 text-sm py-8 text-center">
              Move the slider to the 2nd month or later to see MoM changes.
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={500}>
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="mom_predicted"
                  domain={[momRange.min, momRange.max]}
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                >
                  <Label
                    value="Predicted MoM Change (%)"
                    position="bottom"
                    offset={35}
                  />
                </XAxis>
                <YAxis
                  type="number"
                  dataKey="mom_actual"
                  domain={[momRange.min, momRange.max]}
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                >
                  <Label
                    value="Actual MoM Change (%)"
                    angle={-90}
                    position="left"
                    offset={40}
                  />
                </YAxis>
                <ZAxis range={[60, 60]} />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload?.[0]) return null;
                    const d = payload[0].payload;
                    const fmtPct = (v: number) =>
                      `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
                    return (
                      <div className="bg-white border rounded shadow p-2 text-xs">
                        <b>{d.admin2_name}</b> ({d.country_name})
                        <br />
                        Actual MoM: {fmtPct(d.mom_actual)}
                        <br />
                        Predicted MoM: {fmtPct(d.mom_predicted)}
                      </div>
                    );
                  }}
                />
                <Legend />
                {/* Perfect prediction line */}
                <ReferenceLine
                  segment={[
                    { x: momRange.min, y: momRange.min },
                    { x: momRange.max, y: momRange.max },
                  ]}
                  stroke="#999"
                  strokeDasharray="6 3"
                />
                {/* Zero lines */}
                <ReferenceLine x={0} stroke="#ccc" />
                <ReferenceLine y={0} stroke="#ccc" />
                <Scatter
                  name="Kenya"
                  data={momKen}
                  fill="#2196F3"
                  opacity={0.7}
                />
                <Scatter
                  name="Somalia"
                  data={momSom}
                  fill="#F44336"
                  opacity={0.7}
                />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
