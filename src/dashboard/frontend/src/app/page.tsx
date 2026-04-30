"use client";

import { useEffect, useState } from "react";
import { useFilters } from "@/hooks/useFilters";
import { getMetricsSummary, getPerAdmin } from "@/lib/api";
import { TARGET_DISPLAY } from "@/lib/constants";
import KpiCard from "@/components/KpiCard";
import type { MetricSummary, PerAdminRow } from "@/types";

export default function OverviewPage() {
  const { target, horizon } = useFilters();
  const [metrics, setMetrics] = useState<MetricSummary[]>([]);
  const [perAdmin, setPerAdmin] = useState<PerAdminRow[]>([]);

  useEffect(() => {
    getMetricsSummary().then(setMetrics);
  }, []);

  useEffect(() => {
    getPerAdmin(target, horizon).then(setPerAdmin);
  }, [target, horizon]);

  const current = metrics.find(
    (m) => m.target === target && m.horizon === horizon
  );

  const best = [...perAdmin].sort((a, b) => a.mape - b.mape).slice(0, 10);
  const worst = [...perAdmin].sort((a, b) => b.mape - a.mape).slice(0, 10);

  return (
    <div className="p-8 space-y-8 max-w-7xl">
      <div>
        <p className="text-[11px] font-semibold tracking-[0.18em] text-blue-600 uppercase">
          Held-out evaluation
        </p>
        <h2 className="text-2xl font-semibold text-slate-900 mt-1">
          {TARGET_DISPLAY[target]} · h={horizon}
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Train: 2008–2023 → Test: 2024–2025
        </p>
      </div>

      {current && (
        <div className="grid grid-cols-5 gap-4">
          <KpiCard title="R²" value={current.r2.toFixed(4)} />
          <KpiCard
            title="MAPE"
            value={`${(current.mape * 100).toFixed(1)}%`}
          />
          <KpiCard title="RMSE" value={current.rmse.toFixed(2)} />
          <KpiCard title="N" value={String(current.n)} />
          <KpiCard
            title="By Country"
            value={`KEN ${((current.KEN_mape ?? 0) * 100).toFixed(1)}% / SOM ${((current.SOM_mape ?? 0) * 100).toFixed(1)}%`}
          />
        </div>
      )}

      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <h3 className="text-base font-semibold text-slate-900 px-5 pt-5 pb-3">
          All Configurations
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 uppercase text-[11px] tracking-wide">
              <tr>
                <th className="px-4 py-3 text-left font-medium">Target</th>
                <th className="px-4 py-3 font-medium">Horizon</th>
                <th className="px-4 py-3 font-medium">R²</th>
                <th className="px-4 py-3 font-medium">MAPE</th>
                <th className="px-4 py-3 font-medium">RMSE</th>
                <th className="px-4 py-3 font-medium">KEN MAPE</th>
                <th className="px-4 py-3 font-medium">SOM MAPE</th>
                <th className="px-4 py-3 font-medium">N</th>
              </tr>
            </thead>
            <tbody className="text-slate-700">
              {metrics.map((m, i) => (
                <tr
                  key={i}
                  className={`border-t border-slate-100 ${
                    m.target === target && m.horizon === horizon
                      ? "bg-blue-50/60 text-slate-900 font-medium"
                      : "hover:bg-slate-50"
                  }`}
                >
                  <td className="px-4 py-2.5">
                    {TARGET_DISPLAY[m.target as keyof typeof TARGET_DISPLAY] ||
                      m.target}
                  </td>
                  <td className="px-4 py-2.5 text-center">{m.horizon}</td>
                  <td className="px-4 py-2.5 text-center">
                    {m.r2.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {(m.mape * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {m.rmse.toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {((m.KEN_mape ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {((m.SOM_mape ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2.5 text-center">{m.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <h3 className="text-base font-semibold text-slate-900 px-5 pt-5 pb-3">
            Best 10 Regions
          </h3>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 uppercase text-[11px] tracking-wide">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">Region</th>
                <th className="px-4 py-2.5 font-medium">Country</th>
                <th className="px-4 py-2.5 font-medium">MAPE</th>
              </tr>
            </thead>
            <tbody className="text-slate-700">
              {best.map((r, i) => (
                <tr key={i} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-4 py-2">{r.admin2_name}</td>
                  <td className="px-4 py-2 text-center">{r.country_name}</td>
                  <td className="px-4 py-2 text-center text-emerald-700 font-medium">
                    {(r.mape * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <h3 className="text-base font-semibold text-slate-900 px-5 pt-5 pb-3">
            Worst 10 Regions
          </h3>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 uppercase text-[11px] tracking-wide">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">Region</th>
                <th className="px-4 py-2.5 font-medium">Country</th>
                <th className="px-4 py-2.5 font-medium">MAPE</th>
              </tr>
            </thead>
            <tbody className="text-slate-700">
              {worst.map((r, i) => (
                <tr key={i} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-4 py-2">{r.admin2_name}</td>
                  <td className="px-4 py-2 text-center">{r.country_name}</td>
                  <td className="px-4 py-2 text-center text-rose-700 font-medium">
                    {(r.mape * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
