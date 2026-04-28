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
    <div className="p-6 space-y-8">
      <h2 className="text-xl font-bold">
        Held-Out Test: {TARGET_DISPLAY[target]} h={horizon}
      </h2>
      <p className="text-sm text-gray-500">
        Train: 2008-2023 &rarr; Test: 2024-2025
      </p>

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

      <div>
        <h3 className="text-lg font-semibold mb-3">All Configurations</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-3 py-2 text-left">Target</th>
                <th className="px-3 py-2">Horizon</th>
                <th className="px-3 py-2">R²</th>
                <th className="px-3 py-2">MAPE</th>
                <th className="px-3 py-2">RMSE</th>
                <th className="px-3 py-2">KEN MAPE</th>
                <th className="px-3 py-2">SOM MAPE</th>
                <th className="px-3 py-2">N</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, i) => (
                <tr
                  key={i}
                  className={
                    m.target === target && m.horizon === horizon
                      ? "bg-blue-50 font-medium"
                      : "hover:bg-gray-50"
                  }
                >
                  <td className="px-3 py-1.5">
                    {TARGET_DISPLAY[m.target as keyof typeof TARGET_DISPLAY] ||
                      m.target}
                  </td>
                  <td className="px-3 py-1.5 text-center">{m.horizon}</td>
                  <td className="px-3 py-1.5 text-center">
                    {m.r2.toFixed(3)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    {(m.mape * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    {m.rmse.toFixed(2)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    {((m.KEN_mape ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    {((m.SOM_mape ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-1.5 text-center">{m.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-semibold mb-2">Best 10 Regions</h3>
          <table className="w-full text-sm border">
            <thead className="bg-green-50">
              <tr>
                <th className="px-2 py-1.5 text-left">Region</th>
                <th className="px-2 py-1.5">Country</th>
                <th className="px-2 py-1.5">MAPE</th>
              </tr>
            </thead>
            <tbody>
              {best.map((r, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-2 py-1">{r.admin2_name}</td>
                  <td className="px-2 py-1 text-center">{r.country_name}</td>
                  <td className="px-2 py-1 text-center">
                    {(r.mape * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3 className="text-lg font-semibold mb-2">Worst 10 Regions</h3>
          <table className="w-full text-sm border">
            <thead className="bg-red-50">
              <tr>
                <th className="px-2 py-1.5 text-left">Region</th>
                <th className="px-2 py-1.5">Country</th>
                <th className="px-2 py-1.5">MAPE</th>
              </tr>
            </thead>
            <tbody>
              {worst.map((r, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-2 py-1">{r.admin2_name}</td>
                  <td className="px-2 py-1 text-center">{r.country_name}</td>
                  <td className="px-2 py-1 text-center">
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
