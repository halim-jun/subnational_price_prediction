"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useFilters } from "@/hooks/useFilters";
import { getGeoJson, getAllPredictions } from "@/lib/api";
import { TARGET_DISPLAY } from "@/lib/constants";
import type { PredictionRow } from "@/types";

type MapView = "predicted" | "actual" | "error" | "spike_actual" | "spike_predicted";

function valueToColor(
  val: number,
  min: number,
  max: number,
  mode: MapView
): string {
  if (val === null || val === undefined || isNaN(val)) return "#ccc";

  if (mode === "error") {
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const t = Math.max(-1, Math.min(1, val / absMax));
    if (t < 0) {
      const s = Math.abs(t);
      return `rgb(${Math.round(255 * (1 - s))},${Math.round(255 * (1 - s * 0.6))},255)`;
    }
    return `rgb(255,${Math.round(255 * (1 - t * 0.6))},${Math.round(255 * (1 - t))})`;
  }

  // Spike: diverging green (decrease) -> white (0) -> red (increase)
  if (mode === "spike_actual" || mode === "spike_predicted") {
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const t = Math.max(-1, Math.min(1, val / absMax));
    if (t < 0) {
      // Green for decrease
      const s = Math.abs(t);
      return `rgb(${Math.round(255 * (1 - s * 0.7))},${Math.round(200 + 55 * (1 - s))},${Math.round(255 * (1 - s * 0.7))})`;
    }
    // Red for increase
    const s = t;
    return `rgb(${Math.round(200 + 55 * (1 - s))},${Math.round(255 * (1 - s * 0.7))},${Math.round(255 * (1 - s * 0.7))})`;
  }

  // Sequential viridis-like
  const range = max - min || 1;
  const t = Math.max(0, Math.min(1, (val - min) / range));
  const r = Math.round(68 + t * 185);
  const g = Math.round(1 + t * 230);
  const b = Math.round(84 + (0.5 - Math.abs(t - 0.5)) * 2 * 66);
  return `rgb(${r},${g},${b})`;
}

// Compute MoM spike % for each (admin2, date)
function computeSpikes(
  predictions: PredictionRow[],
  dates: string[]
): Record<string, Record<string, { spike_actual: number; spike_predicted: number }>> {
  // Build lookup: admin2 -> date -> row
  const byAdmin: Record<string, Record<string, PredictionRow>> = {};
  for (const p of predictions) {
    if (!byAdmin[p.admin2_name]) byAdmin[p.admin2_name] = {};
    byAdmin[p.admin2_name][p.date] = p;
  }

  // For each date, compute spike = (current - prev) / prev * 100
  const result: Record<string, Record<string, { spike_actual: number; spike_predicted: number }>> = {};
  for (let i = 0; i < dates.length; i++) {
    const date = dates[i];
    const prevDate = i > 0 ? dates[i - 1] : null;
    result[date] = {};

    for (const admin of Object.keys(byAdmin)) {
      const curr = byAdmin[admin]?.[date];
      const prev = prevDate ? byAdmin[admin]?.[prevDate] : null;

      if (curr && prev && prev.actual !== 0 && prev.predicted !== 0) {
        result[date][admin] = {
          spike_actual: ((curr.actual - prev.actual) / Math.abs(prev.actual)) * 100,
          spike_predicted: ((curr.predicted - prev.predicted) / Math.abs(prev.predicted)) * 100,
        };
      } else {
        result[date][admin] = { spike_actual: NaN, spike_predicted: NaN };
      }
    }
  }
  return result;
}

export default function MapView() {
  const { target, horizon } = useFilters();
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const geoLayerRef = useRef<L.GeoJSON | null>(null);

  const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [dates, setDates] = useState<string[]>([]);
  const [dateIdx, setDateIdx] = useState(0);
  const [mapView, setMapView] = useState<MapView>("predicted");
  const [loading, setLoading] = useState(true);

  // Load data
  useEffect(() => {
    setLoading(true);
    Promise.all([getGeoJson(), getAllPredictions(target, horizon)])
      .then(([geo, preds]) => {
        setGeojson(geo);
        setPredictions(preds);
        const uniqueDates = [...new Set(preds.map((p) => p.date))].sort();
        setDates(uniqueDates);
        setDateIdx(uniqueDates.length - 1);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load data:", err);
        setLoading(false);
      });
  }, [target, horizon]);

  // Build lookup
  const predLookup = useMemo(() => {
    const lookup: Record<string, Record<string, PredictionRow>> = {};
    for (const p of predictions) {
      if (!lookup[p.date]) lookup[p.date] = {};
      lookup[p.date][p.admin2_name] = p;
    }
    return lookup;
  }, [predictions]);

  // Spike lookup
  const spikeLookup = useMemo(() => {
    return computeSpikes(predictions, dates);
  }, [predictions, dates]);

  const currentDate = dates[dateIdx] || "";
  const currentData = predLookup[currentDate] || {};
  const currentSpikes = spikeLookup[currentDate] || {};

  const isSpike = mapView === "spike_actual" || mapView === "spike_predicted";

  const colorRange = useMemo(() => {
    if (predictions.length === 0) return { min: 0, max: 1 };

    if (isSpike) {
      // Compute range from all spike values
      const allSpikes: number[] = [];
      for (const date of dates) {
        const sp = spikeLookup[date];
        if (!sp) continue;
        for (const admin of Object.keys(sp)) {
          const v = mapView === "spike_actual" ? sp[admin].spike_actual : sp[admin].spike_predicted;
          if (!isNaN(v)) allSpikes.push(v);
        }
      }
      if (allSpikes.length === 0) return { min: -10, max: 10 };
      const sorted = allSpikes.sort((a, b) => a - b);
      const p5 = sorted[Math.floor(sorted.length * 0.05)] ?? -10;
      const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 10;
      const absMax = Math.max(Math.abs(p5), Math.abs(p95));
      return { min: -absMax, max: absMax };
    }

    if (mapView === "error") {
      const vals = predictions.map((p) => p.error).filter((v) => !isNaN(v));
      const sorted = vals.sort((a, b) => a - b);
      const p5 = sorted[Math.floor(sorted.length * 0.05)] ?? -0.1;
      const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 0.1;
      const absMax = Math.max(Math.abs(p5), Math.abs(p95));
      return { min: -absMax, max: absMax };
    }

    const vals = predictions.map((p) =>
      mapView === "predicted" ? p.predicted : p.actual
    );
    const sorted = vals.filter((v) => !isNaN(v)).sort((a, b) => a - b);
    return {
      min: sorted[Math.floor(sorted.length * 0.02)] ?? 0,
      max: sorted[Math.floor(sorted.length * 0.98)] ?? 1,
    };
  }, [predictions, mapView, dates, spikeLookup, isSpike]);

  // Initialize Leaflet map once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = L.map(mapContainer.current).setView([1, 42], 5);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 12,
    }).addTo(map);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      geoLayerRef.current = null;
    };
  }, []);

  // Update GeoJSON layer
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !geojson) return;

    if (geoLayerRef.current) {
      map.removeLayer(geoLayerRef.current);
      geoLayerRef.current = null;
    }

    const layer = L.geoJSON(geojson, {
      style: (feature) => {
        const name = feature?.properties?.shapeName;
        const pred = currentData[name];
        const spike = currentSpikes[name];

        let val: number;
        if (mapView === "spike_actual") {
          val = spike?.spike_actual ?? NaN;
        } else if (mapView === "spike_predicted") {
          val = spike?.spike_predicted ?? NaN;
        } else if (mapView === "predicted") {
          val = pred?.predicted ?? NaN;
        } else if (mapView === "actual") {
          val = pred?.actual ?? NaN;
        } else {
          val = pred?.error ?? NaN;
        }

        return {
          fillColor: valueToColor(val, colorRange.min, colorRange.max, mapView),
          fillOpacity: 0.75,
          color: "#333",
          weight: 0.8,
        };
      },
      onEachFeature: (feature, featureLayer) => {
        const name = feature.properties?.shapeName;
        const pred = currentData[name];
        const spike = currentSpikes[name];
        const fmt = (v: number | null | undefined) =>
          v !== null && v !== undefined && !isNaN(v) ? v.toFixed(3) : "N/A";
        const fmtPct = (v: number | null | undefined) =>
          v !== null && v !== undefined && !isNaN(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(1)}%` : "N/A";

        featureLayer.bindTooltip(
          `<div style="font-size:13px;line-height:1.6">
            <strong>${name}</strong> (${pred?.country_name || feature.properties?.country_iso || ""})<br/>
            Actual: <b>${fmt(pred?.actual)}</b><br/>
            Predicted: <b>${fmt(pred?.predicted)}</b><br/>
            Error: <b>${fmt(pred?.error)}</b><br/>
            <hr style="margin:4px 0;border-color:#ddd"/>
            MoM Spike (Actual): <b>${fmtPct(spike?.spike_actual)}</b><br/>
            MoM Spike (Predicted): <b>${fmtPct(spike?.spike_predicted)}</b>
          </div>`,
          { sticky: true }
        );
      },
    }).addTo(map);

    geoLayerRef.current = layer;
  }, [geojson, currentData, currentSpikes, colorRange, mapView]);

  const viewLabels: { key: MapView; label: string }[] = [
    { key: "predicted", label: "Predicted" },
    { key: "actual", label: "Actual" },
    { key: "error", label: "Error" },
    { key: "spike_actual", label: "MoM Spike (Actual)" },
    { key: "spike_predicted", label: "MoM Spike (Predicted)" },
  ];

  const legendGradient = isSpike
    ? "linear-gradient(to right, rgb(76,175,80), white, rgb(244,67,54))"
    : mapView === "error"
      ? "linear-gradient(to right, rgb(100,150,255), white, rgb(255,100,100))"
      : "linear-gradient(to right, rgb(68,1,84), rgb(160,116,84), rgb(253,231,37))";

  const legendMin = isSpike
    ? `${colorRange.min.toFixed(0)}%`
    : colorRange.min.toFixed(3);
  const legendMax = isSpike
    ? `+${colorRange.max.toFixed(0)}%`
    : colorRange.max.toFixed(3);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <div className="flex items-center gap-4 px-6 py-4 bg-white border-b border-slate-200 shrink-0 flex-wrap">
        <div>
          <p className="text-[11px] font-semibold tracking-[0.18em] text-blue-600 uppercase">
            Prediction Map
          </p>
          <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
            {TARGET_DISPLAY[target]} &middot; h={horizon}
          </h2>
        </div>
        <div className="flex gap-1.5 ml-4 flex-wrap">
          {viewLabels.map((v) => (
            <button
              key={v.key}
              onClick={() => setMapView(v.key)}
              className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
                mapView === v.key
                  ? "bg-blue-600 text-white shadow-sm"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
        {loading && (
          <span className="text-sm text-slate-400 ml-4">Loading...</span>
        )}
      </div>

      {/* Date slider */}
      <div className="px-6 py-3 bg-[#f5f7fb] border-b border-slate-200 flex items-center gap-4 shrink-0">
        <span className="text-sm font-semibold text-slate-900 w-24">
          {currentDate}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, dates.length - 1)}
          value={dateIdx}
          onChange={(e) => setDateIdx(Number(e.target.value))}
          className="flex-1 accent-blue-600"
        />
        <span className="text-xs text-slate-400">
          {dates[0] || "..."} ~ {dates[dates.length - 1] || "..."}
        </span>
      </div>

      {/* Map */}
      <div ref={mapContainer} style={{ flex: 1, minHeight: "500px" }} />

      {/* Legend */}
      <div className="px-6 py-2.5 bg-white border-t border-slate-200 flex items-center gap-3 text-xs text-slate-600 shrink-0">
        <span>{legendMin}</span>
        <div
          className="h-2.5 flex-1 rounded-full"
          style={{ background: legendGradient }}
        />
        <span>{legendMax}</span>
        {isSpike && (
          <span className="ml-2 text-slate-400">
            (month-over-month % change)
          </span>
        )}
      </div>
    </div>
  );
}
