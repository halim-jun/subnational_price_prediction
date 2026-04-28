import type {
  PredictionRow,
  MetricSummary,
  TimeSeriesPoint,
  FeatureImportance,
  PerAdminRow,
} from "@/types";

const BASE = "";
const STATIC_MODE = process.env.NEXT_PUBLIC_STATIC_MODE === "true";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

function tag(target: string, horizon: number): string {
  return `${target}_${horizon}`;
}

export async function getGeoJson(): Promise<GeoJSON.FeatureCollection> {
  if (STATIC_MODE) return fetchJson(`${BASE}/data/geo/admin2.json`);
  return fetchJson(`${BASE}/api/geo/admin2`);
}

export async function getAllPredictions(
  target: string,
  horizon: number
): Promise<PredictionRow[]> {
  const url = STATIC_MODE
    ? `${BASE}/data/predictions/map_${tag(target, horizon)}.json`
    : `${BASE}/api/predictions/map/all?target=${target}&horizon=${horizon}`;
  const res = await fetchJson<{ data: PredictionRow[] }>(url);
  return res.data;
}

export async function getTimeSeries(
  target: string,
  horizon: number,
  admin2Name: string
): Promise<TimeSeriesPoint[]> {
  if (STATIC_MODE) {
    const bundle = await fetchJson<{
      regions: Record<string, TimeSeriesPoint[]>;
    }>(`${BASE}/data/predictions/timeseries_${tag(target, horizon)}.json`);
    return bundle.regions[admin2Name] ?? [];
  }
  const res = await fetchJson<{ data: TimeSeriesPoint[] }>(
    `${BASE}/api/predictions/timeseries?target=${target}&horizon=${horizon}&admin2_name=${encodeURIComponent(admin2Name)}`
  );
  return res.data;
}

export async function getRegions(
  target: string,
  horizon: number
): Promise<string[]> {
  const url = STATIC_MODE
    ? `${BASE}/data/predictions/regions_${tag(target, horizon)}.json`
    : `${BASE}/api/predictions/regions?target=${target}&horizon=${horizon}`;
  const res = await fetchJson<{ regions: string[] }>(url);
  return res.regions;
}

export async function getMetricsSummary(
  target?: string,
  horizon?: number
): Promise<MetricSummary[]> {
  if (STATIC_MODE) {
    const res = await fetchJson<{ data: MetricSummary[] }>(
      `${BASE}/data/metrics/summary.json`
    );
    let rows = res.data;
    if (target) rows = rows.filter((r) => r.target === target);
    if (horizon) rows = rows.filter((r) => r.horizon === horizon);
    return rows;
  }
  const params = new URLSearchParams();
  if (target) params.set("target", target);
  if (horizon) params.set("horizon", String(horizon));
  const res = await fetchJson<{ data: MetricSummary[] }>(
    `${BASE}/api/metrics/summary?${params}`
  );
  return res.data;
}

export async function getPerAdmin(
  target: string,
  horizon: number
): Promise<PerAdminRow[]> {
  const url = STATIC_MODE
    ? `${BASE}/data/metrics/per-admin_${tag(target, horizon)}.json`
    : `${BASE}/api/metrics/per-admin?target=${target}&horizon=${horizon}`;
  const res = await fetchJson<{ data: PerAdminRow[] }>(url);
  return res.data;
}

export async function getFeatureImportance(
  target: string,
  horizon: number
): Promise<{ data: FeatureImportance[]; colors: Record<string, string> }> {
  const url = STATIC_MODE
    ? `${BASE}/data/features/importance_${tag(target, horizon)}.json`
    : `${BASE}/api/features/importance?target=${target}&horizon=${horizon}`;
  return fetchJson(url);
}
