import type {
  PredictionRow,
  MetricSummary,
  TimeSeriesPoint,
  FeatureImportance,
  PerAdminRow,
} from "@/types";

// Cloudflare Pages serves /data/*.json from public/data/. When the site is
// hosted under a sub-path, NEXT_PUBLIC_BASE_PATH propagates through here too.
const BASE = (process.env.NEXT_PUBLIC_BASE_PATH ?? "") + "/data";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Static fetch failed: ${url} (${res.status})`);
  return res.json();
}

// In-memory cache for predictions; one (target,horizon) bundle feeds map,
// timeseries, scatter, and regions, so we only fetch it once per session.
const predCache = new Map<
  string,
  Promise<{ data: PredictionRow[]; regions: string[] }>
>();

function predBundle(target: string, horizon: number) {
  const key = `${target}_h${horizon}`;
  let p = predCache.get(key);
  if (!p) {
    p = fetchJson<{ data: PredictionRow[]; regions: string[] }>(
      `${BASE}/predictions/${key}.json`
    );
    predCache.set(key, p);
  }
  return p;
}

export async function getGeoJson(): Promise<GeoJSON.FeatureCollection> {
  return fetchJson(`${BASE}/geo/admin2.json`);
}

export async function getAllPredictions(
  target: string,
  horizon: number
): Promise<PredictionRow[]> {
  const bundle = await predBundle(target, horizon);
  return bundle.data;
}

export async function getTimeSeries(
  target: string,
  horizon: number,
  admin2Name: string
): Promise<TimeSeriesPoint[]> {
  const bundle = await predBundle(target, horizon);
  return bundle.data
    .filter((r) => r.admin2_name === admin2Name)
    .map((r) => ({
      date: r.date,
      actual: r.actual,
      predicted: r.predicted,
      error: r.error,
    }));
}

export async function getRegions(
  target: string,
  horizon: number
): Promise<string[]> {
  const bundle = await predBundle(target, horizon);
  return bundle.regions;
}

export async function getMetricsSummary(
  target?: string,
  horizon?: number
): Promise<MetricSummary[]> {
  const res = await fetchJson<{ data: MetricSummary[] }>(
    `${BASE}/metrics/summary.json`
  );
  return res.data.filter(
    (m) =>
      (!target || m.target === target) &&
      (horizon === undefined || m.horizon === horizon)
  );
}

export async function getPerAdmin(
  target: string,
  horizon: number
): Promise<PerAdminRow[]> {
  const res = await fetchJson<{ data: PerAdminRow[] }>(
    `${BASE}/metrics/per-admin/${target}_h${horizon}.json`
  );
  return res.data;
}

export async function getFeatureImportance(
  target: string,
  horizon: number
): Promise<{ data: FeatureImportance[]; colors: Record<string, string> }> {
  return fetchJson(
    `${BASE}/features/importance/${target}_h${horizon}.json`
  );
}
