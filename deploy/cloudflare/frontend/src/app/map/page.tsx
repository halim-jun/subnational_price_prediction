"use client";

import dynamic from "next/dynamic";

// Leaflet touches `window`/`navigator` at import time, so it must only run in
// the browser. Disabling SSR keeps the static-export build from crashing.
const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => (
    <div className="p-6 text-sm text-gray-500">Loading map…</div>
  ),
});

export default function MapPage() {
  return <MapView />;
}
