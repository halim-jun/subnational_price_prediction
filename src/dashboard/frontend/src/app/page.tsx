"use client";

import dynamic from "next/dynamic";

const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-screen text-sm text-slate-400">
      Loading map...
    </div>
  ),
});

export default function HomePage() {
  return <MapView />;
}
