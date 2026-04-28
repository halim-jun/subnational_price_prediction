"use client";

import dynamic from "next/dynamic";

const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-screen text-sm text-gray-400">
      Loading map...
    </div>
  ),
});

export default function MapPage() {
  return <MapView />;
}
