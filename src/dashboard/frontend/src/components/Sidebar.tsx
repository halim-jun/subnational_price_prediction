"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useFilters } from "@/hooks/useFilters";
import { TARGETS, TARGET_DISPLAY, HORIZONS } from "@/lib/constants";
import type { Target, Horizon } from "@/types";
import LogoutButton from "@/components/LogoutButton";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/map", label: "Prediction Map" },
  { href: "/timeseries", label: "Time Series" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { target, horizon, setTarget, setHorizon } = useFilters();

  return (
    <aside className="w-64 bg-gray-900 text-white flex flex-col h-screen fixed left-0 top-0">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-lg font-bold">EA Food Price</h1>
        <p className="text-xs text-gray-400">Prediction Dashboard</p>
      </div>

      <nav className="p-4 space-y-1">
        {NAV.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`block px-3 py-2 rounded text-sm ${
              pathname === item.href
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-800"
            }`}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="p-4 border-t border-gray-700 space-y-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Target</label>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value as Target)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm"
          >
            {TARGETS.map((t) => (
              <option key={t} value={t}>
                {TARGET_DISPLAY[t]}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1">Horizon</label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value) as Horizon)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm"
          >
            {HORIZONS.map((h) => (
              <option key={h} value={h}>
                {h} month{h > 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-auto p-4 border-t border-gray-700">
        <LogoutButton />
      </div>
    </aside>
  );
}
