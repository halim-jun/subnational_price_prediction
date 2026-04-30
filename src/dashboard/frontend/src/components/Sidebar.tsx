"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useFilters } from "@/hooks/useFilters";
import { TARGETS, TARGET_DISPLAY, HORIZONS } from "@/lib/constants";
import type { Target, Horizon } from "@/types";
import LogoutButton from "@/components/LogoutButton";

const NAV = [
  { href: "/", label: "Prediction Map" },
  { href: "/timeseries", label: "Time Series" },
  { href: "/performance", label: "Performance Evaluation" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { target, horizon, setTarget, setHorizon } = useFilters();

  return (
    <aside className="w-64 bg-white text-slate-800 flex flex-col h-screen fixed left-0 top-0 border-r border-slate-200">
      <div className="px-5 py-6 border-b border-slate-200">
        <p className="text-[10px] font-semibold tracking-[0.18em] text-blue-600 uppercase">
          Prototype
        </p>
        <h1 className="text-base font-semibold text-slate-900 mt-1 leading-snug">
          East Africa food price
          <br />
          prediction prototype
        </h1>
      </div>

      <nav className="px-3 py-4 space-y-1">
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`block px-3 py-2 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-slate-200 space-y-4">
        <div>
          <label className="block text-[11px] font-medium text-slate-500 mb-1.5 tracking-wide uppercase">
            Target
          </label>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value as Target)}
            className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500"
          >
            {TARGETS.map((t) => (
              <option key={t} value={t}>
                {TARGET_DISPLAY[t]}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[11px] font-medium text-slate-500 mb-1.5 tracking-wide uppercase">
            Horizon
          </label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value) as Horizon)}
            className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500"
          >
            {HORIZONS.map((h) => (
              <option key={h} value={h}>
                {h} month{h > 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-auto px-5 py-4 border-t border-slate-200">
        <LogoutButton />
      </div>
    </aside>
  );
}
