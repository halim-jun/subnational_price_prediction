"use client";

import { useState } from "react";

export default function LogoutButton() {
  const [busy, setBusy] = useState(false);

  async function onClick() {
    if (busy) return;
    setBusy(true);
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {
      // Cookie clear is best-effort. Reload anyway so middleware re-checks.
    } finally {
      window.location.replace("/login");
    }
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className="w-full text-left text-xs text-slate-500 hover:text-slate-900 disabled:text-slate-300 transition-colors"
    >
      {busy ? "Signing out..." : "Sign out →"}
    </button>
  );
}
