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
      className="w-full text-left text-xs text-gray-400 hover:text-white disabled:text-gray-600"
    >
      {busy ? "로그아웃 중..." : "로그아웃 →"}
    </button>
  );
}
