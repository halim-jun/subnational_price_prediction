"use client";

import { useState, useEffect, FormEvent } from "react";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("error") === "expired") {
      setError("Your session has expired. Please sign in again.");
    }
  }, []);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        const params = new URLSearchParams(window.location.search);
        const next = params.get("next") || "/";
        window.location.replace(next);
        return;
      }
      if (res.status === 429) {
        setError("Too many attempts. Please try again in a moment.");
      } else {
        setError("Incorrect password.");
      }
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f5f7fb] px-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <span className="inline-block text-[11px] font-semibold tracking-[0.2em] text-blue-600 uppercase border border-blue-100 bg-blue-50 rounded-full px-3 py-1">
            Private preview
          </span>
          <h1 className="text-2xl font-semibold text-slate-900 mt-5 leading-snug">
            East Africa food price
            <br />
            prediction prototype
          </h1>
          <p className="text-sm text-slate-500 mt-3">
            Sign in to access the dashboard.
          </p>
        </div>

        <form
          onSubmit={onSubmit}
          className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 space-y-4"
        >
          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-slate-700 mb-2"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              autoFocus
              required
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500"
              disabled={submitting}
            />
          </div>

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !password}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 text-white text-sm font-medium rounded-full px-4 py-2.5 transition-colors"
          >
            {submitting ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <p className="text-center text-xs text-slate-500 mt-6">
          No access? Please contact the administrator.
        </p>
      </div>
    </div>
  );
}
