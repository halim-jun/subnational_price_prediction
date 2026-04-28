"use client";

import { useState, useEffect, FormEvent } from "react";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("error") === "expired") {
      setError("세션이 만료되었습니다. 다시 로그인해주세요.");
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
        setError("로그인 시도가 너무 많습니다. 잠시 후 다시 시도해주세요.");
      } else {
        setError("비밀번호가 올바르지 않습니다.");
      }
    } catch {
      setError("로그인 중 오류가 발생했습니다. 다시 시도해주세요.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 px-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white">EA Food Price</h1>
          <p className="text-sm text-gray-400 mt-1">Prediction Dashboard</p>
          <p className="text-xs text-gray-500 mt-3">프로토타입 — 비공개 미리보기</p>
        </div>

        <form
          onSubmit={onSubmit}
          className="bg-gray-800 border border-gray-700 rounded-lg p-6 space-y-4"
        >
          <div>
            <label
              htmlFor="password"
              className="block text-sm text-gray-300 mb-2"
            >
              비밀번호
            </label>
            <input
              id="password"
              type="password"
              autoFocus
              required
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              disabled={submitting}
            />
          </div>

          {error && (
            <div className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !password}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded px-3 py-2 transition-colors"
          >
            {submitting ? "확인 중..." : "들어가기"}
          </button>
        </form>

        <p className="text-center text-xs text-gray-600 mt-6">
          접근 권한이 없으시면 관리자에게 문의해주세요.
        </p>
      </div>
    </div>
  );
}
