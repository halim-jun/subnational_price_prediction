// POST /api/auth/login — validates shared password, issues HMAC-signed cookie.
// In-memory IP rate limit (per-isolate). Adequate for prototype scale.

import {
  constantTimeEqual,
  issueCookieValue,
  setCookieHeader,
} from "../../_lib/auth";

interface Env {
  SITE_PASSWORD: string;
  AUTH_SECRET: string;
}

const FAIL_DELAY_MS = 1500;
const LOCKOUT_THRESHOLD = 5;
const LOCKOUT_WINDOW_MS = 60_000;
const LOCKOUT_DURATION_MS = 60_000;

interface AttemptState {
  count: number;
  firstAt: number;
  lockedUntil: number;
}

const attempts = new Map<string, AttemptState>();

function clientIp(req: Request): string {
  return (
    req.headers.get("CF-Connecting-IP") ||
    req.headers.get("X-Forwarded-For")?.split(",")[0].trim() ||
    "unknown"
  );
}

function checkLockout(ip: string): number {
  const now = Date.now();
  const s = attempts.get(ip);
  if (!s) return 0;
  if (s.lockedUntil > now) return s.lockedUntil - now;
  if (now - s.firstAt > LOCKOUT_WINDOW_MS) {
    attempts.delete(ip);
    return 0;
  }
  return 0;
}

function recordFailure(ip: string): void {
  const now = Date.now();
  const s = attempts.get(ip);
  if (!s || now - s.firstAt > LOCKOUT_WINDOW_MS) {
    attempts.set(ip, { count: 1, firstAt: now, lockedUntil: 0 });
    return;
  }
  s.count += 1;
  if (s.count >= LOCKOUT_THRESHOLD) {
    s.lockedUntil = now + LOCKOUT_DURATION_MS;
  }
}

function recordSuccess(ip: string): void {
  attempts.delete(ip);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const onRequestPost: PagesFunction<Env> = async ({ request, env }) => {
  if (!env.SITE_PASSWORD || !env.AUTH_SECRET) {
    return new Response(
      JSON.stringify({
        error: "Server misconfigured: SITE_PASSWORD or AUTH_SECRET missing.",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  const ip = clientIp(request);
  const lockedFor = checkLockout(ip);
  if (lockedFor > 0) {
    return new Response(
      JSON.stringify({
        error: "Too many attempts.",
        retryAfterSeconds: Math.ceil(lockedFor / 1000),
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": String(Math.ceil(lockedFor / 1000)),
        },
      }
    );
  }

  let body: { password?: unknown };
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid request body." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const submitted =
    typeof body.password === "string" ? body.password : "";
  const ok = await constantTimeEqual(submitted, env.SITE_PASSWORD);

  if (!ok) {
    recordFailure(ip);
    await sleep(FAIL_DELAY_MS);
    return new Response(JSON.stringify({ error: "Invalid password." }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  recordSuccess(ip);
  const cookieValue = await issueCookieValue(env.AUTH_SECRET);
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Set-Cookie": setCookieHeader(cookieValue),
    },
  });
};
