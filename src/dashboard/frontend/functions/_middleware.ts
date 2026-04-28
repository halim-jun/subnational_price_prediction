// Cloudflare Pages middleware — gates all requests behind a shared password.
// Allowlists the login page, auth API, and Next.js framework assets.
// Anything else without a valid cookie is redirected to /login?next=<path>.

import { COOKIE_NAME, readCookie, verifyCookieValue } from "./_lib/auth";

interface Env {
  AUTH_SECRET: string;
}

const ALLOW_PREFIXES = [
  "/login",
  "/api/auth/",
  "/_next/",
  "/favicon",
];

const ALLOW_EXACT = new Set<string>([
  "/robots.txt",
]);

function isAllowlisted(pathname: string): boolean {
  if (ALLOW_EXACT.has(pathname)) return true;
  for (const p of ALLOW_PREFIXES) {
    if (pathname === p || pathname.startsWith(p)) return true;
  }
  return false;
}

export const onRequest: PagesFunction<Env> = async (ctx) => {
  const { request, env, next } = ctx;
  const url = new URL(request.url);

  if (isAllowlisted(url.pathname)) {
    return next();
  }

  if (!env.AUTH_SECRET) {
    return new Response(
      "Server misconfigured: AUTH_SECRET env var is not set in Cloudflare Pages.",
      { status: 500 }
    );
  }

  const cookie = readCookie(request, COOKIE_NAME);
  const ok = await verifyCookieValue(cookie, env.AUTH_SECRET);
  if (ok) {
    return next();
  }

  const loginUrl = new URL("/login", url);
  const nextPath = url.pathname + url.search;
  if (nextPath && nextPath !== "/") {
    loginUrl.searchParams.set("next", nextPath);
  }
  if (cookie) {
    loginUrl.searchParams.set("error", "expired");
  }
  return Response.redirect(loginUrl.toString(), 302);
};
