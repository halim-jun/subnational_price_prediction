# Cloudflare Prototype Login Barrier — Design Spec

**Date:** 2026-04-28
**Status:** Approved
**Scope:** Prototype-only Cloudflare Pages deployment with shared-password login gate

## Goal

Deploy the East Africa food price prediction dashboard to Cloudflare Pages as a prototype, gated behind a single shared password so random visitors and search engines cannot access it. This is **not production auth**.

## Architecture

Static Next.js export hosted on Cloudflare Pages, with Cloudflare Pages Functions serving as an authentication edge layer.

```
Cloudflare Pages
├── Static assets (Next.js `out/`)
│   ├── login/index.html
│   ├── index.html, map/, timeseries/   ← dashboard pages
│   └── data/                            ← pre-exported JSON (replaces FastAPI)
└── functions/
    ├── _middleware.ts                   ← gates ALL requests
    └── api/auth/
        ├── login.ts                     ← validate password, set cookie
        └── logout.ts                    ← clear cookie
```

**Why this shape:** A pure static export would leave `/data/*.json` publicly fetchable. Pages Functions intercept all requests at the edge so JSON payloads are also gated.

## Authentication

- Single shared password stored in `SITE_PASSWORD` env var (Cloudflare Pages dashboard).
- HMAC-SHA-256 signed cookie using `AUTH_SECRET` env var (32+ char random string).
- Cookie format: `auth=<base64(expires_unix_ts)>.<hex(hmac_sha256(expires, AUTH_SECRET))>`
- Cookie attributes: `HttpOnly; Secure; SameSite=Lax; Path=/`
- Lifetime: 7 days.
- Stateless — no DB, no session store.

### Flow

1. Request hits `_middleware.ts` → no/invalid cookie → 302 to `/login`
2. User submits password → `POST /api/auth/login` → password compared (constant-time) → cookie set
3. Subsequent requests pass middleware → static assets served

### Allowlisted paths (middleware passes through)

- `/login` — login page itself
- `/api/auth/login`, `/api/auth/logout`
- `/_next/*`, favicon, other static framework assets

### Brute-force defenses

- Failed login: artificial 1.5s delay
- 5 failures from same IP within 60s → 60s lockout (in-memory per Worker isolate; acceptable for prototype scale)

## Static Data Export

`scripts/export_static_data.py` calls FastAPI endpoints via `fastapi.testclient.TestClient` and writes JSON files to `src/dashboard/frontend/public/data/`.

### Output structure

```
public/data/
├── config.json
├── geo/admin2.json
├── predictions/
│   ├── map_<target>_<horizon>.json        (3×3 = 9)
│   ├── timeseries_<target>_<horizon>.json (3×3 = 9, all regions in one file)
│   └── regions_<target>_<horizon>.json    (3×3 = 9)
├── metrics/
│   ├── summary.json
│   └── per-admin_<target>_<horizon>.json  (3×3 = 9)
└── features/
    └── importance_<target>_<horizon>.json (3×3 = 9)
```

Targets: Food Price Index, Maize, Sorghum. Horizons: 1, 2, 3 months.
~40 JSON files total. Largest is the GeoJSON.

### Frontend dual mode

`src/lib/api.ts` branches on `process.env.NEXT_PUBLIC_STATIC_MODE === "true"`:

- **STATIC mode:** fetch `/data/<path>.json` (used in Cloudflare build)
- **API mode:** fetch `/api/<path>` (used in local dev with FastAPI)

For `getTimeSeries`, static mode loads the full `timeseries_<target>_<horizon>.json` and filters by region in-memory.

## File Changes

### Added

```
src/dashboard/frontend/
├── functions/
│   ├── _middleware.ts
│   └── api/auth/{login,logout}.ts
├── src/app/login/page.tsx
├── src/components/LogoutButton.tsx
├── src/lib/auth.ts                  ← Web Crypto HMAC helpers (used by Functions)
├── .env.local.example
scripts/
└── export_static_data.py
docs/
└── cloudflare-prototype.md
```

### Modified

- `src/dashboard/frontend/src/lib/api.ts` — STATIC_MODE branching
- `src/dashboard/frontend/src/components/Sidebar.tsx` — logout button
- `src/dashboard/frontend/next.config.ts` — `output: 'export'`
- `src/dashboard/frontend/.gitignore` — exclude `public/data/`, `out/`
- `CLAUDE.md` — add Cloudflare build commands

## Build Pipeline

```bash
# 1. Export data from FastAPI to JSON
python scripts/export_static_data.py

# 2. Static build with API client in STATIC mode
cd src/dashboard/frontend
NEXT_PUBLIC_STATIC_MODE=true npm run build
# → out/ directory generated

# 3. Deploy to Cloudflare Pages
# Set SITE_PASSWORD and AUTH_SECRET as env vars in Pages project settings
# Build output directory: src/dashboard/frontend/out
# Functions directory: src/dashboard/frontend/functions
```

## Non-goals (explicitly out of scope)

- Per-user accounts, signup, email verification
- Password reset / forgot-password flow
- Role-based access control
- Audit logging
- Rate-limit persistence across Worker isolates
- Production-grade auth (recommend Cloudflare Access or Auth0 when promoting beyond prototype)

## Future migration path

When promoting beyond prototype:
1. Restore FastAPI backend hosting (Railway, Fly.io, Render, etc.)
2. Set `NEXT_PUBLIC_STATIC_MODE=false` (or remove the toggle entirely)
3. Replace shared-password barrier with Cloudflare Access, Auth0, or similar IdP
4. Remove `functions/api/auth/*` and `_middleware.ts` (replaced by IdP middleware)
5. Delete `public/data/` (no longer needed)
