# Cloudflare Pages Deployment

Production-bound, **fully static** deployment of the EA Food Price Prediction
dashboard. All model artifacts are exported to JSON at build time and served
as flat files from Cloudflare's edge — no backend, no servers, no Python in
the request path.

This directory is **independent of `src/dashboard/`** (which is the
research/prototype version with FastAPI). The two can evolve separately.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          BUILD TIME (one-shot)                            │
│                                                                            │
│   artifact/model_output_holdout/    artifact/model_output_stcv/           │
│   data/geoboundaries/                                                      │
│              │                              │                              │
│              ▼                              ▼                              │
│        ┌──────────────────────────────────────────┐                       │
│        │  scripts/export_data.py  (pandas + gpd)   │                       │
│        └──────────────────────────────────────────┘                       │
│                              │                                             │
│                              ▼                                             │
│             frontend/public/data/  (~3.8 MB, 31 JSON files)                │
│              ├─ geo/admin2.json                                            │
│              ├─ config.json                                                │
│              ├─ metrics/summary.json, fold-results.json                    │
│              ├─ metrics/per-admin/{target}_h{h}.json    (×9)               │
│              ├─ predictions/{target}_h{h}.json          (×9)               │
│              ├─ features/importance/{target}_h{h}.json  (×9)               │
│              └─ manifest.json                                              │
│                              │                                             │
│                              ▼                                             │
│        ┌──────────────────────────────────────────┐                       │
│        │  next build  (output: "export")           │                       │
│        └──────────────────────────────────────────┘                       │
│                              │                                             │
│                              ▼                                             │
│                  frontend/out/  (~6.2 MB)                                  │
│                  ├─ index.html, map/, timeseries/                          │
│                  ├─ _next/                                                 │
│                  └─ data/  (copied from public/data/)                      │
│                                                                            │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ git push (or `wrangler pages deploy`)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              REQUEST TIME                                  │
│                                                                            │
│                   ┌─────────────────┐                                     │
│   user ──HTTPS──▶ │ Cloudflare Pages│ ──cache──▶ user (edge, ~ms)          │
│                   │  (out/ as-is)   │                                      │
│                   └─────────────────┘                                      │
│                                                                            │
│   • static HTML/JS/CSS from `_next/`                                       │
│   • JSON from `/data/...` fetched lazily by client React                  │
│   • map tiles still come from openstreetmap.org                            │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why fully static?** Predictions are generated in batch (`train_holdout.py`,
`train_model_stcv.py`) and don't change between user requests. There is no
authentication, write path, or per-user computation. Anything stateless and
read-only deploys faster, cheaper, and more reliably as static files.

---

## Directory layout

```
deploy/cloudflare/
├── README.md                  # this file
├── build.sh                   # local end-to-end build (export + next build)
├── scripts/
│   └── export_data.py         # reads artifacts → writes frontend/public/data/
└── frontend/                  # Next.js 16 app, configured for static export
    ├── next.config.ts         # output: "export", basePath via env
    ├── package.json
    ├── public/
    │   └── data/              # generated; safe to commit (3.8 MB)
    ├── src/
    │   ├── app/               # /, /map, /timeseries
    │   ├── components/        # Sidebar, KpiCard
    │   ├── hooks/useFilters.ts
    │   ├── lib/api.ts         # fetches /data/*.json (no API server)
    │   ├── lib/constants.ts
    │   └── types/index.ts
    └── out/                   # generated by `next build`; what gets deployed
```

---

## Local build

```bash
# 1. From the repo root, with the project's venv activated:
source venv/bin/activate

# 2. Run the full pipeline:
./deploy/cloudflare/build.sh

# Or step-by-step:
python deploy/cloudflare/scripts/export_data.py
cd deploy/cloudflare/frontend
npm install     # first run only
npm run build
```

Verify locally before deploying:

```bash
cd deploy/cloudflare/frontend/out
python3 -m http.server 8765
# open http://localhost:8765/
```

---

## Deploy to Cloudflare Pages

### Option A — connect a Git repo (recommended)

1. Push this repo to GitHub.
2. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** →
   **Connect to Git**.
3. Settings:
   - **Root directory**: `deploy/cloudflare/frontend`
   - **Build command**: `cd ../.. && python deploy/cloudflare/scripts/export_data.py && cd deploy/cloudflare/frontend && npm install && npm run build`
   - **Build output directory**: `out`
   - **Environment variables**:
     - `PYTHON_VERSION` = `3.11.9`
     - `NODE_VERSION` = `20`
4. Cloudflare runs the build on every push to `main`; preview URLs are
   created for every other branch automatically.

> **Note on artifacts**: this build assumes `artifact/model_output_*/` and
> `data/geoboundaries/` are committed (or restored from LFS / a release
> asset). If they're not in the repo you have two options:
> 1. Commit `frontend/public/data/` and skip `export_data.py` in the build
>    command — the frontend works the same way.
> 2. Have CI download the artifacts (e.g. from S3 / a Release asset) before
>    running the export script.

### Option B — direct upload via Wrangler

```bash
npm install -g wrangler
./deploy/cloudflare/build.sh
cd deploy/cloudflare/frontend
wrangler pages deploy out --project-name ea-food-price-dashboard
```

### Free-tier limits (as of 2026)

| Limit                        | Cloudflare Pages free            | This project |
| ---------------------------- | -------------------------------- | ------------ |
| Bandwidth                    | unlimited                        | n/a          |
| Builds / month               | 500                              | n/a          |
| File size per asset          | 25 MB                            | max ~400 KB  |
| Total files per deploy       | 20,000                           | ~50          |
| Request rate                 | 100,000 /day on Workers (n/a here) | n/a        |

---

## Custom domain

Cloudflare dashboard → Pages project → **Custom domains** → **Set up a
domain**. Apex / subdomain both work; DNS is managed automatically if your
domain is on Cloudflare.

For sub-path hosting (e.g. `example.com/dashboard/`), set:

```
NEXT_PUBLIC_BASE_PATH=/dashboard
```

…on the Pages project's environment variables and rebuild.

---

## Path to a "real" production deployment (not a prototype)

The current static setup is good enough for permanent deployment as long as
predictions only change when you retrain. Things you'd add when this stops
being a prototype:

1. **CI separation.** Move `export_data.py` into a GitHub Actions workflow
   that runs on every push (or every model retrain) and pushes the JSON to a
   `published-data` branch / R2 bucket; Pages just consumes the artifact.
   Avoids committing 3.8 MB of generated JSON to `main`.

2. **R2 + signed URLs for raw artifacts.** If model outputs grow past
   ~50 MB, stop committing parquet/CSV to git. Store them in Cloudflare R2
   (S3-compatible, 10 GB free), and have `export_data.py` pull from there.

3. **Versioned data.** Stamp each export with model version + timestamp
   (`/data/v2026-04-28/...`) so the frontend can pin to a known-good build
   and you can roll back without rebuilding.

4. **A lightweight API layer (optional).** If you eventually need anything
   stateful (user auth, custom queries, on-demand inference), add a
   **Cloudflare Worker** at `/api/*`. Same domain, edge-deployed, free up to
   100k req/day. The static frontend stays unchanged.
   - For Python inference at the edge, Workers' Python runtime handles
     numpy-free code; for XGBoost-style inference, port the model to a
     Cloudflare Worker via ONNX or run a small **HuggingFace Spaces** /
     **Fly.io Machines** backend behind the same Pages domain.

5. **Observability.** Pages → **Analytics** is on by default (page views,
   country breakdown). For client errors, drop in **Cloudflare Web
   Analytics** (free, no cookie banner) or Sentry.

6. **Caching headers.** JSON under `/data/` is fingerprint-free, so add a
   `_headers` file at `frontend/public/_headers`:

   ```
   /data/*
     Cache-Control: public, max-age=300, s-maxage=86400
   ```

   The version-pinned alternative (point 3) lets you set `immutable, max-age=31536000` instead.

7. **Locked-down preview deployments.** Pages exposes preview URLs publicly.
   Add **Cloudflare Access** in front of `*.pages.dev` to require login, so
   only the production custom domain is world-readable.
