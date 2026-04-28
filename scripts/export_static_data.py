"""Export FastAPI dashboard data to static JSON files for Cloudflare Pages prototype.

Run this before `npm run build` when building the Cloudflare prototype.
Reads the same data the FastAPI backend serves and writes it to
`src/dashboard/frontend/public/data/` so the frontend can fetch it directly
without needing a running backend.

Usage:
    python scripts/export_static_data.py
"""

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.dashboard.api.config import TARGETS, HORIZONS  # noqa: E402
from src.dashboard.api.main import app  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("export_static_data")

OUT_ROOT = PROJECT_ROOT / "src/dashboard/frontend/public/data"


def write_json(rel_path: str, payload) -> None:
    out = OUT_ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = os.path.getsize(out) / 1024
    log.info("wrote %s (%.1f KB)", rel_path, size_kb)


def fetch_or_die(client: TestClient, url: str):
    r = client.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text[:200]}")
    return r.json()


def main():
    if OUT_ROOT.exists():
        log.info("output dir exists, files will be overwritten: %s", OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        log.info("=== fixed endpoints ===")
        write_json("config.json", fetch_or_die(client, "/api/config"))
        write_json("geo/admin2.json", fetch_or_die(client, "/api/geo/admin2"))
        write_json("metrics/summary.json", fetch_or_die(client, "/api/metrics/summary"))

        log.info("=== per (target, horizon) endpoints ===")
        for target in TARGETS:
            for horizon in HORIZONS:
                tag = f"{target}_{horizon}"
                log.info("--- %s ---", tag)

                # predictions: map (all months for slider)
                payload = fetch_or_die(
                    client,
                    f"/api/predictions/map/all?target={target}&horizon={horizon}",
                )
                write_json(f"predictions/map_{tag}.json", payload)

                # predictions: regions list
                regions_payload = fetch_or_die(
                    client,
                    f"/api/predictions/regions?target={target}&horizon={horizon}",
                )
                write_json(f"predictions/regions_{tag}.json", regions_payload)

                # predictions: timeseries — pre-fetch every region into one file
                # so the static frontend can filter in-memory.
                regions = regions_payload.get("regions", [])
                bundled = {}
                for region in regions:
                    from urllib.parse import quote
                    r = fetch_or_die(
                        client,
                        f"/api/predictions/timeseries?target={target}"
                        f"&horizon={horizon}&admin2_name={quote(region)}",
                    )
                    bundled[region] = r.get("data", [])
                write_json(
                    f"predictions/timeseries_{tag}.json",
                    {"target": target, "horizon": horizon, "regions": bundled},
                )

                # metrics: per-admin
                payload = fetch_or_die(
                    client,
                    f"/api/metrics/per-admin?target={target}&horizon={horizon}",
                )
                write_json(f"metrics/per-admin_{tag}.json", payload)

                # features: importance
                payload = fetch_or_die(
                    client,
                    f"/api/features/importance?target={target}&horizon={horizon}",
                )
                write_json(f"features/importance_{tag}.json", payload)

    log.info("done. output: %s", OUT_ROOT)


if __name__ == "__main__":
    main()
