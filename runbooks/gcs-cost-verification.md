# GCS cost-tracking verification

## Why this exists

The mproxy sidecar attributes per-run GCS cost (Class A ops, Class B ops,
egress) by classifying intercepted `storage.googleapis.com` traffic. For
those numbers to be usable as customer billing, they have to match what
GCP actually charges. This procedure runs that comparison against a clean
GCP project and a known-good BigQuery billing export.

Run it:
- after non-trivial changes to `gcs_classification` or `mitm_addon`,
- before promising new tracked SKUs to a customer,
- on a periodic cadence (quarterly is reasonable).

## One-time setup

1. **Dedicated GCP project**. Clean slate, isolated billing. Nothing else
   should be writing to its buckets while a verification run is in flight.
2. **Enable BigQuery billing export**:
   - Billing → Billing export → BigQuery export → Detailed usage cost.
   - Wait ~24h for the first export to populate.
   - The dataset becomes `<billing_account>_billing` with table
     `gcp_billing_export_resource_v1_<id>`.
3. **Worker image** built from the branch under test (`docker/Dockerfile.all`).

## Workload matrix

Run each end-to-end through the project. Pick paths that exercise the
classifier shapes we care about; the goal is *coverage*, not realistic
production traffic.

| Workload    | Shape                                  | Why                                          |
|-------------|----------------------------------------|----------------------------------------------|
| CPU-bound   | Subchunkable apply on dense volumes    | XML reads/writes (the bulk of real traffic)  |
| GPU         | Inference flow with checkpoint reads   | JSON downloads + XML reads                   |
| Heavy-I/O   | Skeletonization shards                 | Listings, batch ops, large egress            |

Capture the run's start/end timestamps (UTC) for each.

## Diff procedure

For each workload:

1. **Sidecar totals** — query Firestore `pod-stats` for the run, summed
   across pods:

   ```python
   from zetta_utils.run.db import POD_STATS_DB
   docs = POD_STATS_DB.query(column_filter={"run_id": [run_id]})
   class_a = sum(b["class_a_count"] for d in docs.values() for b in d.get("buckets", {}).values())
   class_b = sum(b["class_b_count"] for d in docs.values() for b in d.get("buckets", {}).values())
   egress  = sum(b["egress_bytes"]  for d in docs.values() for b in d.get("buckets", {}).values())
   ```

2. **BigQuery actuals** — over the same time window, per SKU. Edit dataset
   and timestamps:

   ```sql
   SELECT
     sku.description AS sku,
     SUM(usage.amount) AS units,
     usage.unit AS unit,
     SUM(cost) AS usd
   FROM `<billing_account>_billing.gcp_billing_export_resource_v1_<id>`
   WHERE service.description = 'Cloud Storage'
     AND usage_start_time >= TIMESTAMP('<run_start_utc>')
     AND usage_end_time   <= TIMESTAMP('<run_end_utc>')
   GROUP BY sku, unit
   ORDER BY sku;
   ```

   Map BigQuery SKUs to sidecar categories:

   | BigQuery SKU description (substring)        | Sidecar field   |
   |---------------------------------------------|-----------------|
   | "Class A Operations"                        | `class_a_count` |
   | "Class B Operations"                        | `class_b_count` |
   | "Download Worldwide Destinations" (or similar egress SKU) | `egress_bytes` |

3. **Diff and report**. Per category:

   ```
   delta = (sidecar - bigquery) / bigquery
   ```

   Acceptance:
   - Initial: |delta| ≤ 5% on every category, every workload.
   - Stretch: |delta| ≤ 2%.

## When the diff is over budget

1. Look at the sidecar bucket breakdown. If `_unclassified` > 0, the
   classifier missed a request shape — `kubectl logs <pod> -c mproxy` will
   have a `WARNING: unclassified request: <method> <url>` line for every
   one. Add the shape to the fixture in
   `tests/unit/cloud_management/resource_allocation/k8s/test_gcs_classifier.py`
   (`_FIXTURE`), update `extract_bucket_from_api_path` /
   `classify_gcs_request` to handle it, re-run.
2. If `_batch` count is non-trivial relative to billed Class A/B totals,
   the deferred batch sub-op parsing (plan §3D step 2) likely needs to
   land. The batch payload is an HTTP `multipart/mixed` body; each part
   is itself a request that needs classification.
3. Egress drift between sidecar and BigQuery often comes from
   gzip-compressed responses where the addon falls back to decoded body
   length (`Content-Length` absent under chunked transfer). Cross-check
   `Content-Length` presence in `mproxy` logs.

Document every >5% discrepancy and its root cause/fix below in this file
under "History" so future verifications start with the prior state.

## History

_(populate as verifications happen)_

| Date       | Branch       | Workload | Class A Δ | Class B Δ | Egress Δ | Notes |
|------------|--------------|----------|-----------|-----------|----------|-------|
| _pending_  |              |          |           |           |          |       |
