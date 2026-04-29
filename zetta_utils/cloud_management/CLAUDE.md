# Cloud Management

Cloud resource provisioning and lifecycle management for distributed runs (GKE/EKS clusters, GCS, SQS).

## Submodules

- `resource_allocation/k8s/` — Kubernetes resource provisioning, autoscaler, node-pool nudger, pod/deployment helpers, watchers.
- `resource_allocation/aws_sqs.py` — SQS queue context managers used by mazepa.

## K8s Module Architecture

### Core Components

`pod.py` — pod-spec construction, lifecycle helpers, k8s event watchers.
- `get_mazepa_pod_spec` / `get_pod_spec` — build a worker pod spec with mproxy + oom sidecars, CA-cert volume, node selectors, tolerations.
- `_resilient_watch(list_fn, on_event, ...)` — generic resilient `watch.Watch` loop with exponential backoff on transient errors. Forwards arbitrary kwargs (`field_selector`, `label_selector`) to `list_fn` for server-side filtering.
- `_BatchedWarner(name, interval_sec, log_path)` — collects messages, emits one summary warning per `interval_sec` window, deduplicates the sample by bracketed-prefix category so each event class shows up at least once. File log gets every event verbatim. Used by both pod-disruption and run-event watchers.
- `watch_for_pod_disruptions(run_id, cluster_info, ...)` — streams pods labeled `run_id=<run_id>`, classifies SIGKILL / SIGTERM / OOMKilled / Evicted, batches via `_BatchedWarner`.
- `watch_for_run_events(run_id, cluster_info, ...)` — streams k8s events; surfaces curated reason set grouped by category: pod-disruption (Killing, Evicted, ...), pod-failure (BackOff = CrashLoopBackOff), cluster-autoscaler (NotTriggerScaleUp).
- `watch_for_triggered_scale_up(name_prefix, cluster_info, on_event, ...)` — server-side filtered watcher for cluster-autoscaler `TriggeredScaleUp` events on this run's pods. Drops events older than 120s so stream reconnects don't replay stale CA decisions. Parses `{MIG_NAME N->M (max: K)}` and yields `(pod_name, [(mig, target), ...])`.
- `_get_core_v1_api(cluster_info)` / `_reset_core_v1_api()` — process-wide cached CoreV1Api built from `get_cluster_data(cluster_info)` (programmatic ADC auth, no kubeconfig needed). Reset on watcher failure to recover from token expiry.

`autoscaler.py` — master-side autoscaler + node-pool nudger.
- `autoscaling_deployment_ctx_mngr(...)` — public entry point. Wraps `deployment_ctx_mngr` and runs three daemon threads: tick loop (`_run_loop`), TriggeredScaleUp watcher (`_run_triggered_scale_up_watcher`), and nudge loop (`_run_nudge_loop`).
- Tick loop polls SQS depth via `sqs_utils.get_queue_depth`, computes `desired = clamp(visible+in_flight, min, max)`, patches `Deployment.spec.replicas`. Scale-down honors a stabilization window. Replica/min/max/stabilization are read from Deployment annotations (`zetta.ai/autoscaler-*`) every tick so they can be adjusted at runtime via `kubectl annotate` or the `zetta run-update` CLI.
- Nudger pipeline bypasses CA's RESOURCE_POOL_EXHAUSTED backoff. The watcher records `(pool, target_per_zone)` from each TriggeredScaleUp event onto `_GroupNudgeState`. The nudge loop applies the target via `gke.resize_node_pool` (never shrinks; honors per-pool cool-off). Sizing math is fully deferred to CA — we only break the backoff barrier.
- `_get_apps_v1_api(cluster_info)` / `_reset_apps_v1_api()` — programmatic-auth singleton, reset on tick failure.

`gke.py` — wrappers around `ClusterManagerClient` (cluster info, node-pool listing, resize).
- `_get_cluster_manager()` — process-wide cached `ClusterManagerClient` so callers on hot paths share one gRPC channel + ADC.
- `list_node_pools` / `resize_node_pool` / `gke_cluster_data`.

`deployment.py` — deployment construction. Mazepa worker deployments label pods `{run_id: <execution_id>, worker_group: <group>}` so server-side `label_selector` filters work for the watchers.

`sidecar.py` — mproxy (mitmproxy GCS interception) and runtime (OOM tracker) sidecar containers; `wait_for_ca` script gating the main container on CA-cert availability.

`common.py` — `ClusterInfo` (frozen attrs class, hashable), `get_cluster_data` (returns a `k8s_client.Configuration` from cluster info via ADC), `parse_cluster_info`.

### Scaling Annotations

Read every tick by `_GroupScaler._read_scaling_config`. Adjustable at runtime:

- `zetta.ai/autoscaler-max-replicas`
- `zetta.ai/autoscaler-min-replicas`
- `zetta.ai/autoscaler-scale-down-stabilization-sec`

The `zetta run-update <run_id> -g <group> --max-workers N` CLI patches these directly.

### Authentication

Watcher and autoscaler threads use programmatic ADC auth via `get_cluster_data(cluster_info)` so they work without a local kubeconfig. The CLI (`run-update`) uses `config.load_kube_config()` because the user invokes it from a workstation that already has the cluster context loaded.

Token refresh: caches reset on any tick / watcher failure, so the next iteration rebuilds the api client against fresh credentials. No explicit TTL.

### Server-Side Filtering

All watchers push filtering to the apiserver:
- Pod disruption watcher: `label_selector="run_id=<run_id>"` so other runs' pods don't flood this master.
- Run-event watcher: `field_selector="involvedObject.kind=Pod"` to drop service/configmap event chatter.
- TriggeredScaleUp watcher: `field_selector="reason=TriggeredScaleUp,involvedObject.kind=Pod"`.

### Env-Configurable Intervals

- `ZETTA_BATCHED_WARN_FLUSH_INTERVAL_SEC` — default 90s, controls the `_BatchedWarner` summary cadence.
- `ZETTA_UPDATE_COSTS_INTERVAL_SEC` / `ZETTA_POD_STATS_INTERVAL_SEC` — read by `zetta_utils.run` for the cost / pod-stats repeaters.

## Development Guidelines

- Auxiliary daemon threads (autoscaler ticks, watchers, nudger) MUST NOT crash the run. Wrap loop bodies in broad except, log-and-continue. Resource lifecycle (deployment / SQS teardown) is the main flow's responsibility, never an aux thread's.
- Hold api clients as module-level singletons; reset on failure. Do not pass clients across thread boundaries unless they are documented thread-safe.
- New watcher event reasons go into the right category constant in `pod.py` (`_POD_DISRUPTION_EVENT_REASONS`, `_POD_FAILURE_EVENT_REASONS`, `_CLUSTER_AUTOSCALER_EVENT_REASONS`). The union becomes the run-event watcher's filter.
- For new k8s event watchers, prefer server-side `field_selector` / `label_selector` over client-side filtering.
