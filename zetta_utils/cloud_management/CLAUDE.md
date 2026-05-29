# Cloud Management

Cloud resource provisioning and lifecycle management for distributed runs (GKE/EKS clusters, GCS, SQS).

## Submodules

- `resource_allocation/k8s/` — Kubernetes resource provisioning, autoscaler, node-pool nudger, pod/deployment helpers, watchers.
- `resource_allocation/aws_sqs.py` — SQS queue context managers used by mazepa.

## K8s Module Architecture

### Core Components

`watcher.py` — generic infrastructure for k8s watcher loops. Imported by the pod / event watchers in `pod.py`.
- `resilient_watch(list_fn_factory, on_event, ..., on_error=None)` — `watch.Watch` loop with exponential backoff. The factory is invoked once per stream attempt so a fresh client is picked up after `on_error()` invalidates a cached one (token expiry recovery). Transient errors route through a per-loop `BatchedWarner` so a stuck watcher emits one summary per flush interval rather than per retry; flushes immediately on stream recovery. Forwards `**list_fn_kwargs` (e.g. `field_selector`, `label_selector`) to the bound list method for server-side filtering.
- `BatchedWarner(name, log_path)` — batches content events, emits one summary warning per `FLUSH_INTERVAL_SEC` window (env-tunable via `ZETTA_BATCHED_WARN_FLUSH_INTERVAL_SEC`), deduplicates the sample by bracketed-prefix category so each event class shows up at least once. File log gets every event verbatim.
- `get_core_v1_api(cluster_info)` / `reset_core_v1_api()` — process-wide cached `CoreV1Api` built from `get_cluster_data(cluster_info)` (programmatic ADC auth, no kubeconfig needed). Reset wired into `resilient_watch.on_error` so token rejection clears the cache and the next factory call rebuilds against fresh credentials.

`pod.py` — pod-spec construction, lifecycle helpers, run-specific watcher entry points.
- `get_mazepa_pod_spec` / `get_pod_spec` — build a worker pod spec with mproxy + oom sidecars, CA-cert volume, node selectors, tolerations.
- `watch_for_pod_disruptions(run_id, cluster_info, ...)` — streams pods labeled `run_id=<run_id>`, classifies SIGKILL / SIGTERM / OOMKilled / Evicted, batches via `BatchedWarner`.
- `watch_for_run_events(run_id, cluster_info, ...)` — streams k8s events; surfaces curated reason set grouped by category: pod-disruption (Killing, Evicted, ...), pod-failure (BackOff = CrashLoopBackOff), cluster-autoscaler (NotTriggerScaleUp).
- `watch_for_triggered_scale_up(name_prefix, cluster_info, on_event, ...)` — server-side filtered watcher for cluster-autoscaler `TriggeredScaleUp` events on this run's pods. Drops events older than 120s so stream reconnects don't replay stale CA decisions. Parses `{MIG_NAME N->M (max: K)}` and yields `(pod_name, [(mig, target), ...])`.

`autoscaler.py` — master-side autoscaler + node-pool nudger.
- `autoscaling_deployment_ctx_mngr(...)` — public entry point. Calls `verify_cluster_access` first (see `preflight.py` below), then wraps `deployment_ctx_mngr` and runs three daemon threads: tick loop (`_run_loop`), TriggeredScaleUp watcher (`_run_triggered_scale_up_watcher`), and nudge loop (`_run_nudge_loop`).
- Tick loop polls SQS depth via `sqs_utils.get_queue_depth`, computes `desired = clamp(visible+in_flight, min, max)`, patches `Deployment.spec.replicas`. Scale-down honors a stabilization window. Replica/min/max/stabilization are read from Deployment annotations (`zetta.ai/autoscaler-*`) every tick so they can be adjusted at runtime via `kubectl annotate` or the `zetta run-update` CLI.
- Nudger pipeline bypasses CA's RESOURCE_POOL_EXHAUSTED backoff. The watcher records `(pool, target_per_zone)` from each TriggeredScaleUp event onto `_GroupNudgeState`. The nudge loop re-applies the target via `gke.resize_node_pool` every cycle while pending replicas remain (state persists until `pending == 0`), so we keep pestering GCP across CA's backoff window. Sizing math is fully deferred to CA — we only break the backoff barrier. **Known limitation**: when `--max-workers` is bumped *during* CA's pool backoff, CA suppresses fresh `TriggeredScaleUp` events for that pool, so `attempted_target_per_zone` stays at the pre-bump value until backoff expires. Nothing externally clears CA's backoff. The system catches up automatically once CA re-evaluates post-backoff.
- `_get_apps_v1_api(cluster_info)` / `_reset_apps_v1_api()` — programmatic-auth singleton, reset on tick failure.

`preflight.py` — `verify_cluster_access(cluster_info, namespace)` runs at master startup before any threads spawn. Tries `list_namespaced_pod` + `list_namespaced_deployment` (covers the master's auth + IAM) then reads the named Role / RoleBinding / ClusterRole / ClusterRoleBinding objects from `scripts/gcp/rbac.yml` (covers a missing `kubectl apply` of the rbac file that would otherwise surface as scattered 403s in worker pod logs). Errors carry the exact remediation command.

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

### Manual node-pool pestering

`zetta pester-nodes <run_id> [-g <worker-group>] -n <additional_per_zone>` discovers the pool(s) currently hosting the group's pods (by reading the `cloud.google.com/gke-nodepool` label off each scheduled pod's node) and re-issues `SetNodePoolSize(current + N)` on each every `--interval-sec` until pending replicas hit zero or the user sends SIGINT. Use this when CA's backoff window stretches longer than acceptable. Pass `--pool NAME` (repeatable) to bypass pod-based discovery when *zero* pods have scheduled yet (pure cold-start stockout). `-g` is optional; when omitted the deployment is resolved by `run_id` alone (must match exactly one).

### Authentication

Watcher and autoscaler threads use programmatic ADC auth via `get_cluster_data(cluster_info)` so they work without a local kubeconfig. The user-facing CLIs (`run-update`, `pester-nodes`) use `config.load_kube_config()` because the user invokes them from a workstation that already has the cluster context loaded.

Token refresh: caches reset on any tick / watcher failure (see `watcher.resilient_watch.on_error` and `autoscaler._reset_apps_v1_api`), so the next iteration rebuilds the api client against fresh credentials. No explicit TTL.

Preflight check (`preflight.verify_cluster_access`) runs at the top of `autoscaling_deployment_ctx_mngr` so auth / RBAC gaps fail the run at startup with an actionable error rather than as opaque 401-loops in daemon threads.

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
