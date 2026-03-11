# Training

PyTorch Lightning-based training framework for deep learning workflows.

## Core Components
Data Loading Infrastructure: LayerDataset (PyTorch dataset wrapper around zetta_utils.layer.Layer components), JointDataset (supports horizontal simultaneous and vertical sequential dataset joining), build_collection_dataset (creates datasets from neuroglancer annotation collections)

Sample Indexers: SampleIndexer (base class: maps integer indices to layer-specific indices), VolumetricStridedIndexer (uniform sampling from 3D volumes with configurable stride), VolumetricNGLIndexer (samples from neuroglancer annotation points), RandomIndexer (randomizes sampling order with/without replacement), ChainIndexer (chains multiple indexers sequentially), LoopIndexer (loops over inner indexer to match desired sample count)

Lightning Training Framework: lightning_train (main training orchestrator supporting local/remote distributed training), ZettaDefaultTrainer (custom trainer with checkpointing, logging, and model tracing), remote training via Kubernetes with multi-node DDP support, elastic training via min_nodes/max_nodes (torch elastic launch), DCP checkpointing for distributed training (torch.distributed.checkpoint with async_save + process-based staging), AsyncCheckpointIO for single-node async (process-based torch.save, avoids GIL), SIGTERMCheckpointCallback (emergency checkpoint on preemption)

Data Samplers: TorchDataLoader (registered PyTorch DataLoader), TorchRandomSampler (registered PyTorch RandomSampler), SamplerWrapper (DDP-compatible sampler wrapper)

## Builder Registrations
Key Registered Components: LayerDataset (main dataset class), JointDataset (multi-dataset combiner), lightning_train (training orchestrator), ZettaDefaultTrainer (default trainer), VolumetricStridedIndexer (volume sampling), RandomIndexer (randomized sampling), TorchDataLoader (PyTorch DataLoader wrapper)

## Usage Patterns
Training Specification Structure:
```cue
"@type": "lightning_train"
regime: {"@type": "BaseRegimenConfig"}
trainer: {"@type": "ZettaDefaultTrainer"}
train_dataloader: {"@type": "TorchDataLoader", dataset: {"@type": "LayerDataset"}}
val_dataloader: {}
```

Integration Points: PyTorch Lightning for training loops, Weights & Biases for experiment tracking, Kubernetes for distributed training, Neuroglancer for annotation-based sampling, CloudVolume layers for data access

## Development Guidelines
Dataset Creation: Use LayerDataset for layer-based data access, combine datasets with JointDataset for multi-modal training, choose appropriate indexer based on sampling strategy, consider memory usage for large datasets

Training Configuration: Use ZettaDefaultTrainer for standard training workflows, configure appropriate batch sizes for available memory, use distributed training for large datasets, monitor training with W&B integration. For elastic training set min_nodes < num_nodes in lightning_train spec. Enable async_checkpointing=True in ZettaDefaultTrainer for non-blocking GCS checkpoint writes.

## Elastic Training on Spot Instances

Architecture: Master Job pod (standard node) + worker StatefulSet (spot nodes) + headless Service for DNS. Master hosts the c10d rendezvous store on port 29400. Workers resolve master via Kubernetes DNS: `master.run-<RUN_ID>.default.svc.cluster.local`. StatefulSet gives workers stable DNS names that survive rescheduling.

### Spec Fields (`lightning_train`)
- `num_nodes`: max worker count (master + N-1 StatefulSet replicas)
- `min_nodes`: minimum to start/continue training; set < `num_nodes` for elastic
- `max_restarts`: torch elastic agent restart budget per pod (default 1; set high for spot, e.g., 100)
- `provisioning_model`: `"spot"` for workers, master always runs on `"standard"`
- `rdzv_configs`: dict passed to torch elastic rendezvous, key knobs below

### Timeout Knobs

After a peer dies, `ddp_init_timeout_minutes` controls how long NCCL ops wait before timing out. After NCCL aborts (~60s internal cleanup), the elastic agent detects worker failure and triggers re-rendezvous. Total recovery time ≈ ddp_init_timeout + NCCL_cleanup(~60s) + last_call_timeout + overhead.

- `ddp_init_timeout_minutes` (ZettaDefaultTrainer param → DDPStrategy `timeout`): Controls **all** process group timeouts — NCCL, Gloo, and init_process_group barrier. Both backends get the same timeout via `_new_process_group_helper` (`distributed_c10d.py` line 1997/2018). This is the **primary knob** for failure detection speed. After a peer dies, NCCL all_reduce ops wait this long before the watchdog fires. Set too high → slow recovery. Set too low → false failures on transient network issues. Recommended: 1 min for spot instances.
  - **Note**: The `NCCL_TIMEOUT` env var does NOT control PyTorch's process group timeout. It's an NCCL library-level config unrelated to the DDPStrategy timeout. Do not use it for failure detection tuning.

- `rdzv_configs.read_timeout` (seconds): How long a pod waits for a rendezvous round to complete. Must be > ddp_init_timeout + last_call_timeout + 120 (margin). Default 600s. If too low, replacement pods timeout before surviving workers finish their timeout and join re-rendezvous. This timeout covers slow image downloads and pod startup.

- `rdzv_configs.last_call_timeout` (seconds): After min_nodes join a rendezvous round, wait this long for additional nodes before closing. The timer **resets** every time a new agent joins, so trickle-in is handled automatically. Calibration guide:
  - **Lower bound**: Must be > time for the slowest surviving agent to reach rendezvous after failure detection. In practice, surviving agents join within seconds once ddp_init_timeout fires, so even 30s is safe.
  - **Upper bound**: Should be < typical spot node provisioning time. No point waiting 5 min when autoscaler takes 5+ min — those nodes will join the NEXT round via re-rendezvous.
  - **Spot GKE rule of thumb**: 30-60s. Prioritize fast recovery with available nodes. Elastic training handles scale-up/down natively.
  - **On-demand/reserved nodes** (rare preemptions, stable cluster): 120-300s is fine since nodes are already provisioned.

- `checkpointing_kwargs.update_every_n_secs`: Periodic checkpoint interval. Lower = less work lost on preemption but more I/O overhead. 60s is a good starting point.
- `checkpointing_kwargs.backup_every_n_secs`: Backup checkpoint interval. Should be > update_every_n_secs.

Recovery timeline after preemption:
1. Pod dies → StatefulSet creates replacement (immediate)
2. Replacement joins rendezvous, waits for surviving workers
3. Surviving workers' NCCL ops wait ddp_init_timeout (60s), then NCCL watchdog fires
4. NCCL dumps debug info + aborts process (~60s internal cleanup)
5. Elastic agent detects worker failure, triggers re-rendezvous
6. Once min_nodes join + last_call_timeout elapses → training resumes from checkpoint

Total: ~ddp_init_timeout + NCCL_cleanup(~60s) + last_call_timeout + overhead ≈ 3.5-4 min with recommended values.

Rule of thumb: `read_timeout > ddp_init_timeout + last_call_timeout + 120`

### Pitfalls
1. Dead hostname after preemption: With host_network=True, pods reported GKE node hostnames via socket.gethostname(). After preemption, these pointed to permanently dead nodes. Fix: headless Service + StatefulSet for stable DNS.
2. DNS didn't resolve: K8s default search domains (default.svc.cluster.local) do NOT include the headless service subdomain. Using just "master" as rdzv_endpoint fails. Must use "master.run-<RUN_ID>" which resolves via the search domain.
3. Rendezvous timeout on first attempt: Replacement pod joins re-rendezvous immediately, but surviving workers don't join until NCCL timeout elapses. If read_timeout < NCCL_TIMEOUT + overhead, the replacement times out. Fix: default read_timeout=600s. Observed: read_timeout=300 caused first rendezvous attempt to fail (torch elastic agent retried and second attempt succeeded with fewer nodes). 300 is exactly NCCL_TIMEOUT(120) + last_call_timeout(60) + 120 — too tight. Use 600.
4. DNS label 63-char limit (RFC 1123): StatefulSet pod names = <statefulset-name>-<ordinal>. With run_id up to 50 chars, run-<50chars>-<ordinal> stays under 63. Don't add suffixes like -workers to the StatefulSet name.
5. Lightning `Trainer.ckpt_path` property: Returns `self._checkpoint_connector._ckpt_path`, NOT `self._ckpt_path`. Always access `trainer._ckpt_path` directly for the checkpoint path set in ZettaDefaultTrainer constructor.
6. SIGTERM checkpoint deadlock: Never save DCP checkpoints in a SIGTERM handler. DCP requires all ranks to participate in distributed ops. After preemption, dead peers can't respond, and Gloo has NO timeout for transport-level operations — the save hangs forever. Just exit cleanly on SIGTERM; periodic checkpoints provide recovery points.
7. NCCL_TIMEOUT env var is misleading: The `NCCL_TIMEOUT` env var does NOT control PyTorch's NCCL process group timeout. The NCCL process group timeout comes from `DDPStrategy(timeout=...)` which is set by `ddp_init_timeout_minutes`. Both NCCL and Gloo backends receive the same timeout value. To reduce NCCL failure detection time, reduce `ddp_init_timeout_minutes` (recommended: 1 min for spot).
8. Stale TCPStore keys across rendezvous rounds: By default (`TORCHELASTIC_USE_AGENT_STORE=True`), all rendezvous rounds share one TCPStore server. The rendezvous handler namespaces its own keys per round, but `init_process_group` uses a fixed prefix (`"default_pg"`). On re-rendezvous, `store.get("rank:N")` returns immediately with the stale address from the previous round (same pod IP, different Gloo port) → Gloo `connectFullMesh` fails with "Connection refused". Fix: set `TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1` so each round's rank-0 worker creates a fresh TCPStore.

### Recommended Env Vars
`NCCL_SOCKET_IFNAME: "eth0"` (pod network interface), `LOGLEVEL: "INFO"` (torch elastic agent logging), `TORCH_DISABLE_SHARE_RDZV_TCP_STORE: "1"` (create fresh TCPStore per rendezvous round — prevents stale Gloo address keys from previous rounds causing connectFullMesh failures, see Pitfall 8)

### Trainer Settings for Resilience
async_checkpointing=true (non-blocking checkpoint writes), checkpointing_kwargs.update_every_n_secs=60 (frequent checkpoints minimize work lost), max_epochs=-1 (let max_steps control training length since elastic may restart epochs), num_sanity_val_steps=0 (skip sanity validation which runs on every elastic restart), ddp_init_timeout_minutes=1 (fast failure detection for spot instances — controls both NCCL and Gloo timeouts)

### Resumable Dataloaders
Mid-epoch checkpoints (from `update_every_n_secs`) trigger a Lightning warning: "your dataloader is not resumable." On restart, the dataloader replays from epoch start — some batches get trained on twice. This is harmless when `max_steps` controls training length and epochs are short. If epochs become very long (30+ min), implement `state_dict()`/`load_state_dict()` on the dataloader to track batch position.

### Checkpointing Architecture
Dynamic based on world_size: distributed (world_size > 1) uses `torch.distributed.checkpoint` (DCP) with `dcp.async_save(async_checkpointer_type=PROCESS)` — all ranks participate in save, process-based staging avoids GIL. Single-node uses `AsyncCheckpointIO` with `torch.save` in a subprocess. DCP saves to local tmpdir, then a daemon thread waits for async save completion and immediately uploads to GCS (no deferred upload). DCP checkpoints are directories (not single files) containing `.metadata` + `__0_0.distcp`. Loading auto-detects format: DCP directory → `_EmptyStateDictLoadPlanner` + `FileSystemReader`, legacy file → `torch.load`. Backward compatible with existing `torch.save` checkpoints.

Indexing Strategy: Use VolumetricStridedIndexer for uniform volume sampling, use VolumetricNGLIndexer for annotation-based sampling, chain indexers for complex sampling patterns, use RandomIndexer for shuffled training

Performance Optimization: Configure appropriate number of DataLoader workers, use efficient data loading patterns, profile data loading bottlenecks, consider data caching for repeated access
