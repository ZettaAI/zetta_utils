# Distributed Checkpointing in ZettaDefaultTrainer

## What is ZettaDefaultTrainer?

`ZettaDefaultTrainer` extends PyTorch Lightning's `Trainer` for distributed elastic training on spot instances. It automatically selects between two checkpointing backends based on world size:

- **Single-node** (`world_size == 1`): `AsyncCheckpointIO` — process-based `torch.save`, avoids GIL contention with training
- **Multi-node** (`world_size > 1`): `torch.distributed.checkpoint` (DCP) — parallel saves with deduplication across ranks

The switch happens at trainer init based on `WORLD_SIZE` env var. All checkpoint logic lives in `default.py`.

## Why DCP for DDP?

The conventional DDP approach is `torch.save` on rank 0. This works but is wasteful:

- Only rank 0 writes — all other ranks sit idle
- Rank 0 must serialize the entire model — O(model_size) time on a single process
- With async saves, the GIL contention on rank 0 can slow training

DCP solves these problems:

**Parallel I/O.** All ranks write simultaneously, each writing a different subset of tensors. N ranks = N parallel writers.

**Deduplication.** In DDP, every rank holds an identical copy of the model. DCP detects this: during the planning phase, `_dedup_save_plans()` assigns each tensor to exactly one rank, load-balanced by tensor size. Total checkpoint size ≈ 1x model, not Nx model.

**Process-based async.** `dcp.async_save(async_checkpointer_type=PROCESS)` forks a subprocess per rank. The subprocess initializes its own Gloo process group and handles the collective save — the training process continues training immediately with zero GIL contention.

## How DCP Saves Work

Here's what happens when `ZettaDefaultTrainer.save_checkpoint()` is called with `world_size > 1`:

```
Training process                      DCP subprocess (per rank)
─────────────────                     ────────────────────────
1. dump_checkpoint()
   → nested dict: {state_dict,
      optimizer_states, epoch, ...}

2. _dcp_save(checkpoint, filepath)
   → dcp.async_save(checkpoint,
       checkpoint_id=local_dir,
       async_checkpointer_type=PROCESS)
                                      3. Init Gloo process group
   Returns Future immediately            (all ranks connect)
   │
   │                                  4. create_local_plan()
   │                                     Each rank creates WriteItems
   │                                     for all its tensors
   │
   │                                  5. create_global_plan()
   │                                     Coordinator gathers plans,
   │                                     calls _dedup_save_plans():
   │                                     For each tensor on multiple
   │                                     ranks → pick ONE rank
   │                                     (load-balanced by size)
   │
   │                                  6. Each rank writes ONLY its
   │                                     assigned tensors to
   │                                     __<rank>_0.distcp
   │
   │                                  7. Rank 0 writes .metadata
   │                                     (FQN → file + byte offset)
   │
3. Daemon thread waits for Future
4. Thread uploads to GCS:
   individual fs.put() per file
5. Thread cleans up local tmpdir
```

The resulting checkpoint is a directory, not a single file:

```
last.ckpt/
├── .metadata          # Global index: FQN → file + byte offset
├── __0_0.distcp       # Rank 0's assigned tensors
├── __1_0.distcp       # Rank 1's assigned tensors
├── ...
└── __15_0.distcp      # Rank 15's assigned tensors
```

With 16 DDP ranks and a 100MB model, each `.distcp` file is ~6MB (100MB / 16 ranks, load-balanced). Total checkpoint: ~100MB.

## How DCP Loads Work

```
1. AsyncCheckpointIO.load_checkpoint(path)
   │
   ├─ _is_dcp_checkpoint(path)?
   │  Checks if path/.metadata exists
   │
   ├─ YES → _load_dcp_checkpoint(path, map_location)
   │  │
   │  ├─ Download ALL shard files from GCS to local tmpdir
   │  │  (individual fs.get() per file — see Known Issue #3)
   │  │
   │  ├─ _load_state_dict(checkpoint,
   │  │      planner=_EmptyStateDictLoadPlanner(),
   │  │      no_dist=True)
   │  │  → Single-process load, reads .metadata,
   │  │    fetches tensors from whichever .distcp
   │  │    files contain them
   │  │
   │  ├─ _fix_dcp_optimizer_keys(checkpoint)
   │  │  → Convert string-int keys back to ints
   │  │    (see Known Issue #1)
   │  │
   │  └─ Return nested checkpoint dict
   │
   └─ NO → torch.load(path)  # Legacy single-file checkpoint
```

Key detail: `no_dist=True` means each rank loads independently — no collective communication needed. Every rank downloads all `.distcp` files and reconstructs the full checkpoint. This is what makes elastic world size changes possible.

## DCP Internals: flatten/unflatten

Understanding how DCP serializes nested Python structures is essential for understanding the optimizer key bug.

DCP's `flatten_state_dict` traverses the checkpoint dict recursively and produces a flat `{string_key: value}` mapping:

```python
# Input (nested):
{
    "optimizer_states": [           # list index 0
        {
            "state": {              # dict key "state"
                0: {                # dict key 0 (integer!)
                    "step": tensor,
                    "exp_avg": tensor,
                }
            },
            "param_groups": [...]
        }
    ]
}

# DCP traversal builds path tuples:
#   List index → (i,)     where i is an int
#   Dict key   → (str(k),) where str(k) is always a string
#
# For the "step" tensor above:
#   path = ("optimizer_states", 0, "state", "0", "step")
#                                            ↑
#                            int key 0 became string "0"
#
# Flattened key: "optimizer_states.0.state.0.step"
```

On load, `unflatten_state_dict` reconstructs using `set_element`, which decides the container type based on the path element type:
- `str` element → create a `dict`
- `int` element → create a `list`

But since dict keys were converted to strings during flatten, `set_element` sees `"0"` (a string) and creates a dict with string key `"0"` instead of int key `0`. This is the root cause of the optimizer key bug.

**Terminal detection matters too.** Lists of plain scalars — like `param_groups["params"] = [0, 1, 2, ...]` — are detected as "terminal" and stored as single pickle blobs (not traversed into individual elements). Lists containing tensors or dicts are traversed recursively. This is why `param_groups["params"]` survives the round-trip with correct int values, while `state` dict keys don't.

## Elastic DDP and Changing World Size

DDP replicates the full model on every rank — all ranks hold identical weights. DCP deduplicates on save (only one rank writes each tensor), but each rank loads the full state on resume. This means the number of ranks at save time has NO effect on what can be loaded.

**Saving with world_size=N:**
- All N ranks participate in `dcp.save()` — this is a collective operation, all must join
- `_dedup_save_plans()` assigns each tensor to exactly one of N ranks (load-balanced)
- N `.distcp` files created, each containing that rank's assigned tensors
- `.metadata` records which FQN lives in which file + byte offset
- Total checkpoint ≈ 1x model size regardless of N

**Loading with different world_size:**
- Our code uses `no_dist=True` + `_EmptyStateDictLoadPlanner()` — each rank loads independently, no collective needed
- Each rank downloads ALL `.distcp` files from GCS (every file, not just "its own")
- Reads `.metadata` to find which file contains each tensor → loads full checkpoint
- Works for ANY world_size: saved by 16 ranks, loaded by 4 — or vice versa

**Saving again after world_size change:**
- After load, all ranks hold identical full state (as always in DDP)
- New `dcp.save()` creates a fresh dedup plan for the current N ranks
- Completely new set of `.distcp` files replaces the old checkpoint
- Example: checkpoint saved by 16 ranks has 16 small `.distcp` files; after resume with 4 ranks, new save creates 4 larger `.distcp` files (same total size)

### Concrete scenario — spot preemption shrinks 16→4 ranks

1. 16 ranks running, checkpoint saved: 16 `.distcp` files (~1/16 model each)
2. 12 nodes preempted, rendezvous completes with 4 surviving ranks
3. Each of 4 ranks downloads all 16 `.distcp` files, reconstructs full state
4. Training continues with `world_size=4`
5. Next checkpoint: 4 `.distcp` files (~1/4 model each), still ≈ 1x total

### Concrete scenario — nodes restored, grows 4→16 ranks

1. 4 ranks running, checkpoint exists with 4 `.distcp` files
2. 12 new nodes join, rendezvous completes with 16 ranks
3. Each of 16 ranks downloads all 4 `.distcp` files, reconstructs full state
4. Training continues with `world_size=16`
5. Next checkpoint: 16 `.distcp` files, ≈ 1x total

### Contrast with FSDP

FSDP shards model state across ranks — each rank holds only a slice. Changing world size requires resharding (redistributing slices across the new set of ranks). DDP has no sharding boundary to adjust, making elastic scaling trivial with DCP.

## Known Issues and Fixes

### Issue 1: "duplicated flatten key optimizer_states.0.state.0.step"

**Error:**
```
ValueError: duplicated flatten key optimizer_states.0.state.0.step
```

**Root cause chain:**

1. **DCP load**: `flatten_state_dict` converts int dict keys to strings (`str(k)`). After `unflatten_state_dict`, optimizer `state` dict has string keys: `{"0": {...}, "1": {...}}`

2. **Optimizer.load_state_dict**: Builds `id_map` from `param_groups["params"]` which are ints (stored as terminal pickle blob, survives round-trip). Looks up `state["0"]` using int key `0` — doesn't find it (string `"0"` ≠ int `0`). Optimizer state silently not restored.

3. **Training continues**: Optimizer creates fresh state entries with int keys from actual parameter IDs. Now `state` has BOTH: `{"0": {old_from_checkpoint}, 0: {new_from_training}}`

4. **DCP save**: `flatten_state_dict` traverses both entries. `str("0")` and `str(0)` both produce `"0"`. Two paths lead to `optimizer_states.0.state.0.step` → `ValueError: duplicated flatten key`.

**Fix:** `_fix_dcp_optimizer_keys()` in `default.py` — converts string-integer keys back to ints after DCP load, before Lightning's `Optimizer.load_state_dict` processes them:

```python
def _fix_dcp_optimizer_keys(checkpoint):
    for opt_state in checkpoint.get("optimizer_states", []):
        if "state" in opt_state and isinstance(opt_state["state"], dict):
            opt_state["state"] = {
                int(k) if isinstance(k, str) and k.isdigit() else k: v
                for k, v in opt_state["state"].items()
            }
```

### Issue 2: CheckpointException inherits from BaseException

**Symptom:** Training runs indefinitely with zero successful checkpoints. No error in the main training loop. Only visible in stderr as `Exception in thread Thread-N (_wait_and_upload)`.

**Root cause:** `torch.distributed.checkpoint.api.CheckpointException` inherits from `BaseException`, not `Exception`:

```python
>>> CheckpointException.__mro__
(CheckpointException, BaseException, object)
```

The `_wait_and_upload` thread originally had `except Exception as e:` which never caught it. The exception escaped the thread, `error_holder` stayed empty, and `_dcp_wait_upload` never detected the failure.

**Production impact:** In one observed run, training proceeded for 71 steps (epoch 10→20) with all 10 checkpoint save attempts silently failing. All training progress would have been lost on restart.

**Fix:** Changed `except Exception` to `except BaseException` in `_wait_and_upload`.

### Issue 3: gcsfs directory nesting bug

**Problem:** `fs.get(remote_dir, local_dir, recursive=True)` creates an extra nested directory level. For example, `gs://bucket/last.ckpt/` downloaded to `/tmp/load/last.ckpt/` would create `/tmp/load/last.ckpt/last.ckpt/` with the files inside the inner directory.

**Fix:** Both upload (`_upload_checkpoint`) and download (`_load_dcp_checkpoint`) use individual file operations:

```python
# Upload: individual puts
for fname in os.listdir(local_dir):
    fs.put(os.path.join(local_dir, fname), f"{dest_path}/{fname}")

# Download: individual gets
for finfo in fs.ls(path, detail=False):
    fs.get(finfo, os.path.join(load_path, os.path.basename(finfo)))
```

### Issue 4: All ranks must participate in DCP save

DCP save is a collective operation. Even with deduplication (where only one rank writes each tensor), ALL ranks must participate in the planning phase — `create_local_plan()` and `create_global_plan()` involve all-gather communication.

If any rank skips the save (e.g., early exit, conditional save on rank 0 only), the other ranks will hang waiting for the collective. This is why `save_checkpoint()` calls `_dcp_save()` on all ranks before the `if not self.is_global_zero: return` guard (which only gates the spec JSON writing, not the checkpoint itself).

### Issue 5: SIGTERM emergency checkpoint

Spot preemption sends SIGTERM ~30 seconds before termination. `SIGTERMCheckpointCallback` handles this:

1. Installs a SIGTERM handler at `on_fit_start` that sets `_sigterm_received = True`
2. At the end of each training batch (`on_train_batch_end`), checks the flag
3. If set, saves an emergency checkpoint and raises `SystemExit`

This gives the training loop time to finish the current batch and save state before the node is killed. The handler is restored to the original on `on_fit_end`.

## Timeout Configuration

When a node dies in elastic training, recovery involves several timeouts in sequence:

```
Pod dies
  │
  ├─ StatefulSet creates replacement (immediate)
  │  Replacement joins rendezvous, waits for peers
  │
  ├─ Surviving workers don't know yet — still in training loop
  │  They hit NCCL_TIMEOUT (~120s) when the next collective op
  │  tries to communicate with the dead rank
  │
  ├─ Torch elastic agent detects failure, triggers re-rendezvous
  │  Surviving workers join rendezvous
  │
  ├─ Once min_nodes have joined, last_call_timeout (~60s) starts
  │  Waits for additional nodes before closing
  │
  └─ Training resumes with new world_size
```

Total recovery time: **~NCCL_TIMEOUT + last_call_timeout + overhead**

| Knob | Where | Default | What it controls |
|------|-------|---------|-----------------|
| `NCCL_TIMEOUT` | env var (seconds) | 120 | How long NCCL ops wait before declaring a peer dead |
| `rdzv_configs.read_timeout` | spec (seconds) | 600 | How long a pod waits for a rendezvous round to complete |
| `rdzv_configs.last_call_timeout` | spec (seconds) | 60 | After min_nodes join, wait this long for more before closing |
| `ddp_init_timeout_minutes` | trainer (minutes) | 2 | DDP process group initialization timeout |
| `checkpointing_kwargs.update_every_n_secs` | trainer (seconds) | 60 | Periodic checkpoint interval (less = less work lost) |
| `checkpointing_kwargs.backup_every_n_secs` | trainer (seconds) | 900 | Backup checkpoint interval |

**Rule of thumb:** `read_timeout > NCCL_TIMEOUT + last_call_timeout + 120`

With defaults: `600 > 120 + 60 + 120 = 300`. The margin handles variance in detection and scheduling.

**Pitfall:** `read_timeout=300` (old default) is exactly at the boundary and caused first rendezvous attempts to fail in production. The replacement pod would timeout before surviving workers joined. Torch elastic agent retried and the second attempt succeeded, but with fewer nodes. Fixed by setting default to 600.
