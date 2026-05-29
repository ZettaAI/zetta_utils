# Web API Module

FastAPI-based web service for zetta_utils task management and annotations.

## Architecture
- FastAPI with OAuth2 authentication (@zetta.ai email restriction)
- CORS enabled for cross-origin requests
- Modular API design with separate routers for different resources

## API Endpoints

### Tasks API (`/tasks`)
- `GET /projects/{project_name}/task_types/{task_type_id}` - Get task type details
- `GET /projects/{project_name}/tasks/{task_id}` - Get task details
- `POST /projects/{project_name}/start_task` - Start task for user
- `POST /projects/{project_name}/set_task_ng_state` - Update task neuroglancer state
- `PUT /projects/{project_name}/release_task` - Release task with completion status
- `POST /projects/{project_name}/submit_timesheet` - Submit timesheet entry
- `GET /projects/{project_name}/task_feedback` - Get task feedback for user
- `GET /projects/{project_name}/segments/{seed_id}/link` - Generate segment neuroglancer link
- `GET /projects/{project_name}/tasks/{task_id}/trace_state` - Get trace task neuroglancer state with merges
- `GET /projects/{project_name}/tasks/{task_id}/trace_link` - Get trace task spelunker link with merges

#### Segment Link Parameters
- `include_certain_ends` (bool) - Include certain endpoints layer (yellow)
- `include_uncertain_ends` (bool) - Include uncertain endpoints layer (red)
- `include_breadcrumbs` (bool) - Include breadcrumbs layer (blue)

#### Trace Task Link Parameters
- `include_certain_ends` (bool) - Include certain endpoints layer (yellow)
- `include_uncertain_ends` (bool) - Include uncertain endpoints layer (red)
- `include_breadcrumbs` (bool) - Include breadcrumbs layer (blue)
- `include_segment_type_layers` (bool) - Include segment type layers
- `include_merges` (bool) - Include merge edits as line annotations (orange)

### Other APIs
- `/annotations` - Annotation management
- `/collections` - Collection operations
- `/layer_groups` - Layer group management
- `/layers` - Layer operations
- `/painting` - Painting functionality
- `/precomputed` - Precomputed annotations

## Authentication
OAuth2 token verification with @zetta.ai email domain restriction.

## Deployment

Built and pushed via `build_web_api.py` at the repo root. Single script handles both CPU and GPU variants.

### Dependencies

web_api installs only the slim `web_api` / `web_api-gpu` extras (defined in the
root `pyproject.toml`), not the heavy `modules` extra. Both build on a shared
`web-api-base` extra that pulls `tensor_ops`, `cloudvol`, `tensorstore`, `convnet`,
`task_management`, `cutie`, the web stack (fastapi/pydantic/hypercorn/
google-cloud-iap/python-multipart), plus `scipy`, `hydra-core`, `omegaconf`.
`web_api` = base + CPU torch; `web_api-gpu` = base + CUDA torch. `web_api-gpu`
deliberately does **not** pull the `gpu`/tensorrt extra: web_api only calls
`convnet.load_model` with `tensorrt_enabled=False`, so TensorRT is never imported,
and adding it would layer a multi-GB CUDA-13 runtime onto the CUDA-12.1 base for no
gain. When web_api gains an import that needs a new third-party package or a new
`zetta_utils` subpackage, add the covering extra to `web-api-base` and regenerate
the pinned files with `./update_pinned_requirements.sh`.

**torch variant.** `web_api` and `web_api-gpu` are declared conflicting in
`[tool.uv]`, and `[tool.uv.sources]` pins torch to the `pytorch-cpu` index for the
`web_api` extra. So `requirements.web_api.txt` resolves `torch==…+cpu` with **no**
`nvidia-*` CUDA wheels (multi-GB lighter), while `web_api-gpu`/`modules`/`all` keep
the default CUDA torch. This is uv-only: plain `pip install '.[web_api-gpu]'`
ignores it (which is why the GPU image keeps its base cu121 torch).

- **CPU** (`Dockerfile`): installs `requirements.web_api.txt` `--no-deps` with
  `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu` (so the `+cpu` torch
  wheel resolves). Keeps the faust-cchardet/cchardet-stub shim because cutie needs
  `cchardet`, which is pruned and does not build on Py3.12+.
- **GPU** (`gpu.Dockerfile`): `pip install '.[web-api-gpu]'` on the
  `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` base with `PIP_EXTRA_INDEX_URL`
  pinned to cu121 — resolution leaves the base image's torch 2.5.1+cu121 untouched.
- The `web-api-extras-build` CI job installs the CPU extra clean (with the CPU torch
  index), smoke-imports `app.main`, and asserts the heavy extras
  (lightning/wandb/tensorrt/...) stay out. `web-api-gpu-build` does a full GPU
  `docker build`.

### Tag format
`<semver>-<yyyymmddNN>` (e.g., `1.0.46-2026052001`). Semver tracks logic changes; `yyyymmddNN` is a per-day build counter for infra-only rebuilds that don't bump semver.

### Sources of truth
- **`web_api/VERSION`** — current semver (committed to repo). Updated only after a successful push.
- **Git tags** — `webapi_v<semver>-<yyyymmddNN>` (single tag covers both CPU and GPU). Used to compute the next build number for a given day. CPU and GPU always share the same version.

### Modes (mutually exclusive)
- `--bump {major,minor,patch}` — logic change. Computes new semver, builds, pushes, writes VERSION, auto-commits VERSION (unless `--no-commit`), creates git tag.
- `--rebuild` — infra-only change. Reuses current VERSION, increments today's build number, no VERSION write, no commit, still creates git tag.
- `--tag X` — explicit tag override (bypasses VERSION logic and skips git tagging). For ad-hoc local builds.

### Side-effect flags
- `--no-build` — skip docker build (push-only).
- `--no-push` — skip docker push AND git tag (build-only). VERSION is not written in this mode, so `--bump --no-push` lets you preview a bump without side effects.
- `--no-commit` — with `--bump`: write VERSION but don't auto-commit. User commits manually.
- `--no-latest` — skip tagging and pushing the `:latest` image. By default, `--bump` and `--rebuild` retag the just-pushed image as `:latest` and push it (so `docker-compose` and other consumers of `:latest` pick it up automatically). `--tag X` mode never touches `:latest`.

### Defaults
`--variant both`, `--project zetta-research`, `--region us-east1`, `--repo zutils`.

### Maintenance notes
- VERSION is written **after** successful push, not at the start, to avoid stranding a half-bumped file on build/push failure.
- `git fetch --tags` is run before computing the next build number so concurrent builds on different machines don't collide.
- When the deploy workflow changes (new flags, tag format, defaults), update both `web_api/README.md` (user-facing) and this file.

## Usage
```bash
# Get segment link with all endpoints
GET /tasks/projects/kronauer_ant_x0/segments/74732294451380972/link

# Get segment link without breadcrumbs
GET /tasks/projects/kronauer_ant_x0/segments/74732294451380972/link?include_breadcrumbs=false

# Get trace task neuroglancer state with all features including merges
GET /tasks/projects/kronauer_ant_x0/tasks/task_123/trace_state

# Get trace task spelunker link without merge annotations
GET /tasks/projects/kronauer_ant_x0/tasks/task_123/trace_link?include_merges=false

# Get trace task link with only certain endpoints and merges
GET /tasks/projects/kronauer_ant_x0/tasks/task_123/trace_link?include_uncertain_ends=false&include_breadcrumbs=false
```
