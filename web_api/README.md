# web_api — Build & Deploy

This directory contains the FastAPI web service. CPU and GPU container images are built and pushed with a single script at the repo root: `build_web_api.py`.

## Tag format

Container images and git tags use:

```
<semver>-<yyyymmddNN>
```

- `<semver>` — the version in `web_api/VERSION` (e.g. `1.0.46`). Bumped only when **application logic** changes.
- `<yyyymmddNN>` — build counter for the day (e.g. `2026052001` = first build on 2026-05-20). Used so you can publish multiple builds of the same semver on the same day for infrastructure-only changes (Dockerfile, deps, base image, etc).

Example full tag: `1.0.46-2026052001`.

CPU and GPU images always share the same tag. Git tag format: `webapi_v<semver>-<yyyymmddNN>` (one tag for both variants).

In addition to the versioned tag, `--bump` and `--rebuild` retag the pushed image as `:latest` and push that too. `docker-compose.yml` in this directory uses `:latest`, so running `docker compose up` always picks up the most recently published build. Disable with `--no-latest` (e.g., when publishing an old hotfix that shouldn't replace `:latest`).

## Quick start

All commands run from the repo root.

### Ship a logic change (most common)
Bumps semver, builds both images, pushes them, auto-commits the bumped `VERSION` file, creates and pushes a git tag.

```bash
python3 build_web_api.py --bump patch     # 1.0.45 -> 1.0.46
python3 build_web_api.py --bump minor     # 1.0.45 -> 1.1.0
python3 build_web_api.py --bump major     # 1.0.45 -> 2.0.0
```

### Ship an infra-only change
Same logic version, new build number. No semver change, no commit, but still a fresh image and git tag.

```bash
python3 build_web_api.py --rebuild
# Tag: <current-VERSION>-<today><NN+1>
```

### Test locally without publishing
Build images locally, don't push or tag.

```bash
python3 build_web_api.py --bump patch --no-push      # preview a bump (VERSION untouched)
python3 build_web_api.py --rebuild --no-push         # plain local build
```

### Publish a pre-built image
Already built, just push.

```bash
python3 build_web_api.py --rebuild --no-build
```

### Only one variant
```bash
python3 build_web_api.py --variant cpu --rebuild
python3 build_web_api.py --variant gpu --bump patch
```

### Ad-hoc / custom tag
Bypasses `VERSION` logic entirely. No git tagging. For dev experiments.

```bash
python3 build_web_api.py --tag dev-vlad
python3 build_web_api.py --tag mytest --no-push
```

## Flag reference

One of `--bump`, `--rebuild`, or `--tag` is **required**.

| Flag | Description |
| :--- | :--- |
| `--bump {major,minor,patch}` | Logic change. Computes new semver. Writes VERSION + commits after successful push. |
| `--rebuild` | Infra-only change. Keeps current semver, increments today's build number. |
| `--tag X` | Explicit tag override. Skips VERSION logic and git tagging. |
| `--variant {cpu,gpu,both}` | Default: `both`. |
| `--project` | Default: `zetta-research`. |
| `--region` | Default: `us-east1`. |
| `--repo` | Default: `zutils`. |
| `--no-build` | Skip `docker build`. |
| `--no-push` | Skip `docker push` AND git tagging. |
| `--no-commit` | With `--bump`: write VERSION but don't auto-commit. |
| `--no-latest` | Skip retagging and pushing the `:latest` image. |

## How it works

1. Resolves the target tag:
   - `--bump`: reads `VERSION`, computes new semver in memory.
   - `--rebuild`: reads `VERSION` as-is.
   - `--tag X`: uses `X` directly.
2. For `--bump`/`--rebuild`: runs `git fetch --tags` and scans existing `webapi_v<semver>-<yyyymmdd>NN` tags to pick the next `NN`.
3. Builds the selected variants (`web_api/Dockerfile` for CPU, `web_api/gpu.Dockerfile` for GPU).
4. If pushing:
   - Pushes all variants.
   - For `--bump`: writes `VERSION` to disk, then commits it (unless `--no-commit`).
   - Creates `webapi_v<full-tag>` and pushes it to `origin` (unless `--tag` was used).

`VERSION` is only written **after** a successful push. A failed build or `--no-push` leaves the file untouched, so you can preview a bump without side effects.

## When to bump vs rebuild

- **`--bump`**: Application code changed. New endpoints, bug fixes in handlers, new business logic, schema changes.
- **`--rebuild`**: Container code is identical but you need a new image. Dockerfile changes, dependency upgrades, base image refresh, CI/build infra changes.

If unsure, `--bump patch` is always safe.

## Resulting artifacts

A successful `--bump patch` run produces:
- CPU image: `us-east1-docker.pkg.dev/zetta-research/zutils/web_api:1.0.46-2026052001`
- GPU image: `us-east1-docker.pkg.dev/zetta-research/zutils/web_api_gpu:1.0.46-2026052001`
- Git commit: `chore: bump web_api version to 1.0.46`
- Git tag (pushed): `webapi_v1.0.46-2026052001`
