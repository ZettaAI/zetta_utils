# CLI Module

## Task Management CLI

Commands use dashes: `zetta task-mgmt <command>`

Key commands:
- `segment-link -p <project> -s <seed_id>` - Generate neuroglancer segment links
  - `--no-certain-ends` - Exclude certain endpoints (yellow)
  - `--no-uncertain-ends` - Exclude uncertain endpoints (red)
  - `--no-breadcrumbs` - Exclude breadcrumbs (blue)
- `start -p <project> -u <user>` - Start task
- `get -p <project> -t <task>` - Get task details
- `release -p <project> -u <user> -c <status>` - Release task
- `tasks -p <project>` - List tasks
