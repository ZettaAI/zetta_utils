# CLI Module

## Task Management CLI

Commands use dashes: `zetta task-mgmt <command>`

Key commands:
- `segment-link -p <project> -s <seed_id>` - Generate neuroglancer segment links
  - `--include-certain-ends / --no-certain-ends` - Include/exclude certain endpoints layer (yellow); default include
  - `--include-uncertain-ends / --no-uncertain-ends` - Include/exclude uncertain endpoints layer (red); default include
  - `--include-breadcrumbs / --no-breadcrumbs` - Include/exclude breadcrumbs layer (blue); default include
- `start -p <project> -u <user>` - Start task
- `get -p <project> -t <task>` - Get task details
- `release -p <project> -u <user> -c <status>` - Release task
- `tasks -p <project>` - List tasks
- `reactivate -p <project> -t <task>` - Reactivate a completed task by clearing its completion status
- `clear -p <project>` - Clear all tasks and timesheets from a project (DESTRUCTIVE)
  - `--include-users / -u` - Also clear users
  - `--include-task-types / -t` - Also clear task types
  - `--force / -f` - Skip confirmation prompt
