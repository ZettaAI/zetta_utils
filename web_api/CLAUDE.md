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

#### Segment Link Parameters
- `include_certain_ends` (bool) - Include certain endpoints layer (yellow)
- `include_uncertain_ends` (bool) - Include uncertain endpoints layer (red)
- `include_breadcrumbs` (bool) - Include breadcrumbs layer (blue)

### Other APIs
- `/annotations` - Annotation management
- `/collections` - Collection operations
- `/layer_groups` - Layer group management
- `/layers` - Layer operations
- `/painting` - Painting functionality
- `/precomputed` - Precomputed annotations

## Authentication
OAuth2 token verification with @zetta.ai email domain restriction.

## Usage
```bash
# Get segment link with all endpoints
GET /tasks/projects/kronauer_ant_x0/segments/74732294451380972/link

# Get segment link without breadcrumbs
GET /tasks/projects/kronauer_ant_x0/segments/74732294451380972/link?include_breadcrumbs=false
```
