# Task Management

Distributed task management system for annotation workflows.

## Architecture
- PostgreSQL with SQLAlchemy ORM, composite primary keys for multi-tenancy
- Concurrency-safe: extensive row-level locking with `FOR UPDATE` and `skip_locked=True`
- Plugin-based: extensible task type system with verification, completion, creation handlers
- Multi-tenant: composite primary keys with `project_name` for multi-project support

## Database Models (db/models.py)
Core Models: ProjectModel (segmentation configuration: segmentation_path, SV resolution, datastack info), TaskModel (work units with neuroglancer states, priorities, lifecycle management), UserModel (workers with qualifications, hourly rates, active task tracking), TaskTypeModel (task definitions with allowed completion statuses), DependencyModel (inter-task dependencies with satisfaction tracking)

Biological Data Models: SegmentModel (neurological segments with 3D coordinates, synapse counts, status tracking), EndpointModel (segment endpoint locations with certainty status), SegmentTypeModel (segment type definitions with reference segments)

Tracking Models: TimesheetModel (aggregated time tracking per user/task), TimesheetSubmissionModel (individual time submissions for audit trail), TaskFeedbackModel (links trace tasks to their feedback reviews)

## Key Components
Task Lifecycle Management (task.py): Atomic task assignment with 90-second idle takeover and precise locking. Auto-selection strategy: assigned tasks → available tasks → idle takeover. Completion handling: automatic dependency resolution and downstream task activation. Pause/resume: tasks can be paused to prevent auto-selection.
Key Functions: start_task() (atomic task assignment with user qualification checks), release_task() (task completion with side effect handling), _atomic_task_takeover() (complex takeover logic with deadlock prevention), _handle_task_completion() (dependency satisfaction and task activation)

Task Types System (task_types/): Plugin architecture with three extension points: 1) Verification (verification.py): validate task completion, 2) Completion (completion.py): handle post-completion side effects, 3) Creation (creation.py): generate task-specific neuroglancer states
Built-in Task Types: trace_v0 (neuron tracing with neuroglancer annotation layers), trace_feedback_v0 (review and feedback for completed traces), trace_postprocess_v0 (post-processing for completed traces)

Dependency System (dependency.py): Task dependencies with required completion statuses, automatic dependency satisfaction on task completion, deadlock prevention through alphabetical lock ordering

Time Tracking (timesheet.py): Atomic timesheet submissions with race-condition safety, aggregated time tracking per user/task, individual submission audit trail, work history analysis

Automated Workers (automated_workers/): segmentation_auto_verifier.py (automated skeleton verification worker), Slack integration for notifications, CAVE client integration for skeleton analysis

## Design Patterns & Important Features
Concurrency Control: Row-level locking with extensive use of `WITH FOR UPDATE` for atomic operations, skip_locked=True for optimistic task selection, alphabetical ordering to prevent deadlocks, complex multi-step operations in single transactions

Task Assignment Strategy: 1) Assigned tasks (explicitly assigned to user), 2) Available tasks (unassigned tasks user is qualified for), 3) Idle takeover (take over tasks idle for >90 seconds)

Validation System: TypedDict types with comprehensive type definitions in types.py, runtime validation with typeguard decorators for type checking, business logic validation with task-specific verification handlers

Database Session Management: Context manager get_session_context() for session lifecycle, optional sessions (all functions accept optional db_session parameter), transaction management with proper commit/rollback handling

## Configuration & Constants
_MAX_IDLE_SECONDS = 90 (task idle timeout for takeover), MAX_NONUNIQUE_ID = 2**32 - 1 (random ID generation range), PostgreSQL connection with environment-based password, extensive indexing for performance

## Testing Patterns
Test containers (PostgreSQL containers for isolated testing), fixtures (comprehensive test fixtures for projects, users, tasks), concurrent testing (tests for race conditions and locking behavior), mocking (uses mocker fixture, avoids unittest mocks)

## Usage Example
```python
create_project(project_name="test_project", segmentation_path="gs://path/to/segmentation", sv_resolution_x=4.0, sv_resolution_y=4.0, sv_resolution_z=42.0)
create_user(project_name="test_project", data={"user_id": "worker1", "hourly_rate": 25.0, "qualified_task_types": ["trace_v0"], "active_task": ""})
create_task_type(project_name="test_project", data={"task_type": "trace_v0", "completion_statuses": ["Done", "Can't Continue"]})
task_id = start_task(project_name="test_project", user_id="worker1")
release_task(project_name="test_project", user_id="worker1", task_id=task_id, completion_status="Done")
```

## Development Guidelines
Code Patterns: Always use get_session_context() for database operations, implement proper error handling with specific exception types, use typeguard decorators for runtime type checking, follow the plugin pattern for task type extensions

Testing Guidelines: Use PostgreSQL test containers for integration tests, test concurrent operations with proper isolation, mock external dependencies (CAVE, Slack) using mocker fixture, test both success and failure scenarios

Performance Considerations: Use skip_locked=True for optimistic locking, implement proper indexing for query performance, use batch operations for bulk data processing, monitor query execution times and optimize as needed
