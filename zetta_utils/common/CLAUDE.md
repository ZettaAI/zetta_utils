# Common

Essential utility functions and classes used throughout the zetta_utils codebase.

## Core Components
Utility Functions (misc.py): get_unique_id() (generates unique identifiers using coolname slugs and UUIDs), used for execution IDs and temporary resource naming

Partial Functions (partial.py): ComparablePartial (generic partial function wrapper maintaining comparability), critical for builder system and distribution protocols, enables serializable partial functions with kwargs

Path Utilities (path.py): abspath() (converts relative paths to absolute paths with protocol handling), is_local() (determines if a path is local file:// or fq://), strip_prefix() (removes protocol prefixes from paths)

Context Managers (ctx_managers.py): set_env_ctx_mngr() (context manager for temporarily setting environment variables), noop_ctx_mngr() (no-operation context manager)

Timer Utilities (timer.py): RepeatTimer (threading-based repeating timer class), used by mazepa worker system for periodic operations

User Input (user_input.py): get_user_input() (prompts for user input with timeout support), get_user_confirmation() (boolean confirmation prompts with timeout), uses rich library for enhanced terminal interaction

Pretty Printing (pprint.py): lrpad() (left-right padding for formatted string output), utcnow_ISO8601() (UTC timestamp formatting)

Signal Handling (signal_handlers.py): custom_signal_handler_ctx() (context manager for custom signal handling)

## Key Implementation Patterns
ComparablePartial Pattern: Wraps functions with kwargs for delayed execution, maintains comparability for caching and deduplication, used extensively in distributions, mazepa execution, and builder system

Path Handling: Supports multiple protocols (file://, gs://, fq://), handles path expansion and absolute path conversion, used throughout layer system for data access

Environment Management: Temporary environment variable setting, proper cleanup on exit, used in cloud management and execution contexts

## Usage Patterns
Builder System Integration: ComparablePartial used in distribution builders (uniform_distr, normal_distr), path utilities used in layer backends for data access, environment context managers used in builder execution

Mazepa Execution System: get_unique_id() generates execution IDs, RepeatTimer powers worker polling mechanisms, ComparablePartial enables serializable task operations

Layer System: Path utilities handle data source locations across different backends, used in CloudVolume, TensorStore, and annotation layer backends, enables protocol-agnostic data access

## Development Guidelines
Using ComparablePartial: Use for delayed execution of functions with parameters, enables caching and deduplication of operations, maintains function comparability for optimization

Path Handling: Use abspath() for protocol-aware path conversion, check is_local() before local filesystem operations, handle different storage protocols consistently

Environment Management: Use set_env_ctx_mngr() for temporary environment changes, ensure proper cleanup with context managers, avoid global environment modifications

Timer Operations: Use RepeatTimer for periodic background tasks, handle timer lifecycle properly (start/stop), consider thread safety for timer operations

## Integration Points
The common module serves as foundational infrastructure enabling: Builder system serialization and comparability, Layer system protocol-agnostic data access, Mazepa execution system resource management, Distribution system partial function handling
