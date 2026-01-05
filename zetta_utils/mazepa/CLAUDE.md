# Mazepa

Distributed task execution framework with task queues, flow orchestration, dependency management.

## Core Components
Task System (tasks.py): Task (executable unit with unique ID, operation name, worker type, execution settings), TaskableOperation (protocol for operations convertible to tasks via make_task() method), @taskable_operation_cls (class decorator for creating taskable operations - preferred), @taskable_operation (function decorator for taskable operations), TaskUpkeepSettings (configuration for task heartbeat/lease extension)

Flow System (flows.py): FlowSchema (factory that creates Flow instances - defines workflow structure), Flow (actual execution container with generator functions that yield tasks/dependencies), @flow_schema_cls (class decorator for flow schemas - preferred), @flow_schema (function decorator for flow schemas), Dependency (task dependencies and synchronization points), Built-in flows (sequential_flow, concurrent_flow)

Execution Engine (execution.py): Executor (main execution orchestrator with configurable parameters), execute() (primary execution function with task batching, progress tracking, checkpointing), ExecutionState (abstract base for managing flow state and dependencies), InMemoryExecutionState (in-memory implementation tracking flows, tasks, dependencies)

Worker System (worker.py): run_worker() (main worker loop for distributed execution), process_task_message() (task processing with retries and error handling), transient error handling and automatic retries, task upkeep for long-running operations

Resource Management: Semaphores (POSIX semaphores for resource control: read, write, cuda, cpu), configure_semaphores() (context manager for semaphore lifecycle), semaphore() (function to acquire semaphores in operations)

## Advanced Features
Subchunkable Apply Flow: VolumetricApplyFlowSchema (complex flow for processing large volumetric data), subchunkable processing (hierarchical chunking with blending and cropping), DelegatedSubchunkedOperation (multi-level delegation for complex workflows), reduction operations (ReduceByWeightedSum, ReduceNaive for combining overlapping chunks)

Progress and Error Handling: TaskOutcome (execution results with exception handling and timing), TaskStatus (enum for task lifecycle states), ProgressReport (progress tracking with submitted/completed counts), TransientErrorCondition (configurable retry conditions), checkpoint system (execution state persistence and recovery)

## Builder Registrations
Key Registrations: run_worker (worker process entry point), build_subchunkable_apply_flow (complex volumetric processing), build_postpad_subchunkable_apply_flow (simplified interface)

Common Usage in Specs:
```cue
"@type": "mazepa.execute_on_slurm"
target: {
    "@type": "mazepa.sequential_flow"
    stages: [{"@type": "build_subchunkable_apply_flow"}]
}
```

## Core Design Patterns
1. Protocol-based interfaces: TaskableOperation, FlowSchema for extensibility
2. Class-based decorators: Use @taskable_operation_cls and @flow_schema_cls for better structure
3. Hierarchical execution: FlowSchemas create Flows, Flows yield Tasks, Tasks execute Operations
4. Resource-aware scheduling: Worker types, semaphores, and queues
5. Fault tolerance: Retries, checkpointing, and graceful degradation

## Development Guidelines
Creating Operations and Flows:
```python
@taskable_operation_cls
class MyOperation:
    def __init__(self, param: str): self.param = param
    def __call__(self, input_data: str) -> str: return f"{self.param}: {input_data}"

@flow_schema_cls
class MyFlowSchema:
    def __init__(self, operations: list): self.operations = operations
    def __call__(self) -> Flow:
        for op in self.operations: result = yield op(input_data="example")
        return result
```

Flow vs FlowSchema: FlowSchema (template/factory that defines workflow structure, registered with builder), Flow (runtime instance created by FlowSchema, contains actual generator execution logic). Always register FlowSchemas with builder, not Flows directly.

Resource Management: Use semaphores for resource-intensive operations, configure appropriate worker types for different task categories, monitor resource usage and adjust limits as needed

Error Handling: Implement proper exception handling in operations, use transient error conditions for recoverable failures, design operations to be idempotent when possible, use checkpointing for long-running workflows
