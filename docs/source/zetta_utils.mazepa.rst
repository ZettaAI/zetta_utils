``zetta_utils.mazepa``
=======================

Overview
--------

**Mazepa** is a lightweight event-driven task queue library developed by Zetta.AI. It provides an efficient solution for distributing users' workloads to a large number of remote workers, simplifying task management in distributed environments.


Task Execution Model
^^^^^^^^^^^^^^^^^^^^

``mazepa``'s task execution model revolves around the concepts of Tasks, Flows, and Dependencies. Understanding these key elements is essential for effectively utilizing ``mazepa``.

Tasks
"""""

**Tasks** represent individual units of work that need to be executed. They encapsulate the necessary information, such as the function to be executed and its associated arguments, along with additional metadata related to scheduling, execution status, and error handling. When a Task is yielded within a Flow, it will initiate execution.

Flows
"""""

**Flows** define the overall workflow by specifying the order and dependencies between ``Tasks``. When writing a ``Flow``, users need to follow two simple rules:

#. Once a ``Task`` is yielded, it will start execution, and
#. When a ``Dependency`` is yielded, control flow will return to the user's ``Flow`` only once the ``Dependency`` is satisfied.

This allows for precise control over task execution and enables the creation of complex workflows.

For example, consider the following snippet from a ``Flow``:

.. code-block:: python

   ...
   yield [Task1, Task2]
   yield mazepa.Dependency(Task1)
   yield Task3
   ...

In this case, ``Task1`` and ``Task2`` can be executed in any order, and once ``Task1`` is done, ``Task3`` will be executed. By explicitly defining the dependencies using ``yield`` statements, ``mazepa`` ensures that tasks are executed in the desired order, while making the ``Flow`` easy to write and debug.

Dependencies
""""""""""""

**Dependencies** represent completion events that must occur before continuing the execution of subsequent tasks. By yielding a ``Dependency`` within a ``Flow``, control flow will wait until the specified dependency is satisfied before proceeding further. This allows for fine-grained control and synchronization of task execution.

Execution
"""""""""

All executions of ``mazepa`` workflows are initiated by a call to the ``mazepa.execute`` function. The ``mazepa.execute`` function expects a ``target``, which can be a single ``Flow`` or a collection of ``Flows``. Additionally, it can accept other parameters such as the choice of the queueing system, the frequency of querying task completion status, the frequency of checkpointing execution progress, and so on.


Integration and Event-Driven Workflow
"""""""""""""""""""""""""""""""""""""

``mazepa`` can be seen as an event-driven wrapper that can be seamlessly integrated with any other task queue system. Dependencies in ``mazepa`` specify the completion events that must occur before continuing execution, enabling a flexible and event-driven workflow design. By following the rules of task yielding and utilizing ``Dependencies`` effectively, users can leverage ``mazepa``'s event-driven approach to create intricate and efficient workflows.

Currently, ``mazepa`` supports local or AWS SQS-based executions. However, integration with other task queue libraries, such as GCP Pub/Sub or RabbitMQ, can be achieved by implementing a class that conforms to the ``ExecutionQueue`` protocol, and passing an object of this class to ``mazepa.execute``. ``mazepa.execute`` will default to local execution if no execution queue is given.

Task and Flow Creation
^^^^^^^^^^^^^^^^^^^^^^

This subsection showcases different ways of creating Mazepa tasks and flows.

.. note::

   The following examples use default local execution, where tasks are always executed in the order they are yielded, even without explicit dependencies. This behavior is not guaranteed when using remote execution queues.

The following bare-bones example demonstrates how ``Tasks`` and ``Flows`` can be manually constructed by the user from callables, highlighting the simplicity of ``mazepa``:

.. code-block:: python

   from zetta_utils import mazepa
   from zetta_utils.mazepa import Task, Flow

   def greet(name):
       print (f"Hi {name}!")

   def greet_many():
       yield Task(fn=greet, args=("Albert",))
       yield Task(fn=greet, args=("Isaac",))

   flow = Flow(greet_many)
   mazepa.execute(target=flow)
   # Hi Albert!
   # Hi Isaac!

In this example, ``Tasks`` and a ``Flow`` are created manually. The ``greet`` function represents an individual unit of work, and ``greet_many`` is a generator function that yields two ``Tasks``: one for greeting ``"Albert"`` and another for greeting ``"Isaac"``. The ``mazepa.execute`` function is then used to execute the specified ``Flow``, resulting in the printing of the greetings.

While this method of manually creating ``Tasks`` and ``Flows`` is simple, users are encouraged to avoid creating ``Tasks`` and ``Flows`` in this way. Instead, users are encouraged to use Python decorators for creating ``Task/Flow`` blueprints:

.. code-block:: python

   from zetta_utils import mazepa
   from zetta_utils.mazepa import taskable_operation, flow_schema

   @taskable_operation
   def greet_op(name):
       print (f"Hi {name}!")

   @flow_schema
   def greet_many(names):
       yield [
           greet_op.make_task(name)
           for name in names
       ]

   flow = greet_many(["Albert", "Isaac"])
   mazepa.execute(target=flow)

**@taskable_operation** is a non-destructive decorator that adds a ``make_task`` function call to the callable it wraps. The decorator declares that the given function now constitutes a taskable operation - i.e. it’s an operation that can be called normally, but can also be packaged into a task. Note that because the decorator is non-destructive, the function can be still called normally as if it was never wrapped.

**@flow_schema** is analogous to ``@taskable_operation``, except that ``@flow_schema`` is a destructive decorator, meaning that it overrides the ``__call__`` method of the wrapped function. This design choice was made because callables wrapped by ``@taskable_operation`` often need to be used outside of ``mazepa`` context, while flow (schema) generators are only used for ``mazepa``.

Using these blueprint decorators provides a cleaner syntax while also allowing users to specify per-operation task metadata, such as tags and retry counts.

Additionally, ``mazepa`` provides decorators **@taskable_operation_cls** and **@flow_schema_cls** for creating **TaskableOperations** and **FlowSchemas** from class objects. This allows users to define operations and flows as classes, further enhancing code organization and modularity.

.. code-block:: python

   @mazepa.taskable_operation_cls
   class GreetOperation:
       def __init__(self, greeting="Hi"):
         self.greeting = greeting

       def __call__(self, name):
         print (f"{self.greeting} {name}!")

   @mazepa.flow_schema_cls
   class GreedFlowSchema:
       def flow(self, names):
           for name in names:
               yield GreetOperation().make_task(name)

   flow_schema = GreedFlowSchema()
   flow = flow_schema(["Albert", "Isaac"])

   mazepa.execute(flow)
   # Hi Albert!
   # Hi Isaac!


Alternatives
^^^^^^^^^^^^

.. note::

   This section was last updated in July 2023, and it is possible that some of this information is outdated. PRs are alwayws welcome!

Celery
""""""
**Celery** is a popular task queue library that provides distributed task execution. However, Celery's approach to specifying control flow can be more complex and less intuitive compared to Mazepa. Celery requires the use of constructs like groups, chords, and chains, which can make the codebase more convoluted. Additionally, the Celery codebase is spread across multiple projects, making it harder to understand and debug. In contrast, Mazepa uses a pythonic control flow approach based on generators, resulting in concise and easier-to-understand code.

Dramatiq
""""""""
**Dramatiq** is another task queue library that lacks a pythonic way to specify dependencies between tasks, similar to Celery. While Dramatiq's implementation is simpler than Celery's, it still contains more code than Mazepa. Mazepa's focus on reusing built-in Python constructs, such as generators, contributes to its concise and understandable implementation.

Airflow
"""""""
**Airflow** is a workflow management platform that allows users to construct execution DAGs (Directed Acyclic Graphs) for defining workflows. While Airflow provides a more pythonic control flow specification compared to Celery and Dramatiq, its static DAG construction can still be less intuitive and less debuggable than Mazepa's approach. Furthermore, Airflow is a heavyweight solution that requires running a dedicated server, adding complexity to the system architecture.

Prefect
"""""""
**Prefect** is a workflow management library that offers a pythonic way to specify dependencies between tasks. However, Prefect has a couple of limitations. Firstly, it lacks the ability to control the execution environment at the task level, as the environment specification is done at the root flow level. This limitation can be problematic for users who need to execute tasks with different requirements in terms of CPU, GPU, or other resources. Secondly, Prefect's tracking data collection for each sub-flow can lead to significant performance overhead when scheduling a large number of sub-flows.

Seunglab Taskqueue
""""""""""""""""""
Mazepa builds on top of Seunglab **Taskqueue** to provide support for AWS SQS-based execution. By adding pythonic control flow specification and the ability to handle non-JSONifiable task arguments, Mazepa extends the functionality of Seunglab Taskqueue and simplifies implementing complex workflows.
