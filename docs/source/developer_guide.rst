===============
Developer Guide
===============

Introduction
------------

This is a comprehensive guide for new developers to zetta_utils. We will discuss how to set up the environment, the tools that we use in pre-commit, why we use the tools we do (especially type annotations), how to write and run tests, general code quality guidelines, and standards for pull requests.

You might find some (or all) parts of this guide very basic; this is intentional as we want to encourage new developers to familiarise themselves with the tooling we commonly use for both increased productivity and better code quality.


Setting up the development environment
--------------------------------------

Installation
============

You are *strongly* encouraged to use a virtual environment like `conda` or `virtualenv` for development work. While this is encouraged anyway, it is especially important in a development context where you may be changing the dependency versions.

The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_. For development purposes, install the package with ``[all]`` extras, which include tools for running tests and building documentation:

.. code-block:: console

   $ git clone git@github.com:ZettaAI/zetta_utils.git
   $ cd zetta_utils
   $ SETUPTOOLS_ENABLE_FEATURES=legacy-editable pip install -e '.[all]'

The ``SETUPTOOLS_ENABLE_FEATURES=legacy-editable`` is necessary due to a shortcoming of ``setuptools`` (https://github.com/pypa/setuptools/issues/3535).

After installation completes, install ``pre-commit`` hooks:

.. code-block:: console

   $ pre-commit install

For running and writing integration tests, you should also install **Git Large File Storage** (LFS). Follow `this guide <https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>`_ to do so.

tmux configuration (optional)
=============================

If you are using a terminal based IDE (or are just using ``vim``), you are strongly recommended to use **tmux**, which is a terminal multiplexer that lets you switch between multiple terminal sessions. It is similar to ``screen``, but with a lot more flexibility, similar to working with tabs and split screens in ``vim``.

On Ubuntu, run ``sudo apt-get install tmux``. We recommend this tmux config: https://github.com/viktorChekhovoi/workspace-setup, which you can install through

.. code-block:: console

   $ git clone https://github.com/viktorChekhovoi/workspace-setup.git
   $ ./workspace-setup/tmux.sh

To open `tmux`, just run ``tmux`` from the terminal.

With the configuration above, [Control + a] is the prefix used to enter a command.

Some sample commands:

* ``Control + a`` + ``-``: split a window horizontally
* ``Control + a`` + ``\`` (or ``|``): split a window horizontally
* ``Control + a`` + ``backspace`` (or ``del``): close a window
* ``Control + a`` + ``n``: opens a new tab
* ``Control + a`` + ``[arrow key]``: switch between windows (you can also use the mouse)

Code Analysers & pre-commit
---------------------------

``zetta_utils`` relies on a number of code analysers to maintain code quality and readability, and you will not be able to make a commit without passing the analysers. If you need to override the checks for whatever reason, you can use ``git commit -n`` to force the commit, but note that a pull request will not be accepted unless it passes CI (continuous integration), which includes the following checks.

pylint
======

**pylint** is a static code analyser that checks the code for errors and style without actually running it. It can detect things like:

* Unused imports and variables
* Variable names that do not conform to a convention (e.g. ``foo`` is disallowed)
* References to a variable outside scope
* Function definitions being overwritten
* Suggestions for refactoring and rewriting logic
* Long line lengths
* Too many local variables
* Too many branches

To run ``pylint``, you may use:

.. code-block:: console

   $ pylint zetta_utils/path/to/file

Note that the command must be run from the ``zetta_utils`` root directory to use the correct configuration file.

While ``pylint`` is not foolproof, it is a good indication that the code quality could be improved if it has something to say about your code. ``pylint`` will **NOT** modify your code for you; you will have to go in and fix any issues it detects yourself.

.. note::
   In case you have a good reason to override ``pylint``, you may use ``pylint: disable=warning-name``, but this should be used sparingly. An example of acceptable use case can be found in `SubchunkableApplyFlow <https://github.com/ZettaAI/zetta_utils/blob/0b102a8bb737a9f940db667eeffbebc244d04a88/zetta_utils/mazepa_layer_processing/common/subchunkable_apply_flow.py>`_, where the ``build_subchunkable_apply_flow`` function handles the argument checking for a complicated ``Flow`` with many arguments and thus ``too-many-locals`` and ``too-many-branches`` has been overridden.

black
=====


**black** is a Python code formatter that is PEP 8 compliant. `Here is a comprehensive list <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_ of what ``black`` thinks is acceptable, but as a summary, it will enforce:

* Uniform horizontal and vertical whitespaces
* Removal of spurious empty lines
* Addition of trailing commas for multiline expressions split by commas
* Removal or addition of parentheses as deemped necessary
* Consistent usage of **'** versus **"**

To run ``black``, you may use:

.. code-block:: console

   $ black zetta_utils/path/to/file

``black`` **WILL** modify your code for you. This means that if you have run ``black``, you will have to re-add the files that it touched to your commit.

mypy
====

**mypy** is a static type checker that looks at type annotations (see the section below for more details on type annotations) and checks that everything is typed correctly. Among other things, it will check that:

* Function calls match the signature of the function
* Objects have the requested attribute
* Declared return type matches the actual return type
* `Liskov subsitution principle <https://en.wikipedia.org/wiki/Liskov_substitution_principle>`_ is not violated

To run ``mypy``, you may use:

.. code-block:: console

   $ mypy zetta_utils/path/to/file

Note that the command must be run from the ``zetta_utils`` root directory for the ``mypy`` to be able to parse all relevant files.

``mypy`` will **NOT** modify your code for you.

pre-commit
==========

When you try to commit a change to ``zetta_utils``, the **pre-commit** hook will run, which checks for common mistakes and inconsistencies like:

* Trailing whitespace in code
* Files that do not end in an empty line
* Unparsable ``json``, ``toml``, and ``yaml`` files (note: CUE is not yet supported)
* Accidental commit of a large file
* Accidental inclusion of a private key
* Unresolved merges
* Leftover ``reveal_type`` used for typing

In addition, ``pre-commit`` will run ``pylint``, ``black``, and ``mypy``.

It can be a little frustrating at times to get a large commit past the ``pre-commit`` hook when there are a number of errors that all seem very minor (especially when you know that your code runs fine for your use case!). The errors that ``mypy`` throws with respect to types are usually the hardest to fix, sometimes requiring a significant redesign, but we have yet to find a situation where we regretted redesigning or rewriting to placate ``mypy`` in the long run.

In an ideal world, we would want the entire ``zetta_utils`` codebase to look like it was programmed by a single programmer obsessing over the code quality, and the ``pre-commit`` hooks bring us a lot closer to that.

As noted, you can override the ``pre-commit`` with ``git commit -n``, but running ``pre-commit`` locally is a lot faster than waiting for Github's CI hook to run, which can take upwards of 10 minutes.

That covers the code analysers and the ``pre-commit`` hook used in ``zetta_utils``. These tools ensure code quality and maintainability, contributing to a cleaner and more reliable codebase.

Type Annotations
----------------

``zetta_utils`` makes extensive use of **type annotations**. Type annotations are an `optional part of Python syntax <https://peps.python.org/pep-0483/>_` that declare the types of variables, function return types, and attributes.

You can add a type annotation to a variable using a colon, like so:

.. code-block:: python

   foo: int = 5

You can add a type annotation first and then initialise a variable, similar to what you might do in C:

.. code-block:: python

   foo: int
   foo = 5

You can add a type annotation to the arguments of the function as well as annotate its return type using ``->``:

.. code-block:: python

   def sum(foo: int, bar: int) -> int:
       return foo + bar

So, why do we want to bother with type annotations? Isn’t the whole point of Python that it has dynamic, duck typing? There are a number of benefits to type annotations, but it boils down to readability, maintenance, and avoiding runtime errors:

* Improved code readability: Type annotations make the code more explicit and self-documenting. By specifying the types of variables, parameters, and return values, it becomes easier for other developers (including yourself, especially a few months down the road) to understand the purpose and expected usage of different parts of the codebase. ``reveal_type`` can be used to reveal the expected type of a variable inside a function, for instance.
* Enhanced code maintenance: Type annotations serve as a form of documentation that can help with code maintenance. When revisiting or modifying code, type annotations provide valuable information about the expected types, which reduces the risk of introducing bugs or unintended side effects.
* Static type checking: Type annotations enable static type checkers (``mypy`` in our case) to analyze the code for potential type errors before runtime. This can catch certain bugs and issues early in the development process, reducing the likelihood of encountering runtime errors or unexpected behavior.

To highlight the last point, ``mypy`` will raise an error at the following four snippets:

.. code-block:: python

   x = 3 + "test"

``mypy`` output: ``error: Unsupported operand types for + ("int" and "str")  [operator]``

.. code-block:: python

   def sum(foo: int, bar: int) -> int:
       return foo + bar

       sum(3, "test")

``mypy`` output: ``error: Argument 2 to "sum" has incompatible type "str"; expected "int"  [arg-type]``

.. code-block:: python

   @attrs.define
   class SomeClass:
       foo: int

       SomeClass("string")

``mypy`` output: ``error: Argument 1 to "SomeClass" has incompatible type "str"; expected "int"  [arg-type]``

.. code-block:: python

   def prod(foo: int, bar: int) -> str:
       return foo * bar

``mypy`` output: ``error: Incompatible return value type (got "int", expected "str")  [return-value]``


If these four snippets make it to production code, they might result in a runtime error further down the execution path, or worse, result in an unexpected output without an error that leaves you scratching your head trying to figure out what went wrong using a bunch of debug statements. It is much easier to catch the errors before they happen, before we even run the code.

Python's built-in `typing module <https://docs.python.org/3.10/library/typing.html>`_ supports type annotation with types such as ``Any``, ``Sequence``, ``List``, ``Literal``, ``Union`` to indicate any type, any type of sequence (e.g. list, tuple, or a generator), list, a collection of objects (similar to ``enum`` in C), or a union of multiple different types. Furthermore, you can use **TypeVar** to annotate the return type of a function contingent on the type of the arguments, or use **@overload** decorator to declare the contingent return type directly.

In general, **every class attribute, function and method arguments, as well as return values must be type annotated**, preferably with the most restrictive type that fits the bill. (Declaring variables with types is usually unnecessary, because the type inference performed by ``mypy`` is usually smart enough, but it can help sometimes.) In addition, **@typeguard.typechecked** should be added to all declared functions and classes for dynamic type checking, unless:

#. You're using Generics, which are currently `not supported by typeguard <https://github.com/agronholm/typeguard/issues/139>`_, or
#. Dynamic type checking significantly slows down your code (e.g. a small function called many times inside a loop)

If you wish to suppress dynamic type checking in a performance-critical part of the code, you may use **@typeguard.supress_type_checks** decorator.

.. note::

   Do **NOT** manually check input types to raise exceptions.

To help you pass ``mypy``, you can use the annotation **reveal_type(variable)** to reveal its inferred type. This is a fully static keyword (like a compiler directive), and the line should be removed before the final commit since it will result in a runtime error. As an example, running the ``pre-commit`` on

.. code-block:: python

   x = 3 + 3.14
   reveal_type(x)

outputs ``note: Revealed type is "builtins.float"``.

.. note::

   When you are annotating an object of some class within the class declaration (e.g. you are defining ``MyNumber`` class and want to annotate the return type of ``MyNumber.double()`` as ``MyNumber``), you will need to put on ``from __future__ import annotations`` at the top of your Python file to get ``mypy`` to recognise the class before it is fully declared.

.. note::

   In case you have a good reason to override ``mypy``, you may use ``type: ignore`` to indicate to ``mypy`` that the type inference should be ignored for a given line, but this should be used sparingly. ``type: ignore`` is something that we use when ``mypy`` does not support type inference due to limitations of Python or due to a bug with a third-party library, not when we want to avoid typing our code cleanly.


attrs and classes
-----------------

**attrs** is a package that makes class declarations cleaner and easier, with less boilerplate. ``attrs`` writes the ``__init__``, ``__repr__``, and (for frozen classes) ``__eq__`` dunder methods for the programmer.

``attrs`` works seamlessly with type annotations. Using ``attrs``, you can define a class with type annotated attributes like so:

.. code-block:: python

   @attrs.frozen
   class SomeClass:
       foo: AType
       bar: AnotherType

In general, **every class should be defined using either** ``@attrs.frozen`` **(if the class is immutable) or** ``@attrs.mutable`` **(if the class is mutable)**. Unless it is necessary, a class should be immutable for safety.

If you need to return a modified instance of an immutable class, ``attrs.evolve()`` is a concise way to deepcopy and reinitialise a frozen object with some changes.

.. note::

   **Methods that returns a modified copy of an object, rather than the mutated original, should start with** ``with``, **or use the past participle of a verb; methods that mutate the original object should use the base form of a verb**. For instance, ``VolumetricCallableOperation`` has a ``with_added_crop_pad(self, crop_pad)`` method, and ``BBox3D`` has a ``padded(self, pad, resolution)`` method, both of which returns a modified copy. If these were mutating the object, they would be called ``add_crop_pad`` and ``pad``, respectively.

.. note::

   Avoid excessive reliance on inheritance, especially multilevel inheritance: inheritance makes the code harder to read and maintain. Whenever possible, make your base classes `pure abstract <https://en.wikibooks.org/wiki/C%2B%2B_Programming/Classes/Abstract_Classes/Pure_Abstract_Classes>`_.

Logging and Exceptions
----------------------

``zetta_utils`` has a built-in **logger** that uses **rich** to pretty print colour coded logs with **grafana** integration. The ``logger`` can be imported using

.. code-block:: python

   from zetta_utils import log
   logger = log.get_logger("zetta_utils")

``logger`` is backed by Python's built-in logger, and supports a number of different message levels such as: ``debug``, ``info``, ``warning``, ``error`` and ``exception`` (same level), and ``critical``. To use the logger, you can use ``logger.info("string")`` for instance. Any exceptions raised will automatically be collected and output by the logger, so you should not have to write ``logger.exception`` yourself.

The verbosity level for the logger in stdout can be set using ``zetta run -v``, ``zetta run -vv``, or ``zetta run -vvv``, corresponding to ``warning``, ``info``, and ``debug``, respectively. The default is ``info``.

If you need to output information to the user in your code, you should **NEVER** use ``print``, and use ``logger`` instead.


.. note::

   We prefer to not have ``assert`` statements in any of the core modules. ``assert`` is used to catch programmer error rather than user error, because ``AssertionErrors`` are less helpful than typed ``Errors``. Instead, you should make frequent use of detailed exception handling using the ``from`` keyword. However, ``asserts`` may be used for performance when intending to run with ``python -O``.

Tests
-----

Unit Tests vs. Integration Tests
================================

``zetta_utils`` has unit tests with 100% line coverage in the core modules, and some integration tests. The differences between unit tests and integration tests are:

#. Unit tests focus on testing individual units or components in isolation, while integration tests examine the interaction between and the behaviour of multiple components.
#. Unit tests are supposed to verify the correctness of each module, while integration tests are supposed to identify defects that arise when the modules are combined.
#. Unit tests are designed to run independently of other units and external dependencies, using mockers, while integration tests require many dependencies and rely on databases, network connections, and extermal services.
#. Unit tests are fine grained, targetting specific functions or methods, while integration tests are coarser and focus on testing the interfaces, data flow, and the communications between different components.
#. Unit tests provide rapid feedback and debugging, while integration tests provide slower but important feedback on compatibility and correctness of the final product.

The unit tests are in ``tests/unit``, with the directory structure mirroring the ``zetta_utils`` folder, with the addition of the ``assets`` folder.  Unit tests can be run by running ``pytest`` from the main ``zetta_utils`` directory.

The integration tests are in ``tests/integration``, again with the addition of the ``assets`` folder. Since integration tests are testing how the code will behave in production, the ``assets`` are kept in **Git LFS**. Integration tests can be run from the ``tests/integration`` directory (this is due to a limitation of Github Actions) by running ``pytest --run-integration``.

When successfully run, ``pytest`` will generate a code coverage report. Unit tests are automatically run on a pull request, and the code coverage report gets appended to the pull request as part of the CI. Since integration tests take a little longer, the integration tests are run manually through the Github web interface through Actions, rather than automatically.

``zetta_utils`` does not yet have a good integration test coverage, but we hope to increase our coverage in the future.

Unit Tests
==========

As noted, the unit tests are in ``tests/unit``. The directory structure mirrors the codebase, and the files are called ``test_{original_file_name}.py``. Every time you add a file in a covered area, you **need** to add the corresponding test file and write tests.


Good unit tests should use fixtures (a known state against or in which the test is running), using mocked dependencies whenever possible. Using mocked dependencies to spoof calls to external libraries and other parts of the codebase isolates any potential issue to the problem in the unit being tested. In addition, unit tests must be fast: do not use large tensors, send HTTP requests, or read from the file system in your unit tests.

For simple methods, the easiest way to write tests is to use ``@pytest.mark.parametrize``, which lets you run the same piece of code on multiple different inputs. Here is a simple example that tests ``Vec3D`` to see if the indexing works correctly:

.. code-block:: python

   vec3d = Vec3D(1.0, 2.0, 3.0)
   vec3d_diff = Vec3D(4.0, 5.0, 6.0)

   @pytest.mark.parametrize("arg, ind, val", [[vec3d, 2, 3], [vec3d_diff, 0, 4]])
   def test_indexing(arg, ind, val):
       assert arg[ind] == val

For testing exceptions, you can use ``with pytests.raises``. Here is an example that tests ``Vec3D`` to see if it raises a ``TypeError`` when given a tuple of wrong length:

.. code-block:: python

   @pytest.mark.parametrize(
   "constructor, args, expected_exc",
   [
       [Vec3D, (1, 2, 3, 4), TypeError],
   ],
   )
   def test_exc_tuple(constructor, args, expected_exc):
       with pytest.raises(expected_exc):
           constructor(args)

Other useful methods for testing include ``mocker.patch`` (which modifies a single function) and ``mocker.MagicMock`` (which mocks an object or a method). Please see the ``mocker`` documentation for usage. (`This guide <https://www.nerdwallet.com/blog/engineering/5-pytest-best-practices/>`_ may be a good place to start if you are not familiar with ``pytest``.)

.. note::

   Factoring your code into smaller functions and removing code duplication will make it easier to construct test cases. If a part of your code is too difficult to test, consider whether you're using ``mocker`` to its full potential. If it's still too hard to test, consider refactoring.


While we do not have 100% branch coverage, we do want to maintain 100% line coverage. Every method and function should be tested, with an exception for the case where the function:

#. Does not have logic in it.
#. Is under three lines long.
#. Does pure delegation.
#. Only handles visualisation.

If the function or a method meets this criteria, you can use ``# pragma: no cover`` comment on the line where it is declared to exempt it from coverage.

Integration Tests
=================

A good integration test should use real dependencies, and include a variety of possible configurations to catch possible errors. Any assets should be stored using ``Git LFS``. If your integration tests relies on remote data, your data should be in ``gs://zetta_public/`` bucket, and any temporary writes should be written to ``gs://tmp_2w/`` bucket to ensure timely deletion.

To make the test only execute when ``--run-integration`` is given, you should add

.. code-block:: python

   @pytest.mark.skipif(
       "not config.getoption('--run-integration')",
       reason="Only run when `--run-integration` is given",
   )

to each function in your testing file.

Docstrings
----------

Every class / method / function that is user-facing should have a docstring, using the `Sphinx ReadTheDocs <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_ format.

The docstring should briefly outline what the class / method / function does, and then explain its parameters. Since the arguments and the return type should already be type annotated, ``:param [ParamName]: [Description]`` strings for each parameter and ``:return: [Description]`` are the only things required. Here is a minimal example:

.. code-block:: python

   def add(a: int, b: int) -> int:
   """
   Simple function that takes two ints and adds them.

   :param a: First integer.
   :param b: Second integer.
   :return: Sum of the two integers
   """
       return a + b

Once this docstring exists, the new class / function should be included in the documentations under ``zetta_utils/docs/source``. Using ``.. autoclass:: zetta_utils.module.class`` and ``.. autofunction:: zetta_utils.module.function`` will respectively generate the documentation from the docstring. If the module /  class / function necessitates a more detailed write up, then you are encouraged to write more in the correct ``rst`` file.

Code Quality Guidelines
-----------------------

We do not have a comprehensive style guide, but here are some `guidelines that Google uses <https://google.github.io/styleguide/pyguide.html>`_, which we mostly try to follow (one notable difference is that we use the default Sphinx RTD format for docstrings).

In addition to the requirements in the above sections, here are some things that we expect developers to follow:

Don't duplicate code
====================

You should never have to rewrite the same three lines of code. If you notice that you are rewriting the same code in multiple places, chances are that that piece of code might be useful to other people. For brevity and maintainability, the code should be refactored into its own function.

Delegate as much as possible
============================

Related to the last point, if you want to do something simple that sounds useful like padding a bounding box, we probably already have a method or a function to do it. Delegate to the method that we have already written and tested, instead of implementing your own. If it sounds simple and useful but doesn't exist, write it, write the unit test for it, and then delegate to it.

Think about maintainability and expandability
=============================================

Any piece of code that gets merged is code that we will have to maintain in perpetuity. Make sure that your code is modular and that the API is well-thought out: modular code helps code be reused even if the API changes, and well-thought out APIs (especially Protocols) make future feature requests easier to implement.

Comment sparingly
=================

For readability, we encourage you to use comments as sparingly as possible. If your code is written well with good structure and variable / method / function names, it should be self-documenting. Comments should be reserved for delineating blocks that accomplish different things, for noting PEPs that might improve the code in the future, or for things that you know will be gotchas for future you or for other developers.


Be Pythonic
===========

*Pythonic* describes code that uses features of Python to improve readability, maintainability, and efficiency. Make use of unpacking, comprehensions, lambdas, f-strings,  ``enumerate``, ``map``, and ``reduce``. Consider this code snippet:

.. code-block:: python

   # propagate is_consecutive mask to all participating sections
   participation = torch.zeros((x * y, c, z), dtype=torch.uint8)
   for i in range(num_consecutive):
       participation += is_consecutive[:, :, i : z + i]
   # only needs to participate in one consecutive range to be masked
   participates = participation > 1


The same piece of code could be rewritten as:

.. code-block:: python

   # only needs to participate in one consecutive range to be masked
   participates = torch.logical_or(*(is_consecutive[:, :, i : z+i] for i in range(num_consecutive))


Single source of default arguments
==================================

In cases where there is a user-facing function that does argument checking and an internal function that does the heavy lifting, the internal function should **never** have default arguments. Any defaults required should either be set either in the user-facing function declaration or made inside of the user-facing function. This makes the code easier to reason through, and avoids confusion about the roles of each function.

No mutable defaults
===================

Functions should avoid having mutable default arguments, such as a dictionary. This is because **Python's default arguments are evaluated when the function is defined, not when the function is called**. If you need a default argument that is a mutable type, the correct way to do it is to set the default to ``None``, and then check for whether an argument was passed, constructing the default if necessary. Here is an example:

.. code-block:: python

   def foo(x: Dict[str, Any] | None = None):
       if x is None:
           x_ = {}
       else:
           x_ = x
       # use x_ in place of x

This also applies for default arguments that are based on a global variable that might change.



Pull requests
-------------

In order to be considered for review, a pull request must meet the following standards:

* Clean commit history, with clear commit messages
* Must pass CI (continuous integration), which is run automatically for each PR; same as the pre-commit hook
* Must have 100% unit test coverage, as reported automatically by `codecov`

Your code also must not break any examples in documentation, which is dynamically regenerated on every merge to main. We use sphinx ``doctest`` extension to ensure that all examples and provided outputs stay up to date. You can check documentation build status locally by executing the following in ``zetta_utils`` base folder:

.. code-block:: console

   $ cd docs
   $ make doctest
   $ make html

To maintain a clean commit history, we rebase, rather than merge (this may change in the future once we are seeing heavier development); if the ``main`` has changed and there are conflicts, you should rebase your pull request.

Please keep commits self-contained and of a reasonable size, with descriptive messages prefixed by things like ``fix:``, ``feat:``, ``chore:``. To revise the commit history, you can use ``git rebase -i HEAD~[N]`` where ``N`` is the number of previous commits that you wish to modify / squash.
