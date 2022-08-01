===============
Developer Guide
===============

.. _dev-environment:

Dev Environment
---------------

This subsection describes ``zetta_utils`` development setup.

Dev Guidelines
--------------

This subsection describes development guidelines followed in ``zetta_utils``.
*Please read this subsection before contributing to the project.*

Coding Practices
~~~~~~~~~~~~~~~~
1. Use use `attrs decorator <https://www.attrs.org/en/stable/index.html>`_ when declaring new classes
   [`Example 1 <https://github.com/ZettaAI/zetta_utils/blob/main/zetta_utils/bbox.py>`_,
   `Example 2 <https://github.com/ZettaAI/zetta_utils/blob/main/zetta_utils/training/datasets/layer_dataset.py>`_].
   Prefer ``attrs`` over ``dataclass``.

2. Avoid excessive reliance on inheretance. Whenever possible, make your base classes
   `pure abstract <https://en.wikibooks.org/wiki/C%2B%2B_Programming/Classes/Abstract_Classes/Pure_Abstract_Classes>`_.
   Avoid multilevel inheritance.

3. Use annotations for all parameter, return and yield types.

4. Add dynamic type checking through the `@typeguard.typechecked <https://typeguard.readthedocs.io/en/latest/>`_ to all
   declared funcitons/classes, unless: (1) you're using Generics, which are currently `not supported by typeguard <https://github.com/agronholm/typeguard/issues/139>`_ (2) dynamic typechecking significantly slows down your code, eg a small funciton called many times in a loop.
   Do not manually check input types to raise exceptions.

5. When defining non-abstract classes, avoid using overly general names such as ``Dataset``, ``Scheduler``, ``Task``.
   There might be other kinds of datasets, schedulers or tasks that come along later making things awkward.
   Make non-abstract class names more specific to exact implementation, eg ``LayerDataset``, or invent a name for your
   method of doing things, eg ``MetroemDataset``.

Testing
~~~~~~~

1. ``zetta_utils`` strives for 100% unit test coverage. If portions of your code are trivial, use ``#pragma: no cover`` to indicate
   code that should not be considered for coverage, eg:

   .. code::

        def multiply(data: Tensor, x) -> Tensor:  # pragma: no cover
            return x * data
   ..

2. Use ``@pytest.mark.parametrize``, ``@pytest.fixture``, and ``mocker`` fixture for building unit tests. If you're unfamiliar with pytest,
   start off by readying this `guide <https://www.nerdwallet.com/blog/engineering/5-pytest-best-practices/>`_.

3. It's important to keep unit tests isolated. Use ``mocker`` decorator to spoof calls to external libraries and other parts of the codebase.

4. It's important to keep unit tests fast. Do not use large tensors, send HTTP requests or read from the file system. If a test reads from the
   file system, it is not a unit test.

5. Factoring your code into smaller functions will make it easier to construct test cases. Removing code duplication will make it easier
   to write tests.

6. If a part of your code is too difficult to test, consider whether you're using ``mocker`` to its full potential. If it's still too hard
   to test, consider refactoring.

Docstrings
~~~~~~~~~~

1. TODO


Pull Requests
~~~~~~~~~~~~~

To be merged in, your pull request has to sattisfy the CI, which will perform 4 checks.
First, your code has to satisfy ``pylint`` and ``mypy``.
Feel free to use ``# type`` and ``# pylint`` annotations when reasonable.
Make sure to have ``pre-commit`` setup as described in the :ref:`dev-environment`, which will make sure that
``pylint`` and ``mypy`` are sattisfied before letting you commit.

Next, your code has to pass all unit tests, and 100% of the code that's not explicitly marked
with ``# pragma: no cover`` must be covered by the tests. You can run the unit tests locally by executing
the following in ``zetta_utils`` base folder:

.. code-block:: bash

       pytest test/unit


Your code also must not break any examples in documentation, which is dynamically regenerated on every merge to main.
We use sphinx ``doctest`` extension to ensure that all examples and provided outputs stay up to date.
You can check documentation build status locally by executing the following in ``zetta_utils`` base folder:


.. code-block:: bash

        cd docs
        make doctest
        make html

Until ``zetta_utils`` library matures and sees heavy development, we will use ``suqash mode`` for merging PRs to master.
So for now, you don't have to worry about the commit history when forming a PR.

