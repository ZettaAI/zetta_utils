``zetta_utils.log``
===================

Import the ``get_logger`` method from ``zetta_utils.log`` and use it to obtain a `logger object <https://docs.python.org/3/library/logging.html>`_. Then use this instead of ``print`` and ``warnings.warn`` statements.

.. code::

   >>> from zetta_utils.log import get_logger
   >>> logger = get_logger("test")
   >>> logger.warning("This is a warning")   // no output by default, as severity is less than ERROR
   >>> logger.exception(RuntimeError)
   2024-02-06 21:44:22.382 ERROR    test              <stdin>:   1
                                    <class 'RuntimeError'>
                                    NoneType: None
   >>> import logging
   >>> logger.setLevel(logging.INFO)
   >>> logger.warning("This is a warning")   // now outputs, since our logger level is set to INFO
   2024-02-06 21:46:28.439 WARNING  test              <stdin>:   1
                                    This is a warning

In addition to setting the log level of individual loggers, you can set the log level of the "zetta_utils" and "mazepa" loggers (and any subsequently created loggers) by calling the log.set_verbosity method:

.. code::

   >>> import zetta_utils.log
   >>> zetta_utils.log.setVerbosity('INFO')
   >>> zetta_utils.log.get_logger('mazepa').info("Hello mazepa code")
   2024-02-06 22:06:05.189 INFO     mazepa              <stdin>:   1
                                    Hello mazepa code

Note that when zetta_utils code is run via the command-line interface (i.e. ```zetta run``` command), verbosity is set via a command-line flag: ``-v``, ``-vv``, and ``-vvv``, correspond to ``WARNING``, ``INFO``, and ``DEBUG``, respectively. The default is ``INFO``.  (This differs from the normal default, which is ``ERROR``).

All logged messages will be sent by default to ``stderr``.
