``zetta_utils.log``
===================

Import ``logger`` object from ``zetta_utils.log`` and use it instead of ``print`` and ``warnings.warn`` statements.

.. code::

   >>> from zetta_utils.log import logger
   >>> logger.warn("This is a warning")
   [08-01 05:30:03.924, pid 24380, {YOUR FILE NAME}>:   {LINE NUMBER}] WARNING - This is a warning
   >>> logger.info("Info message")
   [08-01 05:30:03.924, pid 24380, {YOUR FILE NAME}>:   {LINE NUMBER}]    INFO - Info message
   >>> logger.exception(RuntimeError)
   [08-01 05:30:03.924, pid 24380, {YOUR FILE NAME}>:   {LINE NUMBER}]   ERROR - <class 'RuntimeError'>

All logged messages will be sent to ``stderr`` and also written to a file in ``/tmp/logs/zetta_logs/zetta.log.{timestamp}.yaml``.
Although the files are not in YAML format, ``.yaml`` extension highlights strings and integers, which is nice.
