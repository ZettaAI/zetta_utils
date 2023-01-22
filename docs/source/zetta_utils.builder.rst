``zetta_utils.builder``
=======================

Overview
--------

``zu.builder`` provides machinery to represent layers, datasets, or any other registered components
as dictionaries. This can be used to pass in flexible parameters to CLI tools and to allow flexible,
readable specifications of training and inference workflow through ``json``/``yaml``/``cue`` fields.

Use ``builder.register()`` to make a callable accessibule with ``zu.builder``:

.. doctest::

   >>> from zetta_utils import builder
   >>> @builder.register("MyClass")
   ... class MyClass:
   ...    def __init__(self, a):
   ...       self.a = a

After a callable is registered, you can represent them as dictionaries by including the matching ``@type`` key
and providing the initialization parameters:

.. doctest::

   >>> spec = {
   ...    "@type": "MyClass",
   ...    "a": 100
   ... }
   >>> obj = builder.build(spec)
   >>> print (type(obj))
   <class 'MyClass'>
   >>> print (obj.a)
   100

All user-facing ``zetta_utils`` objects are all registered with ``zu.builder``. You can check out the state of the current registry
by inspecting ``zu.builder.REGISTRY``

API reference
-------------

.. autofunction:: zetta_utils.builder.register

.. autofunction:: zetta_utils.builder.build

.. autofunction:: zetta_utils.builder.get_callable_from_name
