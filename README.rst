***********************
aas-core3.0-micropython
***********************

.. image:: https://github.com/aas-core-works/aas-core3.0-micropython/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/aas-core-works/aas-core3.0-micropython/actions/workflows/ci.yml
    :alt: Continuous integration

Manipulate, verify and de/serialize asset administration shells in Micropython. 

This is a semantically patched version of the `aas-core3.0-python`_ SDK so that it can run in the Micropython environment.

.. _aas-core3.0-python: https://github.com/aas-core-works/aas-core3.0-python

We continuously patch the original Python SDK, so that the version between the two code bases correspond.

Differences to the original aas-core3.0-python SDK
==================================================
Micropython supports only a subset of the CPython standard library.
This also constraints what we can implement in the SDK.
Due to the limitations we had to **exclude**:

* **XML de/serialization**, as there is no mature XML library in Micropython, and
* **Verification**, as the regular expression module in Micropython lacks counted repetitions and does not work on escaped characters.

Versioning
==========
We follow the versioning of the original SDK that we patched.
