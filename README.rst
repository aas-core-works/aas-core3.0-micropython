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

* **XML de-serialization**, as there is no mature XML parser in Micropython, and
* **Verification**, as the regular expression module in Micropython lacks counted repetitions and does not work on escaped characters.

The **XML serialization**, however, is included as the original library directly writes to a text stream, without dependence on an external library.

Versioning
==========
We follow the versioning of the original SDK that we patched.

Installation
============
We provide a [package.json](package.json), so you can directly install using `mpremote` from this repository:

.. code-block::

   mpremote mip install github:aas-core-works/aas-core3.0-micropython

... or using `mip`:

.. code-block::

   micropython -m mip install github:aas-core-works/aas-core3.0-micropython

Getting Started
===============
We document here a couple of code snippets so that you can quickly get started working with the library.

Please refer to `the original documentation of aas-core3.0-python`_ for more context and detailed information.

.. the original documentation of aas-core3.0-python: https://github.com/aas-core-works/aas-core3.0-python

Create, Get and Set Properties of an AAS Model
----------------------------------------------

.. code-block::

    import aas_core3.types as aas_types

    # Create the first element
    some_element = aas_types.Property(
        id_short="some_property", value_type=aas_types.DataTypeDefXSD.INT, value="1984"
    )

    # Create the second element
    another_element = aas_types.Blob(
        id_short="some_blob",
        content_type="application/octet-stream",
        value=b"\xDE\xAD\xBE\xEF",
    )

    # You can directly access the element properties.
    another_element.value = b"\xDE\xAD\xC0\xDE"

    # Nest the elements in a submodel
    submodel = aas_types.Submodel(
        id="some-unique-global-identifier",
        submodel_elements=[some_element, another_element],
    )

    # Now create the environment to wrap it all up
    environment = aas_types.Environment(submodels=[submodel])

    # You can access the properties from the children as well.
    environment.submodels[0].submodel_elements[1].value = b"\xC0\x01\xCA\xFE"

    # Now you can do something with the environment...

Iterate and Transform
---------------------
``descend`` and ``descend_once``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    import aas_core3.types as aas_types

    # Prepare the environment
    environment = aas_types.Environment(
        submodels=[
            aas_types.Submodel(
                id="some-unique-global-identifier",
                submodel_elements=[
                    aas_types.Property(
                        id_short="some_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1984",
                    ),
                    aas_types.Property(
                        id_short="another_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1985",
                    ),
                    aas_types.Property(
                        id_short="yet_another_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1986",
                    ),
                ],
            )
        ]
    )

    for something in environment.descend():
        if (
            isinstance(something, aas_types.Property)
            and "another" in something.id_short
        ):
            print(something.id_short)

    # Prints:
    # another_property
    # yet_another_property

Visitor
^^^^^^^

.. code-block::

    import aas_core3.types as aas_types

    class Visitor(aas_types.PassThroughVisitor):
    def visit_property(self, that: aas_types.Property):
        if "another" in that.id_short:
            print(that.id_short)

    # Prepare the environment
    environment = aas_types.Environment(
        submodels=[
            aas_types.Submodel(
                id="some-unique-global-identifier",
                submodel_elements=[
                    aas_types.Property(
                        id_short="some_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1984",
                    ),
                    aas_types.Property(
                        id_short="another_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1985",
                    ),
                    aas_types.Property(
                        id_short="yet_another_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1986",
                    ),
                ],
            )
        ]
    )

    # Iterate
    visitor = Visitor()
    visitor.visit(environment)

    # Prints
    # another_property
    # yet_another_property

JSON Serialization
------------------

.. code-block::

    import aas_core3.types as aas_types
    import aas_core3.jsonization as aas_jsonization
    
    # Prepare the environment
    environment = aas_types.Environment(
        submodels=[
            aas_types.Submodel(
                id="some-unique-global-identifier",
                submodel_elements=[
                    aas_types.Property(
                        id_short="some_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1984",
                    )
                ],
            )
        ]
    )

    # Serialize to a JSON-able mapping
    jsonable = aas_jsonization.to_jsonable(environment)

    # Print the mapping as text
    print(json.dumps(jsonable))

    # Prints (as a continuous string without newlines and indention)
    # {
    #   "submodels": [
    #     {
    #       "id": "some-unique-global-identifier",
    #       "submodelElements": [
    #         {
    #           "idShort": "some_property",
    #           "valueType": "xs:int",
    #           "value": "1984",
    #           "modelType": "Property"
    #         }
    #       ],
    #       "modelType": "Submodel"
    #     }
    #   ]
    # }

JSON De-serialization
---------------------

.. code-block::

    import aas_core3.types as aas_types
    import aas_core3.jsonization as aas_jsonization

	    text = """\
        {
          "submodels": [
            {
              "id": "some-unique-global-identifier",
              "submodelElements": [
                {
                  "idShort": "someProperty",
                  "valueType": "xs:boolean",
                  "modelType": "Property"
                }
              ],
              "modelType": "Submodel"
            }
          ]
        }"""

    jsonable = json.loads(text)

    environment = aas_jsonization.environment_from_jsonable(jsonable)

    for something in environment.descend():
        print(type(something))

    # Prints
    # <class 'aas_core3.types.Submodel'>
	# <class 'aas_core3.types.Property'>

XML Serialization
-----------------

.. code-block::

    import aas_core3.types as aas_types
    import aas_core3.xmlization as aas_xmlization

    # Prepare the environment
    environment = aas_types.Environment(
        submodels=[
            aas_types.Submodel(
                id="some-unique-global-identifier",
                submodel_elements=[
                    aas_types.Property(
                        id_short="some_property",
                        value_type=aas_types.DataTypeDefXSD.INT,
                        value="1984",
                    )
                ],
            )
        ]
    )

    # Serialize to an XML-encoded string
    text = aas_xmlization.to_str(environment)

    print(text)

    # Prints (without the newlines and indention)
    # <environment xmlns="https://admin-shell.io/aas/3/0">
    #   <submodels>
    #     <submodel>
    #       <id>some-unique-global-identifier</id>
    #       <submodelElements>
    #         <property>
    #           <idShort>some_property</idShort>
    #           <valueType>xs:int</valueType>
    #           <value>1984</value>
    #         </property>
    #       </submodelElements>
    #     </submodel>
    #   </submodels>
    # </environment>

XML De-serialization
--------------------
As we noted above, there is no mature XML library for Micropython so we could not adapt the original code.
