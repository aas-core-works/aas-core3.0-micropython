#!/usr/bin/env micropython

import json
import os
import sys

# NOTE (mristin, 2024-04-5):
# os.getcwd() is not available on micropython 1.17 available on Ubuntu 22.04.
sys.path.insert(0, os.getenv("AAS_CORE3_MICROPYTHON_REPO"))

import aas_core3.types as aas_types
import aas_core3.jsonization as aas_jsonization


def test_create_get_set():
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


def test_descend_and_descend_once():
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


class Visitor(aas_types.PassThroughVisitor):
    def visit_property(self, that: aas_types.Property):
        if "another" in that.id_short:
            print(that.id_short)


def test_visitor():
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


def test_jsonization_serialize():
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


def test_jsonization_deserialize():
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


def test_xmlization_serialize():
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


if __name__ == "__main__":
    test_create_get_set()
    test_descend_and_descend_once()
    test_visitor()
    test_jsonization_serialize()
    test_jsonization_deserialize()
    test_jsonization_serialize()
    test_xmlization_serialize()
    