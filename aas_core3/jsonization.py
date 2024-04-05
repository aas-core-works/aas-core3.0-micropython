import binascii

import sys


import aas_core3.common as aas_common
import aas_core3.stringification as aas_stringification
import aas_core3.types as aas_types


class PropertySegment:

    def __init__(self, instance, name):

        self.instance = instance
        self.name = name


class IndexSegment:

    def __init__(self, container, index):

        self.container = container
        self.index = index


class Path:

    def __init__(self):

        self._segments = []

    @property
    def segments(self):

        return self._segments

    def _prepend(self, segment):

        self._segments.insert(0, segment)

    def __str__(self):
        if len(self._segments) == 0:
            return ""

        parts = []

        iterator = iter(self._segments)
        first = next(iterator)
        if isinstance(first, PropertySegment):
            parts.append(f"{first.name}")
        elif isinstance(first, IndexSegment):
            parts.append(f"[{first.index}]")
        else:
            aas_common.assert_never(first)

        for segment in iterator:
            if isinstance(segment, PropertySegment):
                parts.append(f".{segment.name}")
            elif isinstance(segment, IndexSegment):
                parts.append(f"[{segment.index}]")
            else:
                aas_common.assert_never(segment)

        return "".join(parts)


class DeserializationException(Exception):

    def __init__(self, cause):

        self.cause = cause
        self.path = Path()


def _bool_from_jsonable(jsonable):

    if not isinstance(jsonable, bool):
        raise DeserializationException(f"Expected a bool, but got: {type(jsonable)}")
    return jsonable


def _int_from_jsonable(jsonable):

    if not isinstance(jsonable, int):
        raise DeserializationException(f"Expected an int, but got: {type(jsonable)}")
    return jsonable


def _float_from_jsonable(jsonable):

    if not isinstance(jsonable, float):
        raise DeserializationException(f"Expected a float, but got: {type(jsonable)}")
    return jsonable


def _str_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException(f"Expected a str, but got: {type(jsonable)}")
    return jsonable


def _bytes_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException(f"Expected a str, but got: {type(jsonable)}")

    return binascii.a2b_base64(jsonable.encode("ascii"))


def has_semantics_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _HAS_SEMANTICS_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for HasSemantics: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForExtension:

    def __init__(self):

        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.name = None
        self.value_type = None
        self.value = None
        self.refers_to = None

    def ignore(self, jsonable):

        pass

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_name_from_jsonable(self, jsonable):

        self.name = _str_from_jsonable(jsonable)

    def set_value_type_from_jsonable(self, jsonable):

        self.value_type = data_type_def_xsd_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_refers_to_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.refers_to = items


def extension_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForExtension()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_EXTENSION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.name is None:
        raise DeserializationException("The required property 'name' is missing")

    return aas_types.Extension(
        setter.name,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.value_type,
        setter.value,
        setter.refers_to,
    )


def has_extensions_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _HAS_EXTENSIONS_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for HasExtensions: {model_type}"
        )

    return dispatch(jsonable)


def referable_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _REFERABLE_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for Referable: {model_type}"
        )

    return dispatch(jsonable)


def identifiable_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _IDENTIFIABLE_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for Identifiable: {model_type}"
        )

    return dispatch(jsonable)


def modelling_kind_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.modelling_kind_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of ModellingKind: {jsonable}"
        )

    return literal


def has_kind_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _HAS_KIND_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for HasKind: {model_type}"
        )

    return dispatch(jsonable)


def has_data_specification_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _HAS_DATA_SPECIFICATION_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for HasDataSpecification: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForAdministrativeInformation:

    def __init__(self):

        self.embedded_data_specifications = None
        self.version = None
        self.revision = None
        self.creator = None
        self.template_id = None

    def ignore(self, jsonable):

        pass

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_version_from_jsonable(self, jsonable):

        self.version = _str_from_jsonable(jsonable)

    def set_revision_from_jsonable(self, jsonable):

        self.revision = _str_from_jsonable(jsonable)

    def set_creator_from_jsonable(self, jsonable):

        self.creator = reference_from_jsonable(jsonable)

    def set_template_id_from_jsonable(self, jsonable):

        self.template_id = _str_from_jsonable(jsonable)


def administrative_information_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForAdministrativeInformation()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_ADMINISTRATIVE_INFORMATION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.AdministrativeInformation(
        setter.embedded_data_specifications,
        setter.version,
        setter.revision,
        setter.creator,
        setter.template_id,
    )


def qualifiable_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _QUALIFIABLE_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for Qualifiable: {model_type}"
        )

    return dispatch(jsonable)


def qualifier_kind_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.qualifier_kind_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of QualifierKind: {jsonable}"
        )

    return literal


class _SetterForQualifier:

    def __init__(self):

        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.kind = None
        self.type = None
        self.value_type = None
        self.value = None
        self.value_id = None

    def ignore(self, jsonable):

        pass

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_kind_from_jsonable(self, jsonable):

        self.kind = qualifier_kind_from_jsonable(jsonable)

    def set_type_from_jsonable(self, jsonable):

        self.type = _str_from_jsonable(jsonable)

    def set_value_type_from_jsonable(self, jsonable):

        self.value_type = data_type_def_xsd_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_value_id_from_jsonable(self, jsonable):

        self.value_id = reference_from_jsonable(jsonable)


def qualifier_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForQualifier()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_QUALIFIER.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.type is None:
        raise DeserializationException("The required property 'type' is missing")

    if setter.value_type is None:
        raise DeserializationException("The required property 'valueType' is missing")

    return aas_types.Qualifier(
        setter.type,
        setter.value_type,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.kind,
        setter.value,
        setter.value_id,
    )


class _SetterForAssetAdministrationShell:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.administration = None
        self.id = None
        self.embedded_data_specifications = None
        self.derived_from = None
        self.asset_information = None
        self.submodels = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_administration_from_jsonable(self, jsonable):

        self.administration = administrative_information_from_jsonable(jsonable)

    def set_id_from_jsonable(self, jsonable):

        self.id = _str_from_jsonable(jsonable)

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_derived_from_from_jsonable(self, jsonable):

        self.derived_from = reference_from_jsonable(jsonable)

    def set_asset_information_from_jsonable(self, jsonable):

        self.asset_information = asset_information_from_jsonable(jsonable)

    def set_submodels_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.submodels = items


def asset_administration_shell_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForAssetAdministrationShell()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_ASSET_ADMINISTRATION_SHELL.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.id is None:
        raise DeserializationException("The required property 'id' is missing")

    if setter.asset_information is None:
        raise DeserializationException(
            "The required property 'assetInformation' is missing"
        )

    return aas_types.AssetAdministrationShell(
        setter.id,
        setter.asset_information,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.administration,
        setter.embedded_data_specifications,
        setter.derived_from,
        setter.submodels,
    )


class _SetterForAssetInformation:

    def __init__(self):

        self.asset_kind = None
        self.global_asset_id = None
        self.specific_asset_ids = None
        self.asset_type = None
        self.default_thumbnail = None

    def ignore(self, jsonable):

        pass

    def set_asset_kind_from_jsonable(self, jsonable):

        self.asset_kind = asset_kind_from_jsonable(jsonable)

    def set_global_asset_id_from_jsonable(self, jsonable):

        self.global_asset_id = _str_from_jsonable(jsonable)

    def set_specific_asset_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = specific_asset_id_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.specific_asset_ids = items

    def set_asset_type_from_jsonable(self, jsonable):

        self.asset_type = _str_from_jsonable(jsonable)

    def set_default_thumbnail_from_jsonable(self, jsonable):

        self.default_thumbnail = resource_from_jsonable(jsonable)


def asset_information_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForAssetInformation()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_ASSET_INFORMATION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.asset_kind is None:
        raise DeserializationException("The required property 'assetKind' is missing")

    return aas_types.AssetInformation(
        setter.asset_kind,
        setter.global_asset_id,
        setter.specific_asset_ids,
        setter.asset_type,
        setter.default_thumbnail,
    )


class _SetterForResource:

    def __init__(self):

        self.path = None
        self.content_type = None

    def ignore(self, jsonable):

        pass

    def set_path_from_jsonable(self, jsonable):

        self.path = _str_from_jsonable(jsonable)

    def set_content_type_from_jsonable(self, jsonable):

        self.content_type = _str_from_jsonable(jsonable)


def resource_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForResource()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_RESOURCE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.path is None:
        raise DeserializationException("The required property 'path' is missing")

    return aas_types.Resource(setter.path, setter.content_type)


def asset_kind_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.asset_kind_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of AssetKind: {jsonable}"
        )

    return literal


class _SetterForSpecificAssetID:

    def __init__(self):

        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.name = None
        self.value = None
        self.external_subject_id = None

    def ignore(self, jsonable):

        pass

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_name_from_jsonable(self, jsonable):

        self.name = _str_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_external_subject_id_from_jsonable(self, jsonable):

        self.external_subject_id = reference_from_jsonable(jsonable)


def specific_asset_id_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForSpecificAssetID()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_SPECIFIC_ASSET_ID.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.name is None:
        raise DeserializationException("The required property 'name' is missing")

    if setter.value is None:
        raise DeserializationException("The required property 'value' is missing")

    return aas_types.SpecificAssetID(
        setter.name,
        setter.value,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.external_subject_id,
    )


class _SetterForSubmodel:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.administration = None
        self.id = None
        self.kind = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.submodel_elements = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_administration_from_jsonable(self, jsonable):

        self.administration = administrative_information_from_jsonable(jsonable)

    def set_id_from_jsonable(self, jsonable):

        self.id = _str_from_jsonable(jsonable)

    def set_kind_from_jsonable(self, jsonable):

        self.kind = modelling_kind_from_jsonable(jsonable)

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_submodel_elements_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = submodel_element_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.submodel_elements = items


def submodel_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForSubmodel()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_SUBMODEL.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.id is None:
        raise DeserializationException("The required property 'id' is missing")

    return aas_types.Submodel(
        setter.id,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.administration,
        setter.kind,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.submodel_elements,
    )


def submodel_element_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _SUBMODEL_ELEMENT_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for SubmodelElement: {model_type}"
        )

    return dispatch(jsonable)


def relationship_element_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _RELATIONSHIP_ELEMENT_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for RelationshipElement: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForRelationshipElement:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.first = None
        self.second = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_first_from_jsonable(self, jsonable):

        self.first = reference_from_jsonable(jsonable)

    def set_second_from_jsonable(self, jsonable):

        self.second = reference_from_jsonable(jsonable)


def _relationship_element_from_jsonable_without_dispatch(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForRelationshipElement()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_RELATIONSHIP_ELEMENT.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.first is None:
        raise DeserializationException("The required property 'first' is missing")

    if setter.second is None:
        raise DeserializationException("The required property 'second' is missing")

    return aas_types.RelationshipElement(
        setter.first,
        setter.second,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
    )


def aas_submodel_elements_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.aas_submodel_elements_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of AASSubmodelElements: {jsonable}"
        )

    return literal


class _SetterForSubmodelElementList:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.order_relevant = None
        self.semantic_id_list_element = None
        self.type_value_list_element = None
        self.value_type_list_element = None
        self.value = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_order_relevant_from_jsonable(self, jsonable):

        self.order_relevant = _bool_from_jsonable(jsonable)

    def set_semantic_id_list_element_from_jsonable(self, jsonable):

        self.semantic_id_list_element = reference_from_jsonable(jsonable)

    def set_type_value_list_element_from_jsonable(self, jsonable):

        self.type_value_list_element = aas_submodel_elements_from_jsonable(jsonable)

    def set_value_type_list_element_from_jsonable(self, jsonable):

        self.value_type_list_element = data_type_def_xsd_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = submodel_element_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.value = items


def submodel_element_list_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForSubmodelElementList()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_SUBMODEL_ELEMENT_LIST.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.type_value_list_element is None:
        raise DeserializationException(
            "The required property 'typeValueListElement' is missing"
        )

    return aas_types.SubmodelElementList(
        setter.type_value_list_element,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.order_relevant,
        setter.semantic_id_list_element,
        setter.value_type_list_element,
        setter.value,
    )


class _SetterForSubmodelElementCollection:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = submodel_element_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.value = items


def submodel_element_collection_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForSubmodelElementCollection()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_SUBMODEL_ELEMENT_COLLECTION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.SubmodelElementCollection(
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.value,
    )


def data_element_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _DATA_ELEMENT_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for DataElement: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForProperty:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value_type = None
        self.value = None
        self.value_id = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_type_from_jsonable(self, jsonable):

        self.value_type = data_type_def_xsd_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_value_id_from_jsonable(self, jsonable):

        self.value_id = reference_from_jsonable(jsonable)


def property_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForProperty()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_PROPERTY.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.value_type is None:
        raise DeserializationException("The required property 'valueType' is missing")

    return aas_types.Property(
        setter.value_type,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.value,
        setter.value_id,
    )


class _SetterForMultiLanguageProperty:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value = None
        self.value_id = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.value = items

    def set_value_id_from_jsonable(self, jsonable):

        self.value_id = reference_from_jsonable(jsonable)


def multi_language_property_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForMultiLanguageProperty()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_MULTI_LANGUAGE_PROPERTY.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.MultiLanguageProperty(
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.value,
        setter.value_id,
    )


class _SetterForRange:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value_type = None
        self.min = None
        self.max = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_type_from_jsonable(self, jsonable):

        self.value_type = data_type_def_xsd_from_jsonable(jsonable)

    def set_min_from_jsonable(self, jsonable):

        self.min = _str_from_jsonable(jsonable)

    def set_max_from_jsonable(self, jsonable):

        self.max = _str_from_jsonable(jsonable)


def range_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForRange()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_RANGE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.value_type is None:
        raise DeserializationException("The required property 'valueType' is missing")

    return aas_types.Range(
        setter.value_type,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.min,
        setter.max,
    )


class _SetterForReferenceElement:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_from_jsonable(self, jsonable):

        self.value = reference_from_jsonable(jsonable)


def reference_element_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForReferenceElement()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_REFERENCE_ELEMENT.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.ReferenceElement(
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.value,
    )


class _SetterForBlob:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value = None
        self.content_type = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_from_jsonable(self, jsonable):

        self.value = _bytes_from_jsonable(jsonable)

    def set_content_type_from_jsonable(self, jsonable):

        self.content_type = _str_from_jsonable(jsonable)


def blob_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForBlob()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_BLOB.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.content_type is None:
        raise DeserializationException("The required property 'contentType' is missing")

    return aas_types.Blob(
        setter.content_type,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.value,
    )


class _SetterForFile:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.value = None
        self.content_type = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_content_type_from_jsonable(self, jsonable):

        self.content_type = _str_from_jsonable(jsonable)


def file_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForFile()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_FILE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.content_type is None:
        raise DeserializationException("The required property 'contentType' is missing")

    return aas_types.File(
        setter.content_type,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.value,
    )


class _SetterForAnnotatedRelationshipElement:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.first = None
        self.second = None
        self.annotations = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_first_from_jsonable(self, jsonable):

        self.first = reference_from_jsonable(jsonable)

    def set_second_from_jsonable(self, jsonable):

        self.second = reference_from_jsonable(jsonable)

    def set_annotations_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = data_element_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.annotations = items


def annotated_relationship_element_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForAnnotatedRelationshipElement()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_ANNOTATED_RELATIONSHIP_ELEMENT.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.first is None:
        raise DeserializationException("The required property 'first' is missing")

    if setter.second is None:
        raise DeserializationException("The required property 'second' is missing")

    return aas_types.AnnotatedRelationshipElement(
        setter.first,
        setter.second,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.annotations,
    )


class _SetterForEntity:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.statements = None
        self.entity_type = None
        self.global_asset_id = None
        self.specific_asset_ids = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_statements_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = submodel_element_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.statements = items

    def set_entity_type_from_jsonable(self, jsonable):

        self.entity_type = entity_type_from_jsonable(jsonable)

    def set_global_asset_id_from_jsonable(self, jsonable):

        self.global_asset_id = _str_from_jsonable(jsonable)

    def set_specific_asset_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = specific_asset_id_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.specific_asset_ids = items


def entity_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForEntity()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_ENTITY.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.entity_type is None:
        raise DeserializationException("The required property 'entityType' is missing")

    return aas_types.Entity(
        setter.entity_type,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.statements,
        setter.global_asset_id,
        setter.specific_asset_ids,
    )


def entity_type_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.entity_type_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of EntityType: {jsonable}"
        )

    return literal


def direction_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.direction_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of Direction: {jsonable}"
        )

    return literal


def state_of_event_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.state_of_event_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of StateOfEvent: {jsonable}"
        )

    return literal


class _SetterForEventPayload:

    def __init__(self):

        self.source = None
        self.source_semantic_id = None
        self.observable_reference = None
        self.observable_semantic_id = None
        self.topic = None
        self.subject_id = None
        self.time_stamp = None
        self.payload = None

    def ignore(self, jsonable):

        pass

    def set_source_from_jsonable(self, jsonable):

        self.source = reference_from_jsonable(jsonable)

    def set_source_semantic_id_from_jsonable(self, jsonable):

        self.source_semantic_id = reference_from_jsonable(jsonable)

    def set_observable_reference_from_jsonable(self, jsonable):

        self.observable_reference = reference_from_jsonable(jsonable)

    def set_observable_semantic_id_from_jsonable(self, jsonable):

        self.observable_semantic_id = reference_from_jsonable(jsonable)

    def set_topic_from_jsonable(self, jsonable):

        self.topic = _str_from_jsonable(jsonable)

    def set_subject_id_from_jsonable(self, jsonable):

        self.subject_id = reference_from_jsonable(jsonable)

    def set_time_stamp_from_jsonable(self, jsonable):

        self.time_stamp = _str_from_jsonable(jsonable)

    def set_payload_from_jsonable(self, jsonable):

        self.payload = _bytes_from_jsonable(jsonable)


def event_payload_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForEventPayload()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_EVENT_PAYLOAD.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.source is None:
        raise DeserializationException("The required property 'source' is missing")

    if setter.observable_reference is None:
        raise DeserializationException(
            "The required property 'observableReference' is missing"
        )

    if setter.time_stamp is None:
        raise DeserializationException("The required property 'timeStamp' is missing")

    return aas_types.EventPayload(
        setter.source,
        setter.observable_reference,
        setter.time_stamp,
        setter.source_semantic_id,
        setter.observable_semantic_id,
        setter.topic,
        setter.subject_id,
        setter.payload,
    )


def event_element_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _EVENT_ELEMENT_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for EventElement: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForBasicEventElement:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.observed = None
        self.direction = None
        self.state = None
        self.message_topic = None
        self.message_broker = None
        self.last_update = None
        self.min_interval = None
        self.max_interval = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_observed_from_jsonable(self, jsonable):

        self.observed = reference_from_jsonable(jsonable)

    def set_direction_from_jsonable(self, jsonable):

        self.direction = direction_from_jsonable(jsonable)

    def set_state_from_jsonable(self, jsonable):

        self.state = state_of_event_from_jsonable(jsonable)

    def set_message_topic_from_jsonable(self, jsonable):

        self.message_topic = _str_from_jsonable(jsonable)

    def set_message_broker_from_jsonable(self, jsonable):

        self.message_broker = reference_from_jsonable(jsonable)

    def set_last_update_from_jsonable(self, jsonable):

        self.last_update = _str_from_jsonable(jsonable)

    def set_min_interval_from_jsonable(self, jsonable):

        self.min_interval = _str_from_jsonable(jsonable)

    def set_max_interval_from_jsonable(self, jsonable):

        self.max_interval = _str_from_jsonable(jsonable)


def basic_event_element_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForBasicEventElement()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_BASIC_EVENT_ELEMENT.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.observed is None:
        raise DeserializationException("The required property 'observed' is missing")

    if setter.direction is None:
        raise DeserializationException("The required property 'direction' is missing")

    if setter.state is None:
        raise DeserializationException("The required property 'state' is missing")

    return aas_types.BasicEventElement(
        setter.observed,
        setter.direction,
        setter.state,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.message_topic,
        setter.message_broker,
        setter.last_update,
        setter.min_interval,
        setter.max_interval,
    )


class _SetterForOperation:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None
        self.input_variables = None
        self.output_variables = None
        self.inoutput_variables = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_input_variables_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = operation_variable_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.input_variables = items

    def set_output_variables_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = operation_variable_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.output_variables = items

    def set_inoutput_variables_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = operation_variable_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.inoutput_variables = items


def operation_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForOperation()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_OPERATION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.Operation(
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
        setter.input_variables,
        setter.output_variables,
        setter.inoutput_variables,
    )


class _SetterForOperationVariable:

    def __init__(self):

        self.value = None

    def ignore(self, jsonable):

        pass

    def set_value_from_jsonable(self, jsonable):

        self.value = submodel_element_from_jsonable(jsonable)


def operation_variable_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForOperationVariable()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_OPERATION_VARIABLE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.value is None:
        raise DeserializationException("The required property 'value' is missing")

    return aas_types.OperationVariable(setter.value)


class _SetterForCapability:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.semantic_id = None
        self.supplemental_semantic_ids = None
        self.qualifiers = None
        self.embedded_data_specifications = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_semantic_id_from_jsonable(self, jsonable):

        self.semantic_id = reference_from_jsonable(jsonable)

    def set_supplemental_semantic_ids_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.supplemental_semantic_ids = items

    def set_qualifiers_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = qualifier_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.qualifiers = items

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items


def capability_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForCapability()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_CAPABILITY.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.Capability(
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.semantic_id,
        setter.supplemental_semantic_ids,
        setter.qualifiers,
        setter.embedded_data_specifications,
    )


class _SetterForConceptDescription:

    def __init__(self):

        self.extensions = None
        self.category = None
        self.id_short = None
        self.display_name = None
        self.description = None
        self.administration = None
        self.id = None
        self.embedded_data_specifications = None
        self.is_case_of = None

    def ignore(self, jsonable):

        pass

    def set_extensions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = extension_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.extensions = items

    def set_category_from_jsonable(self, jsonable):

        self.category = _str_from_jsonable(jsonable)

    def set_id_short_from_jsonable(self, jsonable):

        self.id_short = _str_from_jsonable(jsonable)

    def set_display_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_name_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.display_name = items

    def set_description_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_text_type_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.description = items

    def set_administration_from_jsonable(self, jsonable):

        self.administration = administrative_information_from_jsonable(jsonable)

    def set_id_from_jsonable(self, jsonable):

        self.id = _str_from_jsonable(jsonable)

    def set_embedded_data_specifications_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = embedded_data_specification_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.embedded_data_specifications = items

    def set_is_case_of_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = reference_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.is_case_of = items


def concept_description_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForConceptDescription()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_CONCEPT_DESCRIPTION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.id is None:
        raise DeserializationException("The required property 'id' is missing")

    return aas_types.ConceptDescription(
        setter.id,
        setter.extensions,
        setter.category,
        setter.id_short,
        setter.display_name,
        setter.description,
        setter.administration,
        setter.embedded_data_specifications,
        setter.is_case_of,
    )


def reference_types_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.reference_types_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of ReferenceTypes: {jsonable}"
        )

    return literal


class _SetterForReference:

    def __init__(self):

        self.type = None
        self.referred_semantic_id = None
        self.keys = None

    def ignore(self, jsonable):

        pass

    def set_type_from_jsonable(self, jsonable):

        self.type = reference_types_from_jsonable(jsonable)

    def set_referred_semantic_id_from_jsonable(self, jsonable):

        self.referred_semantic_id = reference_from_jsonable(jsonable)

    def set_keys_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = key_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.keys = items


def reference_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForReference()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_REFERENCE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.type is None:
        raise DeserializationException("The required property 'type' is missing")

    if setter.keys is None:
        raise DeserializationException("The required property 'keys' is missing")

    return aas_types.Reference(setter.type, setter.keys, setter.referred_semantic_id)


class _SetterForKey:

    def __init__(self):

        self.type = None
        self.value = None

    def ignore(self, jsonable):

        pass

    def set_type_from_jsonable(self, jsonable):

        self.type = key_types_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)


def key_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForKey()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_KEY.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.type is None:
        raise DeserializationException("The required property 'type' is missing")

    if setter.value is None:
        raise DeserializationException("The required property 'value' is missing")

    return aas_types.Key(setter.type, setter.value)


def key_types_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.key_types_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of KeyTypes: {jsonable}"
        )

    return literal


def data_type_def_xsd_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.data_type_def_xsd_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of DataTypeDefXSD: {jsonable}"
        )

    return literal


def abstract_lang_string_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _ABSTRACT_LANG_STRING_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for AbstractLangString: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForLangStringNameType:

    def __init__(self):

        self.language = None
        self.text = None

    def ignore(self, jsonable):

        pass

    def set_language_from_jsonable(self, jsonable):

        self.language = _str_from_jsonable(jsonable)

    def set_text_from_jsonable(self, jsonable):

        self.text = _str_from_jsonable(jsonable)


def lang_string_name_type_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForLangStringNameType()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_LANG_STRING_NAME_TYPE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.language is None:
        raise DeserializationException("The required property 'language' is missing")

    if setter.text is None:
        raise DeserializationException("The required property 'text' is missing")

    return aas_types.LangStringNameType(setter.language, setter.text)


class _SetterForLangStringTextType:

    def __init__(self):

        self.language = None
        self.text = None

    def ignore(self, jsonable):

        pass

    def set_language_from_jsonable(self, jsonable):

        self.language = _str_from_jsonable(jsonable)

    def set_text_from_jsonable(self, jsonable):

        self.text = _str_from_jsonable(jsonable)


def lang_string_text_type_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForLangStringTextType()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_LANG_STRING_TEXT_TYPE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.language is None:
        raise DeserializationException("The required property 'language' is missing")

    if setter.text is None:
        raise DeserializationException("The required property 'text' is missing")

    return aas_types.LangStringTextType(setter.language, setter.text)


class _SetterForEnvironment:

    def __init__(self):

        self.asset_administration_shells = None
        self.submodels = None
        self.concept_descriptions = None

    def ignore(self, jsonable):

        pass

    def set_asset_administration_shells_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = asset_administration_shell_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.asset_administration_shells = items

    def set_submodels_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = submodel_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.submodels = items

    def set_concept_descriptions_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = concept_description_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.concept_descriptions = items


def environment_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForEnvironment()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_ENVIRONMENT.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    return aas_types.Environment(
        setter.asset_administration_shells,
        setter.submodels,
        setter.concept_descriptions,
    )


def data_specification_content_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    model_type = jsonable.get("modelType", None)
    if model_type is None:
        raise DeserializationException(
            "Expected the property modelType, but found none"
        )

    if not isinstance(model_type, str):
        raise DeserializationException(
            "Expected the property modelType to be a str, but got: {type(model_type)}"
        )

    dispatch = _DATA_SPECIFICATION_CONTENT_FROM_JSONABLE_DISPATCH.get(model_type, None)
    if dispatch is None:
        raise DeserializationException(
            f"Unexpected model type for DataSpecificationContent: {model_type}"
        )

    return dispatch(jsonable)


class _SetterForEmbeddedDataSpecification:

    def __init__(self):

        self.data_specification = None
        self.data_specification_content = None

    def ignore(self, jsonable):

        pass

    def set_data_specification_from_jsonable(self, jsonable):

        self.data_specification = reference_from_jsonable(jsonable)

    def set_data_specification_content_from_jsonable(self, jsonable):

        self.data_specification_content = data_specification_content_from_jsonable(
            jsonable
        )


def embedded_data_specification_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForEmbeddedDataSpecification()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_EMBEDDED_DATA_SPECIFICATION.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.data_specification is None:
        raise DeserializationException(
            "The required property 'dataSpecification' is missing"
        )

    if setter.data_specification_content is None:
        raise DeserializationException(
            "The required property 'dataSpecificationContent' is missing"
        )

    return aas_types.EmbeddedDataSpecification(
        setter.data_specification, setter.data_specification_content
    )


def data_type_iec_61360_from_jsonable(jsonable):

    if not isinstance(jsonable, str):
        raise DeserializationException("Expected a str, but got: {type(jsonable)}")

    literal = aas_stringification.data_type_iec_61360_from_str(jsonable)
    if literal is None:
        raise DeserializationException(
            f"Not a valid string representation of a literal of DataTypeIEC61360: {jsonable}"
        )

    return literal


class _SetterForLevelType:

    def __init__(self):

        self.min = None
        self.nom = None
        self.typ = None
        self.max = None

    def ignore(self, jsonable):

        pass

    def set_min_from_jsonable(self, jsonable):

        self.min = _bool_from_jsonable(jsonable)

    def set_nom_from_jsonable(self, jsonable):

        self.nom = _bool_from_jsonable(jsonable)

    def set_typ_from_jsonable(self, jsonable):

        self.typ = _bool_from_jsonable(jsonable)

    def set_max_from_jsonable(self, jsonable):

        self.max = _bool_from_jsonable(jsonable)


def level_type_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForLevelType()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_LEVEL_TYPE.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.min is None:
        raise DeserializationException("The required property 'min' is missing")

    if setter.nom is None:
        raise DeserializationException("The required property 'nom' is missing")

    if setter.typ is None:
        raise DeserializationException("The required property 'typ' is missing")

    if setter.max is None:
        raise DeserializationException("The required property 'max' is missing")

    return aas_types.LevelType(setter.min, setter.nom, setter.typ, setter.max)


class _SetterForValueReferencePair:

    def __init__(self):

        self.value = None
        self.value_id = None

    def ignore(self, jsonable):

        pass

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_value_id_from_jsonable(self, jsonable):

        self.value_id = reference_from_jsonable(jsonable)


def value_reference_pair_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForValueReferencePair()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_VALUE_REFERENCE_PAIR.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.value is None:
        raise DeserializationException("The required property 'value' is missing")

    if setter.value_id is None:
        raise DeserializationException("The required property 'valueId' is missing")

    return aas_types.ValueReferencePair(setter.value, setter.value_id)


class _SetterForValueList:

    def __init__(self):

        self.value_reference_pairs = None

    def ignore(self, jsonable):

        pass

    def set_value_reference_pairs_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = value_reference_pair_from_jsonable(jsonable_item)
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.value_reference_pairs = items


def value_list_from_jsonable(jsonable):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForValueList()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_VALUE_LIST.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.value_reference_pairs is None:
        raise DeserializationException(
            "The required property 'valueReferencePairs' is missing"
        )

    return aas_types.ValueList(setter.value_reference_pairs)


class _SetterForLangStringPreferredNameTypeIEC61360:

    def __init__(self):

        self.language = None
        self.text = None

    def ignore(self, jsonable):

        pass

    def set_language_from_jsonable(self, jsonable):

        self.language = _str_from_jsonable(jsonable)

    def set_text_from_jsonable(self, jsonable):

        self.text = _str_from_jsonable(jsonable)


def lang_string_preferred_name_type_iec_61360_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForLangStringPreferredNameTypeIEC61360()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_LANG_STRING_PREFERRED_NAME_TYPE_IEC_61360.get(
            key
        )
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.language is None:
        raise DeserializationException("The required property 'language' is missing")

    if setter.text is None:
        raise DeserializationException("The required property 'text' is missing")

    return aas_types.LangStringPreferredNameTypeIEC61360(setter.language, setter.text)


class _SetterForLangStringShortNameTypeIEC61360:

    def __init__(self):

        self.language = None
        self.text = None

    def ignore(self, jsonable):

        pass

    def set_language_from_jsonable(self, jsonable):

        self.language = _str_from_jsonable(jsonable)

    def set_text_from_jsonable(self, jsonable):

        self.text = _str_from_jsonable(jsonable)


def lang_string_short_name_type_iec_61360_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForLangStringShortNameTypeIEC61360()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_LANG_STRING_SHORT_NAME_TYPE_IEC_61360.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.language is None:
        raise DeserializationException("The required property 'language' is missing")

    if setter.text is None:
        raise DeserializationException("The required property 'text' is missing")

    return aas_types.LangStringShortNameTypeIEC61360(setter.language, setter.text)


class _SetterForLangStringDefinitionTypeIEC61360:

    def __init__(self):

        self.language = None
        self.text = None

    def ignore(self, jsonable):

        pass

    def set_language_from_jsonable(self, jsonable):

        self.language = _str_from_jsonable(jsonable)

    def set_text_from_jsonable(self, jsonable):

        self.text = _str_from_jsonable(jsonable)


def lang_string_definition_type_iec_61360_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForLangStringDefinitionTypeIEC61360()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_LANG_STRING_DEFINITION_TYPE_IEC_61360.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.language is None:
        raise DeserializationException("The required property 'language' is missing")

    if setter.text is None:
        raise DeserializationException("The required property 'text' is missing")

    return aas_types.LangStringDefinitionTypeIEC61360(setter.language, setter.text)


class _SetterForDataSpecificationIEC61360:

    def __init__(self):

        self.preferred_name = None
        self.short_name = None
        self.unit = None
        self.unit_id = None
        self.source_of_definition = None
        self.symbol = None
        self.data_type = None
        self.definition = None
        self.value_format = None
        self.value_list = None
        self.value = None
        self.level_type = None

    def ignore(self, jsonable):

        pass

    def set_preferred_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_preferred_name_type_iec_61360_from_jsonable(
                    jsonable_item
                )
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.preferred_name = items

    def set_short_name_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_short_name_type_iec_61360_from_jsonable(
                    jsonable_item
                )
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.short_name = items

    def set_unit_from_jsonable(self, jsonable):

        self.unit = _str_from_jsonable(jsonable)

    def set_unit_id_from_jsonable(self, jsonable):

        self.unit_id = reference_from_jsonable(jsonable)

    def set_source_of_definition_from_jsonable(self, jsonable):

        self.source_of_definition = _str_from_jsonable(jsonable)

    def set_symbol_from_jsonable(self, jsonable):

        self.symbol = _str_from_jsonable(jsonable)

    def set_data_type_from_jsonable(self, jsonable):

        self.data_type = data_type_iec_61360_from_jsonable(jsonable)

    def set_definition_from_jsonable(self, jsonable):

        if not isinstance(jsonable, list):
            raise DeserializationException(
                f"Expected an iterable, but got: {type(jsonable)}"
            )

        items = []
        for i, jsonable_item in enumerate(jsonable):
            try:
                item = lang_string_definition_type_iec_61360_from_jsonable(
                    jsonable_item
                )
            except DeserializationException as exception:
                exception.path._prepend(IndexSegment(jsonable, i))
                raise

            items.append(item)

        self.definition = items

    def set_value_format_from_jsonable(self, jsonable):

        self.value_format = _str_from_jsonable(jsonable)

    def set_value_list_from_jsonable(self, jsonable):

        self.value_list = value_list_from_jsonable(jsonable)

    def set_value_from_jsonable(self, jsonable):

        self.value = _str_from_jsonable(jsonable)

    def set_level_type_from_jsonable(self, jsonable):

        self.level_type = level_type_from_jsonable(jsonable)


def data_specification_iec_61360_from_jsonable(
    jsonable,
):

    if not isinstance(jsonable, dict):
        raise DeserializationException(f"Expected a mapping, but got: {type(jsonable)}")

    setter = _SetterForDataSpecificationIEC61360()

    for key, jsonable_value in jsonable.items():
        setter_method = _SETTER_MAP_FOR_DATA_SPECIFICATION_IEC_61360.get(key)
        if setter_method is None:
            raise DeserializationException(f"Unexpected property: {key}")

        try:
            setter_method(setter, jsonable_value)
        except DeserializationException as exception:
            exception.path._prepend(PropertySegment(jsonable_value, key))
            raise exception

    if setter.preferred_name is None:
        raise DeserializationException(
            "The required property 'preferredName' is missing"
        )

    return aas_types.DataSpecificationIEC61360(
        setter.preferred_name,
        setter.short_name,
        setter.unit,
        setter.unit_id,
        setter.source_of_definition,
        setter.symbol,
        setter.data_type,
        setter.definition,
        setter.value_format,
        setter.value_list,
        setter.value,
        setter.level_type,
    )


_HAS_SEMANTICS_FROM_JSONABLE_DISPATCH = {
    "RelationshipElement": relationship_element_from_jsonable,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
    "BasicEventElement": basic_event_element_from_jsonable,
    "Blob": blob_from_jsonable,
    "Capability": capability_from_jsonable,
    "Entity": entity_from_jsonable,
    "Extension": extension_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Operation": operation_from_jsonable,
    "Property": property_from_jsonable,
    "Qualifier": qualifier_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
    "SpecificAssetId": specific_asset_id_from_jsonable,
    "Submodel": submodel_from_jsonable,
    "SubmodelElementCollection": submodel_element_collection_from_jsonable,
    "SubmodelElementList": submodel_element_list_from_jsonable,
}


_SETTER_MAP_FOR_EXTENSION = {
    "semanticId": _SetterForExtension.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForExtension.set_supplemental_semantic_ids_from_jsonable,
    "name": _SetterForExtension.set_name_from_jsonable,
    "valueType": _SetterForExtension.set_value_type_from_jsonable,
    "value": _SetterForExtension.set_value_from_jsonable,
    "refersTo": _SetterForExtension.set_refers_to_from_jsonable,
    "modelType": _SetterForExtension.ignore,
}


_HAS_EXTENSIONS_FROM_JSONABLE_DISPATCH = {
    "RelationshipElement": relationship_element_from_jsonable,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
    "AssetAdministrationShell": asset_administration_shell_from_jsonable,
    "BasicEventElement": basic_event_element_from_jsonable,
    "Blob": blob_from_jsonable,
    "Capability": capability_from_jsonable,
    "ConceptDescription": concept_description_from_jsonable,
    "Entity": entity_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Operation": operation_from_jsonable,
    "Property": property_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
    "Submodel": submodel_from_jsonable,
    "SubmodelElementCollection": submodel_element_collection_from_jsonable,
    "SubmodelElementList": submodel_element_list_from_jsonable,
}


_REFERABLE_FROM_JSONABLE_DISPATCH = {
    "RelationshipElement": relationship_element_from_jsonable,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
    "AssetAdministrationShell": asset_administration_shell_from_jsonable,
    "BasicEventElement": basic_event_element_from_jsonable,
    "Blob": blob_from_jsonable,
    "Capability": capability_from_jsonable,
    "ConceptDescription": concept_description_from_jsonable,
    "Entity": entity_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Operation": operation_from_jsonable,
    "Property": property_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
    "Submodel": submodel_from_jsonable,
    "SubmodelElementCollection": submodel_element_collection_from_jsonable,
    "SubmodelElementList": submodel_element_list_from_jsonable,
}


_IDENTIFIABLE_FROM_JSONABLE_DISPATCH = {
    "AssetAdministrationShell": asset_administration_shell_from_jsonable,
    "ConceptDescription": concept_description_from_jsonable,
    "Submodel": submodel_from_jsonable,
}


_HAS_KIND_FROM_JSONABLE_DISPATCH = {
    "Submodel": submodel_from_jsonable,
}


_HAS_DATA_SPECIFICATION_FROM_JSONABLE_DISPATCH = {
    "AdministrativeInformation": administrative_information_from_jsonable,
    "RelationshipElement": relationship_element_from_jsonable,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
    "AssetAdministrationShell": asset_administration_shell_from_jsonable,
    "BasicEventElement": basic_event_element_from_jsonable,
    "Blob": blob_from_jsonable,
    "Capability": capability_from_jsonable,
    "ConceptDescription": concept_description_from_jsonable,
    "Entity": entity_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Operation": operation_from_jsonable,
    "Property": property_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
    "Submodel": submodel_from_jsonable,
    "SubmodelElementCollection": submodel_element_collection_from_jsonable,
    "SubmodelElementList": submodel_element_list_from_jsonable,
}


_SETTER_MAP_FOR_ADMINISTRATIVE_INFORMATION = {
    "embeddedDataSpecifications": _SetterForAdministrativeInformation.set_embedded_data_specifications_from_jsonable,
    "version": _SetterForAdministrativeInformation.set_version_from_jsonable,
    "revision": _SetterForAdministrativeInformation.set_revision_from_jsonable,
    "creator": _SetterForAdministrativeInformation.set_creator_from_jsonable,
    "templateId": _SetterForAdministrativeInformation.set_template_id_from_jsonable,
    "modelType": _SetterForAdministrativeInformation.ignore,
}


_QUALIFIABLE_FROM_JSONABLE_DISPATCH = {
    "RelationshipElement": relationship_element_from_jsonable,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
    "BasicEventElement": basic_event_element_from_jsonable,
    "Blob": blob_from_jsonable,
    "Capability": capability_from_jsonable,
    "Entity": entity_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Operation": operation_from_jsonable,
    "Property": property_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
    "Submodel": submodel_from_jsonable,
    "SubmodelElementCollection": submodel_element_collection_from_jsonable,
    "SubmodelElementList": submodel_element_list_from_jsonable,
}


_SETTER_MAP_FOR_QUALIFIER = {
    "semanticId": _SetterForQualifier.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForQualifier.set_supplemental_semantic_ids_from_jsonable,
    "kind": _SetterForQualifier.set_kind_from_jsonable,
    "type": _SetterForQualifier.set_type_from_jsonable,
    "valueType": _SetterForQualifier.set_value_type_from_jsonable,
    "value": _SetterForQualifier.set_value_from_jsonable,
    "valueId": _SetterForQualifier.set_value_id_from_jsonable,
    "modelType": _SetterForQualifier.ignore,
}


_SETTER_MAP_FOR_ASSET_ADMINISTRATION_SHELL = {
    "extensions": _SetterForAssetAdministrationShell.set_extensions_from_jsonable,
    "category": _SetterForAssetAdministrationShell.set_category_from_jsonable,
    "idShort": _SetterForAssetAdministrationShell.set_id_short_from_jsonable,
    "displayName": _SetterForAssetAdministrationShell.set_display_name_from_jsonable,
    "description": _SetterForAssetAdministrationShell.set_description_from_jsonable,
    "administration": _SetterForAssetAdministrationShell.set_administration_from_jsonable,
    "id": _SetterForAssetAdministrationShell.set_id_from_jsonable,
    "embeddedDataSpecifications": _SetterForAssetAdministrationShell.set_embedded_data_specifications_from_jsonable,
    "derivedFrom": _SetterForAssetAdministrationShell.set_derived_from_from_jsonable,
    "assetInformation": _SetterForAssetAdministrationShell.set_asset_information_from_jsonable,
    "submodels": _SetterForAssetAdministrationShell.set_submodels_from_jsonable,
    "modelType": _SetterForAssetAdministrationShell.ignore,
}


_SETTER_MAP_FOR_ASSET_INFORMATION = {
    "assetKind": _SetterForAssetInformation.set_asset_kind_from_jsonable,
    "globalAssetId": _SetterForAssetInformation.set_global_asset_id_from_jsonable,
    "specificAssetIds": _SetterForAssetInformation.set_specific_asset_ids_from_jsonable,
    "assetType": _SetterForAssetInformation.set_asset_type_from_jsonable,
    "defaultThumbnail": _SetterForAssetInformation.set_default_thumbnail_from_jsonable,
    "modelType": _SetterForAssetInformation.ignore,
}


_SETTER_MAP_FOR_RESOURCE = {
    "path": _SetterForResource.set_path_from_jsonable,
    "contentType": _SetterForResource.set_content_type_from_jsonable,
    "modelType": _SetterForResource.ignore,
}


_SETTER_MAP_FOR_SPECIFIC_ASSET_ID = {
    "semanticId": _SetterForSpecificAssetID.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForSpecificAssetID.set_supplemental_semantic_ids_from_jsonable,
    "name": _SetterForSpecificAssetID.set_name_from_jsonable,
    "value": _SetterForSpecificAssetID.set_value_from_jsonable,
    "externalSubjectId": _SetterForSpecificAssetID.set_external_subject_id_from_jsonable,
    "modelType": _SetterForSpecificAssetID.ignore,
}


_SETTER_MAP_FOR_SUBMODEL = {
    "extensions": _SetterForSubmodel.set_extensions_from_jsonable,
    "category": _SetterForSubmodel.set_category_from_jsonable,
    "idShort": _SetterForSubmodel.set_id_short_from_jsonable,
    "displayName": _SetterForSubmodel.set_display_name_from_jsonable,
    "description": _SetterForSubmodel.set_description_from_jsonable,
    "administration": _SetterForSubmodel.set_administration_from_jsonable,
    "id": _SetterForSubmodel.set_id_from_jsonable,
    "kind": _SetterForSubmodel.set_kind_from_jsonable,
    "semanticId": _SetterForSubmodel.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForSubmodel.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForSubmodel.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForSubmodel.set_embedded_data_specifications_from_jsonable,
    "submodelElements": _SetterForSubmodel.set_submodel_elements_from_jsonable,
    "modelType": _SetterForSubmodel.ignore,
}


_SUBMODEL_ELEMENT_FROM_JSONABLE_DISPATCH = {
    "RelationshipElement": relationship_element_from_jsonable,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
    "BasicEventElement": basic_event_element_from_jsonable,
    "Blob": blob_from_jsonable,
    "Capability": capability_from_jsonable,
    "Entity": entity_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Operation": operation_from_jsonable,
    "Property": property_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
    "SubmodelElementCollection": submodel_element_collection_from_jsonable,
    "SubmodelElementList": submodel_element_list_from_jsonable,
}


_RELATIONSHIP_ELEMENT_FROM_JSONABLE_DISPATCH = {
    "RelationshipElement": _relationship_element_from_jsonable_without_dispatch,
    "AnnotatedRelationshipElement": annotated_relationship_element_from_jsonable,
}


_SETTER_MAP_FOR_RELATIONSHIP_ELEMENT = {
    "extensions": _SetterForRelationshipElement.set_extensions_from_jsonable,
    "category": _SetterForRelationshipElement.set_category_from_jsonable,
    "idShort": _SetterForRelationshipElement.set_id_short_from_jsonable,
    "displayName": _SetterForRelationshipElement.set_display_name_from_jsonable,
    "description": _SetterForRelationshipElement.set_description_from_jsonable,
    "semanticId": _SetterForRelationshipElement.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForRelationshipElement.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForRelationshipElement.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForRelationshipElement.set_embedded_data_specifications_from_jsonable,
    "first": _SetterForRelationshipElement.set_first_from_jsonable,
    "second": _SetterForRelationshipElement.set_second_from_jsonable,
    "modelType": _SetterForRelationshipElement.ignore,
}


_SETTER_MAP_FOR_SUBMODEL_ELEMENT_LIST = {
    "extensions": _SetterForSubmodelElementList.set_extensions_from_jsonable,
    "category": _SetterForSubmodelElementList.set_category_from_jsonable,
    "idShort": _SetterForSubmodelElementList.set_id_short_from_jsonable,
    "displayName": _SetterForSubmodelElementList.set_display_name_from_jsonable,
    "description": _SetterForSubmodelElementList.set_description_from_jsonable,
    "semanticId": _SetterForSubmodelElementList.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForSubmodelElementList.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForSubmodelElementList.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForSubmodelElementList.set_embedded_data_specifications_from_jsonable,
    "orderRelevant": _SetterForSubmodelElementList.set_order_relevant_from_jsonable,
    "semanticIdListElement": _SetterForSubmodelElementList.set_semantic_id_list_element_from_jsonable,
    "typeValueListElement": _SetterForSubmodelElementList.set_type_value_list_element_from_jsonable,
    "valueTypeListElement": _SetterForSubmodelElementList.set_value_type_list_element_from_jsonable,
    "value": _SetterForSubmodelElementList.set_value_from_jsonable,
    "modelType": _SetterForSubmodelElementList.ignore,
}


_SETTER_MAP_FOR_SUBMODEL_ELEMENT_COLLECTION = {
    "extensions": _SetterForSubmodelElementCollection.set_extensions_from_jsonable,
    "category": _SetterForSubmodelElementCollection.set_category_from_jsonable,
    "idShort": _SetterForSubmodelElementCollection.set_id_short_from_jsonable,
    "displayName": _SetterForSubmodelElementCollection.set_display_name_from_jsonable,
    "description": _SetterForSubmodelElementCollection.set_description_from_jsonable,
    "semanticId": _SetterForSubmodelElementCollection.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForSubmodelElementCollection.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForSubmodelElementCollection.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForSubmodelElementCollection.set_embedded_data_specifications_from_jsonable,
    "value": _SetterForSubmodelElementCollection.set_value_from_jsonable,
    "modelType": _SetterForSubmodelElementCollection.ignore,
}


_DATA_ELEMENT_FROM_JSONABLE_DISPATCH = {
    "Blob": blob_from_jsonable,
    "File": file_from_jsonable,
    "MultiLanguageProperty": multi_language_property_from_jsonable,
    "Property": property_from_jsonable,
    "Range": range_from_jsonable,
    "ReferenceElement": reference_element_from_jsonable,
}


_SETTER_MAP_FOR_PROPERTY = {
    "extensions": _SetterForProperty.set_extensions_from_jsonable,
    "category": _SetterForProperty.set_category_from_jsonable,
    "idShort": _SetterForProperty.set_id_short_from_jsonable,
    "displayName": _SetterForProperty.set_display_name_from_jsonable,
    "description": _SetterForProperty.set_description_from_jsonable,
    "semanticId": _SetterForProperty.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForProperty.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForProperty.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForProperty.set_embedded_data_specifications_from_jsonable,
    "valueType": _SetterForProperty.set_value_type_from_jsonable,
    "value": _SetterForProperty.set_value_from_jsonable,
    "valueId": _SetterForProperty.set_value_id_from_jsonable,
    "modelType": _SetterForProperty.ignore,
}


_SETTER_MAP_FOR_MULTI_LANGUAGE_PROPERTY = {
    "extensions": _SetterForMultiLanguageProperty.set_extensions_from_jsonable,
    "category": _SetterForMultiLanguageProperty.set_category_from_jsonable,
    "idShort": _SetterForMultiLanguageProperty.set_id_short_from_jsonable,
    "displayName": _SetterForMultiLanguageProperty.set_display_name_from_jsonable,
    "description": _SetterForMultiLanguageProperty.set_description_from_jsonable,
    "semanticId": _SetterForMultiLanguageProperty.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForMultiLanguageProperty.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForMultiLanguageProperty.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForMultiLanguageProperty.set_embedded_data_specifications_from_jsonable,
    "value": _SetterForMultiLanguageProperty.set_value_from_jsonable,
    "valueId": _SetterForMultiLanguageProperty.set_value_id_from_jsonable,
    "modelType": _SetterForMultiLanguageProperty.ignore,
}


_SETTER_MAP_FOR_RANGE = {
    "extensions": _SetterForRange.set_extensions_from_jsonable,
    "category": _SetterForRange.set_category_from_jsonable,
    "idShort": _SetterForRange.set_id_short_from_jsonable,
    "displayName": _SetterForRange.set_display_name_from_jsonable,
    "description": _SetterForRange.set_description_from_jsonable,
    "semanticId": _SetterForRange.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForRange.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForRange.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForRange.set_embedded_data_specifications_from_jsonable,
    "valueType": _SetterForRange.set_value_type_from_jsonable,
    "min": _SetterForRange.set_min_from_jsonable,
    "max": _SetterForRange.set_max_from_jsonable,
    "modelType": _SetterForRange.ignore,
}


_SETTER_MAP_FOR_REFERENCE_ELEMENT = {
    "extensions": _SetterForReferenceElement.set_extensions_from_jsonable,
    "category": _SetterForReferenceElement.set_category_from_jsonable,
    "idShort": _SetterForReferenceElement.set_id_short_from_jsonable,
    "displayName": _SetterForReferenceElement.set_display_name_from_jsonable,
    "description": _SetterForReferenceElement.set_description_from_jsonable,
    "semanticId": _SetterForReferenceElement.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForReferenceElement.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForReferenceElement.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForReferenceElement.set_embedded_data_specifications_from_jsonable,
    "value": _SetterForReferenceElement.set_value_from_jsonable,
    "modelType": _SetterForReferenceElement.ignore,
}


_SETTER_MAP_FOR_BLOB = {
    "extensions": _SetterForBlob.set_extensions_from_jsonable,
    "category": _SetterForBlob.set_category_from_jsonable,
    "idShort": _SetterForBlob.set_id_short_from_jsonable,
    "displayName": _SetterForBlob.set_display_name_from_jsonable,
    "description": _SetterForBlob.set_description_from_jsonable,
    "semanticId": _SetterForBlob.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForBlob.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForBlob.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForBlob.set_embedded_data_specifications_from_jsonable,
    "value": _SetterForBlob.set_value_from_jsonable,
    "contentType": _SetterForBlob.set_content_type_from_jsonable,
    "modelType": _SetterForBlob.ignore,
}


_SETTER_MAP_FOR_FILE = {
    "extensions": _SetterForFile.set_extensions_from_jsonable,
    "category": _SetterForFile.set_category_from_jsonable,
    "idShort": _SetterForFile.set_id_short_from_jsonable,
    "displayName": _SetterForFile.set_display_name_from_jsonable,
    "description": _SetterForFile.set_description_from_jsonable,
    "semanticId": _SetterForFile.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForFile.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForFile.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForFile.set_embedded_data_specifications_from_jsonable,
    "value": _SetterForFile.set_value_from_jsonable,
    "contentType": _SetterForFile.set_content_type_from_jsonable,
    "modelType": _SetterForFile.ignore,
}


_SETTER_MAP_FOR_ANNOTATED_RELATIONSHIP_ELEMENT = {
    "extensions": _SetterForAnnotatedRelationshipElement.set_extensions_from_jsonable,
    "category": _SetterForAnnotatedRelationshipElement.set_category_from_jsonable,
    "idShort": _SetterForAnnotatedRelationshipElement.set_id_short_from_jsonable,
    "displayName": _SetterForAnnotatedRelationshipElement.set_display_name_from_jsonable,
    "description": _SetterForAnnotatedRelationshipElement.set_description_from_jsonable,
    "semanticId": _SetterForAnnotatedRelationshipElement.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForAnnotatedRelationshipElement.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForAnnotatedRelationshipElement.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForAnnotatedRelationshipElement.set_embedded_data_specifications_from_jsonable,
    "first": _SetterForAnnotatedRelationshipElement.set_first_from_jsonable,
    "second": _SetterForAnnotatedRelationshipElement.set_second_from_jsonable,
    "annotations": _SetterForAnnotatedRelationshipElement.set_annotations_from_jsonable,
    "modelType": _SetterForAnnotatedRelationshipElement.ignore,
}


_SETTER_MAP_FOR_ENTITY = {
    "extensions": _SetterForEntity.set_extensions_from_jsonable,
    "category": _SetterForEntity.set_category_from_jsonable,
    "idShort": _SetterForEntity.set_id_short_from_jsonable,
    "displayName": _SetterForEntity.set_display_name_from_jsonable,
    "description": _SetterForEntity.set_description_from_jsonable,
    "semanticId": _SetterForEntity.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForEntity.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForEntity.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForEntity.set_embedded_data_specifications_from_jsonable,
    "statements": _SetterForEntity.set_statements_from_jsonable,
    "entityType": _SetterForEntity.set_entity_type_from_jsonable,
    "globalAssetId": _SetterForEntity.set_global_asset_id_from_jsonable,
    "specificAssetIds": _SetterForEntity.set_specific_asset_ids_from_jsonable,
    "modelType": _SetterForEntity.ignore,
}


_SETTER_MAP_FOR_EVENT_PAYLOAD = {
    "source": _SetterForEventPayload.set_source_from_jsonable,
    "sourceSemanticId": _SetterForEventPayload.set_source_semantic_id_from_jsonable,
    "observableReference": _SetterForEventPayload.set_observable_reference_from_jsonable,
    "observableSemanticId": _SetterForEventPayload.set_observable_semantic_id_from_jsonable,
    "topic": _SetterForEventPayload.set_topic_from_jsonable,
    "subjectId": _SetterForEventPayload.set_subject_id_from_jsonable,
    "timeStamp": _SetterForEventPayload.set_time_stamp_from_jsonable,
    "payload": _SetterForEventPayload.set_payload_from_jsonable,
    "modelType": _SetterForEventPayload.ignore,
}


_EVENT_ELEMENT_FROM_JSONABLE_DISPATCH = {
    "BasicEventElement": basic_event_element_from_jsonable,
}


_SETTER_MAP_FOR_BASIC_EVENT_ELEMENT = {
    "extensions": _SetterForBasicEventElement.set_extensions_from_jsonable,
    "category": _SetterForBasicEventElement.set_category_from_jsonable,
    "idShort": _SetterForBasicEventElement.set_id_short_from_jsonable,
    "displayName": _SetterForBasicEventElement.set_display_name_from_jsonable,
    "description": _SetterForBasicEventElement.set_description_from_jsonable,
    "semanticId": _SetterForBasicEventElement.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForBasicEventElement.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForBasicEventElement.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForBasicEventElement.set_embedded_data_specifications_from_jsonable,
    "observed": _SetterForBasicEventElement.set_observed_from_jsonable,
    "direction": _SetterForBasicEventElement.set_direction_from_jsonable,
    "state": _SetterForBasicEventElement.set_state_from_jsonable,
    "messageTopic": _SetterForBasicEventElement.set_message_topic_from_jsonable,
    "messageBroker": _SetterForBasicEventElement.set_message_broker_from_jsonable,
    "lastUpdate": _SetterForBasicEventElement.set_last_update_from_jsonable,
    "minInterval": _SetterForBasicEventElement.set_min_interval_from_jsonable,
    "maxInterval": _SetterForBasicEventElement.set_max_interval_from_jsonable,
    "modelType": _SetterForBasicEventElement.ignore,
}


_SETTER_MAP_FOR_OPERATION = {
    "extensions": _SetterForOperation.set_extensions_from_jsonable,
    "category": _SetterForOperation.set_category_from_jsonable,
    "idShort": _SetterForOperation.set_id_short_from_jsonable,
    "displayName": _SetterForOperation.set_display_name_from_jsonable,
    "description": _SetterForOperation.set_description_from_jsonable,
    "semanticId": _SetterForOperation.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForOperation.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForOperation.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForOperation.set_embedded_data_specifications_from_jsonable,
    "inputVariables": _SetterForOperation.set_input_variables_from_jsonable,
    "outputVariables": _SetterForOperation.set_output_variables_from_jsonable,
    "inoutputVariables": _SetterForOperation.set_inoutput_variables_from_jsonable,
    "modelType": _SetterForOperation.ignore,
}


_SETTER_MAP_FOR_OPERATION_VARIABLE = {
    "value": _SetterForOperationVariable.set_value_from_jsonable,
    "modelType": _SetterForOperationVariable.ignore,
}


_SETTER_MAP_FOR_CAPABILITY = {
    "extensions": _SetterForCapability.set_extensions_from_jsonable,
    "category": _SetterForCapability.set_category_from_jsonable,
    "idShort": _SetterForCapability.set_id_short_from_jsonable,
    "displayName": _SetterForCapability.set_display_name_from_jsonable,
    "description": _SetterForCapability.set_description_from_jsonable,
    "semanticId": _SetterForCapability.set_semantic_id_from_jsonable,
    "supplementalSemanticIds": _SetterForCapability.set_supplemental_semantic_ids_from_jsonable,
    "qualifiers": _SetterForCapability.set_qualifiers_from_jsonable,
    "embeddedDataSpecifications": _SetterForCapability.set_embedded_data_specifications_from_jsonable,
    "modelType": _SetterForCapability.ignore,
}


_SETTER_MAP_FOR_CONCEPT_DESCRIPTION = {
    "extensions": _SetterForConceptDescription.set_extensions_from_jsonable,
    "category": _SetterForConceptDescription.set_category_from_jsonable,
    "idShort": _SetterForConceptDescription.set_id_short_from_jsonable,
    "displayName": _SetterForConceptDescription.set_display_name_from_jsonable,
    "description": _SetterForConceptDescription.set_description_from_jsonable,
    "administration": _SetterForConceptDescription.set_administration_from_jsonable,
    "id": _SetterForConceptDescription.set_id_from_jsonable,
    "embeddedDataSpecifications": _SetterForConceptDescription.set_embedded_data_specifications_from_jsonable,
    "isCaseOf": _SetterForConceptDescription.set_is_case_of_from_jsonable,
    "modelType": _SetterForConceptDescription.ignore,
}


_SETTER_MAP_FOR_REFERENCE = {
    "type": _SetterForReference.set_type_from_jsonable,
    "referredSemanticId": _SetterForReference.set_referred_semantic_id_from_jsonable,
    "keys": _SetterForReference.set_keys_from_jsonable,
    "modelType": _SetterForReference.ignore,
}


_SETTER_MAP_FOR_KEY = {
    "type": _SetterForKey.set_type_from_jsonable,
    "value": _SetterForKey.set_value_from_jsonable,
    "modelType": _SetterForKey.ignore,
}


_ABSTRACT_LANG_STRING_FROM_JSONABLE_DISPATCH = {
    "LangStringDefinitionTypeIec61360": lang_string_definition_type_iec_61360_from_jsonable,
    "LangStringNameType": lang_string_name_type_from_jsonable,
    "LangStringPreferredNameTypeIec61360": lang_string_preferred_name_type_iec_61360_from_jsonable,
    "LangStringShortNameTypeIec61360": lang_string_short_name_type_iec_61360_from_jsonable,
    "LangStringTextType": lang_string_text_type_from_jsonable,
}


_SETTER_MAP_FOR_LANG_STRING_NAME_TYPE = {
    "language": _SetterForLangStringNameType.set_language_from_jsonable,
    "text": _SetterForLangStringNameType.set_text_from_jsonable,
    "modelType": _SetterForLangStringNameType.ignore,
}


_SETTER_MAP_FOR_LANG_STRING_TEXT_TYPE = {
    "language": _SetterForLangStringTextType.set_language_from_jsonable,
    "text": _SetterForLangStringTextType.set_text_from_jsonable,
    "modelType": _SetterForLangStringTextType.ignore,
}


_SETTER_MAP_FOR_ENVIRONMENT = {
    "assetAdministrationShells": _SetterForEnvironment.set_asset_administration_shells_from_jsonable,
    "submodels": _SetterForEnvironment.set_submodels_from_jsonable,
    "conceptDescriptions": _SetterForEnvironment.set_concept_descriptions_from_jsonable,
    "modelType": _SetterForEnvironment.ignore,
}


_DATA_SPECIFICATION_CONTENT_FROM_JSONABLE_DISPATCH = {
    "DataSpecificationIec61360": data_specification_iec_61360_from_jsonable,
}


_SETTER_MAP_FOR_EMBEDDED_DATA_SPECIFICATION = {
    "dataSpecification": _SetterForEmbeddedDataSpecification.set_data_specification_from_jsonable,
    "dataSpecificationContent": _SetterForEmbeddedDataSpecification.set_data_specification_content_from_jsonable,
    "modelType": _SetterForEmbeddedDataSpecification.ignore,
}


_SETTER_MAP_FOR_LEVEL_TYPE = {
    "min": _SetterForLevelType.set_min_from_jsonable,
    "nom": _SetterForLevelType.set_nom_from_jsonable,
    "typ": _SetterForLevelType.set_typ_from_jsonable,
    "max": _SetterForLevelType.set_max_from_jsonable,
    "modelType": _SetterForLevelType.ignore,
}


_SETTER_MAP_FOR_VALUE_REFERENCE_PAIR = {
    "value": _SetterForValueReferencePair.set_value_from_jsonable,
    "valueId": _SetterForValueReferencePair.set_value_id_from_jsonable,
    "modelType": _SetterForValueReferencePair.ignore,
}


_SETTER_MAP_FOR_VALUE_LIST = {
    "valueReferencePairs": _SetterForValueList.set_value_reference_pairs_from_jsonable,
    "modelType": _SetterForValueList.ignore,
}


_SETTER_MAP_FOR_LANG_STRING_PREFERRED_NAME_TYPE_IEC_61360 = {
    "language": _SetterForLangStringPreferredNameTypeIEC61360.set_language_from_jsonable,
    "text": _SetterForLangStringPreferredNameTypeIEC61360.set_text_from_jsonable,
    "modelType": _SetterForLangStringPreferredNameTypeIEC61360.ignore,
}


_SETTER_MAP_FOR_LANG_STRING_SHORT_NAME_TYPE_IEC_61360 = {
    "language": _SetterForLangStringShortNameTypeIEC61360.set_language_from_jsonable,
    "text": _SetterForLangStringShortNameTypeIEC61360.set_text_from_jsonable,
    "modelType": _SetterForLangStringShortNameTypeIEC61360.ignore,
}


_SETTER_MAP_FOR_LANG_STRING_DEFINITION_TYPE_IEC_61360 = {
    "language": _SetterForLangStringDefinitionTypeIEC61360.set_language_from_jsonable,
    "text": _SetterForLangStringDefinitionTypeIEC61360.set_text_from_jsonable,
    "modelType": _SetterForLangStringDefinitionTypeIEC61360.ignore,
}


_SETTER_MAP_FOR_DATA_SPECIFICATION_IEC_61360 = {
    "preferredName": _SetterForDataSpecificationIEC61360.set_preferred_name_from_jsonable,
    "shortName": _SetterForDataSpecificationIEC61360.set_short_name_from_jsonable,
    "unit": _SetterForDataSpecificationIEC61360.set_unit_from_jsonable,
    "unitId": _SetterForDataSpecificationIEC61360.set_unit_id_from_jsonable,
    "sourceOfDefinition": _SetterForDataSpecificationIEC61360.set_source_of_definition_from_jsonable,
    "symbol": _SetterForDataSpecificationIEC61360.set_symbol_from_jsonable,
    "dataType": _SetterForDataSpecificationIEC61360.set_data_type_from_jsonable,
    "definition": _SetterForDataSpecificationIEC61360.set_definition_from_jsonable,
    "valueFormat": _SetterForDataSpecificationIEC61360.set_value_format_from_jsonable,
    "valueList": _SetterForDataSpecificationIEC61360.set_value_list_from_jsonable,
    "value": _SetterForDataSpecificationIEC61360.set_value_from_jsonable,
    "levelType": _SetterForDataSpecificationIEC61360.set_level_type_from_jsonable,
    "modelType": _SetterForDataSpecificationIEC61360.ignore,
}


def _bytes_to_base64_str(value):

    return base64.b64encode(value).decode("ascii")


class _Serializer(aas_types.AbstractTransformer):

    def transform_extension(self, that):

        jsonable = dict()

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        jsonable["name"] = that.name

        if that.value_type is not None:
            jsonable["valueType"] = that.value_type.value

        if that.value is not None:
            jsonable["value"] = that.value

        if that.refers_to is not None:
            jsonable["refersTo"] = [self.transform(item) for item in that.refers_to]

        return jsonable

    def transform_administrative_information(self, that):

        jsonable = dict()

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.version is not None:
            jsonable["version"] = that.version

        if that.revision is not None:
            jsonable["revision"] = that.revision

        if that.creator is not None:
            jsonable["creator"] = self.transform(that.creator)

        if that.template_id is not None:
            jsonable["templateId"] = that.template_id

        return jsonable

    def transform_qualifier(self, that):

        jsonable = dict()

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.kind is not None:
            jsonable["kind"] = that.kind.value

        jsonable["type"] = that.type

        jsonable["valueType"] = that.value_type.value

        if that.value is not None:
            jsonable["value"] = that.value

        if that.value_id is not None:
            jsonable["valueId"] = self.transform(that.value_id)

        return jsonable

    def transform_asset_administration_shell(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.administration is not None:
            jsonable["administration"] = self.transform(that.administration)

        jsonable["id"] = that.id

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.derived_from is not None:
            jsonable["derivedFrom"] = self.transform(that.derived_from)

        jsonable["assetInformation"] = self.transform(that.asset_information)

        if that.submodels is not None:
            jsonable["submodels"] = [self.transform(item) for item in that.submodels]

        jsonable["modelType"] = "AssetAdministrationShell"

        return jsonable

    def transform_asset_information(self, that):

        jsonable = dict()

        jsonable["assetKind"] = that.asset_kind.value

        if that.global_asset_id is not None:
            jsonable["globalAssetId"] = that.global_asset_id

        if that.specific_asset_ids is not None:
            jsonable["specificAssetIds"] = [
                self.transform(item) for item in that.specific_asset_ids
            ]

        if that.asset_type is not None:
            jsonable["assetType"] = that.asset_type

        if that.default_thumbnail is not None:
            jsonable["defaultThumbnail"] = self.transform(that.default_thumbnail)

        return jsonable

    def transform_resource(self, that):

        jsonable = dict()

        jsonable["path"] = that.path

        if that.content_type is not None:
            jsonable["contentType"] = that.content_type

        return jsonable

    def transform_specific_asset_id(self, that):

        jsonable = dict()

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        jsonable["name"] = that.name

        jsonable["value"] = that.value

        if that.external_subject_id is not None:
            jsonable["externalSubjectId"] = self.transform(that.external_subject_id)

        return jsonable

    def transform_submodel(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.administration is not None:
            jsonable["administration"] = self.transform(that.administration)

        jsonable["id"] = that.id

        if that.kind is not None:
            jsonable["kind"] = that.kind.value

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.submodel_elements is not None:
            jsonable["submodelElements"] = [
                self.transform(item) for item in that.submodel_elements
            ]

        jsonable["modelType"] = "Submodel"

        return jsonable

    def transform_relationship_element(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        jsonable["first"] = self.transform(that.first)

        jsonable["second"] = self.transform(that.second)

        jsonable["modelType"] = "RelationshipElement"

        return jsonable

    def transform_submodel_element_list(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.order_relevant is not None:
            jsonable["orderRelevant"] = that.order_relevant

        if that.semantic_id_list_element is not None:
            jsonable["semanticIdListElement"] = self.transform(
                that.semantic_id_list_element
            )

        jsonable["typeValueListElement"] = that.type_value_list_element.value

        if that.value_type_list_element is not None:
            jsonable["valueTypeListElement"] = that.value_type_list_element.value

        if that.value is not None:
            jsonable["value"] = [self.transform(item) for item in that.value]

        jsonable["modelType"] = "SubmodelElementList"

        return jsonable

    def transform_submodel_element_collection(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.value is not None:
            jsonable["value"] = [self.transform(item) for item in that.value]

        jsonable["modelType"] = "SubmodelElementCollection"

        return jsonable

    def transform_property(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        jsonable["valueType"] = that.value_type.value

        if that.value is not None:
            jsonable["value"] = that.value

        if that.value_id is not None:
            jsonable["valueId"] = self.transform(that.value_id)

        jsonable["modelType"] = "Property"

        return jsonable

    def transform_multi_language_property(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.value is not None:
            jsonable["value"] = [self.transform(item) for item in that.value]

        if that.value_id is not None:
            jsonable["valueId"] = self.transform(that.value_id)

        jsonable["modelType"] = "MultiLanguageProperty"

        return jsonable

    def transform_range(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        jsonable["valueType"] = that.value_type.value

        if that.min is not None:
            jsonable["min"] = that.min

        if that.max is not None:
            jsonable["max"] = that.max

        jsonable["modelType"] = "Range"

        return jsonable

    def transform_reference_element(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.value is not None:
            jsonable["value"] = self.transform(that.value)

        jsonable["modelType"] = "ReferenceElement"

        return jsonable

    def transform_blob(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.value is not None:
            jsonable["value"] = _bytes_to_base64_str(that.value)

        jsonable["contentType"] = that.content_type

        jsonable["modelType"] = "Blob"

        return jsonable

    def transform_file(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.value is not None:
            jsonable["value"] = that.value

        jsonable["contentType"] = that.content_type

        jsonable["modelType"] = "File"

        return jsonable

    def transform_annotated_relationship_element(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        jsonable["first"] = self.transform(that.first)

        jsonable["second"] = self.transform(that.second)

        if that.annotations is not None:
            jsonable["annotations"] = [
                self.transform(item) for item in that.annotations
            ]

        jsonable["modelType"] = "AnnotatedRelationshipElement"

        return jsonable

    def transform_entity(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.statements is not None:
            jsonable["statements"] = [self.transform(item) for item in that.statements]

        jsonable["entityType"] = that.entity_type.value

        if that.global_asset_id is not None:
            jsonable["globalAssetId"] = that.global_asset_id

        if that.specific_asset_ids is not None:
            jsonable["specificAssetIds"] = [
                self.transform(item) for item in that.specific_asset_ids
            ]

        jsonable["modelType"] = "Entity"

        return jsonable

    def transform_event_payload(self, that):

        jsonable = dict()

        jsonable["source"] = self.transform(that.source)

        if that.source_semantic_id is not None:
            jsonable["sourceSemanticId"] = self.transform(that.source_semantic_id)

        jsonable["observableReference"] = self.transform(that.observable_reference)

        if that.observable_semantic_id is not None:
            jsonable["observableSemanticId"] = self.transform(
                that.observable_semantic_id
            )

        if that.topic is not None:
            jsonable["topic"] = that.topic

        if that.subject_id is not None:
            jsonable["subjectId"] = self.transform(that.subject_id)

        jsonable["timeStamp"] = that.time_stamp

        if that.payload is not None:
            jsonable["payload"] = _bytes_to_base64_str(that.payload)

        return jsonable

    def transform_basic_event_element(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        jsonable["observed"] = self.transform(that.observed)

        jsonable["direction"] = that.direction.value

        jsonable["state"] = that.state.value

        if that.message_topic is not None:
            jsonable["messageTopic"] = that.message_topic

        if that.message_broker is not None:
            jsonable["messageBroker"] = self.transform(that.message_broker)

        if that.last_update is not None:
            jsonable["lastUpdate"] = that.last_update

        if that.min_interval is not None:
            jsonable["minInterval"] = that.min_interval

        if that.max_interval is not None:
            jsonable["maxInterval"] = that.max_interval

        jsonable["modelType"] = "BasicEventElement"

        return jsonable

    def transform_operation(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.input_variables is not None:
            jsonable["inputVariables"] = [
                self.transform(item) for item in that.input_variables
            ]

        if that.output_variables is not None:
            jsonable["outputVariables"] = [
                self.transform(item) for item in that.output_variables
            ]

        if that.inoutput_variables is not None:
            jsonable["inoutputVariables"] = [
                self.transform(item) for item in that.inoutput_variables
            ]

        jsonable["modelType"] = "Operation"

        return jsonable

    def transform_operation_variable(self, that):

        jsonable = dict()

        jsonable["value"] = self.transform(that.value)

        return jsonable

    def transform_capability(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.semantic_id is not None:
            jsonable["semanticId"] = self.transform(that.semantic_id)

        if that.supplemental_semantic_ids is not None:
            jsonable["supplementalSemanticIds"] = [
                self.transform(item) for item in that.supplemental_semantic_ids
            ]

        if that.qualifiers is not None:
            jsonable["qualifiers"] = [self.transform(item) for item in that.qualifiers]

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        jsonable["modelType"] = "Capability"

        return jsonable

    def transform_concept_description(self, that):

        jsonable = dict()

        if that.extensions is not None:
            jsonable["extensions"] = [self.transform(item) for item in that.extensions]

        if that.category is not None:
            jsonable["category"] = that.category

        if that.id_short is not None:
            jsonable["idShort"] = that.id_short

        if that.display_name is not None:
            jsonable["displayName"] = [
                self.transform(item) for item in that.display_name
            ]

        if that.description is not None:
            jsonable["description"] = [
                self.transform(item) for item in that.description
            ]

        if that.administration is not None:
            jsonable["administration"] = self.transform(that.administration)

        jsonable["id"] = that.id

        if that.embedded_data_specifications is not None:
            jsonable["embeddedDataSpecifications"] = [
                self.transform(item) for item in that.embedded_data_specifications
            ]

        if that.is_case_of is not None:
            jsonable["isCaseOf"] = [self.transform(item) for item in that.is_case_of]

        jsonable["modelType"] = "ConceptDescription"

        return jsonable

    def transform_reference(self, that):

        jsonable = dict()

        jsonable["type"] = that.type.value

        if that.referred_semantic_id is not None:
            jsonable["referredSemanticId"] = self.transform(that.referred_semantic_id)

        jsonable["keys"] = [self.transform(item) for item in that.keys]

        return jsonable

    def transform_key(self, that):

        jsonable = dict()

        jsonable["type"] = that.type.value

        jsonable["value"] = that.value

        return jsonable

    def transform_lang_string_name_type(self, that):

        jsonable = dict()

        jsonable["language"] = that.language

        jsonable["text"] = that.text

        return jsonable

    def transform_lang_string_text_type(self, that):

        jsonable = dict()

        jsonable["language"] = that.language

        jsonable["text"] = that.text

        return jsonable

    def transform_environment(self, that):

        jsonable = dict()

        if that.asset_administration_shells is not None:
            jsonable["assetAdministrationShells"] = [
                self.transform(item) for item in that.asset_administration_shells
            ]

        if that.submodels is not None:
            jsonable["submodels"] = [self.transform(item) for item in that.submodels]

        if that.concept_descriptions is not None:
            jsonable["conceptDescriptions"] = [
                self.transform(item) for item in that.concept_descriptions
            ]

        return jsonable

    def transform_embedded_data_specification(self, that):

        jsonable = dict()

        jsonable["dataSpecification"] = self.transform(that.data_specification)

        jsonable["dataSpecificationContent"] = self.transform(
            that.data_specification_content
        )

        return jsonable

    def transform_level_type(self, that):

        jsonable = dict()

        jsonable["min"] = that.min

        jsonable["nom"] = that.nom

        jsonable["typ"] = that.typ

        jsonable["max"] = that.max

        return jsonable

    def transform_value_reference_pair(self, that):

        jsonable = dict()

        jsonable["value"] = that.value

        jsonable["valueId"] = self.transform(that.value_id)

        return jsonable

    def transform_value_list(self, that):

        jsonable = dict()

        jsonable["valueReferencePairs"] = [
            self.transform(item) for item in that.value_reference_pairs
        ]

        return jsonable

    def transform_lang_string_preferred_name_type_iec_61360(self, that):

        jsonable = dict()

        jsonable["language"] = that.language

        jsonable["text"] = that.text

        return jsonable

    def transform_lang_string_short_name_type_iec_61360(self, that):

        jsonable = dict()

        jsonable["language"] = that.language

        jsonable["text"] = that.text

        return jsonable

    def transform_lang_string_definition_type_iec_61360(self, that):

        jsonable = dict()

        jsonable["language"] = that.language

        jsonable["text"] = that.text

        return jsonable

    def transform_data_specification_iec_61360(self, that):

        jsonable = dict()

        jsonable["preferredName"] = [
            self.transform(item) for item in that.preferred_name
        ]

        if that.short_name is not None:
            jsonable["shortName"] = [self.transform(item) for item in that.short_name]

        if that.unit is not None:
            jsonable["unit"] = that.unit

        if that.unit_id is not None:
            jsonable["unitId"] = self.transform(that.unit_id)

        if that.source_of_definition is not None:
            jsonable["sourceOfDefinition"] = that.source_of_definition

        if that.symbol is not None:
            jsonable["symbol"] = that.symbol

        if that.data_type is not None:
            jsonable["dataType"] = that.data_type.value

        if that.definition is not None:
            jsonable["definition"] = [self.transform(item) for item in that.definition]

        if that.value_format is not None:
            jsonable["valueFormat"] = that.value_format

        if that.value_list is not None:
            jsonable["valueList"] = self.transform(that.value_list)

        if that.value is not None:
            jsonable["value"] = that.value

        if that.level_type is not None:
            jsonable["levelType"] = self.transform(that.level_type)

        jsonable["modelType"] = "DataSpecificationIec61360"

        return jsonable


_SERIALIZER = _Serializer()


def to_jsonable(that):

    return _SERIALIZER.transform(that)
