import binascii
import io
import math
import os
import sys


import aas_core3.stringification as aas_stringification
import aas_core3.types as aas_types


NAMESPACE = "https://admin-shell.io/aas/3/0"


class _Serializer(aas_types.AbstractVisitor):

    def _write_first_start_element_with_namespace(self, name):

        self.stream.write(f'<{name} xmlns="{NAMESPACE}">')

        self._write_start_element = self._write_start_element_without_namespace
        self._write_empty_element = self._write_empty_element_without_namespace

    def _write_start_element_without_namespace(self, name):

        self.stream.write(f"<{name}>")

    def _escape_and_write_text(self, text):

        self.stream.write(
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    def _write_end_element(self, name):

        self.stream.write(f"</{name}>")

    def _write_first_empty_element_with_namespace(self, name):

        self.stream.write(f'<{name} xmlns="{NAMESPACE}"/>')
        self._write_empty_element = self._rase_if_write_element_called_again
        self._write_start_element = self._rase_if_write_element_called_again

    def _rase_if_write_element_called_again(self, name):
        raise AssertionError(
            f"We expected to call ``_write_first_empty_element_with_namespace`` only once. This is an unexpected second call for writing an (empty or non-empty) element with the tag name: {name}"
        )

    def _write_empty_element_without_namespace(self, name):

        self.stream.write(f"<{name}/>")

    def _write_bool_property(self, name, value):

        self._write_start_element(name)
        self.stream.write("true" if value else "false")
        self._write_end_element(name)

    def _write_int_property(self, name, value):

        self._write_start_element(name)
        self.stream.write(str(value))
        self._write_end_element(name)

    def _write_float_property(self, name, value):

        self._write_start_element(name)

        if value == math.inf:
            self.stream.write("INF")
        elif value == -math.inf:
            self.stream.write("-INF")
        elif math.isnan(value):
            self.stream.write("NaN")
        elif value == 0:
            if math.copysign(1.0, value) < 0.0:
                self.stream.write("-0.0")
            else:
                self.stream.write("0.0")
        else:
            self.stream.write(str(value))

    def _write_str_property(self, name, value):

        self._write_start_element(name)
        self._escape_and_write_text(value)
        self._write_end_element(name)

    def _write_bytes_property(self, name, value):

        self._write_start_element(name)

        encoded = binascii.b2a_base64(value).decode("ascii")

        self.stream.write(encoded)
        self._write_end_element(name)

    def __init__(self, stream):

        self.stream = stream
        self._write_start_element = self._write_first_start_element_with_namespace
        self._write_empty_element = self._write_first_empty_element_with_namespace

    def _write_extension_as_sequence(self, that):

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for an_item in that.supplemental_semantic_ids:
                    self.visit(an_item)
                self._write_end_element("supplementalSemanticIds")

        self._write_str_property("name", that.name)

        if that.value_type is not None:
            self._write_str_property("valueType", that.value_type.value)

        if that.value is not None:
            self._write_str_property("value", that.value)

        if that.refers_to is not None:
            if len(that.refers_to) == 0:
                self._write_empty_element("refersTo")
            else:
                self._write_start_element("refersTo")
                for another_item in that.refers_to:
                    self.visit(another_item)
                self._write_end_element("refersTo")

    def visit_extension(self, that):

        self._write_start_element("extension")
        self._write_extension_as_sequence(that)
        self._write_end_element("extension")

    def _write_administrative_information_as_sequence(self, that):

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for an_item in that.embedded_data_specifications:
                    self.visit(an_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.version is not None:
            self._write_str_property("version", that.version)

        if that.revision is not None:
            self._write_str_property("revision", that.revision)

        if that.creator is not None:
            self._write_start_element("creator")
            self._write_reference_as_sequence(that.creator)
            self._write_end_element("creator")

        if that.template_id is not None:
            self._write_str_property("templateId", that.template_id)

    def visit_administrative_information(self, that):

        if (
            that.embedded_data_specifications is None
            and that.version is None
            and that.revision is None
            and that.creator is None
            and that.template_id is None
        ):
            self._write_empty_element("administrativeInformation")
        else:
            self._write_start_element("administrativeInformation")
            self._write_administrative_information_as_sequence(that)
            self._write_end_element("administrativeInformation")

    def _write_qualifier_as_sequence(self, that):

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for an_item in that.supplemental_semantic_ids:
                    self.visit(an_item)
                self._write_end_element("supplementalSemanticIds")

        if that.kind is not None:
            self._write_str_property("kind", that.kind.value)

        self._write_str_property("type", that.type)

        self._write_str_property("valueType", that.value_type.value)

        if that.value is not None:
            self._write_str_property("value", that.value)

        if that.value_id is not None:
            self._write_start_element("valueId")
            self._write_reference_as_sequence(that.value_id)
            self._write_end_element("valueId")

    def visit_qualifier(self, that):

        self._write_start_element("qualifier")
        self._write_qualifier_as_sequence(that)
        self._write_end_element("qualifier")

    def _write_asset_administration_shell_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.administration is not None:
            the_administration = that.administration

            if (
                the_administration.embedded_data_specifications is None
                and the_administration.version is None
                and the_administration.revision is None
                and the_administration.creator is None
                and the_administration.template_id is None
            ):
                self._write_empty_element("administration")
            else:
                self._write_start_element("administration")
                self._write_administrative_information_as_sequence(the_administration)
                self._write_end_element("administration")

        self._write_str_property("id", that.id)

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.derived_from is not None:
            self._write_start_element("derivedFrom")
            self._write_reference_as_sequence(that.derived_from)
            self._write_end_element("derivedFrom")

        self._write_start_element("assetInformation")
        self._write_asset_information_as_sequence(that.asset_information)
        self._write_end_element("assetInformation")

        if that.submodels is not None:
            if len(that.submodels) == 0:
                self._write_empty_element("submodels")
            else:
                self._write_start_element("submodels")
                for yet_yet_yet_another_item in that.submodels:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("submodels")

    def visit_asset_administration_shell(self, that):

        self._write_start_element("assetAdministrationShell")
        self._write_asset_administration_shell_as_sequence(that)
        self._write_end_element("assetAdministrationShell")

    def _write_asset_information_as_sequence(self, that):

        self._write_str_property("assetKind", that.asset_kind.value)

        if that.global_asset_id is not None:
            self._write_str_property("globalAssetId", that.global_asset_id)

        if that.specific_asset_ids is not None:
            if len(that.specific_asset_ids) == 0:
                self._write_empty_element("specificAssetIds")
            else:
                self._write_start_element("specificAssetIds")
                for an_item in that.specific_asset_ids:
                    self.visit(an_item)
                self._write_end_element("specificAssetIds")

        if that.asset_type is not None:
            self._write_str_property("assetType", that.asset_type)

        if that.default_thumbnail is not None:
            self._write_start_element("defaultThumbnail")
            self._write_resource_as_sequence(that.default_thumbnail)
            self._write_end_element("defaultThumbnail")

    def visit_asset_information(self, that):

        self._write_start_element("assetInformation")
        self._write_asset_information_as_sequence(that)
        self._write_end_element("assetInformation")

    def _write_resource_as_sequence(self, that):

        self._write_str_property("path", that.path)

        if that.content_type is not None:
            self._write_str_property("contentType", that.content_type)

    def visit_resource(self, that):

        self._write_start_element("resource")
        self._write_resource_as_sequence(that)
        self._write_end_element("resource")

    def _write_specific_asset_id_as_sequence(self, that):

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for an_item in that.supplemental_semantic_ids:
                    self.visit(an_item)
                self._write_end_element("supplementalSemanticIds")

        self._write_str_property("name", that.name)

        self._write_str_property("value", that.value)

        if that.external_subject_id is not None:
            self._write_start_element("externalSubjectId")
            self._write_reference_as_sequence(that.external_subject_id)
            self._write_end_element("externalSubjectId")

    def visit_specific_asset_id(self, that):

        self._write_start_element("specificAssetId")
        self._write_specific_asset_id_as_sequence(that)
        self._write_end_element("specificAssetId")

    def _write_submodel_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.administration is not None:
            the_administration = that.administration

            if (
                the_administration.embedded_data_specifications is None
                and the_administration.version is None
                and the_administration.revision is None
                and the_administration.creator is None
                and the_administration.template_id is None
            ):
                self._write_empty_element("administration")
            else:
                self._write_start_element("administration")
                self._write_administrative_information_as_sequence(the_administration)
                self._write_end_element("administration")

        self._write_str_property("id", that.id)

        if that.kind is not None:
            self._write_str_property("kind", that.kind.value)

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.submodel_elements is not None:
            if len(that.submodel_elements) == 0:
                self._write_empty_element("submodelElements")
            else:
                self._write_start_element("submodelElements")
                for yet_yet_yet_yet_yet_another_item in that.submodel_elements:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("submodelElements")

    def visit_submodel(self, that):

        self._write_start_element("submodel")
        self._write_submodel_as_sequence(that)
        self._write_end_element("submodel")

    def _write_relationship_element_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        self._write_start_element("first")
        self._write_reference_as_sequence(that.first)
        self._write_end_element("first")

        self._write_start_element("second")
        self._write_reference_as_sequence(that.second)
        self._write_end_element("second")

    def visit_relationship_element(self, that):

        self._write_start_element("relationshipElement")
        self._write_relationship_element_as_sequence(that)
        self._write_end_element("relationshipElement")

    def _write_submodel_element_list_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.order_relevant is not None:
            self._write_bool_property("orderRelevant", that.order_relevant)

        if that.semantic_id_list_element is not None:
            self._write_start_element("semanticIdListElement")
            self._write_reference_as_sequence(that.semantic_id_list_element)
            self._write_end_element("semanticIdListElement")

        self._write_str_property(
            "typeValueListElement", that.type_value_list_element.value
        )

        if that.value_type_list_element is not None:
            self._write_str_property(
                "valueTypeListElement", that.value_type_list_element.value
            )

        if that.value is not None:
            if len(that.value) == 0:
                self._write_empty_element("value")
            else:
                self._write_start_element("value")
                for yet_yet_yet_yet_yet_another_item in that.value:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("value")

    def visit_submodel_element_list(self, that):

        self._write_start_element("submodelElementList")
        self._write_submodel_element_list_as_sequence(that)
        self._write_end_element("submodelElementList")

    def _write_submodel_element_collection_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.value is not None:
            if len(that.value) == 0:
                self._write_empty_element("value")
            else:
                self._write_start_element("value")
                for yet_yet_yet_yet_yet_another_item in that.value:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("value")

    def visit_submodel_element_collection(self, that):

        if (
            that.extensions is None
            and that.category is None
            and that.id_short is None
            and that.display_name is None
            and that.description is None
            and that.semantic_id is None
            and that.supplemental_semantic_ids is None
            and that.qualifiers is None
            and that.embedded_data_specifications is None
            and that.value is None
        ):
            self._write_empty_element("submodelElementCollection")
        else:
            self._write_start_element("submodelElementCollection")
            self._write_submodel_element_collection_as_sequence(that)
            self._write_end_element("submodelElementCollection")

    def _write_property_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        self._write_str_property("valueType", that.value_type.value)

        if that.value is not None:
            self._write_str_property("value", that.value)

        if that.value_id is not None:
            self._write_start_element("valueId")
            self._write_reference_as_sequence(that.value_id)
            self._write_end_element("valueId")

    def visit_property(self, that):

        self._write_start_element("property")
        self._write_property_as_sequence(that)
        self._write_end_element("property")

    def _write_multi_language_property_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.value is not None:
            if len(that.value) == 0:
                self._write_empty_element("value")
            else:
                self._write_start_element("value")
                for yet_yet_yet_yet_yet_another_item in that.value:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("value")

        if that.value_id is not None:
            self._write_start_element("valueId")
            self._write_reference_as_sequence(that.value_id)
            self._write_end_element("valueId")

    def visit_multi_language_property(self, that):

        if (
            that.extensions is None
            and that.category is None
            and that.id_short is None
            and that.display_name is None
            and that.description is None
            and that.semantic_id is None
            and that.supplemental_semantic_ids is None
            and that.qualifiers is None
            and that.embedded_data_specifications is None
            and that.value is None
            and that.value_id is None
        ):
            self._write_empty_element("multiLanguageProperty")
        else:
            self._write_start_element("multiLanguageProperty")
            self._write_multi_language_property_as_sequence(that)
            self._write_end_element("multiLanguageProperty")

    def _write_range_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        self._write_str_property("valueType", that.value_type.value)

        if that.min is not None:
            self._write_str_property("min", that.min)

        if that.max is not None:
            self._write_str_property("max", that.max)

    def visit_range(self, that):

        self._write_start_element("range")
        self._write_range_as_sequence(that)
        self._write_end_element("range")

    def _write_reference_element_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.value is not None:
            self._write_start_element("value")
            self._write_reference_as_sequence(that.value)
            self._write_end_element("value")

    def visit_reference_element(self, that):

        if (
            that.extensions is None
            and that.category is None
            and that.id_short is None
            and that.display_name is None
            and that.description is None
            and that.semantic_id is None
            and that.supplemental_semantic_ids is None
            and that.qualifiers is None
            and that.embedded_data_specifications is None
            and that.value is None
        ):
            self._write_empty_element("referenceElement")
        else:
            self._write_start_element("referenceElement")
            self._write_reference_element_as_sequence(that)
            self._write_end_element("referenceElement")

    def _write_blob_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.value is not None:
            self._write_bytes_property("value", that.value)

        self._write_str_property("contentType", that.content_type)

    def visit_blob(self, that):

        self._write_start_element("blob")
        self._write_blob_as_sequence(that)
        self._write_end_element("blob")

    def _write_file_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.value is not None:
            self._write_str_property("value", that.value)

        self._write_str_property("contentType", that.content_type)

    def visit_file(self, that):

        self._write_start_element("file")
        self._write_file_as_sequence(that)
        self._write_end_element("file")

    def _write_annotated_relationship_element_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        self._write_start_element("first")
        self._write_reference_as_sequence(that.first)
        self._write_end_element("first")

        self._write_start_element("second")
        self._write_reference_as_sequence(that.second)
        self._write_end_element("second")

        if that.annotations is not None:
            if len(that.annotations) == 0:
                self._write_empty_element("annotations")
            else:
                self._write_start_element("annotations")
                for yet_yet_yet_yet_yet_another_item in that.annotations:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("annotations")

    def visit_annotated_relationship_element(self, that):

        self._write_start_element("annotatedRelationshipElement")
        self._write_annotated_relationship_element_as_sequence(that)
        self._write_end_element("annotatedRelationshipElement")

    def _write_entity_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.statements is not None:
            if len(that.statements) == 0:
                self._write_empty_element("statements")
            else:
                self._write_start_element("statements")
                for yet_yet_yet_yet_yet_another_item in that.statements:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("statements")

        self._write_str_property("entityType", that.entity_type.value)

        if that.global_asset_id is not None:
            self._write_str_property("globalAssetId", that.global_asset_id)

        if that.specific_asset_ids is not None:
            if len(that.specific_asset_ids) == 0:
                self._write_empty_element("specificAssetIds")
            else:
                self._write_start_element("specificAssetIds")
                for yet_yet_yet_yet_yet_yet_another_item in that.specific_asset_ids:
                    self.visit(yet_yet_yet_yet_yet_yet_another_item)
                self._write_end_element("specificAssetIds")

    def visit_entity(self, that):

        self._write_start_element("entity")
        self._write_entity_as_sequence(that)
        self._write_end_element("entity")

    def _write_event_payload_as_sequence(self, that):

        self._write_start_element("source")
        self._write_reference_as_sequence(that.source)
        self._write_end_element("source")

        if that.source_semantic_id is not None:
            self._write_start_element("sourceSemanticId")
            self._write_reference_as_sequence(that.source_semantic_id)
            self._write_end_element("sourceSemanticId")

        self._write_start_element("observableReference")
        self._write_reference_as_sequence(that.observable_reference)
        self._write_end_element("observableReference")

        if that.observable_semantic_id is not None:
            self._write_start_element("observableSemanticId")
            self._write_reference_as_sequence(that.observable_semantic_id)
            self._write_end_element("observableSemanticId")

        if that.topic is not None:
            self._write_str_property("topic", that.topic)

        if that.subject_id is not None:
            self._write_start_element("subjectId")
            self._write_reference_as_sequence(that.subject_id)
            self._write_end_element("subjectId")

        self._write_str_property("timeStamp", that.time_stamp)

        if that.payload is not None:
            self._write_bytes_property("payload", that.payload)

    def visit_event_payload(self, that):

        self._write_start_element("eventPayload")
        self._write_event_payload_as_sequence(that)
        self._write_end_element("eventPayload")

    def _write_basic_event_element_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        self._write_start_element("observed")
        self._write_reference_as_sequence(that.observed)
        self._write_end_element("observed")

        self._write_str_property("direction", that.direction.value)

        self._write_str_property("state", that.state.value)

        if that.message_topic is not None:
            self._write_str_property("messageTopic", that.message_topic)

        if that.message_broker is not None:
            self._write_start_element("messageBroker")
            self._write_reference_as_sequence(that.message_broker)
            self._write_end_element("messageBroker")

        if that.last_update is not None:
            self._write_str_property("lastUpdate", that.last_update)

        if that.min_interval is not None:
            self._write_str_property("minInterval", that.min_interval)

        if that.max_interval is not None:
            self._write_str_property("maxInterval", that.max_interval)

    def visit_basic_event_element(self, that):

        self._write_start_element("basicEventElement")
        self._write_basic_event_element_as_sequence(that)
        self._write_end_element("basicEventElement")

    def _write_operation_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.input_variables is not None:
            if len(that.input_variables) == 0:
                self._write_empty_element("inputVariables")
            else:
                self._write_start_element("inputVariables")
                for yet_yet_yet_yet_yet_another_item in that.input_variables:
                    self.visit(yet_yet_yet_yet_yet_another_item)
                self._write_end_element("inputVariables")

        if that.output_variables is not None:
            if len(that.output_variables) == 0:
                self._write_empty_element("outputVariables")
            else:
                self._write_start_element("outputVariables")
                for yet_yet_yet_yet_yet_yet_another_item in that.output_variables:
                    self.visit(yet_yet_yet_yet_yet_yet_another_item)
                self._write_end_element("outputVariables")

        if that.inoutput_variables is not None:
            if len(that.inoutput_variables) == 0:
                self._write_empty_element("inoutputVariables")
            else:
                self._write_start_element("inoutputVariables")
                for yet_yet_yet_yet_yet_yet_yet_another_item in that.inoutput_variables:
                    self.visit(yet_yet_yet_yet_yet_yet_yet_another_item)
                self._write_end_element("inoutputVariables")

    def visit_operation(self, that):

        if (
            that.extensions is None
            and that.category is None
            and that.id_short is None
            and that.display_name is None
            and that.description is None
            and that.semantic_id is None
            and that.supplemental_semantic_ids is None
            and that.qualifiers is None
            and that.embedded_data_specifications is None
            and that.input_variables is None
            and that.output_variables is None
            and that.inoutput_variables is None
        ):
            self._write_empty_element("operation")
        else:
            self._write_start_element("operation")
            self._write_operation_as_sequence(that)
            self._write_end_element("operation")

    def _write_operation_variable_as_sequence(self, that):

        self._write_start_element("value")
        self.visit(that.value)
        self._write_end_element("value")

    def visit_operation_variable(self, that):

        self._write_start_element("operationVariable")
        self._write_operation_variable_as_sequence(that)
        self._write_end_element("operationVariable")

    def _write_capability_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.semantic_id is not None:
            self._write_start_element("semanticId")
            self._write_reference_as_sequence(that.semantic_id)
            self._write_end_element("semanticId")

        if that.supplemental_semantic_ids is not None:
            if len(that.supplemental_semantic_ids) == 0:
                self._write_empty_element("supplementalSemanticIds")
            else:
                self._write_start_element("supplementalSemanticIds")
                for yet_yet_another_item in that.supplemental_semantic_ids:
                    self.visit(yet_yet_another_item)
                self._write_end_element("supplementalSemanticIds")

        if that.qualifiers is not None:
            if len(that.qualifiers) == 0:
                self._write_empty_element("qualifiers")
            else:
                self._write_start_element("qualifiers")
                for yet_yet_yet_another_item in that.qualifiers:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("qualifiers")

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

    def visit_capability(self, that):

        if (
            that.extensions is None
            and that.category is None
            and that.id_short is None
            and that.display_name is None
            and that.description is None
            and that.semantic_id is None
            and that.supplemental_semantic_ids is None
            and that.qualifiers is None
            and that.embedded_data_specifications is None
        ):
            self._write_empty_element("capability")
        else:
            self._write_start_element("capability")
            self._write_capability_as_sequence(that)
            self._write_end_element("capability")

    def _write_concept_description_as_sequence(self, that):

        if that.extensions is not None:
            if len(that.extensions) == 0:
                self._write_empty_element("extensions")
            else:
                self._write_start_element("extensions")
                for an_item in that.extensions:
                    self.visit(an_item)
                self._write_end_element("extensions")

        if that.category is not None:
            self._write_str_property("category", that.category)

        if that.id_short is not None:
            self._write_str_property("idShort", that.id_short)

        if that.display_name is not None:
            if len(that.display_name) == 0:
                self._write_empty_element("displayName")
            else:
                self._write_start_element("displayName")
                for another_item in that.display_name:
                    self.visit(another_item)
                self._write_end_element("displayName")

        if that.description is not None:
            if len(that.description) == 0:
                self._write_empty_element("description")
            else:
                self._write_start_element("description")
                for yet_another_item in that.description:
                    self.visit(yet_another_item)
                self._write_end_element("description")

        if that.administration is not None:
            the_administration = that.administration

            if (
                the_administration.embedded_data_specifications is None
                and the_administration.version is None
                and the_administration.revision is None
                and the_administration.creator is None
                and the_administration.template_id is None
            ):
                self._write_empty_element("administration")
            else:
                self._write_start_element("administration")
                self._write_administrative_information_as_sequence(the_administration)
                self._write_end_element("administration")

        self._write_str_property("id", that.id)

        if that.embedded_data_specifications is not None:
            if len(that.embedded_data_specifications) == 0:
                self._write_empty_element("embeddedDataSpecifications")
            else:
                self._write_start_element("embeddedDataSpecifications")
                for yet_yet_another_item in that.embedded_data_specifications:
                    self.visit(yet_yet_another_item)
                self._write_end_element("embeddedDataSpecifications")

        if that.is_case_of is not None:
            if len(that.is_case_of) == 0:
                self._write_empty_element("isCaseOf")
            else:
                self._write_start_element("isCaseOf")
                for yet_yet_yet_another_item in that.is_case_of:
                    self.visit(yet_yet_yet_another_item)
                self._write_end_element("isCaseOf")

    def visit_concept_description(self, that):

        self._write_start_element("conceptDescription")
        self._write_concept_description_as_sequence(that)
        self._write_end_element("conceptDescription")

    def _write_reference_as_sequence(self, that):

        self._write_str_property("type", that.type.value)

        if that.referred_semantic_id is not None:
            self._write_start_element("referredSemanticId")
            self._write_reference_as_sequence(that.referred_semantic_id)
            self._write_end_element("referredSemanticId")

        if len(that.keys) == 0:
            self._write_empty_element("keys")
        else:
            self._write_start_element("keys")
            for an_item in that.keys:
                self.visit(an_item)
            self._write_end_element("keys")

    def visit_reference(self, that):

        self._write_start_element("reference")
        self._write_reference_as_sequence(that)
        self._write_end_element("reference")

    def _write_key_as_sequence(self, that):

        self._write_str_property("type", that.type.value)

        self._write_str_property("value", that.value)

    def visit_key(self, that):

        self._write_start_element("key")
        self._write_key_as_sequence(that)
        self._write_end_element("key")

    def _write_lang_string_name_type_as_sequence(self, that):

        self._write_str_property("language", that.language)

        self._write_str_property("text", that.text)

    def visit_lang_string_name_type(self, that):

        self._write_start_element("langStringNameType")
        self._write_lang_string_name_type_as_sequence(that)
        self._write_end_element("langStringNameType")

    def _write_lang_string_text_type_as_sequence(self, that):

        self._write_str_property("language", that.language)

        self._write_str_property("text", that.text)

    def visit_lang_string_text_type(self, that):

        self._write_start_element("langStringTextType")
        self._write_lang_string_text_type_as_sequence(that)
        self._write_end_element("langStringTextType")

    def _write_environment_as_sequence(self, that):

        if that.asset_administration_shells is not None:
            if len(that.asset_administration_shells) == 0:
                self._write_empty_element("assetAdministrationShells")
            else:
                self._write_start_element("assetAdministrationShells")
                for an_item in that.asset_administration_shells:
                    self.visit(an_item)
                self._write_end_element("assetAdministrationShells")

        if that.submodels is not None:
            if len(that.submodels) == 0:
                self._write_empty_element("submodels")
            else:
                self._write_start_element("submodels")
                for another_item in that.submodels:
                    self.visit(another_item)
                self._write_end_element("submodels")

        if that.concept_descriptions is not None:
            if len(that.concept_descriptions) == 0:
                self._write_empty_element("conceptDescriptions")
            else:
                self._write_start_element("conceptDescriptions")
                for yet_another_item in that.concept_descriptions:
                    self.visit(yet_another_item)
                self._write_end_element("conceptDescriptions")

    def visit_environment(self, that):

        if (
            that.asset_administration_shells is None
            and that.submodels is None
            and that.concept_descriptions is None
        ):
            self._write_empty_element("environment")
        else:
            self._write_start_element("environment")
            self._write_environment_as_sequence(that)
            self._write_end_element("environment")

    def _write_embedded_data_specification_as_sequence(self, that):

        self._write_start_element("dataSpecification")
        self._write_reference_as_sequence(that.data_specification)
        self._write_end_element("dataSpecification")

        self._write_start_element("dataSpecificationContent")
        self.visit(that.data_specification_content)
        self._write_end_element("dataSpecificationContent")

    def visit_embedded_data_specification(self, that):

        self._write_start_element("embeddedDataSpecification")
        self._write_embedded_data_specification_as_sequence(that)
        self._write_end_element("embeddedDataSpecification")

    def _write_level_type_as_sequence(self, that):

        self._write_bool_property("min", that.min)

        self._write_bool_property("nom", that.nom)

        self._write_bool_property("typ", that.typ)

        self._write_bool_property("max", that.max)

    def visit_level_type(self, that):

        self._write_start_element("levelType")
        self._write_level_type_as_sequence(that)
        self._write_end_element("levelType")

    def _write_value_reference_pair_as_sequence(self, that):

        self._write_str_property("value", that.value)

        self._write_start_element("valueId")
        self._write_reference_as_sequence(that.value_id)
        self._write_end_element("valueId")

    def visit_value_reference_pair(self, that):

        self._write_start_element("valueReferencePair")
        self._write_value_reference_pair_as_sequence(that)
        self._write_end_element("valueReferencePair")

    def _write_value_list_as_sequence(self, that):

        if len(that.value_reference_pairs) == 0:
            self._write_empty_element("valueReferencePairs")
        else:
            self._write_start_element("valueReferencePairs")
            for an_item in that.value_reference_pairs:
                self.visit(an_item)
            self._write_end_element("valueReferencePairs")

    def visit_value_list(self, that):

        self._write_start_element("valueList")
        self._write_value_list_as_sequence(that)
        self._write_end_element("valueList")

    def _write_lang_string_preferred_name_type_iec_61360_as_sequence(self, that):

        self._write_str_property("language", that.language)

        self._write_str_property("text", that.text)

    def visit_lang_string_preferred_name_type_iec_61360(self, that):

        self._write_start_element("langStringPreferredNameTypeIec61360")
        self._write_lang_string_preferred_name_type_iec_61360_as_sequence(that)
        self._write_end_element("langStringPreferredNameTypeIec61360")

    def _write_lang_string_short_name_type_iec_61360_as_sequence(self, that):

        self._write_str_property("language", that.language)

        self._write_str_property("text", that.text)

    def visit_lang_string_short_name_type_iec_61360(self, that):

        self._write_start_element("langStringShortNameTypeIec61360")
        self._write_lang_string_short_name_type_iec_61360_as_sequence(that)
        self._write_end_element("langStringShortNameTypeIec61360")

    def _write_lang_string_definition_type_iec_61360_as_sequence(self, that):

        self._write_str_property("language", that.language)

        self._write_str_property("text", that.text)

    def visit_lang_string_definition_type_iec_61360(self, that):

        self._write_start_element("langStringDefinitionTypeIec61360")
        self._write_lang_string_definition_type_iec_61360_as_sequence(that)
        self._write_end_element("langStringDefinitionTypeIec61360")

    def _write_data_specification_iec_61360_as_sequence(self, that):

        if len(that.preferred_name) == 0:
            self._write_empty_element("preferredName")
        else:
            self._write_start_element("preferredName")
            for an_item in that.preferred_name:
                self.visit(an_item)
            self._write_end_element("preferredName")

        if that.short_name is not None:
            if len(that.short_name) == 0:
                self._write_empty_element("shortName")
            else:
                self._write_start_element("shortName")
                for another_item in that.short_name:
                    self.visit(another_item)
                self._write_end_element("shortName")

        if that.unit is not None:
            self._write_str_property("unit", that.unit)

        if that.unit_id is not None:
            self._write_start_element("unitId")
            self._write_reference_as_sequence(that.unit_id)
            self._write_end_element("unitId")

        if that.source_of_definition is not None:
            self._write_str_property("sourceOfDefinition", that.source_of_definition)

        if that.symbol is not None:
            self._write_str_property("symbol", that.symbol)

        if that.data_type is not None:
            self._write_str_property("dataType", that.data_type.value)

        if that.definition is not None:
            if len(that.definition) == 0:
                self._write_empty_element("definition")
            else:
                self._write_start_element("definition")
                for yet_another_item in that.definition:
                    self.visit(yet_another_item)
                self._write_end_element("definition")

        if that.value_format is not None:
            self._write_str_property("valueFormat", that.value_format)

        if that.value_list is not None:
            self._write_start_element("valueList")
            self._write_value_list_as_sequence(that.value_list)
            self._write_end_element("valueList")

        if that.value is not None:
            self._write_str_property("value", that.value)

        if that.level_type is not None:
            self._write_start_element("levelType")
            self._write_level_type_as_sequence(that.level_type)
            self._write_end_element("levelType")

    def visit_data_specification_iec_61360(self, that):

        self._write_start_element("dataSpecificationIec61360")
        self._write_data_specification_iec_61360_as_sequence(that)
        self._write_end_element("dataSpecificationIec61360")


def write(instance, stream):

    serializer = _Serializer(stream)
    serializer.visit(instance)


def to_str(that):

    writer = io.StringIO()
    write(that, writer)
    return writer.getvalue()
