from aas_core3 import enum as aas_enum


class Class:

    def descend_once(self):

        raise NotImplementedError()

    def descend(self):

        raise NotImplementedError()

    def accept(self, visitor):

        raise NotImplementedError()

    def accept_with_context(self, visitor, context):

        raise NotImplementedError()

    def transform(self, transformer):

        raise NotImplementedError()

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        raise NotImplementedError()


class HasSemantics(Class):

    def over_supplemental_semantic_ids_or_empty(self):

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

    def __init__(
        self,
        semantic_id=None,
        supplemental_semantic_ids=None,
    ):

        self.semantic_id = semantic_id
        self.supplemental_semantic_ids = supplemental_semantic_ids


class Extension(HasSemantics):

    def over_refers_to_or_empty(self):

        if self.refers_to is not None:
            yield from self.refers_to

    def value_type_or_default(self):

        return self.value_type if self.value_type is not None else DataTypeDefXSD.STRING

    def descend_once(self):

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.refers_to is not None:
            yield from self.refers_to

    def descend(self):

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for an_item in self.supplemental_semantic_ids:
                yield an_item

                yield from an_item.descend()

        if self.refers_to is not None:
            for another_item in self.refers_to:
                yield another_item

                yield from another_item.descend()

    def accept(self, visitor):

        visitor.visit_extension(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_extension_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_extension(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_extension_with_context(self, context)

    def __init__(
        self,
        name,
        semantic_id=None,
        supplemental_semantic_ids=None,
        value_type=None,
        value=None,
        refers_to=None,
    ):

        HasSemantics.__init__(self, semantic_id, supplemental_semantic_ids)
        self.name = name
        self.value_type = value_type
        self.value = value
        self.refers_to = refers_to


class HasExtensions(Class):

    def over_extensions_or_empty(self):

        if self.extensions is not None:
            yield from self.extensions

    def __init__(self, extensions=None):

        self.extensions = extensions


class Referable(HasExtensions):

    def over_display_name_or_empty(self):

        if self.display_name is not None:
            yield from self.display_name

    def over_description_or_empty(self):

        if self.description is not None:
            yield from self.description

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
    ):

        HasExtensions.__init__(self, extensions)
        self.id_short = id_short
        self.display_name = display_name
        self.category = category
        self.description = description


class Identifiable(Referable):

    def __init__(
        self,
        id,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        administration=None,
    ):

        Referable.__init__(
            self, extensions, category, id_short, display_name, description
        )
        self.id = id
        self.administration = administration


@aas_enum.decorator
class ModellingKind:

    TEMPLATE = "Template"

    INSTANCE = "Instance"


class HasKind(Class):

    def kind_or_default(self):

        return self.kind if self.kind is not None else ModellingKind.INSTANCE

    def __init__(self, kind=None):

        self.kind = kind


class HasDataSpecification(Class):

    def over_embedded_data_specifications_or_empty(
        self,
    ):

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

    def __init__(
        self,
        embedded_data_specifications=None,
    ):

        self.embedded_data_specifications = embedded_data_specifications


class AdministrativeInformation(HasDataSpecification):

    def descend_once(self):

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.creator is not None:
            yield self.creator

    def descend(self):

        if self.embedded_data_specifications is not None:
            for an_item in self.embedded_data_specifications:
                yield an_item

                yield from an_item.descend()

        if self.creator is not None:
            yield self.creator

            yield from self.creator.descend()

    def accept(self, visitor):

        visitor.visit_administrative_information(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_administrative_information_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_administrative_information(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_administrative_information_with_context(
            self, context
        )

    def __init__(
        self,
        embedded_data_specifications=None,
        version=None,
        revision=None,
        creator=None,
        template_id=None,
    ):

        HasDataSpecification.__init__(self, embedded_data_specifications)
        self.version = version
        self.revision = revision
        self.creator = creator
        self.template_id = template_id


class Qualifiable(Class):

    def over_qualifiers_or_empty(self):

        if self.qualifiers is not None:
            yield from self.qualifiers

    def __init__(self, qualifiers=None):

        self.qualifiers = qualifiers


@aas_enum.decorator
class QualifierKind:

    VALUE_QUALIFIER = "ValueQualifier"

    CONCEPT_QUALIFIER = "ConceptQualifier"

    TEMPLATE_QUALIFIER = "TemplateQualifier"


class Qualifier(HasSemantics):

    def kind_or_default(self):

        return self.kind if self.kind is not None else QualifierKind.CONCEPT_QUALIFIER

    def descend_once(self):

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.value_id is not None:
            yield self.value_id

    def descend(self):

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for an_item in self.supplemental_semantic_ids:
                yield an_item

                yield from an_item.descend()

        if self.value_id is not None:
            yield self.value_id

            yield from self.value_id.descend()

    def accept(self, visitor):

        visitor.visit_qualifier(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_qualifier_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_qualifier(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_qualifier_with_context(self, context)

    def __init__(
        self,
        type,
        value_type,
        semantic_id=None,
        supplemental_semantic_ids=None,
        kind=None,
        value=None,
        value_id=None,
    ):

        HasSemantics.__init__(self, semantic_id, supplemental_semantic_ids)
        self.type = type
        self.value_type = value_type
        self.kind = kind
        self.value = value
        self.value_id = value_id


class AssetAdministrationShell(Identifiable, HasDataSpecification):

    def over_submodels_or_empty(self):

        if self.submodels is not None:
            yield from self.submodels

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.administration is not None:
            yield self.administration

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.derived_from is not None:
            yield self.derived_from

        yield self.asset_information

        if self.submodels is not None:
            yield from self.submodels

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.administration is not None:
            yield self.administration

            yield from self.administration.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.derived_from is not None:
            yield self.derived_from

            yield from self.derived_from.descend()

        yield self.asset_information

        yield from self.asset_information.descend()

        if self.submodels is not None:
            for yet_yet_yet_another_item in self.submodels:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_asset_administration_shell(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_asset_administration_shell_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_asset_administration_shell(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_asset_administration_shell_with_context(
            self, context
        )

    def __init__(
        self,
        id,
        asset_information,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        administration=None,
        embedded_data_specifications=None,
        derived_from=None,
        submodels=None,
    ):

        Identifiable.__init__(
            self,
            id,
            extensions,
            category,
            id_short,
            display_name,
            description,
            administration,
        )
        HasDataSpecification.__init__(self, embedded_data_specifications)
        self.derived_from = derived_from
        self.asset_information = asset_information
        self.submodels = submodels


class AssetInformation(Class):

    def over_specific_asset_ids_or_empty(self):

        if self.specific_asset_ids is not None:
            yield from self.specific_asset_ids

    def descend_once(self):

        if self.specific_asset_ids is not None:
            yield from self.specific_asset_ids

        if self.default_thumbnail is not None:
            yield self.default_thumbnail

    def descend(self):

        if self.specific_asset_ids is not None:
            for an_item in self.specific_asset_ids:
                yield an_item

                yield from an_item.descend()

        if self.default_thumbnail is not None:
            yield self.default_thumbnail

            yield from self.default_thumbnail.descend()

    def accept(self, visitor):

        visitor.visit_asset_information(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_asset_information_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_asset_information(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_asset_information_with_context(self, context)

    def __init__(
        self,
        asset_kind,
        global_asset_id=None,
        specific_asset_ids=None,
        asset_type=None,
        default_thumbnail=None,
    ):

        self.asset_kind = asset_kind
        self.global_asset_id = global_asset_id
        self.specific_asset_ids = specific_asset_ids
        self.asset_type = asset_type
        self.default_thumbnail = default_thumbnail


class Resource(Class):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_resource(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_resource_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_resource(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_resource_with_context(self, context)

    def __init__(self, path, content_type=None):

        self.path = path
        self.content_type = content_type


@aas_enum.decorator
class AssetKind:

    TYPE = "Type"

    INSTANCE = "Instance"

    NOT_APPLICABLE = "NotApplicable"


class SpecificAssetID(HasSemantics):

    def descend_once(self):

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.external_subject_id is not None:
            yield self.external_subject_id

    def descend(self):

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for an_item in self.supplemental_semantic_ids:
                yield an_item

                yield from an_item.descend()

        if self.external_subject_id is not None:
            yield self.external_subject_id

            yield from self.external_subject_id.descend()

    def accept(self, visitor):

        visitor.visit_specific_asset_id(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_specific_asset_id_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_specific_asset_id(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_specific_asset_id_with_context(self, context)

    def __init__(
        self,
        name,
        value,
        semantic_id=None,
        supplemental_semantic_ids=None,
        external_subject_id=None,
    ):

        HasSemantics.__init__(self, semantic_id, supplemental_semantic_ids)
        self.name = name
        self.value = value
        self.external_subject_id = external_subject_id


class Submodel(Identifiable, HasKind, HasSemantics, Qualifiable, HasDataSpecification):

    def over_submodel_elements_or_empty(self):

        if self.submodel_elements is not None:
            yield from self.submodel_elements

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.administration is not None:
            yield self.administration

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.submodel_elements is not None:
            yield from self.submodel_elements

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.administration is not None:
            yield self.administration

            yield from self.administration.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.submodel_elements is not None:
            for yet_yet_yet_yet_yet_another_item in self.submodel_elements:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_submodel(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_submodel_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_submodel(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_submodel_with_context(self, context)

    def __init__(
        self,
        id,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        administration=None,
        kind=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        submodel_elements=None,
    ):

        Identifiable.__init__(
            self,
            id,
            extensions,
            category,
            id_short,
            display_name,
            description,
            administration,
        )
        HasKind.__init__(self, kind)
        HasSemantics.__init__(self, semantic_id, supplemental_semantic_ids)
        Qualifiable.__init__(self, qualifiers)
        HasDataSpecification.__init__(self, embedded_data_specifications)
        self.submodel_elements = submodel_elements


class SubmodelElement(Referable, HasSemantics, Qualifiable, HasDataSpecification):

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
    ):

        Referable.__init__(
            self, extensions, category, id_short, display_name, description
        )
        HasSemantics.__init__(self, semantic_id, supplemental_semantic_ids)
        Qualifiable.__init__(self, qualifiers)
        HasDataSpecification.__init__(self, embedded_data_specifications)


class RelationshipElement(SubmodelElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        yield self.first

        yield self.second

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        yield self.first

        yield from self.first.descend()

        yield self.second

        yield from self.second.descend()

    def accept(self, visitor):

        visitor.visit_relationship_element(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_relationship_element_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_relationship_element(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_relationship_element_with_context(self, context)

    def __init__(
        self,
        first,
        second,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.first = first
        self.second = second


@aas_enum.decorator
class AASSubmodelElements:

    ANNOTATED_RELATIONSHIP_ELEMENT = "AnnotatedRelationshipElement"

    BASIC_EVENT_ELEMENT = "BasicEventElement"

    BLOB = "Blob"

    CAPABILITY = "Capability"

    DATA_ELEMENT = "DataElement"

    ENTITY = "Entity"

    EVENT_ELEMENT = "EventElement"

    FILE = "File"

    MULTI_LANGUAGE_PROPERTY = "MultiLanguageProperty"

    OPERATION = "Operation"

    PROPERTY = "Property"

    RANGE = "Range"

    REFERENCE_ELEMENT = "ReferenceElement"

    RELATIONSHIP_ELEMENT = "RelationshipElement"

    SUBMODEL_ELEMENT = "SubmodelElement"

    SUBMODEL_ELEMENT_LIST = "SubmodelElementList"

    SUBMODEL_ELEMENT_COLLECTION = "SubmodelElementCollection"


class SubmodelElementList(SubmodelElement):

    def over_value_or_empty(self):

        if self.value is not None:
            yield from self.value

    def order_relevant_or_default(self):

        return self.order_relevant if self.order_relevant is not None else True

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.semantic_id_list_element is not None:
            yield self.semantic_id_list_element

        if self.value is not None:
            yield from self.value

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.semantic_id_list_element is not None:
            yield self.semantic_id_list_element

            yield from self.semantic_id_list_element.descend()

        if self.value is not None:
            for yet_yet_yet_yet_yet_another_item in self.value:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_submodel_element_list(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_submodel_element_list_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_submodel_element_list(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_submodel_element_list_with_context(self, context)

    def __init__(
        self,
        type_value_list_element,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        order_relevant=None,
        semantic_id_list_element=None,
        value_type_list_element=None,
        value=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.type_value_list_element = type_value_list_element
        self.order_relevant = order_relevant
        self.semantic_id_list_element = semantic_id_list_element
        self.value_type_list_element = value_type_list_element
        self.value = value


class SubmodelElementCollection(SubmodelElement):

    def over_value_or_empty(self):

        if self.value is not None:
            yield from self.value

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.value is not None:
            yield from self.value

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.value is not None:
            for yet_yet_yet_yet_yet_another_item in self.value:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_submodel_element_collection(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_submodel_element_collection_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_submodel_element_collection(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_submodel_element_collection_with_context(
            self, context
        )

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        value=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.value = value


class DataElement(SubmodelElement):

    def category_or_default(self):

        return self.category if self.category else "VARIABLE"

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )


class Property(DataElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.value_id is not None:
            yield self.value_id

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.value_id is not None:
            yield self.value_id

            yield from self.value_id.descend()

    def accept(self, visitor):

        visitor.visit_property(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_property_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_property(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_property_with_context(self, context)

    def __init__(
        self,
        value_type,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        value=None,
        value_id=None,
    ):

        DataElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.value_type = value_type
        self.value = value
        self.value_id = value_id


class MultiLanguageProperty(DataElement):

    def over_value_or_empty(self):

        if self.value is not None:
            yield from self.value

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.value is not None:
            yield from self.value

        if self.value_id is not None:
            yield self.value_id

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.value is not None:
            for yet_yet_yet_yet_yet_another_item in self.value:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

        if self.value_id is not None:
            yield self.value_id

            yield from self.value_id.descend()

    def accept(self, visitor):

        visitor.visit_multi_language_property(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_multi_language_property_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_multi_language_property(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_multi_language_property_with_context(self, context)

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        value=None,
        value_id=None,
    ):

        DataElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.value = value
        self.value_id = value_id


class Range(DataElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_range(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_range_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_range(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_range_with_context(self, context)

    def __init__(
        self,
        value_type,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        min=None,
        max=None,
    ):

        DataElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.value_type = value_type
        self.min = min
        self.max = max


class ReferenceElement(DataElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.value is not None:
            yield self.value

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.value is not None:
            yield self.value

            yield from self.value.descend()

    def accept(self, visitor):

        visitor.visit_reference_element(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_reference_element_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_reference_element(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_reference_element_with_context(self, context)

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        value=None,
    ):

        DataElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.value = value


class Blob(DataElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_blob(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_blob_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_blob(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_blob_with_context(self, context)

    def __init__(
        self,
        content_type,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        value=None,
    ):

        DataElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.content_type = content_type
        self.value = value


class File(DataElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_file(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_file_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_file(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_file_with_context(self, context)

    def __init__(
        self,
        content_type,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        value=None,
    ):

        DataElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.content_type = content_type
        self.value = value


class AnnotatedRelationshipElement(RelationshipElement):

    def over_annotations_or_empty(self):

        if self.annotations is not None:
            yield from self.annotations

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        yield self.first

        yield self.second

        if self.annotations is not None:
            yield from self.annotations

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        yield self.first

        yield from self.first.descend()

        yield self.second

        yield from self.second.descend()

        if self.annotations is not None:
            for yet_yet_yet_yet_yet_another_item in self.annotations:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_annotated_relationship_element(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_annotated_relationship_element_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_annotated_relationship_element(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_annotated_relationship_element_with_context(
            self, context
        )

    def __init__(
        self,
        first,
        second,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        annotations=None,
    ):

        RelationshipElement.__init__(
            self,
            first,
            second,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.annotations = annotations


class Entity(SubmodelElement):

    def over_statements_or_empty(self):

        if self.statements is not None:
            yield from self.statements

    def over_specific_asset_ids_or_empty(self):

        if self.specific_asset_ids is not None:
            yield from self.specific_asset_ids

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.statements is not None:
            yield from self.statements

        if self.specific_asset_ids is not None:
            yield from self.specific_asset_ids

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.statements is not None:
            for yet_yet_yet_yet_yet_another_item in self.statements:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

        if self.specific_asset_ids is not None:
            for yet_yet_yet_yet_yet_yet_another_item in self.specific_asset_ids:
                yield yet_yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_entity(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_entity_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_entity(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_entity_with_context(self, context)

    def __init__(
        self,
        entity_type,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        statements=None,
        global_asset_id=None,
        specific_asset_ids=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.statements = statements
        self.entity_type = entity_type
        self.global_asset_id = global_asset_id
        self.specific_asset_ids = specific_asset_ids


@aas_enum.decorator
class EntityType:

    CO_MANAGED_ENTITY = "CoManagedEntity"

    SELF_MANAGED_ENTITY = "SelfManagedEntity"


@aas_enum.decorator
class Direction:

    INPUT = "input"

    OUTPUT = "output"


@aas_enum.decorator
class StateOfEvent:

    ON = "on"

    OFF = "off"


class EventPayload(Class):

    def descend_once(self):

        yield self.source

        if self.source_semantic_id is not None:
            yield self.source_semantic_id

        yield self.observable_reference

        if self.observable_semantic_id is not None:
            yield self.observable_semantic_id

        if self.subject_id is not None:
            yield self.subject_id

    def descend(self):

        yield self.source

        yield from self.source.descend()

        if self.source_semantic_id is not None:
            yield self.source_semantic_id

            yield from self.source_semantic_id.descend()

        yield self.observable_reference

        yield from self.observable_reference.descend()

        if self.observable_semantic_id is not None:
            yield self.observable_semantic_id

            yield from self.observable_semantic_id.descend()

        if self.subject_id is not None:
            yield self.subject_id

            yield from self.subject_id.descend()

    def accept(self, visitor):

        visitor.visit_event_payload(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_event_payload_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_event_payload(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_event_payload_with_context(self, context)

    def __init__(
        self,
        source,
        observable_reference,
        time_stamp,
        source_semantic_id=None,
        observable_semantic_id=None,
        topic=None,
        subject_id=None,
        payload=None,
    ):

        self.source = source
        self.observable_reference = observable_reference
        self.time_stamp = time_stamp
        self.source_semantic_id = source_semantic_id
        self.observable_semantic_id = observable_semantic_id
        self.topic = topic
        self.subject_id = subject_id
        self.payload = payload


class EventElement(SubmodelElement):

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )


class BasicEventElement(EventElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        yield self.observed

        if self.message_broker is not None:
            yield self.message_broker

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        yield self.observed

        yield from self.observed.descend()

        if self.message_broker is not None:
            yield self.message_broker

            yield from self.message_broker.descend()

    def accept(self, visitor):

        visitor.visit_basic_event_element(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_basic_event_element_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_basic_event_element(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_basic_event_element_with_context(self, context)

    def __init__(
        self,
        observed,
        direction,
        state,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        message_topic=None,
        message_broker=None,
        last_update=None,
        min_interval=None,
        max_interval=None,
    ):

        EventElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.observed = observed
        self.direction = direction
        self.state = state
        self.message_topic = message_topic
        self.message_broker = message_broker
        self.last_update = last_update
        self.min_interval = min_interval
        self.max_interval = max_interval


class Operation(SubmodelElement):

    def over_input_variables_or_empty(self):

        if self.input_variables is not None:
            yield from self.input_variables

    def over_output_variables_or_empty(self):

        if self.output_variables is not None:
            yield from self.output_variables

    def over_inoutput_variables_or_empty(self):

        if self.inoutput_variables is not None:
            yield from self.inoutput_variables

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.input_variables is not None:
            yield from self.input_variables

        if self.output_variables is not None:
            yield from self.output_variables

        if self.inoutput_variables is not None:
            yield from self.inoutput_variables

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

        if self.input_variables is not None:
            for yet_yet_yet_yet_yet_another_item in self.input_variables:
                yield yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_another_item.descend()

        if self.output_variables is not None:
            for yet_yet_yet_yet_yet_yet_another_item in self.output_variables:
                yield yet_yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_yet_another_item.descend()

        if self.inoutput_variables is not None:
            for yet_yet_yet_yet_yet_yet_yet_another_item in self.inoutput_variables:
                yield yet_yet_yet_yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_operation(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_operation_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_operation(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_operation_with_context(self, context)

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
        input_variables=None,
        output_variables=None,
        inoutput_variables=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.inoutput_variables = inoutput_variables


class OperationVariable(Class):

    def descend_once(self):

        yield self.value

    def descend(self):

        yield self.value

        yield from self.value.descend()

    def accept(self, visitor):

        visitor.visit_operation_variable(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_operation_variable_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_operation_variable(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_operation_variable_with_context(self, context)

    def __init__(self, value):

        self.value = value


class Capability(SubmodelElement):

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.semantic_id is not None:
            yield self.semantic_id

        if self.supplemental_semantic_ids is not None:
            yield from self.supplemental_semantic_ids

        if self.qualifiers is not None:
            yield from self.qualifiers

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.semantic_id is not None:
            yield self.semantic_id

            yield from self.semantic_id.descend()

        if self.supplemental_semantic_ids is not None:
            for yet_yet_another_item in self.supplemental_semantic_ids:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.qualifiers is not None:
            for yet_yet_yet_another_item in self.qualifiers:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_yet_yet_another_item

                yield from yet_yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_capability(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_capability_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_capability(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_capability_with_context(self, context)

    def __init__(
        self,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        semantic_id=None,
        supplemental_semantic_ids=None,
        qualifiers=None,
        embedded_data_specifications=None,
    ):

        SubmodelElement.__init__(
            self,
            extensions,
            category,
            id_short,
            display_name,
            description,
            semantic_id,
            supplemental_semantic_ids,
            qualifiers,
            embedded_data_specifications,
        )


class ConceptDescription(Identifiable, HasDataSpecification):

    def over_is_case_of_or_empty(self):

        if self.is_case_of is not None:
            yield from self.is_case_of

    def descend_once(self):

        if self.extensions is not None:
            yield from self.extensions

        if self.display_name is not None:
            yield from self.display_name

        if self.description is not None:
            yield from self.description

        if self.administration is not None:
            yield self.administration

        if self.embedded_data_specifications is not None:
            yield from self.embedded_data_specifications

        if self.is_case_of is not None:
            yield from self.is_case_of

    def descend(self):

        if self.extensions is not None:
            for an_item in self.extensions:
                yield an_item

                yield from an_item.descend()

        if self.display_name is not None:
            for another_item in self.display_name:
                yield another_item

                yield from another_item.descend()

        if self.description is not None:
            for yet_another_item in self.description:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.administration is not None:
            yield self.administration

            yield from self.administration.descend()

        if self.embedded_data_specifications is not None:
            for yet_yet_another_item in self.embedded_data_specifications:
                yield yet_yet_another_item

                yield from yet_yet_another_item.descend()

        if self.is_case_of is not None:
            for yet_yet_yet_another_item in self.is_case_of:
                yield yet_yet_yet_another_item

                yield from yet_yet_yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_concept_description(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_concept_description_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_concept_description(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_concept_description_with_context(self, context)

    def __init__(
        self,
        id,
        extensions=None,
        category=None,
        id_short=None,
        display_name=None,
        description=None,
        administration=None,
        embedded_data_specifications=None,
        is_case_of=None,
    ):

        Identifiable.__init__(
            self,
            id,
            extensions,
            category,
            id_short,
            display_name,
            description,
            administration,
        )
        HasDataSpecification.__init__(self, embedded_data_specifications)
        self.is_case_of = is_case_of


@aas_enum.decorator
class ReferenceTypes:

    EXTERNAL_REFERENCE = "ExternalReference"

    MODEL_REFERENCE = "ModelReference"


class Reference(Class):

    def descend_once(self):

        if self.referred_semantic_id is not None:
            yield self.referred_semantic_id

        yield from self.keys

    def descend(self):

        if self.referred_semantic_id is not None:
            yield self.referred_semantic_id

            yield from self.referred_semantic_id.descend()

        for an_item in self.keys:
            yield an_item

            yield from an_item.descend()

    def accept(self, visitor):

        visitor.visit_reference(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_reference_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_reference(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_reference_with_context(self, context)

    def __init__(
        self,
        type,
        keys,
        referred_semantic_id=None,
    ):

        self.type = type
        self.keys = keys
        self.referred_semantic_id = referred_semantic_id


class Key(Class):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_key(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_key_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_key(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_key_with_context(self, context)

    def __init__(self, type, value):

        self.type = type
        self.value = value


@aas_enum.decorator
class KeyTypes:

    ANNOTATED_RELATIONSHIP_ELEMENT = "AnnotatedRelationshipElement"

    ASSET_ADMINISTRATION_SHELL = "AssetAdministrationShell"

    BASIC_EVENT_ELEMENT = "BasicEventElement"

    BLOB = "Blob"

    CAPABILITY = "Capability"

    CONCEPT_DESCRIPTION = "ConceptDescription"

    DATA_ELEMENT = "DataElement"

    ENTITY = "Entity"

    EVENT_ELEMENT = "EventElement"

    FILE = "File"

    FRAGMENT_REFERENCE = "FragmentReference"

    GLOBAL_REFERENCE = "GlobalReference"

    IDENTIFIABLE = "Identifiable"

    MULTI_LANGUAGE_PROPERTY = "MultiLanguageProperty"

    OPERATION = "Operation"

    PROPERTY = "Property"

    RANGE = "Range"

    REFERABLE = "Referable"

    REFERENCE_ELEMENT = "ReferenceElement"

    RELATIONSHIP_ELEMENT = "RelationshipElement"

    SUBMODEL = "Submodel"

    SUBMODEL_ELEMENT = "SubmodelElement"

    SUBMODEL_ELEMENT_COLLECTION = "SubmodelElementCollection"

    SUBMODEL_ELEMENT_LIST = "SubmodelElementList"


@aas_enum.decorator
class DataTypeDefXSD:

    ANY_URI = "xs:anyURI"

    BASE_64_BINARY = "xs:base64Binary"

    BOOLEAN = "xs:boolean"

    BYTE = "xs:byte"

    DATE = "xs:date"

    DATE_TIME = "xs:dateTime"

    DECIMAL = "xs:decimal"

    DOUBLE = "xs:double"

    DURATION = "xs:duration"

    FLOAT = "xs:float"

    G_DAY = "xs:gDay"

    G_MONTH = "xs:gMonth"

    G_MONTH_DAY = "xs:gMonthDay"

    G_YEAR = "xs:gYear"

    G_YEAR_MONTH = "xs:gYearMonth"

    HEX_BINARY = "xs:hexBinary"

    INT = "xs:int"

    INTEGER = "xs:integer"

    LONG = "xs:long"

    NEGATIVE_INTEGER = "xs:negativeInteger"

    NON_NEGATIVE_INTEGER = "xs:nonNegativeInteger"

    NON_POSITIVE_INTEGER = "xs:nonPositiveInteger"

    POSITIVE_INTEGER = "xs:positiveInteger"

    SHORT = "xs:short"

    STRING = "xs:string"

    TIME = "xs:time"

    UNSIGNED_BYTE = "xs:unsignedByte"

    UNSIGNED_INT = "xs:unsignedInt"

    UNSIGNED_LONG = "xs:unsignedLong"

    UNSIGNED_SHORT = "xs:unsignedShort"


class AbstractLangString(Class):

    def __init__(self, language, text):

        self.language = language
        self.text = text


class LangStringNameType(AbstractLangString):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_lang_string_name_type(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_lang_string_name_type_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_lang_string_name_type(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_lang_string_name_type_with_context(self, context)

    def __init__(self, language, text):

        AbstractLangString.__init__(self, language, text)


class LangStringTextType(AbstractLangString):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_lang_string_text_type(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_lang_string_text_type_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_lang_string_text_type(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_lang_string_text_type_with_context(self, context)

    def __init__(self, language, text):

        AbstractLangString.__init__(self, language, text)


class Environment(Class):

    def over_asset_administration_shells_or_empty(
        self,
    ):

        if self.asset_administration_shells is not None:
            yield from self.asset_administration_shells

    def over_submodels_or_empty(self):

        if self.submodels is not None:
            yield from self.submodels

    def over_concept_descriptions_or_empty(self):

        if self.concept_descriptions is not None:
            yield from self.concept_descriptions

    def descend_once(self):

        if self.asset_administration_shells is not None:
            yield from self.asset_administration_shells

        if self.submodels is not None:
            yield from self.submodels

        if self.concept_descriptions is not None:
            yield from self.concept_descriptions

    def descend(self):

        if self.asset_administration_shells is not None:
            for an_item in self.asset_administration_shells:
                yield an_item

                yield from an_item.descend()

        if self.submodels is not None:
            for another_item in self.submodels:
                yield another_item

                yield from another_item.descend()

        if self.concept_descriptions is not None:
            for yet_another_item in self.concept_descriptions:
                yield yet_another_item

                yield from yet_another_item.descend()

    def accept(self, visitor):

        visitor.visit_environment(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_environment_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_environment(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_environment_with_context(self, context)

    def __init__(
        self,
        asset_administration_shells=None,
        submodels=None,
        concept_descriptions=None,
    ):

        self.asset_administration_shells = asset_administration_shells
        self.submodels = submodels
        self.concept_descriptions = concept_descriptions


class DataSpecificationContent(Class):
    pass


class EmbeddedDataSpecification(Class):

    def descend_once(self):

        yield self.data_specification

        yield self.data_specification_content

    def descend(self):

        yield self.data_specification

        yield from self.data_specification.descend()

        yield self.data_specification_content

        yield from self.data_specification_content.descend()

    def accept(self, visitor):

        visitor.visit_embedded_data_specification(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_embedded_data_specification_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_embedded_data_specification(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_embedded_data_specification_with_context(
            self, context
        )

    def __init__(
        self,
        data_specification,
        data_specification_content,
    ):

        self.data_specification = data_specification
        self.data_specification_content = data_specification_content


@aas_enum.decorator
class DataTypeIEC61360:

    DATE = "DATE"

    STRING = "STRING"

    STRING_TRANSLATABLE = "STRING_TRANSLATABLE"

    INTEGER_MEASURE = "INTEGER_MEASURE"

    INTEGER_COUNT = "INTEGER_COUNT"

    INTEGER_CURRENCY = "INTEGER_CURRENCY"

    REAL_MEASURE = "REAL_MEASURE"

    REAL_COUNT = "REAL_COUNT"

    REAL_CURRENCY = "REAL_CURRENCY"

    BOOLEAN = "BOOLEAN"

    IRI = "IRI"

    IRDI = "IRDI"

    RATIONAL = "RATIONAL"

    RATIONAL_MEASURE = "RATIONAL_MEASURE"

    TIME = "TIME"

    TIMESTAMP = "TIMESTAMP"

    FILE = "FILE"

    HTML = "HTML"

    BLOB = "BLOB"


class LevelType(Class):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_level_type(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_level_type_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_level_type(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_level_type_with_context(self, context)

    def __init__(self, min, nom, typ, max):

        self.min = min
        self.nom = nom
        self.typ = typ
        self.max = max


class ValueReferencePair(Class):

    def descend_once(self):

        yield self.value_id

    def descend(self):

        yield self.value_id

        yield from self.value_id.descend()

    def accept(self, visitor):

        visitor.visit_value_reference_pair(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_value_reference_pair_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_value_reference_pair(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_value_reference_pair_with_context(self, context)

    def __init__(self, value, value_id):

        self.value = value
        self.value_id = value_id


class ValueList(Class):

    def descend_once(self):

        yield from self.value_reference_pairs

    def descend(self):

        for an_item in self.value_reference_pairs:
            yield an_item

            yield from an_item.descend()

    def accept(self, visitor):

        visitor.visit_value_list(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_value_list_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_value_list(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_value_list_with_context(self, context)

    def __init__(self, value_reference_pairs):

        self.value_reference_pairs = value_reference_pairs


class LangStringPreferredNameTypeIEC61360(AbstractLangString):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_lang_string_preferred_name_type_iec_61360(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_lang_string_preferred_name_type_iec_61360_with_context(
            self, context
        )

    def transform(self, transformer):

        return transformer.transform_lang_string_preferred_name_type_iec_61360(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_lang_string_preferred_name_type_iec_61360_with_context(
            self, context
        )

    def __init__(self, language, text):

        AbstractLangString.__init__(self, language, text)


class LangStringShortNameTypeIEC61360(AbstractLangString):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_lang_string_short_name_type_iec_61360(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_lang_string_short_name_type_iec_61360_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_lang_string_short_name_type_iec_61360(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_lang_string_short_name_type_iec_61360_with_context(
            self, context
        )

    def __init__(self, language, text):

        AbstractLangString.__init__(self, language, text)


class LangStringDefinitionTypeIEC61360(AbstractLangString):

    def descend_once(self):

        return

        yield

    def descend(self):

        return

        yield

    def accept(self, visitor):

        visitor.visit_lang_string_definition_type_iec_61360(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_lang_string_definition_type_iec_61360_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_lang_string_definition_type_iec_61360(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_lang_string_definition_type_iec_61360_with_context(
            self, context
        )

    def __init__(self, language, text):

        AbstractLangString.__init__(self, language, text)


class DataSpecificationIEC61360(DataSpecificationContent):

    def over_short_name_or_empty(self):

        if self.short_name is not None:
            yield from self.short_name

    def over_definition_or_empty(self):

        if self.definition is not None:
            yield from self.definition

    def descend_once(self):

        yield from self.preferred_name

        if self.short_name is not None:
            yield from self.short_name

        if self.unit_id is not None:
            yield self.unit_id

        if self.definition is not None:
            yield from self.definition

        if self.value_list is not None:
            yield self.value_list

        if self.level_type is not None:
            yield self.level_type

    def descend(self):

        for an_item in self.preferred_name:
            yield an_item

            yield from an_item.descend()

        if self.short_name is not None:
            for another_item in self.short_name:
                yield another_item

                yield from another_item.descend()

        if self.unit_id is not None:
            yield self.unit_id

            yield from self.unit_id.descend()

        if self.definition is not None:
            for yet_another_item in self.definition:
                yield yet_another_item

                yield from yet_another_item.descend()

        if self.value_list is not None:
            yield self.value_list

            yield from self.value_list.descend()

        if self.level_type is not None:
            yield self.level_type

            yield from self.level_type.descend()

    def accept(self, visitor):

        visitor.visit_data_specification_iec_61360(self)

    def accept_with_context(self, visitor, context):

        visitor.visit_data_specification_iec_61360_with_context(self, context)

    def transform(self, transformer):

        return transformer.transform_data_specification_iec_61360(self)

    def transform_with_context(
        self,
        transformer,
        context,
    ):

        return transformer.transform_data_specification_iec_61360_with_context(
            self, context
        )

    def __init__(
        self,
        preferred_name,
        short_name=None,
        unit=None,
        unit_id=None,
        source_of_definition=None,
        symbol=None,
        data_type=None,
        definition=None,
        value_format=None,
        value_list=None,
        value=None,
        level_type=None,
    ):

        self.preferred_name = preferred_name
        self.short_name = short_name
        self.unit = unit
        self.unit_id = unit_id
        self.source_of_definition = source_of_definition
        self.symbol = symbol
        self.data_type = data_type
        self.definition = definition
        self.value_format = value_format
        self.value_list = value_list
        self.value = value
        self.level_type = level_type


class AbstractVisitor:

    def visit(self, that):

        that.accept(self)

    def visit_extension(self, that):

        raise NotImplementedError()

    def visit_administrative_information(self, that):

        raise NotImplementedError()

    def visit_qualifier(self, that):

        raise NotImplementedError()

    def visit_asset_administration_shell(self, that):

        raise NotImplementedError()

    def visit_asset_information(self, that):

        raise NotImplementedError()

    def visit_resource(self, that):

        raise NotImplementedError()

    def visit_specific_asset_id(self, that):

        raise NotImplementedError()

    def visit_submodel(self, that):

        raise NotImplementedError()

    def visit_relationship_element(self, that):

        raise NotImplementedError()

    def visit_submodel_element_list(self, that):

        raise NotImplementedError()

    def visit_submodel_element_collection(self, that):

        raise NotImplementedError()

    def visit_property(self, that):

        raise NotImplementedError()

    def visit_multi_language_property(self, that):

        raise NotImplementedError()

    def visit_range(self, that):

        raise NotImplementedError()

    def visit_reference_element(self, that):

        raise NotImplementedError()

    def visit_blob(self, that):

        raise NotImplementedError()

    def visit_file(self, that):

        raise NotImplementedError()

    def visit_annotated_relationship_element(self, that):

        raise NotImplementedError()

    def visit_entity(self, that):

        raise NotImplementedError()

    def visit_event_payload(self, that):

        raise NotImplementedError()

    def visit_basic_event_element(self, that):

        raise NotImplementedError()

    def visit_operation(self, that):

        raise NotImplementedError()

    def visit_operation_variable(self, that):

        raise NotImplementedError()

    def visit_capability(self, that):

        raise NotImplementedError()

    def visit_concept_description(self, that):

        raise NotImplementedError()

    def visit_reference(self, that):

        raise NotImplementedError()

    def visit_key(self, that):

        raise NotImplementedError()

    def visit_lang_string_name_type(self, that):

        raise NotImplementedError()

    def visit_lang_string_text_type(self, that):

        raise NotImplementedError()

    def visit_environment(self, that):

        raise NotImplementedError()

    def visit_embedded_data_specification(self, that):

        raise NotImplementedError()

    def visit_level_type(self, that):

        raise NotImplementedError()

    def visit_value_reference_pair(self, that):

        raise NotImplementedError()

    def visit_value_list(self, that):

        raise NotImplementedError()

    def visit_lang_string_preferred_name_type_iec_61360(self, that):

        raise NotImplementedError()

    def visit_lang_string_short_name_type_iec_61360(self, that):

        raise NotImplementedError()

    def visit_lang_string_definition_type_iec_61360(self, that):

        raise NotImplementedError()

    def visit_data_specification_iec_61360(self, that):

        raise NotImplementedError()


class AbstractVisitorWithContext:

    def visit_with_context(self, that, context):

        that.accept_with_context(self, context)

    def visit_extension_with_context(self, that, context):

        raise NotImplementedError()

    def visit_administrative_information_with_context(self, that, context):

        raise NotImplementedError()

    def visit_qualifier_with_context(self, that, context):

        raise NotImplementedError()

    def visit_asset_administration_shell_with_context(self, that, context):

        raise NotImplementedError()

    def visit_asset_information_with_context(self, that, context):

        raise NotImplementedError()

    def visit_resource_with_context(self, that, context):

        raise NotImplementedError()

    def visit_specific_asset_id_with_context(self, that, context):

        raise NotImplementedError()

    def visit_submodel_with_context(self, that, context):

        raise NotImplementedError()

    def visit_relationship_element_with_context(self, that, context):

        raise NotImplementedError()

    def visit_submodel_element_list_with_context(self, that, context):

        raise NotImplementedError()

    def visit_submodel_element_collection_with_context(self, that, context):

        raise NotImplementedError()

    def visit_property_with_context(self, that, context):

        raise NotImplementedError()

    def visit_multi_language_property_with_context(self, that, context):

        raise NotImplementedError()

    def visit_range_with_context(self, that, context):

        raise NotImplementedError()

    def visit_reference_element_with_context(self, that, context):

        raise NotImplementedError()

    def visit_blob_with_context(self, that, context):

        raise NotImplementedError()

    def visit_file_with_context(self, that, context):

        raise NotImplementedError()

    def visit_annotated_relationship_element_with_context(self, that, context):

        raise NotImplementedError()

    def visit_entity_with_context(self, that, context):

        raise NotImplementedError()

    def visit_event_payload_with_context(self, that, context):

        raise NotImplementedError()

    def visit_basic_event_element_with_context(self, that, context):

        raise NotImplementedError()

    def visit_operation_with_context(self, that, context):

        raise NotImplementedError()

    def visit_operation_variable_with_context(self, that, context):

        raise NotImplementedError()

    def visit_capability_with_context(self, that, context):

        raise NotImplementedError()

    def visit_concept_description_with_context(self, that, context):

        raise NotImplementedError()

    def visit_reference_with_context(self, that, context):

        raise NotImplementedError()

    def visit_key_with_context(self, that, context):

        raise NotImplementedError()

    def visit_lang_string_name_type_with_context(self, that, context):

        raise NotImplementedError()

    def visit_lang_string_text_type_with_context(self, that, context):

        raise NotImplementedError()

    def visit_environment_with_context(self, that, context):

        raise NotImplementedError()

    def visit_embedded_data_specification_with_context(self, that, context):

        raise NotImplementedError()

    def visit_level_type_with_context(self, that, context):

        raise NotImplementedError()

    def visit_value_reference_pair_with_context(self, that, context):

        raise NotImplementedError()

    def visit_value_list_with_context(self, that, context):

        raise NotImplementedError()

    def visit_lang_string_preferred_name_type_iec_61360_with_context(
        self, that, context
    ):

        raise NotImplementedError()

    def visit_lang_string_short_name_type_iec_61360_with_context(self, that, context):

        raise NotImplementedError()

    def visit_lang_string_definition_type_iec_61360_with_context(self, that, context):

        raise NotImplementedError()

    def visit_data_specification_iec_61360_with_context(self, that, context):

        raise NotImplementedError()


class PassThroughVisitor(AbstractVisitor):

    def visit(self, that):

        that.accept(self)

    def visit_extension(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_administrative_information(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_qualifier(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_asset_administration_shell(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_asset_information(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_resource(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_specific_asset_id(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_submodel(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_relationship_element(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_submodel_element_list(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_submodel_element_collection(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_property(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_multi_language_property(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_range(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_reference_element(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_blob(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_file(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_annotated_relationship_element(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_entity(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_event_payload(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_basic_event_element(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_operation(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_operation_variable(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_capability(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_concept_description(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_reference(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_key(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_lang_string_name_type(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_lang_string_text_type(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_environment(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_embedded_data_specification(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_level_type(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_value_reference_pair(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_value_list(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_lang_string_preferred_name_type_iec_61360(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_lang_string_short_name_type_iec_61360(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_lang_string_definition_type_iec_61360(self, that):

        for another in that.descend_once():
            self.visit(another)

    def visit_data_specification_iec_61360(self, that):

        for another in that.descend_once():
            self.visit(another)


class PassThroughVisitorWithContext(AbstractVisitorWithContext):

    def visit_with_context(self, that, context):

        that.accept_with_context(self, context)

    def visit_extension_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_administrative_information_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_qualifier_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_asset_administration_shell_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_asset_information_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_resource_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_specific_asset_id_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_submodel_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_relationship_element_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_submodel_element_list_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_submodel_element_collection_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_property_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_multi_language_property_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_range_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_reference_element_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_blob_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_file_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_annotated_relationship_element_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_entity_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_event_payload_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_basic_event_element_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_operation_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_operation_variable_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_capability_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_concept_description_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_reference_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_key_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_lang_string_name_type_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_lang_string_text_type_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_environment_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_embedded_data_specification_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_level_type_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_value_reference_pair_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_value_list_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_lang_string_preferred_name_type_iec_61360_with_context(
        self, that, context
    ):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_lang_string_short_name_type_iec_61360_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_lang_string_definition_type_iec_61360_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)

    def visit_data_specification_iec_61360_with_context(self, that, context):

        for another in that.descend_once():
            self.visit_with_context(another, context)


class AbstractTransformer:

    def transform(self, that):

        return that.transform(self)

    def transform_extension(self, that):

        raise NotImplementedError()

    def transform_administrative_information(self, that):

        raise NotImplementedError()

    def transform_qualifier(self, that):

        raise NotImplementedError()

    def transform_asset_administration_shell(self, that):

        raise NotImplementedError()

    def transform_asset_information(self, that):

        raise NotImplementedError()

    def transform_resource(self, that):

        raise NotImplementedError()

    def transform_specific_asset_id(self, that):

        raise NotImplementedError()

    def transform_submodel(self, that):

        raise NotImplementedError()

    def transform_relationship_element(self, that):

        raise NotImplementedError()

    def transform_submodel_element_list(self, that):

        raise NotImplementedError()

    def transform_submodel_element_collection(self, that):

        raise NotImplementedError()

    def transform_property(self, that):

        raise NotImplementedError()

    def transform_multi_language_property(self, that):

        raise NotImplementedError()

    def transform_range(self, that):

        raise NotImplementedError()

    def transform_reference_element(self, that):

        raise NotImplementedError()

    def transform_blob(self, that):

        raise NotImplementedError()

    def transform_file(self, that):

        raise NotImplementedError()

    def transform_annotated_relationship_element(self, that):

        raise NotImplementedError()

    def transform_entity(self, that):

        raise NotImplementedError()

    def transform_event_payload(self, that):

        raise NotImplementedError()

    def transform_basic_event_element(self, that):

        raise NotImplementedError()

    def transform_operation(self, that):

        raise NotImplementedError()

    def transform_operation_variable(self, that):

        raise NotImplementedError()

    def transform_capability(self, that):

        raise NotImplementedError()

    def transform_concept_description(self, that):

        raise NotImplementedError()

    def transform_reference(self, that):

        raise NotImplementedError()

    def transform_key(self, that):

        raise NotImplementedError()

    def transform_lang_string_name_type(self, that):

        raise NotImplementedError()

    def transform_lang_string_text_type(self, that):

        raise NotImplementedError()

    def transform_environment(self, that):

        raise NotImplementedError()

    def transform_embedded_data_specification(self, that):

        raise NotImplementedError()

    def transform_level_type(self, that):

        raise NotImplementedError()

    def transform_value_reference_pair(self, that):

        raise NotImplementedError()

    def transform_value_list(self, that):

        raise NotImplementedError()

    def transform_lang_string_preferred_name_type_iec_61360(self, that):

        raise NotImplementedError()

    def transform_lang_string_short_name_type_iec_61360(self, that):

        raise NotImplementedError()

    def transform_lang_string_definition_type_iec_61360(self, that):

        raise NotImplementedError()

    def transform_data_specification_iec_61360(self, that):

        raise NotImplementedError()


class AbstractTransformerWithContext:

    def transform_with_context(self, that, context):

        return that.transform_with_context(self, context)

    def transform_extension_with_context(self, that, context):

        raise NotImplementedError()

    def transform_administrative_information_with_context(self, that, context):

        raise NotImplementedError()

    def transform_qualifier_with_context(self, that, context):

        raise NotImplementedError()

    def transform_asset_administration_shell_with_context(self, that, context):

        raise NotImplementedError()

    def transform_asset_information_with_context(self, that, context):

        raise NotImplementedError()

    def transform_resource_with_context(self, that, context):

        raise NotImplementedError()

    def transform_specific_asset_id_with_context(self, that, context):

        raise NotImplementedError()

    def transform_submodel_with_context(self, that, context):

        raise NotImplementedError()

    def transform_relationship_element_with_context(self, that, context):

        raise NotImplementedError()

    def transform_submodel_element_list_with_context(self, that, context):

        raise NotImplementedError()

    def transform_submodel_element_collection_with_context(self, that, context):

        raise NotImplementedError()

    def transform_property_with_context(self, that, context):

        raise NotImplementedError()

    def transform_multi_language_property_with_context(self, that, context):

        raise NotImplementedError()

    def transform_range_with_context(self, that, context):

        raise NotImplementedError()

    def transform_reference_element_with_context(self, that, context):

        raise NotImplementedError()

    def transform_blob_with_context(self, that, context):

        raise NotImplementedError()

    def transform_file_with_context(self, that, context):

        raise NotImplementedError()

    def transform_annotated_relationship_element_with_context(self, that, context):

        raise NotImplementedError()

    def transform_entity_with_context(self, that, context):

        raise NotImplementedError()

    def transform_event_payload_with_context(self, that, context):

        raise NotImplementedError()

    def transform_basic_event_element_with_context(self, that, context):

        raise NotImplementedError()

    def transform_operation_with_context(self, that, context):

        raise NotImplementedError()

    def transform_operation_variable_with_context(self, that, context):

        raise NotImplementedError()

    def transform_capability_with_context(self, that, context):

        raise NotImplementedError()

    def transform_concept_description_with_context(self, that, context):

        raise NotImplementedError()

    def transform_reference_with_context(self, that, context):

        raise NotImplementedError()

    def transform_key_with_context(self, that, context):

        raise NotImplementedError()

    def transform_lang_string_name_type_with_context(self, that, context):

        raise NotImplementedError()

    def transform_lang_string_text_type_with_context(self, that, context):

        raise NotImplementedError()

    def transform_environment_with_context(self, that, context):

        raise NotImplementedError()

    def transform_embedded_data_specification_with_context(self, that, context):

        raise NotImplementedError()

    def transform_level_type_with_context(self, that, context):

        raise NotImplementedError()

    def transform_value_reference_pair_with_context(self, that, context):

        raise NotImplementedError()

    def transform_value_list_with_context(self, that, context):

        raise NotImplementedError()

    def transform_lang_string_preferred_name_type_iec_61360_with_context(
        self, that, context
    ):

        raise NotImplementedError()

    def transform_lang_string_short_name_type_iec_61360_with_context(
        self, that, context
    ):

        raise NotImplementedError()

    def transform_lang_string_definition_type_iec_61360_with_context(
        self, that, context
    ):

        raise NotImplementedError()

    def transform_data_specification_iec_61360_with_context(self, that, context):

        raise NotImplementedError()


class TransformerWithDefault(AbstractTransformer):

    def __init__(self, default):

        self.default = default

    def transform(self, that):

        return that.transform(self)

    def transform_extension(self, that):

        return self.default

    def transform_administrative_information(self, that):

        return self.default

    def transform_qualifier(self, that):

        return self.default

    def transform_asset_administration_shell(self, that):

        return self.default

    def transform_asset_information(self, that):

        return self.default

    def transform_resource(self, that):

        return self.default

    def transform_specific_asset_id(self, that):

        return self.default

    def transform_submodel(self, that):

        return self.default

    def transform_relationship_element(self, that):

        return self.default

    def transform_submodel_element_list(self, that):

        return self.default

    def transform_submodel_element_collection(self, that):

        return self.default

    def transform_property(self, that):

        return self.default

    def transform_multi_language_property(self, that):

        return self.default

    def transform_range(self, that):

        return self.default

    def transform_reference_element(self, that):

        return self.default

    def transform_blob(self, that):

        return self.default

    def transform_file(self, that):

        return self.default

    def transform_annotated_relationship_element(self, that):

        return self.default

    def transform_entity(self, that):

        return self.default

    def transform_event_payload(self, that):

        return self.default

    def transform_basic_event_element(self, that):

        return self.default

    def transform_operation(self, that):

        return self.default

    def transform_operation_variable(self, that):

        return self.default

    def transform_capability(self, that):

        return self.default

    def transform_concept_description(self, that):

        return self.default

    def transform_reference(self, that):

        return self.default

    def transform_key(self, that):

        return self.default

    def transform_lang_string_name_type(self, that):

        return self.default

    def transform_lang_string_text_type(self, that):

        return self.default

    def transform_environment(self, that):

        return self.default

    def transform_embedded_data_specification(self, that):

        return self.default

    def transform_level_type(self, that):

        return self.default

    def transform_value_reference_pair(self, that):

        return self.default

    def transform_value_list(self, that):

        return self.default

    def transform_lang_string_preferred_name_type_iec_61360(self, that):

        return self.default

    def transform_lang_string_short_name_type_iec_61360(self, that):

        return self.default

    def transform_lang_string_definition_type_iec_61360(self, that):

        return self.default

    def transform_data_specification_iec_61360(self, that):

        return self.default


class TransformerWithDefaultAndContext(AbstractTransformerWithContext):

    def __init__(self, default):

        self.default = default

    def transform_with_context(self, that, context):

        return that.transform_with_context(self, context)

    def transform_extension_with_context(self, that, context):

        return self.default

    def transform_administrative_information_with_context(self, that, context):

        return self.default

    def transform_qualifier_with_context(self, that, context):

        return self.default

    def transform_asset_administration_shell_with_context(self, that, context):

        return self.default

    def transform_asset_information_with_context(self, that, context):

        return self.default

    def transform_resource_with_context(self, that, context):

        return self.default

    def transform_specific_asset_id_with_context(self, that, context):

        return self.default

    def transform_submodel_with_context(self, that, context):

        return self.default

    def transform_relationship_element_with_context(self, that, context):

        return self.default

    def transform_submodel_element_list_with_context(self, that, context):

        return self.default

    def transform_submodel_element_collection_with_context(self, that, context):

        return self.default

    def transform_property_with_context(self, that, context):

        return self.default

    def transform_multi_language_property_with_context(self, that, context):

        return self.default

    def transform_range_with_context(self, that, context):

        return self.default

    def transform_reference_element_with_context(self, that, context):

        return self.default

    def transform_blob_with_context(self, that, context):

        return self.default

    def transform_file_with_context(self, that, context):

        return self.default

    def transform_annotated_relationship_element_with_context(self, that, context):

        return self.default

    def transform_entity_with_context(self, that, context):

        return self.default

    def transform_event_payload_with_context(self, that, context):

        return self.default

    def transform_basic_event_element_with_context(self, that, context):

        return self.default

    def transform_operation_with_context(self, that, context):

        return self.default

    def transform_operation_variable_with_context(self, that, context):

        return self.default

    def transform_capability_with_context(self, that, context):

        return self.default

    def transform_concept_description_with_context(self, that, context):

        return self.default

    def transform_reference_with_context(self, that, context):

        return self.default

    def transform_key_with_context(self, that, context):

        return self.default

    def transform_lang_string_name_type_with_context(self, that, context):

        return self.default

    def transform_lang_string_text_type_with_context(self, that, context):

        return self.default

    def transform_environment_with_context(self, that, context):

        return self.default

    def transform_embedded_data_specification_with_context(self, that, context):

        return self.default

    def transform_level_type_with_context(self, that, context):

        return self.default

    def transform_value_reference_pair_with_context(self, that, context):

        return self.default

    def transform_value_list_with_context(self, that, context):

        return self.default

    def transform_lang_string_preferred_name_type_iec_61360_with_context(
        self, that, context
    ):

        return self.default

    def transform_lang_string_short_name_type_iec_61360_with_context(
        self, that, context
    ):

        return self.default

    def transform_lang_string_definition_type_iec_61360_with_context(
        self, that, context
    ):

        return self.default

    def transform_data_specification_iec_61360_with_context(self, that, context):

        return self.default
