import aas_core3.types as aas_types


_MODELLING_KIND_FROM_STR = {
    "Template": aas_types.ModellingKind.TEMPLATE,
    "Instance": aas_types.ModellingKind.INSTANCE,
}


def modelling_kind_from_str(text):

    return _MODELLING_KIND_FROM_STR.get(text, None)


_QUALIFIER_KIND_FROM_STR = {
    "ValueQualifier": aas_types.QualifierKind.VALUE_QUALIFIER,
    "ConceptQualifier": aas_types.QualifierKind.CONCEPT_QUALIFIER,
    "TemplateQualifier": aas_types.QualifierKind.TEMPLATE_QUALIFIER,
}


def qualifier_kind_from_str(text):

    return _QUALIFIER_KIND_FROM_STR.get(text, None)


_ASSET_KIND_FROM_STR = {
    "Type": aas_types.AssetKind.TYPE,
    "Instance": aas_types.AssetKind.INSTANCE,
    "NotApplicable": aas_types.AssetKind.NOT_APPLICABLE,
}


def asset_kind_from_str(text):

    return _ASSET_KIND_FROM_STR.get(text, None)


_AAS_SUBMODEL_ELEMENTS_FROM_STR = {
    "AnnotatedRelationshipElement": aas_types.AASSubmodelElements.ANNOTATED_RELATIONSHIP_ELEMENT,
    "BasicEventElement": aas_types.AASSubmodelElements.BASIC_EVENT_ELEMENT,
    "Blob": aas_types.AASSubmodelElements.BLOB,
    "Capability": aas_types.AASSubmodelElements.CAPABILITY,
    "DataElement": aas_types.AASSubmodelElements.DATA_ELEMENT,
    "Entity": aas_types.AASSubmodelElements.ENTITY,
    "EventElement": aas_types.AASSubmodelElements.EVENT_ELEMENT,
    "File": aas_types.AASSubmodelElements.FILE,
    "MultiLanguageProperty": aas_types.AASSubmodelElements.MULTI_LANGUAGE_PROPERTY,
    "Operation": aas_types.AASSubmodelElements.OPERATION,
    "Property": aas_types.AASSubmodelElements.PROPERTY,
    "Range": aas_types.AASSubmodelElements.RANGE,
    "ReferenceElement": aas_types.AASSubmodelElements.REFERENCE_ELEMENT,
    "RelationshipElement": aas_types.AASSubmodelElements.RELATIONSHIP_ELEMENT,
    "SubmodelElement": aas_types.AASSubmodelElements.SUBMODEL_ELEMENT,
    "SubmodelElementList": aas_types.AASSubmodelElements.SUBMODEL_ELEMENT_LIST,
    "SubmodelElementCollection": aas_types.AASSubmodelElements.SUBMODEL_ELEMENT_COLLECTION,
}


def aas_submodel_elements_from_str(
    text,
):

    return _AAS_SUBMODEL_ELEMENTS_FROM_STR.get(text, None)


_ENTITY_TYPE_FROM_STR = {
    "CoManagedEntity": aas_types.EntityType.CO_MANAGED_ENTITY,
    "SelfManagedEntity": aas_types.EntityType.SELF_MANAGED_ENTITY,
}


def entity_type_from_str(text):

    return _ENTITY_TYPE_FROM_STR.get(text, None)


_DIRECTION_FROM_STR = {
    "input": aas_types.Direction.INPUT,
    "output": aas_types.Direction.OUTPUT,
}


def direction_from_str(text):

    return _DIRECTION_FROM_STR.get(text, None)


_STATE_OF_EVENT_FROM_STR = {
    "on": aas_types.StateOfEvent.ON,
    "off": aas_types.StateOfEvent.OFF,
}


def state_of_event_from_str(text):

    return _STATE_OF_EVENT_FROM_STR.get(text, None)


_REFERENCE_TYPES_FROM_STR = {
    "ExternalReference": aas_types.ReferenceTypes.EXTERNAL_REFERENCE,
    "ModelReference": aas_types.ReferenceTypes.MODEL_REFERENCE,
}


def reference_types_from_str(text):

    return _REFERENCE_TYPES_FROM_STR.get(text, None)


_KEY_TYPES_FROM_STR = {
    "AnnotatedRelationshipElement": aas_types.KeyTypes.ANNOTATED_RELATIONSHIP_ELEMENT,
    "AssetAdministrationShell": aas_types.KeyTypes.ASSET_ADMINISTRATION_SHELL,
    "BasicEventElement": aas_types.KeyTypes.BASIC_EVENT_ELEMENT,
    "Blob": aas_types.KeyTypes.BLOB,
    "Capability": aas_types.KeyTypes.CAPABILITY,
    "ConceptDescription": aas_types.KeyTypes.CONCEPT_DESCRIPTION,
    "DataElement": aas_types.KeyTypes.DATA_ELEMENT,
    "Entity": aas_types.KeyTypes.ENTITY,
    "EventElement": aas_types.KeyTypes.EVENT_ELEMENT,
    "File": aas_types.KeyTypes.FILE,
    "FragmentReference": aas_types.KeyTypes.FRAGMENT_REFERENCE,
    "GlobalReference": aas_types.KeyTypes.GLOBAL_REFERENCE,
    "Identifiable": aas_types.KeyTypes.IDENTIFIABLE,
    "MultiLanguageProperty": aas_types.KeyTypes.MULTI_LANGUAGE_PROPERTY,
    "Operation": aas_types.KeyTypes.OPERATION,
    "Property": aas_types.KeyTypes.PROPERTY,
    "Range": aas_types.KeyTypes.RANGE,
    "Referable": aas_types.KeyTypes.REFERABLE,
    "ReferenceElement": aas_types.KeyTypes.REFERENCE_ELEMENT,
    "RelationshipElement": aas_types.KeyTypes.RELATIONSHIP_ELEMENT,
    "Submodel": aas_types.KeyTypes.SUBMODEL,
    "SubmodelElement": aas_types.KeyTypes.SUBMODEL_ELEMENT,
    "SubmodelElementCollection": aas_types.KeyTypes.SUBMODEL_ELEMENT_COLLECTION,
    "SubmodelElementList": aas_types.KeyTypes.SUBMODEL_ELEMENT_LIST,
}


def key_types_from_str(text):

    return _KEY_TYPES_FROM_STR.get(text, None)


_DATA_TYPE_DEF_XSD_FROM_STR = {
    "xs:anyURI": aas_types.DataTypeDefXSD.ANY_URI,
    "xs:base64Binary": aas_types.DataTypeDefXSD.BASE_64_BINARY,
    "xs:boolean": aas_types.DataTypeDefXSD.BOOLEAN,
    "xs:byte": aas_types.DataTypeDefXSD.BYTE,
    "xs:date": aas_types.DataTypeDefXSD.DATE,
    "xs:dateTime": aas_types.DataTypeDefXSD.DATE_TIME,
    "xs:decimal": aas_types.DataTypeDefXSD.DECIMAL,
    "xs:double": aas_types.DataTypeDefXSD.DOUBLE,
    "xs:duration": aas_types.DataTypeDefXSD.DURATION,
    "xs:float": aas_types.DataTypeDefXSD.FLOAT,
    "xs:gDay": aas_types.DataTypeDefXSD.G_DAY,
    "xs:gMonth": aas_types.DataTypeDefXSD.G_MONTH,
    "xs:gMonthDay": aas_types.DataTypeDefXSD.G_MONTH_DAY,
    "xs:gYear": aas_types.DataTypeDefXSD.G_YEAR,
    "xs:gYearMonth": aas_types.DataTypeDefXSD.G_YEAR_MONTH,
    "xs:hexBinary": aas_types.DataTypeDefXSD.HEX_BINARY,
    "xs:int": aas_types.DataTypeDefXSD.INT,
    "xs:integer": aas_types.DataTypeDefXSD.INTEGER,
    "xs:long": aas_types.DataTypeDefXSD.LONG,
    "xs:negativeInteger": aas_types.DataTypeDefXSD.NEGATIVE_INTEGER,
    "xs:nonNegativeInteger": aas_types.DataTypeDefXSD.NON_NEGATIVE_INTEGER,
    "xs:nonPositiveInteger": aas_types.DataTypeDefXSD.NON_POSITIVE_INTEGER,
    "xs:positiveInteger": aas_types.DataTypeDefXSD.POSITIVE_INTEGER,
    "xs:short": aas_types.DataTypeDefXSD.SHORT,
    "xs:string": aas_types.DataTypeDefXSD.STRING,
    "xs:time": aas_types.DataTypeDefXSD.TIME,
    "xs:unsignedByte": aas_types.DataTypeDefXSD.UNSIGNED_BYTE,
    "xs:unsignedInt": aas_types.DataTypeDefXSD.UNSIGNED_INT,
    "xs:unsignedLong": aas_types.DataTypeDefXSD.UNSIGNED_LONG,
    "xs:unsignedShort": aas_types.DataTypeDefXSD.UNSIGNED_SHORT,
}


def data_type_def_xsd_from_str(text):

    return _DATA_TYPE_DEF_XSD_FROM_STR.get(text, None)


_DATA_TYPE_IEC_61360_FROM_STR = {
    "DATE": aas_types.DataTypeIEC61360.DATE,
    "STRING": aas_types.DataTypeIEC61360.STRING,
    "STRING_TRANSLATABLE": aas_types.DataTypeIEC61360.STRING_TRANSLATABLE,
    "INTEGER_MEASURE": aas_types.DataTypeIEC61360.INTEGER_MEASURE,
    "INTEGER_COUNT": aas_types.DataTypeIEC61360.INTEGER_COUNT,
    "INTEGER_CURRENCY": aas_types.DataTypeIEC61360.INTEGER_CURRENCY,
    "REAL_MEASURE": aas_types.DataTypeIEC61360.REAL_MEASURE,
    "REAL_COUNT": aas_types.DataTypeIEC61360.REAL_COUNT,
    "REAL_CURRENCY": aas_types.DataTypeIEC61360.REAL_CURRENCY,
    "BOOLEAN": aas_types.DataTypeIEC61360.BOOLEAN,
    "IRI": aas_types.DataTypeIEC61360.IRI,
    "IRDI": aas_types.DataTypeIEC61360.IRDI,
    "RATIONAL": aas_types.DataTypeIEC61360.RATIONAL,
    "RATIONAL_MEASURE": aas_types.DataTypeIEC61360.RATIONAL_MEASURE,
    "TIME": aas_types.DataTypeIEC61360.TIME,
    "TIMESTAMP": aas_types.DataTypeIEC61360.TIMESTAMP,
    "FILE": aas_types.DataTypeIEC61360.FILE,
    "HTML": aas_types.DataTypeIEC61360.HTML,
    "BLOB": aas_types.DataTypeIEC61360.BLOB,
}


def data_type_iec_61360_from_str(text):

    return _DATA_TYPE_IEC_61360_FROM_STR.get(text, None)
