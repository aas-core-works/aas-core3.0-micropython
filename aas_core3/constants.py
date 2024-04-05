import aas_core3.types as aas_types


VALID_CATEGORIES_FOR_DATA_ELEMENT = {"CONSTANT", "PARAMETER", "VARIABLE"}


GENERIC_FRAGMENT_KEYS = {aas_types.KeyTypes.FRAGMENT_REFERENCE}


GENERIC_GLOBALLY_IDENTIFIABLES = {aas_types.KeyTypes.GLOBAL_REFERENCE}


AAS_IDENTIFIABLES = {
    aas_types.KeyTypes.ASSET_ADMINISTRATION_SHELL,
    aas_types.KeyTypes.CONCEPT_DESCRIPTION,
    aas_types.KeyTypes.IDENTIFIABLE,
    aas_types.KeyTypes.SUBMODEL,
}


AAS_SUBMODEL_ELEMENTS_AS_KEYS = {
    aas_types.KeyTypes.ANNOTATED_RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.BASIC_EVENT_ELEMENT,
    aas_types.KeyTypes.BLOB,
    aas_types.KeyTypes.CAPABILITY,
    aas_types.KeyTypes.DATA_ELEMENT,
    aas_types.KeyTypes.ENTITY,
    aas_types.KeyTypes.EVENT_ELEMENT,
    aas_types.KeyTypes.FILE,
    aas_types.KeyTypes.MULTI_LANGUAGE_PROPERTY,
    aas_types.KeyTypes.OPERATION,
    aas_types.KeyTypes.PROPERTY,
    aas_types.KeyTypes.RANGE,
    aas_types.KeyTypes.REFERENCE_ELEMENT,
    aas_types.KeyTypes.RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_COLLECTION,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_LIST,
}


AAS_REFERABLE_NON_IDENTIFIABLES = {
    aas_types.KeyTypes.ANNOTATED_RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.BASIC_EVENT_ELEMENT,
    aas_types.KeyTypes.BLOB,
    aas_types.KeyTypes.CAPABILITY,
    aas_types.KeyTypes.DATA_ELEMENT,
    aas_types.KeyTypes.ENTITY,
    aas_types.KeyTypes.EVENT_ELEMENT,
    aas_types.KeyTypes.FILE,
    aas_types.KeyTypes.MULTI_LANGUAGE_PROPERTY,
    aas_types.KeyTypes.OPERATION,
    aas_types.KeyTypes.PROPERTY,
    aas_types.KeyTypes.RANGE,
    aas_types.KeyTypes.REFERENCE_ELEMENT,
    aas_types.KeyTypes.RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_COLLECTION,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_LIST,
}


AAS_REFERABLES = {
    aas_types.KeyTypes.ASSET_ADMINISTRATION_SHELL,
    aas_types.KeyTypes.CONCEPT_DESCRIPTION,
    aas_types.KeyTypes.IDENTIFIABLE,
    aas_types.KeyTypes.SUBMODEL,
    aas_types.KeyTypes.ANNOTATED_RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.BASIC_EVENT_ELEMENT,
    aas_types.KeyTypes.BLOB,
    aas_types.KeyTypes.CAPABILITY,
    aas_types.KeyTypes.DATA_ELEMENT,
    aas_types.KeyTypes.ENTITY,
    aas_types.KeyTypes.EVENT_ELEMENT,
    aas_types.KeyTypes.FILE,
    aas_types.KeyTypes.MULTI_LANGUAGE_PROPERTY,
    aas_types.KeyTypes.OPERATION,
    aas_types.KeyTypes.PROPERTY,
    aas_types.KeyTypes.RANGE,
    aas_types.KeyTypes.REFERENCE_ELEMENT,
    aas_types.KeyTypes.REFERABLE,
    aas_types.KeyTypes.RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_COLLECTION,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_LIST,
}


GLOBALLY_IDENTIFIABLES = {
    aas_types.KeyTypes.GLOBAL_REFERENCE,
    aas_types.KeyTypes.ASSET_ADMINISTRATION_SHELL,
    aas_types.KeyTypes.CONCEPT_DESCRIPTION,
    aas_types.KeyTypes.IDENTIFIABLE,
    aas_types.KeyTypes.SUBMODEL,
}


FRAGMENT_KEYS = {
    aas_types.KeyTypes.ANNOTATED_RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.BASIC_EVENT_ELEMENT,
    aas_types.KeyTypes.BLOB,
    aas_types.KeyTypes.CAPABILITY,
    aas_types.KeyTypes.DATA_ELEMENT,
    aas_types.KeyTypes.ENTITY,
    aas_types.KeyTypes.EVENT_ELEMENT,
    aas_types.KeyTypes.FILE,
    aas_types.KeyTypes.FRAGMENT_REFERENCE,
    aas_types.KeyTypes.MULTI_LANGUAGE_PROPERTY,
    aas_types.KeyTypes.OPERATION,
    aas_types.KeyTypes.PROPERTY,
    aas_types.KeyTypes.RANGE,
    aas_types.KeyTypes.REFERENCE_ELEMENT,
    aas_types.KeyTypes.RELATIONSHIP_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_COLLECTION,
    aas_types.KeyTypes.SUBMODEL_ELEMENT_LIST,
}


DATA_TYPE_IEC_61360_FOR_PROPERTY_OR_VALUE = {
    aas_types.DataTypeIEC61360.DATE,
    aas_types.DataTypeIEC61360.STRING,
    aas_types.DataTypeIEC61360.STRING_TRANSLATABLE,
    aas_types.DataTypeIEC61360.INTEGER_MEASURE,
    aas_types.DataTypeIEC61360.INTEGER_COUNT,
    aas_types.DataTypeIEC61360.INTEGER_CURRENCY,
    aas_types.DataTypeIEC61360.REAL_MEASURE,
    aas_types.DataTypeIEC61360.REAL_COUNT,
    aas_types.DataTypeIEC61360.REAL_CURRENCY,
    aas_types.DataTypeIEC61360.BOOLEAN,
    aas_types.DataTypeIEC61360.RATIONAL,
    aas_types.DataTypeIEC61360.RATIONAL_MEASURE,
    aas_types.DataTypeIEC61360.TIME,
    aas_types.DataTypeIEC61360.TIMESTAMP,
}


DATA_TYPE_IEC_61360_FOR_REFERENCE = {
    aas_types.DataTypeIEC61360.STRING,
    aas_types.DataTypeIEC61360.IRI,
    aas_types.DataTypeIEC61360.IRDI,
}


DATA_TYPE_IEC_61360_FOR_DOCUMENT = {
    aas_types.DataTypeIEC61360.FILE,
    aas_types.DataTypeIEC61360.BLOB,
    aas_types.DataTypeIEC61360.HTML,
}


IEC_61360_DATA_TYPES_WITH_UNIT = {
    aas_types.DataTypeIEC61360.INTEGER_MEASURE,
    aas_types.DataTypeIEC61360.REAL_MEASURE,
    aas_types.DataTypeIEC61360.RATIONAL_MEASURE,
    aas_types.DataTypeIEC61360.INTEGER_CURRENCY,
    aas_types.DataTypeIEC61360.REAL_CURRENCY,
}
