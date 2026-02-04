"""Serialize AAS models to XML."""

import base64
import io
import math
import os
import sys
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if sys.version_info >= (3, 8):
    from typing import Final, Protocol
else:
    from typing_extensions import Final, Protocol

import aas_core3.stringification as aas_stringification
import aas_core3.types as aas_types

if TYPE_CHECKING:
    PathLike = os.PathLike[Any]
else:
    PathLike = os.PathLike

NAMESPACE = "https://admin-shell.io/aas/3/0"

def from_iterparse(iterator: Iterator[Tuple[str, Element]]) -> aas_types.Class:
    """
    Read an instance from the :paramref:`iterator`.

    The type of the instance is determined by the very first start element.

    Example usage:

    .. code-block::

        import pathlib
        import xml.etree.ElementTree as ET

        import aas_core3.xmlization as aas_xmlization

        path = pathlib.Path(...)
        with path.open("rt") as fid:
            iterator = ET.iterparse(
                source=fid,
                events=['start', 'end']
            )
            instance = aas_xmlization.from_iterparse(
                iterator
            )

        # Do something with the ``instance``

    :param iterator:
        Input stream of ``(event, element)`` coming from
        :py:func:`xml.etree.ElementTree.iterparse` with the argument
        ``events=["start", "end"]``
    :raise: :py:class:`DeserializationException` if unexpected input
    :return:
        Instance of :py:class:`.types.Class` read from the :paramref:`iterator`
    """
    ...

def from_stream(
    stream: TextIO, has_iterparse: HasIterparse = xml.etree.ElementTree
) -> aas_types.Class:
    """
    Read an instance from the :paramref:`stream`.

    The type of the instance is determined by the very first start element.

    Example usage:

    .. code-block::

        import aas_core3.xmlization as aas_xmlization

        with open_some_stream_over_network(...) as stream:
            instance = aas_xmlization.from_stream(
                stream
            )

        # Do something with the ``instance``

    :param stream:
        representing an instance in XML
    :param has_iterparse:
        Module containing ``iterparse`` function.

        Default is to use :py:mod:`xml.etree.ElementTree` from the standard
        library. If you have to deal with malicious input, consider using
        a library such as `defusedxml.ElementTree`_.
    :raise: :py:class:`DeserializationException` if unexpected input
    :return:
        Instance read from :paramref:`stream`
    """
    ...

def from_file(
    path: PathLike, has_iterparse: HasIterparse = xml.etree.ElementTree
) -> aas_types.Class:
    """
    Read an instance from the file at the :paramref:`path`.

    Example usage:

    .. code-block::

        import pathlib
        import aas_core3.xmlization as aas_xmlization

        path = pathlib.Path(...)
        instance = aas_xmlization.from_file(
            path
        )

        # Do something with the ``instance``

    :param path:
        to the file representing an instance in XML
    :param has_iterparse:
        Module containing ``iterparse`` function.

        Default is to use :py:mod:`xml.etree.ElementTree` from the standard
        library. If you have to deal with malicious input, consider using
        a library such as `defusedxml.ElementTree`_.
    :raise: :py:class:`DeserializationException` if unexpected input
    :return:
        Instance read from the file at :paramref:`path`
    """
    ...

def from_str(
    text: str, has_iterparse: HasIterparse = xml.etree.ElementTree
) -> aas_types.Class:
    """
    Read an instance from the :paramref:`text`.

    Example usage:

    .. code-block::

        import pathlib
        import aas_core3.xmlization as aas_xmlization

        text = "<...>...</...>"
        instance = aas_xmlization.from_str(
            text
        )

        # Do something with the ``instance``

    :param text:
        representing an instance in XML
    :param has_iterparse:
        Module containing ``iterparse`` function.

        Default is to use :py:mod:`xml.etree.ElementTree` from the standard
        library. If you have to deal with malicious input, consider using
        a library such as `defusedxml.ElementTree`_.
    :raise: :py:class:`DeserializationException` if unexpected input
    :return:
        Instance read from :paramref:`text`
    """
    ...

def _read_as_element(
    element: Element, iterator: Iterator[Tuple[str, Element]]
) -> aas_types.Class:
    """
    Read an instance from :paramref:`iterator`, including the end element.

    :param element: start element
    :param iterator:
        Input stream of ``(event, element)`` coming from
        :py:func:`xml.etree.ElementTree.iterparse` with the argument
        ``events=["start", "end"]``
    :raise: :py:class:`DeserializationException` if unexpected input
    :return: parsed instance
    """
    ...

_GENERAL_DISPATCH: Mapping[
    str, Callable[[Element, Iterator[Tuple[str, Element]]], aas_types.Class]
]

class _Serializer(aas_types.AbstractVisitor):
    """Encode instances as XML and write them to :py:attr:`~stream`."""

    stream: Final[TextIO]

    _write_start_element: Callable[[str], None]

    _write_empty_element: Callable[[str], None]

    def _write_first_start_element_with_namespace(self, name: str) -> None:
        """
        Write the start element with the tag name :paramref:`name` and specify
        its namespace.

        The :py:attr:`~_write_start_element` is set to
        :py:meth:`~_write_start_element_without_namespace` after the first invocation
        of this method.

        :param name: of the element tag. Expected to contain no XML special characters.
        """
        ...

    def _write_start_element_without_namespace(self, name: str) -> None:
        """
        Write the start element with the tag name :paramref:`name`.

        The first element, written *before* this one, is expected to have been
        already written with the namespace specified.

        :param name: of the element tag. Expected to contain no XML special characters.
        """
        ...

    def _escape_and_write_text(self, text: str) -> None:
        """
        Escape :paramref:`text` for XML and write it.

        :param text: to be escaped and written
        """

        ...

    def _write_end_element(self, name: str) -> None:
        """
        Write the end element with the tag name :paramref:`name`.

        :param name: of the element tag. Expected to contain no XML special characters.
        """
        ...

    def _write_first_empty_element_with_namespace(self, name: str) -> None:
        """
        Write the first (and only) empty element with the tag name :paramref:`name`.

        No elements are expected to be written to the stream afterwards. The element
        includes the namespace specification.

        :param name: of the element tag. Expected to contain no XML special characters.
        """
        ...

    def _rase_if_write_element_called_again(self, name: str) -> None: ...
    def _write_empty_element_without_namespace(self, name: str) -> None:
        """
        Write the empty element with the tag name :paramref:`name`.

        The call to this method is expected to occur *after* the enclosing element with
        a specified namespace has been written.

        :param name: of the element tag. Expected to contain no XML special characters.
        """
        ...

    def _write_bool_property(self, name: str, value: bool) -> None:
        """
        Write the :paramref:`value` of a boolean property enclosed in
        the :paramref:`name` element.

        :param name: of the corresponding element tag
        :param value: of the property
        """
        ...

    def _write_int_property(self, name: str, value: int) -> None:
        """
        Write the :paramref:`value` of an integer property enclosed in
        the :paramref:`name` element.

        :param name: of the corresponding element tag
        :param value: of the property
        """
        ...

    def _write_float_property(self, name: str, value: float) -> None:
        """
        Write the :paramref:`value` of a floating-point property enclosed in
        the :paramref:`name` element.

        :param name: of the corresponding element tag
        :param value: of the property
        """
        ...

    def _write_str_property(self, name: str, value: str) -> None:
        """
        Write the :paramref:`value` of a string property enclosed in
        the :paramref:`name` element.

        :param name: of the corresponding element tag
        :param value: of the property
        """
        ...

    def _write_bytes_property(self, name: str, value: bytes) -> None:
        """
        Write the :paramref:`value` of a binary-content property enclosed in
        the :paramref:`name` element.

        :param name: of the corresponding element tag
        :param value: of the property
        """
        ...

    def __init__(self, stream: TextIO) -> None:
        """
        Initialize the visitor to write to :paramref:`stream`.

        The first element will include the :py:attr:`~.NAMESPACE`. Every other
        element will not have the namespace specified.

        :param stream: where to write to
        """
        ...

    def _write_extension_as_sequence(self, that: aas_types.Extension) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_extension(self, that: aas_types.Extension) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_administrative_information_as_sequence(
        self, that: aas_types.AdministrativeInformation
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_administrative_information(
        self, that: aas_types.AdministrativeInformation
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_qualifier_as_sequence(self, that: aas_types.Qualifier) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_qualifier(self, that: aas_types.Qualifier) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_asset_administration_shell_as_sequence(
        self, that: aas_types.AssetAdministrationShell
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_asset_administration_shell(
        self, that: aas_types.AssetAdministrationShell
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_asset_information_as_sequence(
        self, that: aas_types.AssetInformation
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_asset_information(self, that: aas_types.AssetInformation) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_resource_as_sequence(self, that: aas_types.Resource) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_resource(self, that: aas_types.Resource) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_specific_asset_id_as_sequence(
        self, that: aas_types.SpecificAssetID
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_specific_asset_id(self, that: aas_types.SpecificAssetID) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_submodel_as_sequence(self, that: aas_types.Submodel) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_submodel(self, that: aas_types.Submodel) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_relationship_element_as_sequence(
        self, that: aas_types.RelationshipElement
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_relationship_element(self, that: aas_types.RelationshipElement) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_submodel_element_list_as_sequence(
        self, that: aas_types.SubmodelElementList
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_submodel_element_list(self, that: aas_types.SubmodelElementList) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_submodel_element_collection_as_sequence(
        self, that: aas_types.SubmodelElementCollection
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_submodel_element_collection(
        self, that: aas_types.SubmodelElementCollection
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_property_as_sequence(self, that: aas_types.Property) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_property(self, that: aas_types.Property) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_multi_language_property_as_sequence(
        self, that: aas_types.MultiLanguageProperty
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_multi_language_property(
        self, that: aas_types.MultiLanguageProperty
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_range_as_sequence(self, that: aas_types.Range) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_range(self, that: aas_types.Range) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_reference_element_as_sequence(
        self, that: aas_types.ReferenceElement
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_reference_element(self, that: aas_types.ReferenceElement) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_blob_as_sequence(self, that: aas_types.Blob) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_blob(self, that: aas_types.Blob) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_file_as_sequence(self, that: aas_types.File) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_file(self, that: aas_types.File) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_annotated_relationship_element_as_sequence(
        self, that: aas_types.AnnotatedRelationshipElement
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_annotated_relationship_element(
        self, that: aas_types.AnnotatedRelationshipElement
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_entity_as_sequence(self, that: aas_types.Entity) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_entity(self, that: aas_types.Entity) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_event_payload_as_sequence(self, that: aas_types.EventPayload) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_event_payload(self, that: aas_types.EventPayload) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_basic_event_element_as_sequence(
        self, that: aas_types.BasicEventElement
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_basic_event_element(self, that: aas_types.BasicEventElement) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_operation_as_sequence(self, that: aas_types.Operation) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_operation(self, that: aas_types.Operation) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_operation_variable_as_sequence(
        self, that: aas_types.OperationVariable
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_operation_variable(self, that: aas_types.OperationVariable) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_capability_as_sequence(self, that: aas_types.Capability) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_capability(self, that: aas_types.Capability) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_concept_description_as_sequence(
        self, that: aas_types.ConceptDescription
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_concept_description(self, that: aas_types.ConceptDescription) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_reference_as_sequence(self, that: aas_types.Reference) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_reference(self, that: aas_types.Reference) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_key_as_sequence(self, that: aas_types.Key) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_key(self, that: aas_types.Key) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_lang_string_name_type_as_sequence(
        self, that: aas_types.LangStringNameType
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_lang_string_name_type(self, that: aas_types.LangStringNameType) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_lang_string_text_type_as_sequence(
        self, that: aas_types.LangStringTextType
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_lang_string_text_type(self, that: aas_types.LangStringTextType) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_environment_as_sequence(self, that: aas_types.Environment) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_environment(self, that: aas_types.Environment) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """

        ...

    def _write_embedded_data_specification_as_sequence(
        self, that: aas_types.EmbeddedDataSpecification
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_embedded_data_specification(
        self, that: aas_types.EmbeddedDataSpecification
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_level_type_as_sequence(self, that: aas_types.LevelType) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_level_type(self, that: aas_types.LevelType) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_value_reference_pair_as_sequence(
        self, that: aas_types.ValueReferencePair
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_value_reference_pair(self, that: aas_types.ValueReferencePair) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_value_list_as_sequence(self, that: aas_types.ValueList) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_value_list(self, that: aas_types.ValueList) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_lang_string_preferred_name_type_iec_61360_as_sequence(
        self, that: aas_types.LangStringPreferredNameTypeIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_lang_string_preferred_name_type_iec_61360(
        self, that: aas_types.LangStringPreferredNameTypeIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_lang_string_short_name_type_iec_61360_as_sequence(
        self, that: aas_types.LangStringShortNameTypeIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_lang_string_short_name_type_iec_61360(
        self, that: aas_types.LangStringShortNameTypeIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_lang_string_definition_type_iec_61360_as_sequence(
        self, that: aas_types.LangStringDefinitionTypeIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_lang_string_definition_type_iec_61360(
        self, that: aas_types.LangStringDefinitionTypeIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

    def _write_data_specification_iec_61360_as_sequence(
        self, that: aas_types.DataSpecificationIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as a sequence of
        XML elements.

        Each element in the sequence corresponds to a property. If no properties
        are set, nothing is written to the :py:attr:`~stream`.

        :param that: instance to be serialized
        """
        ...

    def visit_data_specification_iec_61360(
        self, that: aas_types.DataSpecificationIEC61360
    ) -> None:
        """
        Serialize :paramref:`that` to :py:attr:`~stream` as an XML element.

        The enclosing XML element designates the class of the instance, where its
        children correspond to the properties of the instance.

        :param that: instance to be serialized
        """
        ...

def write(instance: aas_types.Class, stream: TextIO) -> None:
    """
    Write the XML representation of :paramref:`instance` to :paramref:`stream`.

    Example usage:

    .. code-block::

        import pathlib

        import aas_core3.types as aas_types
        import aas_core3.xmlization as aas_xmlization

        instance = Extension(
           ... # some constructor arguments
        )

        pth = pathlib.Path(...)
        with pth.open("wt") as fid:
            aas_xmlization.write(instance, fid)

    :param instance: to be serialized
    :param stream: to write to
    """
    ...

def to_str(that: aas_types.Class) -> str:
    """
    Serialize :paramref:`that` to an XML-encoded text.

    :param that: instance to be serialized
    :return: :paramref:`that` serialized to XML serialized to text
    """
    ...
