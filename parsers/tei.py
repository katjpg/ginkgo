import string
from typing import Generator, Any

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PageElement, Tag

from models.grobid import (
    Affiliation,
    Article,
    Author,
    Citation,
    CitationIDs,
    Date,
    Marker,
    PageRange,
    PersonName,
    Ref,
    RefText,
    Scope,
    Section,
    Table,
)


class ParserError(Exception):
    """Exception raised when TEI XML parsing fails."""

    pass


class Parser:
    """TEI XML Parser for GROBID-generated documents.

    Attributes:
        soup: BeautifulSoup object containing parsed XML tree.
    """

    def __init__(self, stream: bytes) -> None:
        """Initialize parser with raw XML bytes.

        Args:
            stream: Raw TEI XML document as bytes.
        """
        self.soup = BeautifulSoup(stream, "lxml-xml")

    def parse(self) -> Article:
        """Parse complete TEI XML document into Article model.

        Returns:
            Article model with all extracted components.

        Raises:
            ParserError: If required XML elements are missing or malformed.
        """
        body = self.soup.body
        if not isinstance(body, Tag):
            raise ParserError("Missing body element in TEI document")

        abstract: Section | None = self.section(self.soup.abstract, title="Abstract")

        sections: list[Section] = []
        for div in body.find_all("div"):
            if (section := self.section(div)) is not None:
                sections.append(section)

        tables: dict[str, Table] = {}
        for table_tag in body.find_all("figure", {"type": "table"}):
            if isinstance(table_tag, Tag) and "xml:id" in table_tag.attrs:
                table_id = str(table_tag.attrs["xml:id"])
                if (table_obj := self.table(table_tag)) is not None:
                    tables[table_id] = table_obj

        source = self.soup.find("sourceDesc")
        if source is None:
            raise ParserError("Missing sourceDesc element")

        biblstruct_tag = source.find("biblStruct")
        if not isinstance(biblstruct_tag, Tag):
            raise ParserError("Missing biblStruct in sourceDesc")

        bibliography = self.citation(biblstruct_tag)
        keywords = self.keywords(self.soup.keywords)

        listbibl_tag = self.soup.find("listBibl")
        if not isinstance(listbibl_tag, Tag):
            raise ParserError("Missing listBibl element")

        citations: dict[str, Citation] = {}
        for struct_tag in listbibl_tag.find_all("biblStruct"):
            if isinstance(struct_tag, Tag):
                citation_id = struct_tag.get("xml:id")
                if citation_id and isinstance(citation_id, str):
                    citations[citation_id] = self.citation(struct_tag)

        return Article(
            abstract=abstract,
            sections=sections,
            tables=tables,
            bibliography=bibliography,
            keywords=keywords,
            citations=citations,
        )

    def citation(self, source_tag: Tag) -> Citation:
        """Parse biblStruct tag into Citation model.

        Args:
            source_tag: biblStruct XML tag.

        Returns:
            Citation model with extracted metadata.
        """
        title = self.title(source_tag, attrs={"type": "main"})
        if not title:
            title = self.title(source_tag, attrs={"level": "m"})

        citation = Citation(title=title)
        citation.authors = self.authors(source_tag)

        ids = CitationIDs(
            DOI=self.idno(source_tag, attrs={"type": "DOI"}),
            arXiv=self.idno(source_tag, attrs={"type": "arXiv"}),
        )
        if not ids.is_empty():
            citation.ids = ids

        citation.date = self.date(source_tag)
        citation.target = self.target(source_tag)
        citation.publisher = self.publisher(source_tag)
        citation.scope = self.scope(source_tag)

        if journal := self.title(source_tag, attrs={"level": "j"}):
            if journal != citation.title:
                citation.journal = journal
        if series := self.title(source_tag, attrs={"level": "s"}):
            if series != citation.title:
                citation.series = series

        return citation

    def title(self, source_tag: Tag | None, attrs: dict[str, Any] | None = None) -> str:
        """Extract text content from title tag.

        Args:
            source_tag: XML tag potentially containing title element.
            attrs: Optional attribute filters for finding specific title types.

        Returns:
            Title text if found, empty string otherwise.
        """
        if attrs is None:
            attrs = {}

        title: str = ""
        if source_tag is not None:
            if (title_tag := source_tag.find("title", attrs=attrs)) is not None:
                title = title_tag.text or ""

        return title

    def target(self, source_tag: Tag | None) -> str | None:
        """Extract target URL from ptr tag.

        Args:
            source_tag: XML tag potentially containing ptr element.

        Returns:
            Target URL if present, None otherwise.
        """
        if source_tag is not None:
            if (ptr_tag := source_tag.ptr) is not None:
                if "target" in ptr_tag.attrs:
                    target_val = ptr_tag.attrs["target"]
                    if isinstance(target_val, str):
                        return target_val
        return None

    def idno(
        self, source_tag: Tag | None, attrs: dict[str, Any] | None = None
    ) -> str | None:
        """Extract identifier from idno tag.

        Args:
            source_tag: XML tag potentially containing idno element.
            attrs: Optional attribute filters for finding specific identifier types.

        Returns:
            Identifier string if found, None otherwise.
        """
        if attrs is None:
            attrs = {}

        if source_tag is not None:
            if (idno_tag := source_tag.find("idno", attrs=attrs)) is not None:
                return idno_tag.text or None
        return None

    def keywords(self, source_tag: Tag | None) -> set[str]:
        """Extract all keywords from term tags.

        Args:
            source_tag: XML tag containing term elements.

        Returns:
            Set of cleaned keyword strings.
        """
        keywords: set[str] = set()

        if source_tag is not None:
            for term_tag in source_tag.find_all("term"):
                if term_tag.text:
                    if clean_keyword := self._clean_title_string(term_tag.text):
                        keywords.add(clean_keyword)

        return keywords

    def publisher(self, source_tag: Tag | None) -> str | None:
        """Extract publisher name from publisher tag.

        Args:
            source_tag: XML tag potentially containing publisher element.

        Returns:
            Publisher name if present, None otherwise.
        """
        if source_tag is not None:
            if (publisher_tag := source_tag.find("publisher")) is not None:
                return publisher_tag.text or None
        return None

    def date(self, source_tag: Tag | None) -> Date | None:
        """Parse date from date tag with 'when' attribute.

        Args:
            source_tag: XML tag potentially containing date element.

        Returns:
            Date model if valid date found, None otherwise.
        """
        if source_tag is not None:
            if (date_tag := source_tag.date) is not None:
                if "when" in date_tag.attrs:
                    when = date_tag.attrs["when"]
                    if isinstance(when, str):
                        return self._parse_date(when)
        return None

    def scope(self, source_tag: Tag | None) -> Scope | None:
        """Parse bibliographic scope from biblScope tags.

        Args:
            source_tag: XML tag potentially containing biblScope elements.

        Returns:
            Scope model if valid scope found, None if empty.
        """
        if source_tag is not None:
            scope = Scope()
            for scope_tag in source_tag.find_all("biblScope"):
                unit = scope_tag.attrs.get("unit")
                if not isinstance(unit, str):
                    continue

                match unit:
                    case "page":
                        try:
                            if "from" in scope_tag.attrs and "to" in scope_tag.attrs:
                                from_val = scope_tag.attrs["from"]
                                to_val = scope_tag.attrs["to"]
                                if isinstance(from_val, str) and isinstance(
                                    to_val, str
                                ):
                                    from_page = int(from_val)
                                    to_page = int(to_val)
                                else:
                                    continue
                            elif scope_tag.text:
                                from_page = int(scope_tag.text)
                                to_page = from_page
                            else:
                                continue

                            scope.pages = PageRange(
                                from_page=from_page, to_page=to_page
                            )
                        except (ValueError, KeyError):
                            continue

                    case "volume":
                        try:
                            if scope_tag.text:
                                volume = int(scope_tag.text)
                                scope.volume = volume
                        except ValueError:
                            continue

            if not scope.is_empty():
                return scope
        return None

    def authors(self, source_tag: Tag | None) -> list[Author]:
        """Parse all author tags into Author models.

        Args:
            source_tag: XML tag potentially containing author elements.

        Returns:
            List of Author models.
        """
        authors: list[Author] = []

        if source_tag is not None:
            for author in source_tag.find_all("author"):
                if (persname := author.find("persName")) is not None:
                    if (surname_tag := persname.find("surname")) is not None:
                        person_name = PersonName(surname=surname_tag.text or "")

                        if forename_tag := persname.find("forename", {"type": "first"}):
                            person_name.first_name = forename_tag.text

                        author_obj = Author(person_name=person_name)
                        authors.append(author_obj)

                        if email_tag := author.find("email"):
                            author_obj.email = email_tag.text

                        for affiliation_tag in author.find_all("affiliation"):
                            affiliation_obj = Affiliation()

                            for orgname_tag in affiliation_tag.find_all("orgName"):
                                org_type = orgname_tag.get("type")
                                org_text = orgname_tag.text

                                if not isinstance(org_type, str):
                                    continue

                                match org_type:
                                    case "institution":
                                        affiliation_obj.institution = org_text
                                    case "department":
                                        affiliation_obj.department = org_text
                                    case "laboratory":
                                        affiliation_obj.laboratory = org_text

                            if not affiliation_obj.is_empty():
                                author_obj.affiliations.append(affiliation_obj)

        return authors

    def section(self, source_tag: Tag | None, title: str = "") -> Section | None:
        """Parse div tag with head tag into Section model.

        Args:
            source_tag: div XML tag containing section content.
            title: Optional forced title for sections without head tag.

        Returns:
            Section model if valid section found, None otherwise.
        """
        if source_tag is not None:
            head = source_tag.find("head")

            if isinstance(head, Tag):
                head_text: str = head.get_text()
                if "n" in head.attrs or (
                    head_text and head_text[0] in string.ascii_letters
                ):
                    if head_text.isupper() or head_text.islower():
                        head_text = head_text.capitalize()
                    section = Section(title=head_text)
                else:
                    return None
            elif title:
                section = Section(title=title)
            else:
                return None

            paragraphs = source_tag.find_all("p")
            for p in paragraphs:
                if p and (ref_text := self.ref_text(p)) is not None:
                    section.paragraphs.append(ref_text)

            return section
        return None

    def ref_text(self, source_tag: Tag | None) -> RefText | None:
        """Parse paragraph text with embedded reference tags.

        Args:
            source_tag: p (paragraph) XML tag.

        Returns:
            RefText model containing text and reference positions.
        """
        if source_tag is not None:
            text_and_refs = self._text_and_refs(source_tag)
            ref_text = RefText(text="")

            for el in text_and_refs:
                start = len(ref_text.text)

                if isinstance(el, Tag):
                    ref_tag_text = el.text or ""
                    end = start + len(ref_tag_text)
                    ref = Ref(start=start, end=end)

                    if (el_type := el.attrs.get("type")) is not None:
                        if isinstance(el_type, str):
                            try:
                                ref.marker = Marker[el_type]
                            except KeyError:
                                pass

                    if (el_target := el.attrs.get("target")) is not None:
                        if isinstance(el_target, str):
                            ref.target = el_target

                    ref_text.refs.append(ref)
                    ref_text.text += ref_tag_text
                else:
                    ref_text.text += str(el)

            return ref_text
        return None

    def table(self, source_tag: Tag | None) -> Table | None:
        """Parse figure tag with type='table' into Table model.

        Args:
            source_tag: figure XML tag with type="table".

        Returns:
            Table model if valid table found, None otherwise.
        """
        if source_tag is not None:
            if (head_tag := source_tag.find("head")) is not None:
                if head_text := head_tag.get_text():
                    table = Table(heading=head_text)

                    if (desc_tag := source_tag.find("figDesc")) is not None:
                        table.description = desc_tag.get_text()

                    rows = source_tag.find_all("row")
                    for row in rows:
                        row_list = []
                        for cell in row.find_all("cell"):
                            row_list.append(cell.get_text())
                        table.rows.append(row_list)

                    return table
        return None

    def _parse_date(self, date: str) -> Date | None:
        """Parse ISO 8601 date string into Date model.

        Args:
            date: ISO 8601 date string.

        Returns:
            Date model if parseable, None otherwise.
        """
        tokens = date.split(sep="-")
        tokens = list(filter(None, tokens))

        match len(tokens):
            case 0:
                return None
            case 1:
                return Date(year=tokens[0])
            case 2:
                return Date(year=tokens[0], month=tokens[1])
            case _:
                return Date(year=tokens[0], month=tokens[1], day=tokens[2])

    def _text_and_refs(self, source_tag: Tag) -> Generator[PageElement, None, None]:
        """Generate sequence of text nodes and reference tags.

        Args:
            source_tag: XML tag to traverse.

        Yields:
            PageElement: Either a NavigableString or ref Tag.
        """
        for descendant in source_tag.descendants:
            descendant_type = type(descendant)
            if descendant_type is Tag and descendant.name == "ref":  # type: ignore
                yield descendant
            elif descendant_type is NavigableString:
                yield descendant

    @staticmethod
    def _clean_title_string(s: str) -> str:
        """Remove leading non-alphabetic characters and capitalize.

        Args:
            s: Raw title string.

        Returns:
            Cleaned and capitalized title string.
        """
        s = s.strip()

        while s and not s[0].isalpha():
            s = s[1:]

        return s.capitalize()
