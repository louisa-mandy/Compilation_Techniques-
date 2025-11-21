"""
Natural-Language-to-SQL compiler.

This module implements a miniature compiler pipeline that turns a narrow band
of English requests (e.g., “Get the names and emails of customers who live in
Jakarta”) into SQL SELECT/INSERT statements. Each classical compiler phase is made
explicit so the flow is easy to follow and extend:

1. Lexer            – tokenises free-form English text.
2. Parser (LL(1))   – produces an abstract syntax tree (AST) for the NL query.
3. Semantic Mapping – uses a systematic mapping table + patterns to interpret
                      business terminology as concrete SQL schema elements.
4. Code Generator   – renders the interpreted AST into executable SQL.

The implementation is intentionally compact yet showcases how compiler ideas
apply outside traditional programming languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    COMMA = auto()
    PERIOD = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str


class NLLexer:
    """Tokenizes a restricted natural-language query format."""

    KEYWORDS = {
        "GET",
        "INSERT",
        "INTO",
        "VALUES",
        "VALUE",
        "SET",
        "THE",
        "A",
        "AN",
        "NEW",
        "RECORD",
        "RECORDS",
        "OF",
        "WHO",
        "THAT",
        "WHERE",
        "WITH",
        "AND",
        "OR",
        "IN",
        "IS",
        "ARE",
        "AS",
        "TO",
    }
    PUNCTUATION = {
        ",": TokenType.COMMA,
        ".": TokenType.PERIOD,
    }

    def __init__(self, source: str) -> None:
        self.source = source
        self.position = 0

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.position < len(self.source):
            current = self.source[self.position]

            if current.isspace():
                self.position += 1
                continue

            if current in self.PUNCTUATION:
                tokens.append(Token(self.PUNCTUATION[current], current))
                self.position += 1
                continue

            if current in {"'", '"'}:
                tokens.append(self._consume_string(current))
                continue

            if current.isdigit():
                tokens.append(self._consume_number())
                continue

            if current.isalpha():
                tokens.append(self._consume_word())
                continue

            raise ValueError(f"Unexpected character '{current}' at {self.position}")

        tokens.append(Token(TokenType.EOF, ""))
        return tokens

    def _consume_word(self) -> Token:
        start = self.position
        while self.position < len(self.source) and self.source[self.position].isalpha():
            self.position += 1
        word = self.source[start : self.position]
        upper_word = word.upper()
        if upper_word in self.KEYWORDS:
            return Token(TokenType.KEYWORD, upper_word)
        return Token(TokenType.IDENTIFIER, word)

    def _consume_string(self, quote: str) -> Token:
        self.position += 1
        start = self.position
        literal: List[str] = []
        while self.position < len(self.source):
            char = self.source[self.position]
            if char == quote:
                literal.append(self.source[start : self.position])
                self.position += 1
                return Token(TokenType.STRING, "".join(literal))
            self.position += 1
        raise ValueError("Unterminated quoted literal.")

    def _consume_number(self) -> Token:
        start = self.position
        while self.position < len(self.source) and self.source[self.position].isdigit():
            self.position += 1
        return Token(TokenType.NUMBER, self.source[start : self.position])


# --------------------------------------------------------------------------- #
# AST definitions
# --------------------------------------------------------------------------- #


@dataclass
class NLCondition:
    words: List[str]
    connector: Optional[str] = None
    next_condition: Optional["NLCondition"] = None


@dataclass
class NLQuery:
    columns: List[str]
    table: str
    where: Optional[NLCondition] = None


@dataclass
class NLInsert:
    table: str
    assignments: List[tuple[str, str]]


@dataclass
class SQLCondition:
    left: str
    operator: str
    right: str
    connector: Optional[str] = None
    next_condition: Optional["SQLCondition"] = None


@dataclass
class SQLSelect:
    columns: List[str]
    table: str
    where: Optional[SQLCondition] = None


@dataclass
class SQLInsert:
    table: str
    columns: List[str]
    values: List[str]


NLStatement = NLQuery | NLInsert
SQLStatement = SQLSelect | SQLInsert


# --------------------------------------------------------------------------- #
# Parser (LL(1))
# --------------------------------------------------------------------------- #


class LL1Parser:
    """LL(1) parser for the small natural-language query grammar."""

    CONDITION_INTRODUCERS = {"WHO", "THAT", "WHERE", "WITH"}
    CONNECTORS = {"AND", "OR"}
    ASSIGNMENT_INTRODUCERS = {"WITH", "VALUES", "VALUE", "SET"}
    ASSIGNMENT_BRIDGES = {"IS", "ARE", "AS", "TO", "EQUALS"}
    INSERT_FILLERS = {"NEW", "RECORD", "RECORDS"}
    ARTICLES = {"A", "AN", "THE"}

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def parse(self) -> NLStatement:
        token = self._lookahead()
        if token.type == TokenType.KEYWORD:
            if token.value == "GET":
                return self._parse_select_command()
            if token.value == "INSERT":
                return self._parse_insert_command()
        raise ValueError("Only GET ... or INSERT ... statements are supported.")

    def _parse_select_command(self) -> NLQuery:
        self._expect_keyword("GET")
        self._match_keyword("THE")
        columns = self._parse_column_list()
        self._expect_keyword("OF")
        table_phrase = self._parse_table_phrase()
        where_clause = None
        if self._match_condition_introducer():
            where_clause = self._parse_condition_chain()
        self._accept_optional(TokenType.PERIOD)
        self._expect(TokenType.EOF)
        return NLQuery(columns, table_phrase, where_clause)

    def _parse_insert_command(self) -> NLInsert:
        self._expect_keyword("INSERT")
        while self._match_any(self.ARTICLES | self.INSERT_FILLERS):
            continue
        if not self._lookahead_is_keyword("INTO"):
            self._skip_until_keyword("INTO")
        self._expect_keyword("INTO")
        table_phrase = self._parse_table_phrase(stop_keywords=self.ASSIGNMENT_INTRODUCERS)
        if not self._match_any(self.ASSIGNMENT_INTRODUCERS):
            raise ValueError("Expected 'with', 'values', or 'set' to introduce assignments.")
        assignments = self._parse_assignment_list()
        if not assignments:
            raise ValueError("Insert statements require at least one column/value pair.")
        self._accept_optional(TokenType.PERIOD)
        self._expect(TokenType.EOF)
        return NLInsert(table_phrase, assignments)

    def _parse_column_list(self) -> List[str]:
        items: List[List[Token]] = []
        current: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type == TokenType.EOF:
                raise ValueError("Expected column list before end of input.")
            if token.type == TokenType.KEYWORD and token.value == "OF":
                break
            if token.type == TokenType.COMMA or (
                token.type == TokenType.KEYWORD and token.value == "AND"
            ):
                if current:
                    items.append(current)
                    current = []
                self._advance()
                continue
            if token.type == TokenType.KEYWORD and token.value == "THE":
                self._advance()
                continue
            if token.type not in (TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER):
                raise ValueError(f"Unexpected token '{token.value}' in column list.")
            current.append(self._advance())
        if current:
            items.append(current)
        if not items:
            raise ValueError("Column list cannot be empty.")
        return [self._tokens_to_phrase(part) for part in items]

    def _parse_table_phrase(self, stop_keywords: Optional[Iterable[str]] = None) -> str:
        stop_set = set(stop_keywords or [])
        self._match_keyword("THE")
        tokens: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type in {TokenType.EOF, TokenType.PERIOD}:
                break
            if token.type == TokenType.KEYWORD and token.value in self.CONDITION_INTRODUCERS:
                break
            if token.type == TokenType.KEYWORD and token.value in stop_set and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value in self.CONNECTORS and not tokens:
                break
            if token.type == TokenType.COMMA:
                self._advance()
                continue
            if token.type not in (TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER):
                break
            tokens.append(self._advance())
        if not tokens:
            raise ValueError("Expected a table description after 'of'.")
        return self._tokens_to_phrase(tokens)

    def _parse_condition_chain(self) -> NLCondition:
        head = self._parse_condition_fragment()
        current = head
        while True:
            connector = self._match_connector()
            if not connector:
                break
            self._match_condition_introducer()
            next_fragment = self._parse_condition_fragment()
            current.connector = connector
            current.next_condition = next_fragment
            current = next_fragment
        return head

    def _parse_condition_fragment(self) -> NLCondition:
        tokens: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type in {TokenType.EOF, TokenType.PERIOD}:
                break
            if token.type == TokenType.KEYWORD and token.value in self.CONNECTORS:
                break
            if token.type == TokenType.KEYWORD and token.value in self.CONDITION_INTRODUCERS:
                if tokens:
                    break
                self._advance()
                continue
            if token.type in {TokenType.COMMA} and not tokens:
                self._advance()
                continue
            if token.type not in (TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER, TokenType.KEYWORD):
                break
            tokens.append(self._advance())
        if not tokens:
            raise ValueError("Expected a condition description.")
        words = [token.value for token in tokens]
        return NLCondition(words=words)

    def _parse_assignment_list(self) -> List[tuple[str, str]]:
        assignments: List[tuple[str, str]] = []
        while True:
            tokens = self._collect_assignment_tokens()
            column_tokens, value_tokens = self._split_assignment_tokens(tokens)
            if not column_tokens or not value_tokens:
                raise ValueError("Each assignment needs both a column and a value.")
            column = self._tokens_to_phrase(column_tokens)
            value = self._tokens_to_phrase(value_tokens)
            assignments.append((column, value))
            if not self._match_assignment_separator():
                break
        return assignments

    def _collect_assignment_tokens(self) -> List[Token]:
        tokens: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type in {TokenType.EOF, TokenType.PERIOD}:
                break
            if token.type == TokenType.COMMA and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value in self.ASSIGNMENT_INTRODUCERS and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value in self.CONNECTORS and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value == "AND" and tokens:
                break
            tokens.append(self._advance())
        if not tokens:
            raise ValueError("Expected column/value pair after assignments introducer.")
        return tokens

    def _split_assignment_tokens(self, tokens: Sequence[Token]) -> tuple[List[Token], List[Token]]:
        for idx, token in enumerate(tokens):
            if token.type in (TokenType.STRING, TokenType.NUMBER):
                left = list(tokens[:idx])
                right = list(tokens[idx:])
                if left and right:
                    return left, right
        for idx, token in enumerate(tokens):
            if token.type == TokenType.KEYWORD and token.value in self.ASSIGNMENT_BRIDGES:
                left = list(tokens[:idx])
                right = list(tokens[idx + 1 :])
                if left and right:
                    return left, right
        if len(tokens) >= 2:
            return [tokens[0]], list(tokens[1:])
        raise ValueError("Could not determine column/value split.")

    def _match_assignment_separator(self) -> bool:
        token = self._lookahead()
        if token.type == TokenType.COMMA:
            self._advance()
            return True
        if token.type == TokenType.KEYWORD and token.value == "AND":
            self._advance()
            return True
        return False

    def _match_any(self, keywords: Iterable[str]) -> bool:
        token = self._lookahead()
        if token.type == TokenType.KEYWORD and token.value in keywords:
            self.index += 1
            return True
        return False

    def _lookahead_is_keyword(self, keyword: str) -> bool:
        token = self._lookahead()
        return token.type == TokenType.KEYWORD and token.value == keyword

    def _skip_until_keyword(self, keyword: str) -> None:
        while True:
            token = self._lookahead()
            if token.type == TokenType.EOF:
                raise ValueError(f"Expected keyword {keyword} in statement.")
            if token.type == TokenType.KEYWORD and token.value == keyword:
                return
            self._advance()

    # Utility helpers ----------------------------------------------------- #

    def _advance(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _lookahead(self) -> Token:
        return self.tokens[self.index]

    def _expect(self, token_type: TokenType) -> Token:
        token = self._lookahead()
        if token.type != token_type:
            raise ValueError(f"Expected {token_type.name}, found {token.type.name}")
        self.index += 1
        return token

    def _expect_keyword(self, keyword: str) -> None:
        token = self._expect(TokenType.KEYWORD)
        if token.value != keyword:
            raise ValueError(f"Expected keyword {keyword}, found {token.value}")

    def _match_keyword(self, keyword: str) -> bool:
        token = self._lookahead()
        if token.type == TokenType.KEYWORD and token.value == keyword:
            self.index += 1
            return True
        return False

    def _match_condition_introducer(self) -> bool:
        token = self._lookahead()
        if token.type == TokenType.KEYWORD and token.value in self.CONDITION_INTRODUCERS:
            self.index += 1
            return True
        return False

    def _match_connector(self) -> Optional[str]:
        token = self._lookahead()
        if token.type == TokenType.KEYWORD and token.value in self.CONNECTORS:
            self.index += 1
            return token.value
        return None

    def _accept_optional(self, token_type: TokenType) -> None:
        if self._lookahead().type == token_type:
            self.index += 1

    def _tokens_to_phrase(self, tokens: Sequence[Token]) -> str:
        return " ".join(token.value for token in tokens).strip()


# --------------------------------------------------------------------------- #
# Semantic Mapping
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() or char.isspace() or char == "_" else " " for char in text)
    return " ".join(cleaned.split())


def _snake_case(text: str) -> str:
    normalized = _normalize(text)
    return normalized.replace(" ", "_")


@dataclass(frozen=True)
class AttributePattern:
    sequences: Sequence[Sequence[str]]
    column: str
    operator: str
    value_prefix: Optional[str] = None

    def extract_value(self, words: Sequence[str]) -> Optional[str]:
        lowered = [word.lower() for word in words]
        for sequence in self.sequences:
            seq_len = len(sequence)
            if len(lowered) < seq_len:
                continue
            if lowered[:seq_len] != list(sequence):
                continue
            remainder = list(words[seq_len:])
            if self.value_prefix:
                if remainder and remainder[0].lower() == self.value_prefix:
                    remainder = remainder[1:]
                else:
                    continue
            value = " ".join(remainder).strip(" ,.")
            if value:
                return value
        return None


SYSTEMATIC_MAPPING_TABLE: Dict[str, Dict[str, str]] = {
    "columns": {
        "name": "name",
        "names": "name",
        "customer name": "name",
        "customer names": "name",
        "email": "email",
        "emails": "email",
        "email address": "email",
        "email addresses": "email",
        "phone": "phone",
        "phone number": "phone",
        "phone numbers": "phone",
        "all information": "*",
        "everything": "*",
    },
    "tables": {
        "customer": "Customers",
        "customers": "Customers",
        "client": "Customers",
        "clients": "Customers",
        "order": "Orders",
        "orders": "Orders",
        "employee": "Employees",
        "employees": "Employees",
    },
}

ATTRIBUTE_PATTERNS: List[AttributePattern] = [
    AttributePattern(
        sequences=[("live", "in"), ("lives", "in"), ("living", "in"), ("reside", "in"), ("resides", "in")],
        column="city",
        operator="=",
    ),
    AttributePattern(
        sequences=[("are", "in"), ("are", "located", "in"), ("located", "in"), ("based", "in")],
        column="city",
        operator="=",
    ),
    AttributePattern(
        sequences=[("are", "from"), ("come", "from"), ("came", "from")],
        column="country",
        operator="=",
    ),
    AttributePattern(
        sequences=[("have", "status"), ("with", "status")],
        column="status",
        operator="=",
    ),
]


class SemanticMapper:
    """Converts NL AST nodes into SQL-ready representations."""

    def __init__(self, mapping_table: Dict[str, Dict[str, str]], attribute_patterns: Sequence[AttributePattern]) -> None:
        self.mapping = mapping_table
        self.attribute_patterns = attribute_patterns

    def map(self, statement: NLStatement) -> SQLStatement:
        if isinstance(statement, NLQuery):
            return self._map_select(statement)
        if isinstance(statement, NLInsert):
            return self._map_insert(statement)
        raise TypeError(f"Unsupported NL statement: {type(statement).__name__}")

    def _map_select(self, query: NLQuery) -> SQLSelect:
        columns = [self._map_column(col) for col in query.columns] or ["*"]
        table = self._map_table(query.table)
        where = self._map_condition_chain(query.where)
        return SQLSelect(columns, table, where)

    def _map_insert(self, statement: NLInsert) -> SQLInsert:
        columns: List[str] = []
        values: List[str] = []
        for column_phrase, value_phrase in statement.assignments:
            columns.append(self._map_column(column_phrase))
            values.append(self._format_literal(value_phrase))
        table = self._map_table(statement.table)
        return SQLInsert(table=table, columns=columns, values=values)

    def _map_column(self, phrase: str) -> str:
        normalized = _normalize(phrase)
        if normalized in self.mapping["columns"]:
            return self.mapping["columns"][normalized]
        if normalized.endswith("s") and normalized[:-1] in self.mapping["columns"]:
            return self.mapping["columns"][normalized[:-1]]
        return _snake_case(phrase) or "column"

    def _map_table(self, phrase: str) -> str:
        normalized = _normalize(phrase)
        if normalized in self.mapping["tables"]:
            return self.mapping["tables"][normalized]
        words = normalized.split()
        return "".join(word.capitalize() for word in words) or "Table"

    def _map_condition_chain(self, condition: Optional[NLCondition]) -> Optional[SQLCondition]:
        if not condition:
            return None
        head = self._map_single_condition(condition)
        current_sql = head
        current_nl = condition
        while current_nl.next_condition:
            current_sql.connector = current_nl.connector
            next_sql = self._map_single_condition(current_nl.next_condition)
            current_sql.next_condition = next_sql
            current_sql = next_sql
            current_nl = current_nl.next_condition
        return head

    def _map_single_condition(self, condition: NLCondition) -> SQLCondition:
        value = self._match_attribute(condition.words)
        if not value:
            value = self._fallback_condition(condition.words)
        if not value:
            text = " ".join(condition.words)
            raise ValueError(f"Unable to interpret condition '{text}'.")
        column, operator, literal = value
        return SQLCondition(column, operator, literal)

    def _match_attribute(self, words: Sequence[str]) -> Optional[tuple[str, str, str]]:
        for pattern in self.attribute_patterns:
            extracted = pattern.extract_value(words)
            if extracted is not None:
                literal = self._format_literal(extracted)
                return pattern.column, pattern.operator, literal
        return None

    def _fallback_condition(self, words: Sequence[str]) -> Optional[tuple[str, str, str]]:
        """Handle simple '<column> is <value>' style phrases."""
        if len(words) < 3:
            return None
        lowered = [word.lower() for word in words]
        operators = {
            "is": "=",
            "are": "=",
            "equals": "=",
            "equal": "=",
            "greater": ">",
            "greater than": ">",
            "less": "<",
            "less than": "<",
        }
        for op_phrase, operator in operators.items():
            op_words = op_phrase.split()
            if lowered[1 : 1 + len(op_words)] == op_words:
                column_phrase = words[0]
                value_words = words[1 + len(op_words) :]
                if operator in {">", "<"} and value_words and value_words[0].lower() == "than":
                    value_words = value_words[1:]
                value = " ".join(value_words).strip(" ,.")
                if not value:
                    return None
                column = self._map_column(column_phrase)
                literal = self._format_literal(value)
                return column, operator, literal
        return None

    def _format_literal(self, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Missing value in condition.")
        if stripped[0] in {"'", '"'} and stripped[-1] == stripped[0]:
            inner = stripped[1:-1].replace("'", "''")
            return f"'{inner}'"
        if self._is_numeric(stripped):
            return stripped
        escaped = stripped.replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def _is_numeric(value: str) -> bool:
        if not value:
            return False
        try:
            float(value)
        except ValueError:
            return False
        return True


# --------------------------------------------------------------------------- #
# Code Generator
# --------------------------------------------------------------------------- #


class CodeGenerator:
    """Renders SQL AST nodes into executable SQL strings."""

    def generate(self, ast: SQLStatement) -> str:
        columns = ", ".join(ast.columns) if ast.columns else "*"
        if isinstance(ast, SQLSelect):
            sql = f"SELECT {columns} FROM {ast.table}"
            where_clause = self._render_conditions(ast.where)
            if where_clause:
                sql += f" WHERE {where_clause}"
            return sql + ";"
        if isinstance(ast, SQLInsert):
            columns_clause = f"({', '.join(ast.columns)})" if ast.columns else ""
            values_clause = ", ".join(ast.values)
            if columns_clause:
                return f"INSERT INTO {ast.table} {columns_clause} VALUES ({values_clause});"
            return f"INSERT INTO {ast.table} VALUES ({values_clause});"
        raise TypeError(f"Unsupported SQL AST node: {type(ast).__name__}")

    def _render_conditions(self, condition: Optional[SQLCondition]) -> str:
        if not condition:
            return ""
        parts: List[str] = []
        current = condition
        while current:
            parts.append(f"{current.left} {current.operator} {current.right}")
            if current.connector and current.next_condition:
                parts.append(current.connector)
            current = current.next_condition
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Facade API
# --------------------------------------------------------------------------- #


class NLToSQLCompiler:
    """Convenience façade tying all compiler phases together."""

    def __init__(self) -> None:
        self.mapper = SemanticMapper(SYSTEMATIC_MAPPING_TABLE, ATTRIBUTE_PATTERNS)
        self.generator = CodeGenerator()

    def compile(self, text: str) -> str:
        tokens = NLLexer(text).tokenize()
        statement = LL1Parser(tokens).parse()
        sql_ast = self.mapper.map(statement)
        return self.generator.generate(sql_ast)


# --------------------------------------------------------------------------- #
# Demonstration
# --------------------------------------------------------------------------- #


def main() -> None:
    compiler = NLToSQLCompiler()
    print("Natural Language → SQL compiler\n")
    print("How to use:")
    print("  1. Type an English request such as:")
    print("       Get the names and emails of customers who live in Jakarta.")
    print("       Insert a new record into customers with name Sarah and status Active.")
    print("  2. Press Enter to compile it into SQL.\n")
    select_example = "Get the names and emails of customers who live in Jakarta."
    insert_example = "Insert a new record into customers with name Sarah and status Active."
    print("Examples:")
    print(f"  NL : {select_example}")
    print(f"  SQL: {compiler.compile(select_example)}\n")
    print(f"  NL : {insert_example}")
    print(f"  SQL: {compiler.compile(insert_example)}\n")
    print("Enter natural-language requests (blank line exits):\n")

    try:
        while True:
            line = input("NL> ").strip()
            if not line:
                break
            try:
                print("SQL:", compiler.compile(line))
            except Exception as exc:
                print("Error:", exc)
    except EOFError:
        pass


if __name__ == "__main__":
    main()

