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
4. DSL Builder      – emits a compact DSL that captures the interpreted intent.
5. LALR Parser      – validates DSL and rehydrates an executable SQL AST.
6. Code Generator   – renders the final AST into executable SQL.

The implementation is intentionally compact yet showcases how compiler ideas
apply outside traditional programming languages.
"""

from __future__ import annotations

import argparse
import json
import difflib
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
        "DELETE",
        "UPDATE",
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
        "FROM",
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
        "BY",
        "GROUP",
        "HAVING",
        "ORDER",
        "LIMIT",
        "OFFSET",
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
class NLUpdate:
    table: str
    assignments: List[tuple[str, str]]
    where: Optional[NLCondition] = None


@dataclass
class NLDelete:
    table: str
    where: Optional[NLCondition] = None


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
    joins: List["SQLJoin"] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    having: Optional[SQLCondition] = None
    order_by: List["SQLOrder"] = field(default_factory=list)
    distinct: bool = False
    limit: Optional[int | str] = None
    offset: Optional[int | str] = None


@dataclass
class SQLInsert:
    table: str
    columns: List[str]
    values: List[str]


@dataclass
class SQLAssignment:
    column: str
    value: str


@dataclass
class SQLUpdate:
    table: str
    assignments: List[SQLAssignment]
    where: Optional[SQLCondition] = None


@dataclass
class SQLDelete:
    table: str
    where: Optional[SQLCondition] = None


@dataclass
class SQLJoin:
    table: str
    join_type: str = "INNER"
    on: Optional[SQLCondition | str] = None


@dataclass
class SQLOrder:
    expression: str
    direction: str = "ASC"


@dataclass
class DSLConditionSpec:
    column: str
    operator: str
    literal: str
    connector: Optional[str] = None


@dataclass
class DSLSelectSpec:
    columns: List[str]
    table: str
    conditions: List[DSLConditionSpec]


@dataclass
class DSLInsertSpec:
    table: str
    columns: List[str]
    values: List[str]


@dataclass
class DSLDeleteSpec:
    table: str
    conditions: List[DSLConditionSpec]


@dataclass
class DSLUpdateSpec:
    table: str
    assignments: List[tuple[str, str]]
    conditions: List[DSLConditionSpec]


NLStatement = NLQuery | NLInsert | NLUpdate | NLDelete
DSLStatementSpec = DSLSelectSpec | DSLInsertSpec | DSLUpdateSpec | DSLDeleteSpec
SQLStatement = SQLSelect | SQLInsert | SQLUpdate | SQLDelete


@dataclass
class CompilerArtifacts:
    sql: str
    dsl: str
    recommendations: List[str]


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
    DELETE_FILLERS = {"RECORD", "RECORDS", "ROW", "ROWS"}
    ARTICLES = {"A", "AN", "THE"}
    COLUMN_KEYWORDS = {"ALL", "ANY", "DISTINCT", "RECORD", "RECORDS"}

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
            if token.value == "UPDATE":
                return self._parse_update_command()
            if token.value == "DELETE":
                return self._parse_delete_command()
        raise ValueError("Only GET, INSERT, UPDATE, or DELETE statements are supported.")

    def _parse_select_command(self) -> NLQuery:
        self._expect_keyword("GET")
        self._match_keyword("THE")
        columns = self._parse_column_list()
        if self._match_keyword("OF"):
            pass
        elif self._match_keyword("FROM"):
            pass
        else:
            raise ValueError("Expected 'of' or 'from' before the table description.")
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

    def _parse_update_command(self) -> NLUpdate:
        self._expect_keyword("UPDATE")
        self._match_keyword("THE")
        table_phrase = self._parse_table_phrase(stop_keywords=self.ASSIGNMENT_INTRODUCERS | {"SET"})
        if not self._match_any({"SET"} | self.ASSIGNMENT_INTRODUCERS):
            raise ValueError("Expected 'set' or assignment introducer in update statement.")
        assignments = self._parse_assignment_list()
        where_clause = None
        if self._match_condition_introducer():
            where_clause = self._parse_condition_chain()
        self._accept_optional(TokenType.PERIOD)
        self._expect(TokenType.EOF)
        return NLUpdate(table_phrase, assignments, where_clause)

    def _parse_delete_command(self) -> NLDelete:
        self._expect_keyword("DELETE")
        while self._match_any(self.ARTICLES | self.DELETE_FILLERS):
            continue
        if not self._lookahead_is_keyword("FROM"):
            self._skip_until_keyword("FROM")
        self._expect_keyword("FROM")
        table_phrase = self._parse_table_phrase()
        where_clause = None
        if self._match_condition_introducer():
            where_clause = self._parse_condition_chain()
        self._accept_optional(TokenType.PERIOD)
        self._expect(TokenType.EOF)
        return NLDelete(table_phrase, where_clause)

    def _parse_column_list(self) -> List[str]:
        items: List[List[Token]] = []
        current: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type == TokenType.EOF:
                raise ValueError("Expected column list before end of input.")
            if token.type == TokenType.KEYWORD and token.value in {"OF", "FROM"}:
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
            if (
                token.type == TokenType.KEYWORD
                and token.value in self.COLUMN_KEYWORDS
            ):
                current.append(self._advance())
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
            if token.type == TokenType.KEYWORD and token.value == "FROM":
                tokens.clear()
                self._advance()
                continue
            if token.type not in (TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER):
                break
            tokens.append(self._advance())
        if not tokens:
            raise ValueError("Expected a table description after the table introducer.")
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

    def map(self, statement: NLStatement) -> DSLStatementSpec:
        if isinstance(statement, NLQuery):
            return self._map_select(statement)
        if isinstance(statement, NLInsert):
            return self._map_insert(statement)
        if isinstance(statement, NLUpdate):
            return self._map_update(statement)
        if isinstance(statement, NLDelete):
            return self._map_delete(statement)
        raise TypeError(f"Unsupported NL statement: {type(statement).__name__}")

    def _map_select(self, query: NLQuery) -> DSLSelectSpec:
        columns = [self._map_column(col) for col in query.columns] or ["*"]
        table = self._map_table(query.table)
        conditions = self._map_condition_chain(query.where)
        return DSLSelectSpec(columns=columns, table=table, conditions=conditions)

    def _map_insert(self, statement: NLInsert) -> DSLInsertSpec:
        columns: List[str] = []
        values: List[str] = []
        for column_phrase, value_phrase in statement.assignments:
            columns.append(self._map_column(column_phrase))
            values.append(self._format_literal(value_phrase))
        table = self._map_table(statement.table)
        return DSLInsertSpec(table=table, columns=columns, values=values)

    def _map_update(self, statement: NLUpdate) -> DSLUpdateSpec:
        assignments: List[tuple[str, str]] = []
        for column_phrase, value_phrase in statement.assignments:
            column = self._map_column(column_phrase)
            value = self._format_literal(value_phrase)
            assignments.append((column, value))
        table = self._map_table(statement.table)
        conditions = self._map_condition_chain(statement.where)
        return DSLUpdateSpec(table=table, assignments=assignments, conditions=conditions)

    def _map_delete(self, statement: NLDelete) -> DSLDeleteSpec:
        table = self._map_table(statement.table)
        conditions = self._map_condition_chain(statement.where)
        return DSLDeleteSpec(table=table, conditions=conditions)

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

    def _map_condition_chain(self, condition: Optional[NLCondition]) -> List[DSLConditionSpec]:
        specs: List[DSLConditionSpec] = []
        current = condition
        while current:
            spec = self._map_single_condition(current)
            specs.append(spec)
            if current.connector and current.next_condition:
                spec.connector = current.connector
            current = current.next_condition
        return specs

    def _map_single_condition(self, condition: NLCondition) -> DSLConditionSpec:
        value = self._match_attribute(condition.words)
        if not value:
            value = self._fallback_condition(condition.words)
        if not value:
            text = " ".join(condition.words)
            raise ValueError(f"Unable to interpret condition '{text}'.")
        column, operator, literal = value
        return DSLConditionSpec(column=column, operator=operator, literal=literal)

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
        if isinstance(ast, SQLSelect):
            return self._generate_select(ast)
        if isinstance(ast, SQLInsert):
            return self._generate_insert(ast)
        if isinstance(ast, SQLUpdate):
            return self._generate_update(ast)
        if isinstance(ast, SQLDelete):
            return self._generate_delete(ast)
        raise TypeError(f"Unsupported SQL AST node: {type(ast).__name__}")

    def _generate_select(self, ast: SQLSelect) -> str:
        select_kw = "SELECT DISTINCT" if ast.distinct else "SELECT"
        columns = ", ".join(ast.columns) if ast.columns else "*"
        sql_parts = [f"{select_kw} {columns}", f"FROM {ast.table}"]
        for join in ast.joins:
            sql_parts.append(self._render_join(join))
        where_clause = self._render_conditions(ast.where)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
        if ast.group_by:
            sql_parts.append(f"GROUP BY {', '.join(ast.group_by)}")
        having_clause = self._render_conditions(ast.having)
        if having_clause:
            sql_parts.append(f"HAVING {having_clause}")
        if ast.order_by:
            order_tokens = [self._render_order(order) for order in ast.order_by if order.expression]
            if order_tokens:
                sql_parts.append(f"ORDER BY {', '.join(order_tokens)}")
        if ast.limit is not None:
            sql_parts.append(f"LIMIT {ast.limit}")
        if ast.offset is not None:
            sql_parts.append(f"OFFSET {ast.offset}")
        return " ".join(sql_parts) + ";"

    def _generate_insert(self, ast: SQLInsert) -> str:
        columns_clause = f"({', '.join(ast.columns)})" if ast.columns else ""
        values_clause = ", ".join(ast.values)
        if columns_clause:
            return f"INSERT INTO {ast.table} {columns_clause} VALUES ({values_clause});"
        return f"INSERT INTO {ast.table} VALUES ({values_clause});"

    def _generate_update(self, ast: SQLUpdate) -> str:
        if not ast.assignments:
            raise ValueError("UPDATE statements require at least one assignment.")
        assignments_clause = self._render_assignments(ast.assignments)
        sql = f"UPDATE {ast.table} SET {assignments_clause}"
        where_clause = self._render_conditions(ast.where)
        if where_clause:
            sql += f" WHERE {where_clause}"
        return sql + ";"

    def _generate_delete(self, ast: SQLDelete) -> str:
        sql = f"DELETE FROM {ast.table}"
        where_clause = self._render_conditions(ast.where)
        if where_clause:
            sql += f" WHERE {where_clause}"
        return sql + ";"

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

    def _render_join(self, join: SQLJoin) -> str:
        join_keyword = join.join_type.strip() if join.join_type else ""
        if not join_keyword:
            join_keyword = "INNER"
        if "join" in join_keyword.lower():
            clause = f"{join_keyword} {join.table}".strip()
        else:
            clause = f"{join_keyword.upper()} JOIN {join.table}"
        if join.on:
            if isinstance(join.on, SQLCondition):
                clause += f" ON {self._render_conditions(join.on)}"
            else:
                clause += f" ON {join.on}"
        return clause

    def _render_order(self, order: SQLOrder) -> str:
        direction = order.direction.strip().upper() if order.direction else ""
        token = f"{order.expression} {direction}".strip()
        return token or order.expression

    def _render_assignments(self, assignments: Sequence[SQLAssignment | Tuple[str, str]]) -> str:
        rendered: List[str] = []
        for assignment in assignments:
            if isinstance(assignment, SQLAssignment):
                column, value = assignment.column, assignment.value
            else:
                column, value = assignment
            rendered.append(f"{column} = {value}")
        return ", ".join(rendered)


# --------------------------------------------------------------------------- #
# DSL Builder + LALR Parser
# --------------------------------------------------------------------------- #


class DSLBuilder:
    """Produces a deterministic DSL representation from interpreted NL statements."""

    def build(self, spec: DSLStatementSpec) -> str:
        if isinstance(spec, DSLSelectSpec):
            return self._render_select(spec)
        if isinstance(spec, DSLInsertSpec):
            return self._render_insert(spec)
        if isinstance(spec, DSLUpdateSpec):
            return self._render_update(spec)
        if isinstance(spec, DSLDeleteSpec):
            return self._render_delete(spec)
        raise TypeError(f"Unsupported DSL spec: {type(spec).__name__}")

    def _render_select(self, spec: DSLSelectSpec) -> str:
        tokens: List[str] = ["SELECT", "TABLE", spec.table, "COLUMNS"]
        tokens.extend(self._render_identifiers(spec.columns or ["*"]))
        if spec.conditions:
            tokens.append("WHERE")
            tokens.extend(self._render_conditions(spec.conditions))
        return " ".join(tokens)

    def _render_insert(self, spec: DSLInsertSpec) -> str:
        tokens: List[str] = ["INSERT", "TABLE", spec.table, "COLUMNS"]
        tokens.extend(self._render_identifiers(spec.columns))
        tokens.append("VALUES")
        tokens.extend(self._render_literals(spec.values))
        return " ".join(tokens)

    def _render_update(self, spec: DSLUpdateSpec) -> str:
        tokens: List[str] = ["UPDATE", "TABLE", spec.table, "SET"]
        tokens.extend(self._render_assignments(spec.assignments))
        if spec.conditions:
            tokens.append("WHERE")
            tokens.extend(self._render_conditions(spec.conditions))
        return " ".join(tokens)

    def _render_delete(self, spec: DSLDeleteSpec) -> str:
        tokens: List[str] = ["DELETE", "TABLE", spec.table]
        if spec.conditions:
            tokens.append("WHERE")
            tokens.extend(self._render_conditions(spec.conditions))
        return " ".join(tokens)

    def _render_identifiers(self, identifiers: Sequence[str]) -> List[str]:
        if not identifiers:
            return []
        if len(identifiers) == 1 and identifiers[0] == "*":
            return ["*"]
        tokens: List[str] = []
        for index, name in enumerate(identifiers):
            if index:
                tokens.append(",")
            tokens.append(name)
        return tokens

    def _render_literals(self, values: Sequence[str]) -> List[str]:
        tokens: List[str] = []
        for index, value in enumerate(values):
            if index:
                tokens.append(",")
            tokens.append(value)
        return tokens

    def _render_conditions(self, conditions: Sequence[DSLConditionSpec]) -> List[str]:
        tokens: List[str] = []
        for index, condition in enumerate(conditions):
            tokens.extend([condition.column, condition.operator, condition.literal])
            if condition.connector and index < len(conditions) - 1:
                tokens.append(condition.connector)
        return tokens

    def _render_assignments(self, assignments: Sequence[tuple[str, str]]) -> List[str]:
        tokens: List[str] = []
        for index, (column, literal) in enumerate(assignments):
            if index:
                tokens.append(",")
            tokens.extend([column, "=", literal])
        return tokens


@dataclass(frozen=True)
class DSLToken:
    type: str
    value: str


class DSLTokenizer:
    """Tokenizes the DSL emitted by :class:`DSLBuilder`."""

    KEYWORDS = {
        "SELECT",
        "TABLE",
        "COLUMNS",
        "WHERE",
        "INSERT",
        "UPDATE",
        "SET",
        "DELETE",
        "VALUES",
        "AND",
        "OR",
    }
    SIMPLE_TOKENS = {
        ",": "COMMA",
        "(": "LPAREN",
        ")": "RPAREN",
        "=": "EQUAL",
        ">": "GT",
        "<": "LT",
    }

    def __init__(self, source: str) -> None:
        self.source = source
        self.position = 0

    def tokenize(self) -> List[DSLToken]:
        tokens: List[DSLToken] = []
        while self.position < len(self.source):
            char = self.source[self.position]
            if char.isspace():
                self.position += 1
                continue
            if char in self.SIMPLE_TOKENS:
                tokens.append(DSLToken(self.SIMPLE_TOKENS[char], char))
                self.position += 1
                continue
            if char == "*":
                tokens.append(DSLToken("STAR", "*"))
                self.position += 1
                continue
            if char in {"'", '"'}:
                tokens.append(self._consume_string(char))
                continue
            if char.isdigit():
                tokens.append(self._consume_number())
                continue
            if char.isalpha() or char == "_":
                tokens.append(self._consume_word())
                continue
            raise ValueError(f"Unexpected DSL character '{char}' at {self.position}.")
        tokens.append(DSLToken("$", ""))
        return tokens

    def _consume_word(self) -> DSLToken:
        start = self.position
        while self.position < len(self.source) and (
            self.source[self.position].isalnum() or self.source[self.position] == "_"
        ):
            self.position += 1
        word = self.source[start : self.position]
        upper_word = word.upper()
        if upper_word in self.KEYWORDS:
            return DSLToken(upper_word, upper_word)
        return DSLToken("IDENT", word)

    def _consume_number(self) -> DSLToken:
        start = self.position
        while self.position < len(self.source) and (
            self.source[self.position].isdigit() or self.source[self.position] == "."
        ):
            self.position += 1
        return DSLToken("NUMBER", self.source[start : self.position])

    def _consume_string(self, quote: str) -> DSLToken:
        self.position += 1
        start = self.position
        literal: List[str] = []
        while self.position < len(self.source):
            char = self.source[self.position]
            if char == quote:
                literal.append(self.source[start : self.position])
                self.position += 1
                value = quote + "".join(literal) + quote
                return DSLToken("STRING", value)
            self.position += 1
        raise ValueError("Unterminated DSL string literal.")


@dataclass(frozen=True)
class Production:
    head: str
    body: Tuple[str, ...]
    action: Callable[[Sequence[object]], object]


@dataclass(frozen=True)
class LR1Item:
    production_index: int
    position: int
    lookahead: str

    def core(self) -> Tuple[int, int]:
        return (self.production_index, self.position)


@dataclass
class Grammar:
    start_symbol: str
    productions: List[Production]


class LALRParserEngine:
    """Constructs and executes a compact LALR(1) parser."""

    EPSILON = "__ε__"

    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar
        self.productions = [Production("$S'", (grammar.start_symbol,), lambda values: values[0])] + grammar.productions
        self.non_terminals = {prod.head for prod in self.productions}
        self.terminals = self._compute_terminals()
        self.first_sets = self._compute_first_sets()
        canonical_states, transitions = self._build_canonical_lr1_states()
        self.states, self.transitions = self._merge_states(canonical_states, transitions)
        self.action_table, self.goto_table = self._build_parse_tables()

    def parse(self, tokens: Sequence[DSLToken]) -> object:
        stack: List[int] = [0]
        values: List[object] = []
        index = 0
        while True:
            state = stack[-1]
            token = tokens[index]
            action = self.action_table.get(state, {}).get(token.type)
            if not action:
                raise ValueError(f"Unexpected token {token.type}('{token.value}') at DSL position {index}.")
            kind, target = action
            if kind == "shift":
                stack.append(target)
                values.append(token)
                index += 1
            elif kind == "reduce":
                production = self.productions[target]
                body_len = len(production.body)
                reduction_values = values[-body_len:] if body_len else []
                if body_len:
                    del values[-body_len:]
                    del stack[-body_len:]
                result = production.action(reduction_values)
                goto_state = self.goto_table.get(stack[-1], {}).get(production.head)
                if goto_state is None:
                    raise ValueError(f"No goto state for {production.head}.")
                stack.append(goto_state)
                values.append(result)
            elif kind == "accept":
                return values[-1]

    def _compute_terminals(self) -> Set[str]:
        terminals: Set[str] = set()
        non_terminals = {prod.head for prod in self.productions}
        for production in self.productions:
            for symbol in production.body:
                if symbol and symbol not in non_terminals:
                    terminals.add(symbol)
        terminals.add("$")
        return terminals

    def _compute_first_sets(self) -> Dict[str, Set[str]]:
        first: Dict[str, Set[str]] = {symbol: {symbol} for symbol in self.terminals}
        for non_terminal in self.non_terminals:
            first[non_terminal] = set()
        updated = True
        while updated:
            updated = False
            for production in self.productions:
                head_first = first[production.head]
                before_size = len(head_first)
                sequence_first = self._first_of_sequence(production.body, first)
                head_first.update(sequence_first - {self.EPSILON})
                if self.EPSILON in sequence_first:
                    head_first.add(self.EPSILON)
                if len(head_first) != before_size:
                    updated = True
        return first

    def _first_of_sequence(self, sequence: Sequence[str], first_sets: Dict[str, Set[str]]) -> Set[str]:
        if not sequence:
            return {self.EPSILON}
        result: Set[str] = set()
        for symbol in sequence:
            symbol_first = first_sets[symbol]
            result.update(symbol_first - {self.EPSILON})
            if self.EPSILON not in symbol_first:
                break
        else:
            result.add(self.EPSILON)
        return result

    def _closure(self, items: Set[LR1Item]) -> Set[LR1Item]:
        closure_set = set(items)
        added = True
        while added:
            added = False
            for item in list(closure_set):
                production = self.productions[item.production_index]
                if item.position >= len(production.body):
                    continue
                symbol = production.body[item.position]
                if symbol not in self.non_terminals:
                    continue
                beta = production.body[item.position + 1 :]
                lookaheads = self._first_of_sequence(beta + (item.lookahead,), self.first_sets)
                for idx, prod in enumerate(self.productions):
                    if prod.head != symbol:
                        continue
                    for lookahead in lookaheads:
                        if lookahead == self.EPSILON:
                            lookahead = item.lookahead
                        new_item = LR1Item(idx, 0, lookahead)
                        if new_item not in closure_set:
                            closure_set.add(new_item)
                            added = True
        return closure_set

    def _goto(self, items: Set[LR1Item], symbol: str) -> Set[LR1Item]:
        goto_items: Set[LR1Item] = set()
        for item in items:
            production = self.productions[item.production_index]
            if item.position < len(production.body) and production.body[item.position] == symbol:
                goto_items.add(LR1Item(item.production_index, item.position + 1, item.lookahead))
        return self._closure(goto_items) if goto_items else set()

    def _build_canonical_lr1_states(self) -> Tuple[List[Set[LR1Item]], Dict[Tuple[int, str], int]]:
        start_item = LR1Item(0, 0, "$")
        start_state = self._closure({start_item})
        states: List[Set[LR1Item]] = [start_state]
        transitions: Dict[Tuple[int, str], int] = {}
        queue = [0]
        while queue:
            state_index = queue.pop()
            state = states[state_index]
            symbols = {symbol for item in state for symbol in self.productions[item.production_index].body[item.position : item.position + 1]}
            for symbol in symbols:
                target = self._goto(state, symbol)
                if not target:
                    continue
                if target not in states:
                    states.append(target)
                    queue.append(len(states) - 1)
                target_index = states.index(target)
                transitions[(state_index, symbol)] = target_index
        return states, transitions

    def _merge_states(
        self,
        states: List[Set[LR1Item]],
        transitions: Dict[Tuple[int, str], int],
    ) -> Tuple[List[Set[LR1Item]], Dict[Tuple[int, str], int]]:
        kernel_map: Dict[frozenset[Tuple[int, int]], List[int]] = {}
        for index, state in enumerate(states):
            kernel = frozenset(
                item.core()
                for item in state
                if item.position != 0 or self.productions[item.production_index].head == "$S'"
            )
            kernel_map.setdefault(kernel, []).append(index)

        merged_states: List[Set[LR1Item]] = []
        old_to_new: Dict[int, int] = {}
        for kernel_states in kernel_map.values():
            merged: Dict[Tuple[int, int], Set[str]] = {}
            for state_index in kernel_states:
                for item in states[state_index]:
                    merged.setdefault(item.core(), set()).add(item.lookahead)
            merged_set = {
                LR1Item(prod_idx, position, lookahead)
                for (prod_idx, position), lookaheads in merged.items()
                for lookahead in lookaheads
            }
            new_index = len(merged_states)
            merged_states.append(merged_set)
            for state_index in kernel_states:
                old_to_new[state_index] = new_index

        merged_transitions: Dict[Tuple[int, str], int] = {}
        for (state_index, symbol), target in transitions.items():
            merged_transitions[(old_to_new[state_index], symbol)] = old_to_new[target]
        return merged_states, merged_transitions

    def _build_parse_tables(self) -> Tuple[Dict[int, Dict[str, Tuple[str, int]]], Dict[int, Dict[str, int]]]:
        action_table: Dict[int, Dict[str, Tuple[str, int]]] = {}
        goto_table: Dict[int, Dict[str, int]] = {}
        for state_index, state in enumerate(self.states):
            for item in state:
                production = self.productions[item.production_index]
                if item.position < len(production.body):
                    symbol = production.body[item.position]
                    target = self.transitions.get((state_index, symbol))
                    if symbol in self.terminals and target is not None:
                        action_table.setdefault(state_index, {})[symbol] = ("shift", target)
                    elif symbol in self.non_terminals and target is not None:
                        goto_table.setdefault(state_index, {})[symbol] = target
                else:
                    if production.head == "$S'":
                        action_table.setdefault(state_index, {})["$"] = ("accept", 0)
                    else:
                        action_table.setdefault(state_index, {})[item.lookahead] = ("reduce", item.production_index)
        return action_table, goto_table


class DSLParser:
    """Parses DSL text via an automatically generated LALR parser."""

    def __init__(self) -> None:
        grammar = self._build_grammar()
        self.engine = LALRParserEngine(grammar)

    def parse(self, source: str) -> SQLStatement:
        tokens = DSLTokenizer(source).tokenize()
        result = self.engine.parse(tokens)
        if not isinstance(result, (SQLSelect, SQLInsert)):
            raise ValueError("DSL parsing did not produce a SQL statement.")
        return result

    def _build_grammar(self) -> Grammar:
        productions: List[Production] = []

        def add(head: str, body: Sequence[str], action: Callable[[Sequence[object]], object]) -> None:
            productions.append(Production(head, tuple(body), action))

        def conditions_to_chain(conditions: List[Dict[str, str]]) -> Optional[SQLCondition]:
            if not conditions:
                return None
            head = SQLCondition(conditions[0]["column"], conditions[0]["operator"], conditions[0]["literal"])
            current = head
            for condition in conditions[1:]:
                connector = condition.get("connector")
                next_node = SQLCondition(condition["column"], condition["operator"], condition["literal"])
                current.connector = connector
                current.next_condition = next_node
                current = next_node
            return head

        def assignments_to_nodes(pairs: List[Dict[str, str]]) -> List[SQLAssignment]:
            return [SQLAssignment(item["column"], item["literal"]) for item in pairs]

        add(
            "Statement",
            ("SelectStmt",),
            lambda values: values[0],
        )
        add(
            "Statement",
            ("InsertStmt",),
            lambda values: values[0],
        )
        add(
            "Statement",
            ("UpdateStmt",),
            lambda values: values[0],
        )
        add(
            "Statement",
            ("DeleteStmt",),
            lambda values: values[0],
        )
        add(
            "SelectStmt",
            ("SELECT", "TABLE", "IDENT", "COLUMNS", "SelectColumns", "WhereOpt"),
            lambda values: SQLSelect(
                columns=values[4],
                table=values[2].value,
                where=conditions_to_chain(values[5]),
            ),
        )
        add(
            "WhereOpt",
            ("WHERE", "ConditionList"),
            lambda values: values[1],
        )
        add(
            "WhereOpt",
            tuple(),
            lambda values: [],
        )
        add(
            "ConditionList",
            ("ConditionList", "Connector", "Condition"),
            lambda values: values[0] + [dict(values[2], connector=values[1])],
        )
        add(
            "ConditionList",
            ("Condition",),
            lambda values: [values[0]],
        )
        add(
            "Condition",
            ("IDENT", "Operator", "Literal"),
            lambda values: {
                "column": values[0].value,
                "operator": values[1],
                "literal": values[2],
                "connector": None,
            },
        )
        add(
            "Operator",
            ("EQUAL",),
            lambda values: "=",
        )
        add(
            "Operator",
            ("GT",),
            lambda values: ">",
        )
        add(
            "Operator",
            ("LT",),
            lambda values: "<",
        )
        add(
            "Connector",
            ("AND",),
            lambda values: values[0].value,
        )
        add(
            "Connector",
            ("OR",),
            lambda values: values[0].value,
        )
        add(
            "Literal",
            ("STRING",),
            lambda values: values[0].value,
        )
        add(
            "Literal",
            ("NUMBER",),
            lambda values: values[0].value,
        )
        add(
            "Literal",
            ("IDENT",),
            lambda values: values[0].value,
        )
        add(
            "SelectColumns",
            ("STAR",),
            lambda values: ["*"],
        )
        add(
            "SelectColumns",
            ("ColumnSeq",),
            lambda values: values[0],
        )
        add(
            "ColumnSeq",
            ("ColumnSeq", "COMMA", "IDENT"),
            lambda values: values[0] + [values[2].value],
        )
        add(
            "ColumnSeq",
            ("IDENT",),
            lambda values: [values[0].value],
        )
        add(
            "InsertStmt",
            ("INSERT", "TABLE", "IDENT", "COLUMNS", "ColumnSeq", "VALUES", "LiteralSeq"),
            lambda values: SQLInsert(
                table=values[2].value,
                columns=values[4],
                values=values[6],
            ),
        )
        add(
            "UpdateStmt",
            ("UPDATE", "TABLE", "IDENT", "SET", "AssignmentList", "WhereOpt"),
            lambda values: SQLUpdate(
                table=values[2].value,
                assignments=assignments_to_nodes(values[4]),
                where=conditions_to_chain(values[5]),
            ),
        )
        add(
            "DeleteStmt",
            ("DELETE", "TABLE", "IDENT", "WhereOpt"),
            lambda values: SQLDelete(
                table=values[2].value,
                where=conditions_to_chain(values[3]),
            ),
        )
        add(
            "AssignmentList",
            ("AssignmentList", "COMMA", "Assignment"),
            lambda values: values[0] + [values[2]],
        )
        add(
            "AssignmentList",
            ("Assignment",),
            lambda values: [values[0]],
        )
        add(
            "Assignment",
            ("IDENT", "EQUAL", "Literal"),
            lambda values: {"column": values[0].value, "literal": values[2]},
        )
        add(
            "LiteralSeq",
            ("LiteralSeq", "COMMA", "Literal"),
            lambda values: values[0] + [values[2]],
        )
        add(
            "LiteralSeq",
            ("Literal",),
            lambda values: [values[0]],
        )

        return Grammar(start_symbol="Statement", productions=productions)


class QueryRecommender:
    """Suggests replacements for unrecognised words in the NL query."""

    def __init__(self, vocabulary: Iterable[str], cutoff: float = 0.78) -> None:
        normalized_vocab = {_normalize(word): word for word in vocabulary}
        self.vocabulary = normalized_vocab
        self.cutoff = cutoff

    def enhance(self, text: str) -> Tuple[str, List[str]]:
        parts = re.split(r"(\W+)", text)
        recommendations: List[str] = []
        for index, fragment in enumerate(parts):
            if not fragment or not fragment.strip().isalpha():
                continue
            normalized = _normalize(fragment)
            if normalized in self.vocabulary:
                parts[index] = self._preserve_case(fragment, self.vocabulary[normalized])
                continue
            suggestion = self._suggest_word(normalized)
            if suggestion:
                replacement = self._preserve_case(fragment, suggestion)
                recommendations.append(f"Replaced '{fragment}' with '{replacement}'")
                parts[index] = replacement
        return "".join(parts), recommendations

    def _suggest_word(self, token: str) -> Optional[str]:
        if not token:
            return None
        matches = difflib.get_close_matches(token, self.vocabulary.keys(), n=1, cutoff=self.cutoff)
        if not matches:
            return None
        return self.vocabulary[matches[0]]

    @staticmethod
    def _preserve_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement.capitalize()
        return replacement


# --------------------------------------------------------------------------- #
# Spider Dataset Augmentation & Testing Hooks
# --------------------------------------------------------------------------- #


class SpiderDatasetSupport:
    """Utility helpers for incorporating Spider dataset vocabulary and samples."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root)
        self.tables_path = self.root / "tables.json"
        if not self.tables_path.exists():
            raise FileNotFoundError(f"Could not find tables.json under {self.root}")

    def build_schema_vocabulary(self) -> tuple[Set[str], Set[str]]:
        tables_data = self._read_json(self.tables_path)
        table_names: Set[str] = set()
        column_names: Set[str] = set()
        for schema in tables_data:
            for table in schema.get("table_names_original", []):
                table_names.add(table)
                table_names.add(table.lower())
            for _, column in schema.get("column_names_original", []):
                if column == "*" or not column:
                    continue
                column_names.add(column)
                column_names.add(column.lower())
        return table_names, column_names

    def augment_mapping_table(
        self,
        mapping_table: Dict[str, Dict[str, str]],
        table_limit: int = 400,
        column_limit: int = 800,
    ) -> Dict[str, int]:
        table_names, column_names = self.build_schema_vocabulary()
        summary = {"tables_added": 0, "columns_added": 0}
        for name in sorted(table_names):
            normalized = _normalize(name)
            if not normalized:
                continue
            if normalized in mapping_table["tables"]:
                continue
            mapping_table["tables"][normalized] = self._canonical_table_name(name)
            summary["tables_added"] += 1
            if summary["tables_added"] >= table_limit:
                break
        for name in sorted(column_names):
            normalized = _normalize(name)
            if not normalized:
                continue
            if normalized in mapping_table["columns"]:
                continue
            mapping_table["columns"][normalized] = _snake_case(name)
            summary["columns_added"] += 1
            if summary["columns_added"] >= column_limit:
                break
        return summary

    def load_examples(self, split_filename: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
        split_path = self.root / split_filename
        if not split_path.exists():
            return []
        payload = self._read_json(split_path)
        return payload[:limit] if limit is not None else payload

    @staticmethod
    def _canonical_table_name(name: str) -> str:
        normalized = _normalize(name)
        return "".join(word.capitalize() for word in normalized.split()) or "Table"

    @staticmethod
    def _read_json(path: Path) -> List[Dict[str, object]]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


class SpiderCorpusValidator:
    """Runs NL→SQL compilation over Spider samples to expose unsupported syntax."""

    def __init__(self, compiler: NLToSQLCompiler, dataset_support: SpiderDatasetSupport) -> None:
        self.compiler = compiler
        self.support = dataset_support

    def run(self, split_filename: str = "train_spider.json", limit: int = 25) -> Dict[str, object]:
        samples = self.support.load_examples(split_filename, limit=limit)
        successes = 0
        failures: List[Dict[str, str]] = []
        for sample in samples:
            question = str(sample.get("question", "")).strip()
            if not question:
                continue
            try:
                self.compiler.compile(question)
                successes += 1
            except Exception as exc:
                failures.append(
                    {
                        "question": question,
                        "db_id": str(sample.get("db_id", "")),
                        "error": str(exc),
                    }
                )
        return {
            "total": len(samples),
            "successes": successes,
            "failures": failures,
        }


# --------------------------------------------------------------------------- #
# Facade API
# --------------------------------------------------------------------------- #


class NLToSQLCompiler:
    """Convenience façade tying all compiler phases together."""

    def __init__(self) -> None:
        vocabulary = self._build_vocabulary()
        self.recommender = QueryRecommender(vocabulary)
        self.mapper = SemanticMapper(SYSTEMATIC_MAPPING_TABLE, ATTRIBUTE_PATTERNS)
        self.dsl_builder = DSLBuilder()
        self.dsl_parser = DSLParser()
        self.generator = CodeGenerator()

    def compile(self, text: str) -> str:
        sql, _, _ = self._run_pipeline(text)
        return sql

    def compile_with_artifacts(self, text: str) -> CompilerArtifacts:
        sql, dsl, recommendations = self._run_pipeline(text)
        return CompilerArtifacts(sql=sql, dsl=dsl, recommendations=recommendations)

    def _run_pipeline(self, text: str) -> Tuple[str, str, List[str]]:
        enhanced_text, recommendations = self.recommender.enhance(text)
        tokens = NLLexer(enhanced_text).tokenize()
        statement = LL1Parser(tokens).parse()
        interpretation = self.mapper.map(statement)
        dsl_script = self.dsl_builder.build(interpretation)
        reconstructed_ast = self.dsl_parser.parse(dsl_script)
        sql = self.generator.generate(reconstructed_ast)
        return sql, dsl_script, recommendations

    def _build_vocabulary(self) -> Set[str]:
        vocabulary: Set[str] = set(NLLexer.KEYWORDS)
        for mapping in SYSTEMATIC_MAPPING_TABLE.values():
            vocabulary.update(mapping.keys())
            vocabulary.update(mapping.values())
        vocabulary.update({"record", "records", "table", "tables"})
        return vocabulary


# --------------------------------------------------------------------------- #
# Demonstration
# --------------------------------------------------------------------------- #


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Natural language → SQL compiler demo.")
    parser.add_argument("--spider-path", help="Path to a local Spider dataset checkout.")
    parser.add_argument(
        "--augment-mappings",
        action="store_true",
        help="Augment table/column vocabularies with Spider schema names.",
    )
    parser.add_argument(
        "--spider-table-limit",
        type=int,
        default=400,
        help="Maximum number of Spider tables to inject into the mapping table.",
    )
    parser.add_argument(
        "--spider-column-limit",
        type=int,
        default=800,
        help="Maximum number of Spider columns to inject into the mapping table.",
    )
    parser.add_argument(
        "--spider-test",
        type=int,
        default=0,
        help="If >0, run this many Spider NL samples through the pipeline.",
    )
    parser.add_argument(
        "--spider-split",
        default="train_spider.json",
        help="Spider JSON split filename to use for testing (default train_spider.json).",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Skip the interactive prompt (useful when running automated checks).",
    )
    return parser


def _guess_spider_path() -> Optional[Path]:
    """Try to auto-detect a local Spider dataset checkout."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "spider_data",
        script_dir / "__MACOSX" / "spider_data",
        script_dir / "__MACOSX",
        script_dir.parent / "spider_data",
    ]
    for candidate in candidates:
        tables = candidate / "tables.json"
        if tables.exists():
            return candidate
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    spider_support: Optional[SpiderDatasetSupport] = None
    spider_path: Optional[Path] = Path(args.spider_path) if args.spider_path else _guess_spider_path()
    if spider_path:
        try:
            spider_support = SpiderDatasetSupport(spider_path)
            print(f"[Spider] Using dataset at {spider_path}")
            if args.augment_mappings:
                summary = spider_support.augment_mapping_table(
                    SYSTEMATIC_MAPPING_TABLE,
                    table_limit=args.spider_table_limit,
                    column_limit=args.spider_column_limit,
                )
                print(
                    f"[Spider] Added {summary['tables_added']} tables and {summary['columns_added']} columns"
                    " to the mapping vocabulary."
                )
        except Exception as exc:
            print(f"[Spider] Skipping dataset features due to: {exc}")
            spider_support = None
    elif args.augment_mappings or args.spider_test:
        print("[Spider] No dataset found. Pass --spider-path or place 'spider_data' next to this script.")

    compiler = NLToSQLCompiler()
    print("Natural Language → SQL compiler\n")
    print("How to use:")
    print("  1. Type an English request such as:")
    print("       Get the names and emails of customers who live in Jakarta.")
    print("       Insert a new record into customers with name Sarah and status Active.")
    print("  2. Press Enter to compile it into SQL.\n")
    print("Available SQL outputs:")
    print("  - SELECT (supports DISTINCT, JOIN, WHERE, GROUP BY/HAVING, ORDER BY, LIMIT/OFFSET)")
    print("  - INSERT (INSERT INTO ... VALUES ...)")
    print("  - UPDATE (UPDATE ... SET ... WHERE ...)")
    print("  - DELETE (DELETE FROM ... WHERE ...)\n")
    select_example = "Get the names and emails of customers who live in Jakarta."
    insert_example = "Insert a new record into customers with name Sarah and status Active."
    print("Examples:")
    select_artifacts = compiler.compile_with_artifacts(select_example)
    insert_artifacts = compiler.compile_with_artifacts(insert_example)
    print(f"  NL : {select_example}")
    print(f"  DSL: {select_artifacts.dsl}")
    print(f"  SQL: {select_artifacts.sql}\n")
    print(f"  NL : {insert_example}")
    print(f"  DSL: {insert_artifacts.dsl}")
    print(f"  SQL: {insert_artifacts.sql}\n")

    if args.spider_test:
        if spider_support is None:
            print("[Spider] Cannot run tests without --spider-path.")
        else:
            validator = SpiderCorpusValidator(compiler, spider_support)
            summary = validator.run(split_filename=args.spider_split, limit=args.spider_test)
            print(
                f"[Spider] Tested {summary['total']} samples: "
                f"{summary['successes']} succeeded, {len(summary['failures'])} failed."
            )
            if summary["failures"]:
                print("  Example failure:", summary["failures"][0])

    if args.no_repl:
        return

    print("Enter natural-language requests (blank line exits):\n")
    try:
        while True:
            line = input("NL> ").strip()
            if not line:
                break
            try:
                artifacts = compiler.compile_with_artifacts(line)
                if artifacts.recommendations:
                    print("Recommendations:")
                    for recommendation in artifacts.recommendations:
                        print(f"  - {recommendation}")
                print("DSL:", artifacts.dsl)
                print("SQL:", artifacts.sql)
            except Exception as exc:
                print("Error:", exc)
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()

