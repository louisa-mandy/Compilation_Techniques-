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

import difflib
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


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
        "DELETE",
        "UPDATE",
        "FROM",
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
        "DISTINCT",
        "ORDER",
        "BY",
        "LIMIT",
        "OFFSET",
        "PAGE",
        "PER",
        "UNIQUE",
        "LIST",
        "SHOW",
        "DISPLAY",
        "SORT",
        "SORTED",
        "ALL",
        "THEN",
        "FIRST",
        "TOP",
        "MOST",
        "EXPENSIVE",
        "CHEAPEST",
        "DESC",
        "ASC",
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
    distinct: bool = False
    order_clauses: List[tuple[str, bool]] = None  # list of (column, is_desc)
    limit: Optional[int] = None
    offset: Optional[int] = None

    def __post_init__(self):
        if self.order_clauses is None:
            self.order_clauses = []


@dataclass
class NLInsert:
    table: str
    assignments: List[tuple[str, str]]


@dataclass
class NLDelete:
    table: str
    where: Optional[NLCondition] = None


@dataclass
class NLUpdate:
    table: str
    assignments: List[tuple[str, str]]
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
    distinct: bool = False
    order_clauses: List[tuple[str, bool]] = None  # list of (column, is_desc)
    limit: Optional[int] = None
    offset: Optional[int] = None

    def __post_init__(self):
        if self.order_clauses is None:
            self.order_clauses = []


@dataclass
class SQLInsert:
    table: str
    columns: List[str]
    values: List[str]


@dataclass
class SQLDelete:
    table: str
    where: Optional[SQLCondition] = None


@dataclass
class SQLUpdate:
    table: str
    assignments: List[tuple[str, str]]
    where: Optional[SQLCondition] = None


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
    distinct: bool = False
    order_clauses: List[tuple[str, bool]] = None  # list of (column, is_desc)
    limit: Optional[int] = None
    offset: Optional[int] = None

    def __post_init__(self):
        if self.order_clauses is None:
            self.order_clauses = []


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


NLStatement = NLQuery | NLInsert | NLDelete | NLUpdate
DSLStatementSpec = DSLSelectSpec | DSLInsertSpec | DSLDeleteSpec | DSLUpdateSpec
SQLStatement = SQLSelect | SQLInsert | SQLDelete | SQLUpdate


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
    ARTICLES = {"A", "AN", "THE"}

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def parse(self) -> NLStatement:
        token = self._lookahead()
        if token.type == TokenType.KEYWORD:
            if token.value in {"GET", "LIST", "SHOW", "DISPLAY", "SORT", "SORTED"}:
                return self._parse_select_command()
            if token.value == "INSERT":
                return self._parse_insert_command()
            if token.value == "DELETE":
                return self._parse_delete_command()
            if token.value == "UPDATE":
                return self._parse_update_command()
        raise ValueError("Only GET ..., INSERT ..., DELETE ..., or UPDATE ... statements are supported.")

    def _parse_select_command(self) -> NLQuery:
        # Accept several verbs that introduce a select-style request
        verb = self._expect(TokenType.KEYWORD)
        if verb.value not in {"GET", "LIST", "SHOW", "DISPLAY", "SORT", "SORTED"}:
            raise ValueError(f"Expected select introducer, found {verb.value}")
        distinct = False
        order_clauses: List[tuple[str, bool]] = []
        limit = None
        offset = None
        self._match_keyword("THE")
        # Check for DISTINCT or UNIQUE
        if self._match_any({"DISTINCT", "UNIQUE"}):
            distinct = True

        # Special-case forms that start with a number/page/most (e.g. "show the 5 most expensive products"
        # or "display page 2 of customers with 20 per page"). If the next token indicates these forms,
        # parse them with heuristics rather than a normal column list.
        next_token = self._lookahead()
        columns: List[str] = ["*"]
        table_phrase: Optional[str] = None
        where_clause: Optional[NLCondition] = None

        if next_token.type == TokenType.NUMBER or (
            next_token.type == TokenType.KEYWORD and next_token.value in {"PAGE", "MOST", "TOP", "FIRST"}
        ):
            # Handle LIMIT-prefixed forms
            if next_token.type == TokenType.NUMBER:
                limit = int(self._advance().value)
            elif next_token.type == TokenType.KEYWORD and next_token.value == "PAGE":
                # page N ... (we'll compute offset when we find 'per' value)
                self._advance()
                if self._lookahead().type == TokenType.NUMBER:
                    page_num = int(self._advance().value)
                else:
                    raise ValueError("Expected page number after 'page'.")
            else:
                page_num = None

            # Look for modifiers like MOST EXPENSIVE
            if self._match_any({"MOST", "EXPENSIVE", "CHEAPEST", "TOP", "FIRST"}):
                # consume any consecutive adjective keywords
                modifiers: List[str] = []
                while self._lookahead().type == TokenType.KEYWORD and self._lookahead().value in {
                    "MOST",
                    "EXPENSIVE",
                    "CHEAPEST",
                    "TOP",
                    "FIRST",
                }:
                    modifiers.append(self._advance().value)
                # common heuristic: 'most expensive' -> order by 'price' desc
                if "EXPENSIVE" in modifiers or "MOST" in modifiers:
                    order_clauses.append(("price", True))
                if "CHEAPEST" in modifiers:
                    order_clauses.append(("price", False))

            # Parse table phrase (allow optional 'OF')
            if self._match_keyword("OF"):
                table_phrase = self._parse_table_phrase()
            else:
                table_phrase = self._parse_table_phrase()

            # If we saw a 'page N' earlier and there's a trailing 'with M per page', handle it
            if 'page_num' in locals():
                if self._match_any({"WITH"}):
                    if self._lookahead().type == TokenType.NUMBER:
                        per_page = int(self._advance().value)
                        if self._match_any({"PER"}):
                            self._match_keyword("PAGE")
                            limit = per_page
                            offset = (page_num - 1) * per_page
            columns = ["*"]
        elif verb.value in {"SORT", "SORTED"}:
            # SORT/SORTED can be followed by table name then BY for ordering
            # Form: "SORT <table> BY <col> <direction>"
            # or: "SORT <col> BY <another_col>" (less likely but possible)
            # We'll try to parse the next identifier/phrase as a table
            start_index = self.index
            table_phrase = self._parse_table_phrase_for_sort_context()
            columns = ["*"]
            # Match optional "all" keyword that may appear between table and ORDER BY
            self._match_any({"ALL"})
            # Now we expect BY for the sort clause
            if self._match_any({"BY"}):
                order_clauses = self._parse_order_by_clause()
            elif self._match_any({"SORTED"}):
                # Handle "SORT table SORTED BY col" variant
                self._expect_keyword("BY")
                order_clauses = self._parse_order_by_clause()
        else:
            # Normal form: parse a column list and optionally 'of' table
            start_index = self.index
            try:
                columns = self._parse_column_list()
            except ValueError:
                # fall back to treat as select-all and continue
                self.index = start_index
                columns = ["*"]

            # If 'OF' present, parse table phrase, otherwise leave it None for mapper to infer
            if self._match_keyword("OF"):
                table_phrase = self._parse_table_phrase()
            else:
                table_phrase = None

            if self._match_condition_introducer():
                where_clause = self._parse_condition_chain()
            
            # Match optional "all" keyword that may appear between table and ORDER BY
            self._match_any({"ALL"})
            
            # Check for ORDER BY with multi-column support
            if self._match_any({"ORDER"}):
                self._expect_keyword("BY")
                order_clauses = self._parse_order_by_clause()
        
        # Check for direction phrases like "oldest to newest", "highest to lowest", "alphabetically"
        order_clauses.extend(self._detect_order_direction_phrases(columns))
        
        # Check for LIMIT
        if self._match_any({"LIMIT", "TOP", "FIRST"}):
            limit_token = self._lookahead()
            if limit_token.type == TokenType.NUMBER:
                limit = int(self._advance().value)
        # Check for OFFSET or PAGE
        if self._match_any({"OFFSET"}):
            offset_token = self._lookahead()
            if offset_token.type == TokenType.NUMBER:
                offset = int(self._advance().value)
        elif self._match_any({"PAGE"}):
            page_token = self._lookahead()
            if page_token.type == TokenType.NUMBER:
                page_num = int(self._advance().value)
                # Look for 'PER' and number
                if self._match_any({"PER"}):
                    per_token = self._lookahead()
                    if per_token.type == TokenType.NUMBER:
                        per_page = int(self._advance().value)
                        limit = per_page
                        offset = (page_num - 1) * per_page
        self._accept_optional(TokenType.PERIOD)
        self._expect(TokenType.EOF)
        return NLQuery(
            columns,
            table_phrase,
            where_clause,
            distinct=distinct,
            order_clauses=order_clauses,
            limit=limit,
            offset=offset,
        )

    def _parse_table_phrase_for_sort_context(self) -> str:
        """Parse a table name in SORT context.
        
        For SORT/SORTED contexts, parse table name(s) and stop at BY or other keywords.
        """
        self._match_keyword("THE")
        tokens: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type in {TokenType.EOF, TokenType.PERIOD}:
                break
            if token.type == TokenType.KEYWORD and token.value in {"BY", "ORDER", "WHERE", "LIMIT", "OFFSET", "SORTED", "ALL"}:
                break
            if token.type == TokenType.COMMA:
                self._advance()
                continue
            if token.type not in (TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER):
                break
            tokens.append(self._advance())
        if not tokens:
            raise ValueError("Expected a table description.")
        return self._tokens_to_phrase(tokens)


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

    def _parse_delete_command(self) -> NLDelete:
        self._expect_keyword("DELETE")
        self._match_keyword("THE")
        while self._match_any(self.ARTICLES | {"RECORD", "RECORDS"}):
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

    def _parse_update_command(self) -> NLUpdate:
        self._expect_keyword("UPDATE")
        self._match_keyword("THE")
        table_phrase = self._parse_table_phrase(stop_keywords=self.ASSIGNMENT_INTRODUCERS)
        if not self._match_any(self.ASSIGNMENT_INTRODUCERS):
            raise ValueError("Expected 'with', 'values', 'set' to introduce assignments.")
        assignments = self._parse_assignment_list()
        if not assignments:
            raise ValueError("Update statements require at least one column/value pair.")
        where_clause = None
        if self._match_condition_introducer():
            where_clause = self._parse_condition_chain()
        self._accept_optional(TokenType.PERIOD)
        self._expect(TokenType.EOF)
        return NLUpdate(table_phrase, assignments, where_clause)

    def _parse_column_list(self) -> List[str]:
        items: List[List[Token]] = []
        current: List[Token] = []
        while True:
            token = self._lookahead()
            if token.type == TokenType.EOF:
                break
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
            if token.type == TokenType.KEYWORD and token.value == "WHERE" and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value == "ORDER" and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value == "LIMIT" and tokens:
                break
            if token.type == TokenType.KEYWORD and token.value == "OFFSET" and tokens:
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

    def _parse_order_by_clause(self) -> List[tuple[str, bool]]:
        """Parse ORDER BY clause with multi-column support.
        
        Returns list of (column, is_desc) tuples.
        Recognizes forms like:
          - ORDER BY column
          - ORDER BY column DESC
          - ORDER BY column ASC
          - ORDER BY col1, col2
          - ORDER BY col1 DESC, col2 ASC
          - ORDER BY col1, then col2
        """
        order_clauses: List[tuple[str, bool]] = []
        while True:
            # Parse column name(s)
            col_tokens = []
            while True:
                token = self._lookahead()
                if token.type in {TokenType.EOF, TokenType.PERIOD}:
                    break
                if token.type == TokenType.KEYWORD and token.value in {"LIMIT", "OFFSET"}:
                    break
                if token.type == TokenType.COMMA:
                    break
                if token.type == TokenType.KEYWORD and token.value == "THEN":
                    break
                if token.type == TokenType.KEYWORD and token.value in {"DESC", "ASC"}:
                    break
                if token.type not in (TokenType.IDENTIFIER, TokenType.STRING, TokenType.NUMBER):
                    break
                col_tokens.append(self._advance())
            
            if col_tokens:
                col_name = self._tokens_to_phrase(col_tokens)
                is_desc = False
                # Check for DESC/ASC modifier
                if self._match_any({"DESC"}):
                    is_desc = True
                elif self._match_any({"ASC"}):
                    is_desc = False
                order_clauses.append((col_name, is_desc))
            
            # Check for comma or THEN separator (accept both)
            if self._match_keyword("THEN"):
                continue
            elif self._match_keyword("COMMA"):
                continue
            elif self._lookahead().type == TokenType.COMMA:
                self._advance()
                continue
            else:
                break
        
        return order_clauses

    def _detect_order_direction_phrases(self, columns: List[str]) -> List[tuple[str, bool]]:
        """Detect and consume natural language direction phrases like 'oldest to newest', 'highest to lowest', 'alphabetically'.
        
        Scans and consumes tokens (until LIMIT/OFFSET/EOF) for direction indicators and maps them to
        implicit ORDER BY clauses based on column names.
        
        Returns list of additional (column, is_desc) tuples to append to order_clauses.
        """
        additional_orders: List[tuple[str, bool]] = []
        
        # Scan ahead for direction keywords and collect them
        tokens_ahead = []
        while True:
            token = self._lookahead()
            if token.type in {TokenType.EOF, TokenType.PERIOD}:
                break
            if token.type == TokenType.KEYWORD and token.value in {"LIMIT", "OFFSET", "ORDER", "WHERE"}:
                break
            tokens_ahead.append(token.value.lower() if token.type == TokenType.KEYWORD else token.value)
            self._advance()  # Actually consume the token
        
        text_ahead = " ".join(tokens_ahead)
        
        # Heuristics for direction inference
        if "alphabetically" in text_ahead:
            # alphabetically -> sort first column by name ASC
            if columns and columns[0] != "*":
                additional_orders.append((columns[0], False))
        elif "oldest" in text_ahead and "newest" in text_ahead:
            # "oldest to newest" -> ASC on first date-like column
            # Infer from columns: hire_date, date_of_birth, etc.
            for col in columns:
                if "date" in col.lower() or "birth" in col.lower():
                    additional_orders.append((col, False))
                    break
        elif "newest" in text_ahead and "oldest" in text_ahead:
            # "newest to oldest" -> DESC
            for col in columns:
                if "date" in col.lower() or "birth" in col.lower():
                    additional_orders.append((col, True))
                    break
        elif "highest" in text_ahead and "lowest" in text_ahead:
            # "highest to lowest" -> DESC on numeric column
            for col in columns:
                if "gpa" in col.lower() or "price" in col.lower() or "amount" in col.lower():
                    additional_orders.append((col, True))
                    break
        elif "lowest" in text_ahead and "highest" in text_ahead:
            # "lowest to highest" -> ASC
            for col in columns:
                if "gpa" in col.lower() or "price" in col.lower() or "amount" in col.lower():
                    additional_orders.append((col, False))
                    break
        
        return additional_orders

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
        "customer name": "customer_name",
        "customer names": "customer_name",
        "email": "email",
        "emails": "email",
        "email address": "email",
        "email addresses": "email",
        "job title": "job_title",
        "job titles": "job_title",
        "product name": "product_name",
        "product names": "product_name",
        "price": "price",
        "phone": "phone",
        "phone number": "phone",
        "phone numbers": "phone",
        "hire date": "hire_date",
        "hire_date": "hire_date",
        "gpa": "gpa",
        "student name": "student_name",
        "student_name": "student_name",
        "employee name": "employee_name",
        "employee_name": "employee_name",
        "title": "title",
        "author": "author",
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
        "student": "Students",
        "students": "Students",
        "book": "Books",
        "books": "Books",
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
        if isinstance(statement, NLDelete):
            return self._map_delete(statement)
        if isinstance(statement, NLUpdate):
            return self._map_update(statement)
        raise TypeError(f"Unsupported NL statement: {type(statement).__name__}")

    def _map_select(self, query: NLQuery) -> DSLSelectSpec:
        columns = [self._map_column(col) for col in query.columns] or ["*"]
        # Table may be omitted in terse requests; attempt to infer from columns when missing
        if query.table:
            table = self._map_table(query.table)
        else:
            table = None
            lowered_cols = " ".join(col.lower() for col in query.columns)
            if "job" in lowered_cols or "title" in lowered_cols or "hire" in lowered_cols:
                table = "Employees"
            elif "customer" in lowered_cols or "customer_id" in lowered_cols:
                table = "Customers"
            elif "product" in lowered_cols or "price" in lowered_cols:
                table = "Products"
            elif "student" in lowered_cols or "gpa" in lowered_cols:
                table = "Students"
            elif "book" in lowered_cols or "author" in lowered_cols:
                table = "Books"
            else:
                # fallback to generic Table name when not inferrable
                table = "Table"
        # normalize table for DSL (lowercase) since DSL emits lowercased names
        table = table.lower() if table else table
        conditions = self._map_condition_chain(query.where)

        # Heuristics: when user requests the "most expensive products" prefer product name + price
        mapped_order_clauses: List[tuple[str, bool]] = []
        for col_name, is_desc in query.order_clauses:
            mapped_col = self._map_column(col_name)
            mapped_order_clauses.append((mapped_col, is_desc))
        
        if columns == ["*"] and table == "products":
            for col_name, _ in mapped_order_clauses:
                if col_name == "price":
                    columns = [self._map_column("product name"), self._map_column("price")]
                    break

        # For paging requests, if no explicit ORDER BY, set sensible default ordering
        if (query.limit or query.offset) and not mapped_order_clauses:
            if table == "customers":
                mapped_order_clauses.append((self._map_column("customer id"), False))
            elif table == "products":
                mapped_order_clauses.append((self._map_column("product id"), False))
            elif table == "employees":
                mapped_order_clauses.append((self._map_column("employee id"), False))
            elif table == "students":
                mapped_order_clauses.append((self._map_column("student id"), False))
            elif table == "books":
                mapped_order_clauses.append((self._map_column("book id"), False))

        return DSLSelectSpec(
            columns=columns,
            table=table,
            conditions=conditions,
            distinct=query.distinct,
            order_clauses=mapped_order_clauses,
            limit=query.limit,
            offset=query.offset,
        )

    def _map_insert(self, statement: NLInsert) -> DSLInsertSpec:
        columns: List[str] = []
        values: List[str] = []
        for column_phrase, value_phrase in statement.assignments:
            columns.append(self._map_column(column_phrase))
            values.append(self._format_literal(value_phrase))
        table = self._map_table(statement.table)
        return DSLInsertSpec(table=table, columns=columns, values=values)

    def _map_delete(self, statement: NLDelete) -> DSLDeleteSpec:
        table = self._map_table(statement.table)
        conditions = self._map_condition_chain(statement.where)
        return DSLDeleteSpec(table=table, conditions=conditions)

    def _map_update(self, statement: NLUpdate) -> DSLUpdateSpec:
        table = self._map_table(statement.table)
        assignments: List[tuple[str, str]] = []
        for column_phrase, value_phrase in statement.assignments:
            column = self._map_column(column_phrase)
            value = self._format_literal(value_phrase)
            assignments.append((column, value))
        conditions = self._map_condition_chain(statement.where)
        return DSLUpdateSpec(table=table, assignments=assignments, conditions=conditions)

    def _map_column(self, phrase: str) -> str:
        if phrase is None:
            return "column"
        if phrase.strip() == "*":
            return "*"
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
            columns = ", ".join(ast.columns) if ast.columns else "*"
            sql = "SELECT"
            if getattr(ast, "distinct", False):
                sql += " DISTINCT"
            sql += f" {columns} FROM {ast.table}"
            where_clause = self._render_conditions(ast.where)
            if where_clause:
                sql += f" WHERE {where_clause}"
            # Render multi-column ORDER BY
            if getattr(ast, "order_clauses", None):
                order_parts = []
                for col, is_desc in ast.order_clauses:
                    part = col
                    if is_desc:
                        part += " DESC"
                    order_parts.append(part)
                if order_parts:
                    sql += " ORDER BY " + ", ".join(order_parts)
            if getattr(ast, "limit", None) is not None:
                sql += f" LIMIT {ast.limit}"
            if getattr(ast, "offset", None) is not None:
                sql += f" OFFSET {ast.offset}"
            return sql + ";"
        if isinstance(ast, SQLInsert):
            columns_clause = f"({', '.join(ast.columns)})" if ast.columns else ""
            values_clause = ", ".join(ast.values)
            if columns_clause:
                return f"INSERT INTO {ast.table} {columns_clause} VALUES ({values_clause});"
            return f"INSERT INTO {ast.table} VALUES ({values_clause});"
        if isinstance(ast, SQLDelete):
            sql = f"DELETE FROM {ast.table}"
            where_clause = self._render_conditions(ast.where)
            if where_clause:
                sql += f" WHERE {where_clause}"
            return sql + ";"
        if isinstance(ast, SQLUpdate):
            set_clause = ", ".join(f"{col} = {val}" for col, val in ast.assignments)
            sql = f"UPDATE {ast.table} SET {set_clause}"
            where_clause = self._render_conditions(ast.where)
            if where_clause:
                sql += f" WHERE {where_clause}"
            return sql + ";"
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
# DSL Builder + LALR Parser
# --------------------------------------------------------------------------- #


class DSLBuilder:
    """Produces a deterministic DSL representation from interpreted NL statements."""

    def build(self, spec: DSLStatementSpec) -> str:
        if isinstance(spec, DSLSelectSpec):
            return self._render_select(spec)
        if isinstance(spec, DSLInsertSpec):
            return self._render_insert(spec)
        if isinstance(spec, DSLDeleteSpec):
            return self._render_delete(spec)
        if isinstance(spec, DSLUpdateSpec):
            return self._render_update(spec)
        raise TypeError(f"Unsupported DSL spec: {type(spec).__name__}")

    def _render_select(self, spec: DSLSelectSpec) -> str:
        tokens: List[str] = ["SELECT"]
        if getattr(spec, "distinct", False):
            tokens.append("DISTINCT")
        tokens.extend(["TABLE", spec.table.lower(), "COLUMNS"])
        tokens.extend(self._render_identifiers(spec.columns or ["*"]))
        if spec.conditions:
            tokens.append("WHERE")
            tokens.extend(self._render_conditions(spec.conditions))
        # Render multi-column ORDER BY
        if getattr(spec, "order_clauses", None):
            tokens.append("ORDER")
            tokens.append("BY")
            for idx, (col, is_desc) in enumerate(spec.order_clauses):
                if idx > 0:
                    tokens.append("COMMA")
                tokens.append(col)
                if is_desc:
                    tokens.append("DESC")
        if getattr(spec, "limit", None) is not None:
            tokens.append("LIMIT")
            tokens.append(str(spec.limit))
            if getattr(spec, "offset", None) is not None:
                tokens.append("OFFSET")
                tokens.append(str(spec.offset))
        return " ".join(tokens)

    def _render_insert(self, spec: DSLInsertSpec) -> str:
        tokens: List[str] = ["INSERT", "TABLE", spec.table, "COLUMNS"]
        tokens.extend(self._render_identifiers(spec.columns))
        tokens.append("VALUES")
        tokens.extend(self._render_literals(spec.values))
        return " ".join(tokens)

    def _render_delete(self, spec: DSLDeleteSpec) -> str:
        tokens: List[str] = ["DELETE", "TABLE", spec.table]
        if spec.conditions:
            tokens.append("WHERE")
            tokens.extend(self._render_conditions(spec.conditions))
        return " ".join(tokens)

    def _render_update(self, spec: DSLUpdateSpec) -> str:
        tokens: List[str] = ["UPDATE", "TABLE", spec.table, "SET"]
        assignment_tokens: List[str] = []
        for index, (column, value) in enumerate(spec.assignments):
            if index:
                assignment_tokens.append(",")
            assignment_tokens.extend([column, "=", value])
        tokens.extend(assignment_tokens)
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
        "DISTINCT",
        "ORDER",
        "BY",
        "LIMIT",
        "OFFSET",
        "INSERT",
        "DELETE",
        "UPDATE",
        "SET",
        "VALUES",
        "AND",
        "OR",
        "DESC",
        "ASC",
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
        if not isinstance(result, (SQLSelect, SQLInsert, SQLDelete, SQLUpdate)):
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
            ("DeleteStmt",),
            lambda values: values[0],
        )
        add(
            "Statement",
            ("UpdateStmt",),
            lambda values: values[0],
        )
        add(
            "SelectStmt",
            ("SELECT", "DistinctOpt", "TABLE", "IDENT", "COLUMNS", "SelectColumns", "WhereOpt", "OrderOpt", "LimitOpt"),
            lambda values: SQLSelect(
                columns=values[5],
                table=values[3].value,
                where=conditions_to_chain(values[6]),
                distinct=values[1],
                order_clauses=values[7],
                limit=values[8][0],
                offset=values[8][1],
            ),
        )

        add(
            "DistinctOpt",
            ("DISTINCT",),
            lambda values: True,
        )
        add(
            "DistinctOpt",
            tuple(),
            lambda values: False,
        )

        add(
            "OrderOpt",
            ("ORDER", "BY", "OrderList"),
            lambda values: values[2],
        )
        add(
            "OrderOpt",
            tuple(),
            lambda values: [],
        )

        add(
            "OrderList",
            ("OrderList", "COMMA", "OrderItem"),
            lambda values: values[0] + [values[2]],
        )
        add(
            "OrderList",
            ("OrderItem",),
            lambda values: [values[0]],
        )

        add(
            "OrderItem",
            ("IDENT", "OrderDir"),
            lambda values: (values[0].value, values[1]),
        )

        add(
            "OrderDir",
            ("DESC",),
            lambda values: True,
        )
        add(
            "OrderDir",
            ("ASC",),
            lambda values: False,
        )
        add(
            "OrderDir",
            tuple(),
            lambda values: False,
        )

        add(
            "LimitOpt",
            ("LIMIT", "NUMBER", "OffsetOpt"),
            lambda values: (int(values[1].value), values[2]),
        )
        add(
            "LimitOpt",
            tuple(),
            lambda values: (None, None),
        )

        add(
            "OffsetOpt",
            ("OFFSET", "NUMBER"),
            lambda values: int(values[1].value),
        )
        add(
            "OffsetOpt",
            tuple(),
            lambda values: None,
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
            "LiteralSeq",
            ("LiteralSeq", "COMMA", "Literal"),
            lambda values: values[0] + [values[2]],
        )
        add(
            "LiteralSeq",
            ("Literal",),
            lambda values: [values[0]],
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
            "UpdateStmt",
            ("UPDATE", "TABLE", "IDENT", "SET", "AssignmentSeq", "WhereOpt"),
            lambda values: SQLUpdate(
                table=values[2].value,
                assignments=values[4],
                where=conditions_to_chain(values[5]),
            ),
        )
        add(
            "AssignmentSeq",
            ("AssignmentSeq", "COMMA", "IDENT", "EQUAL", "Literal"),
            lambda values: values[0] + [(values[2].value, values[4])],
        )
        add(
            "AssignmentSeq",
            ("IDENT", "EQUAL", "Literal"),
            lambda values: [(values[0].value, values[2])],
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
# GUI
# --------------------------------------------------------------------------- #


class NLToSQLGUI:
    def __init__(self, root):
        self.compiler = NLToSQLCompiler()
        self.root = root
        self.root.title("NL to SQL Compiler")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 9))

        # Scrollable frame
        self.scrollable_frame = ScrollableFrame(root)
        self.scrollable_frame.pack(fill="both", expand=True)
        main_frame = self.scrollable_frame.scrollable_frame

        # Title
        title_label = ttk.Label(main_frame, text="Natural Language → SQL Compiler", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Instructions
        instr_frame = ttk.LabelFrame(main_frame, text="Available SQL Syntax", padding=5)
        instr_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(instr_frame, text="• SELECT: Get data from tables").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(instr_frame, text="• INSERT: Add new records").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(instr_frame, text="• DELETE: Remove records").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(instr_frame, text="• UPDATE: Modify records").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(instr_frame, text="• ORDER BY: Sort records").grid(row=4, column=0, sticky=tk.W)

        # Input
        ttk.Label(main_frame, text="Enter query:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.input_text = scrolledtext.ScrolledText(main_frame, height=3, wrap=tk.WORD, font=("Arial", 9))
        self.input_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        ttk.Button(button_frame, text="Compile", command=self.compile_query).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Examples", command=self.show_examples).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_output).grid(row=0, column=2, padx=5)

        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding=5)
        output_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # DSL
        ttk.Label(output_frame, text="DSL:").grid(row=0, column=0, sticky=tk.W)
        self.dsl_text = scrolledtext.ScrolledText(output_frame, height=2, wrap=tk.WORD, state='disabled', font=("Courier", 9))
        self.dsl_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        # SQL
        ttk.Label(output_frame, text="SQL:").grid(row=2, column=0, sticky=tk.W)
        self.sql_text = scrolledtext.ScrolledText(output_frame, height=2, wrap=tk.WORD, state='disabled', font=("Courier", 9))
        self.sql_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        # Recommendations
        ttk.Label(output_frame, text="Recommendations:").grid(row=4, column=0, sticky=tk.W)
        self.rec_text = scrolledtext.ScrolledText(output_frame, height=2, wrap=tk.WORD, state='disabled', font=("Arial", 9))
        self.rec_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(1, weight=1)

    def compile_query(self):
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query.")
            return
        try:
            artifacts = self.compiler.compile_with_artifacts(query)
            self.update_output(artifacts)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_output(self, artifacts):
        self.dsl_text.config(state='normal')
        self.dsl_text.delete("1.0", tk.END)
        self.dsl_text.insert(tk.END, artifacts.dsl)
        self.dsl_text.config(state='disabled')

        self.sql_text.config(state='normal')
        self.sql_text.delete("1.0", tk.END)
        self.sql_text.insert(tk.END, artifacts.sql)
        self.sql_text.config(state='disabled')

        self.rec_text.config(state='normal')
        self.rec_text.delete("1.0", tk.END)
        if artifacts.recommendations:
            self.rec_text.insert(tk.END, "\n".join(artifacts.recommendations))
        else:
            self.rec_text.insert(tk.END, "None")
        self.rec_text.config(state='disabled')

    def show_examples(self):
        examples = [
            "Get the names and emails of customers who live in Jakarta.",
            "Insert a new record into customers with name Sarah and status Active.",
            "Delete the records from customers who live in Jakarta.",
            "Update the customers with status Active where city is Jakarta."
        ]
        example_window = tk.Toplevel(self.root)
        example_window.title("Examples")
        example_window.geometry("550x350")
        example_window.resizable(False, False)
        text = scrolledtext.ScrolledText(example_window, wrap=tk.WORD, font=("Arial", 9))
        text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        for ex in examples:
            text.insert(tk.END, ex + "\n\n")
        text.config(state='disabled')

    def clear_output(self):
        self.input_text.delete("1.0", tk.END)
        self.dsl_text.config(state='normal')
        self.dsl_text.delete("1.0", tk.END)
        self.dsl_text.config(state='disabled')
        self.sql_text.config(state='normal')
        self.sql_text.delete("1.0", tk.END)
        self.sql_text.config(state='disabled')
        self.rec_text.config(state='normal')
        self.rec_text.delete("1.0", tk.END)
        self.rec_text.config(state='disabled')


def main() -> None:
    root = tk.Tk()
    app = NLToSQLGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
