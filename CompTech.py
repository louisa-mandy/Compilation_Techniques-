"""
Natural-Language-to-SQL compiler.

This module implements a miniature compiler pipeline that turns a narrow band
of English requests (e.g., “Get the names and emails of customers who live in
Jakarta”) into SQL SELECT/INSERT statements. Each classical compiler phase is made
explicit so the flow is easy to follow and extend:

1. Lexer            - tokenises free-form English text.
2. Parser (LL(1))   - produces an abstract syntax tree (AST) for the NL query.
3. Semantic Mapping - uses a systematic mapping table + patterns to interpret
                      business terminology as concrete SQL schema elements.
4. DSL Builder      - emits a compact DSL that captures the interpreted intent.
5. LALR Parser      - validates DSL and rehydrates an executable SQL AST.
6. Semantic Analysis - validates queries using symbol table, detects semantic errors,
                      and produces an annotated AST with type and metadata information.
7. Code Generator   - renders the final AST into executable SQL.

The implementation is intentionally compact yet showcases how compiler ideas
apply outside traditional programming languages.
"""

from __future__ import annotations

import difflib
import json
import hashlib
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import streamlit as st


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

        # Suggestion-only typo handling: if the word is close to a keyword,
        # raise a LexicalError with a suggestion to help the user correct typos.
        try:
            suggestion = difflib.get_close_matches(upper_word, self.KEYWORDS, n=1, cutoff=0.8)
        except Exception:
            suggestion = []
        if suggestion:
            raise LexicalError(
                f"Unknown keyword '{word}'. Did you mean '{suggestion[0]}'?"
            )

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


# --- Intermediate Representation (IR) for DSL (tree form) -----------------


@dataclass
class IRCondition:
    column: str
    operator: str
    literal: str
    connector: Optional[str] = None


@dataclass
class IRSelect:
    projections: List[str]
    table: str
    conditions: List[IRCondition]
    distinct: bool = False
    order_clauses: List[tuple[str, bool]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None

    def __post_init__(self):
        if self.order_clauses is None:
            self.order_clauses = []


IRStatement = IRSelect


@dataclass
class IRInsert:
    table: str
    columns: List[str]
    values: List[str]


@dataclass
class IRDelete:
    table: str
    conditions: List[IRCondition]


@dataclass
class IRUpdate:
    table: str
    assignments: List[tuple[str, str]]
    conditions: List[IRCondition]


IRStatement = IRSelect | IRInsert | IRDelete | IRUpdate



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


# --------------------------------------------------------------------------- #
# Semantic Error Handling
# --------------------------------------------------------------------------- #


class SemanticError(Exception):
    """Base class for semantic errors (errors in meaning, not syntax)."""
    
    def __init__(self, message: str, location: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.location = location
    
    def __str__(self) -> str:
        if self.location:
            return f"Semantic Error at {self.location}: {self.message}"
        return f"Semantic Error: {self.message}"


class UnknownColumnError(SemanticError):
    """Raised when a column is referenced but doesn't exist in any table."""
    
    def __init__(self, column: str, table: Optional[str] = None, location: Optional[str] = None):
        if table:
            message = f"Column '{column}' does not exist in table '{table}'"
        else:
            message = f"Column '{column}' does not exist in any known table"
        super().__init__(message, location)
        self.column = column
        self.table = table


class AmbiguousColumnError(SemanticError):
    """Raised when a column name is ambiguous (exists in multiple tables)."""
    
    def __init__(self, column: str, tables: List[str], location: Optional[str] = None):
        tables_str = ", ".join(tables)
        message = f"Column '{column}' is ambiguous - exists in multiple tables: {tables_str}"
        super().__init__(message, location)
        self.column = column
        self.tables = tables


class UnknownTableError(SemanticError):
    """Raised when a table is referenced but doesn't exist."""
    
    def __init__(self, table: str, location: Optional[str] = None):
        message = f"Table '{table}' does not exist"
        super().__init__(message, location)
        self.table = table


class TypeMismatchError(SemanticError):
    """Raised when there's a type mismatch in operations."""
    
    def __init__(self, expected_type: str, actual_type: str, context: str, location: Optional[str] = None):
        message = f"Type mismatch in {context}: expected {expected_type}, got {actual_type}"
        super().__init__(message, location)
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.context = context


class InvalidOperationError(SemanticError):
    """Raised when an operation is invalid for the given types."""
    
    def __init__(self, operation: str, details: str, location: Optional[str] = None):
        message = f"Invalid operation '{operation}': {details}"
        super().__init__(message, location)
        self.operation = operation
        self.details = details


class LexicalError(Exception):
    """Raised for lexical/tokenization errors, such as unknown keywords.

    This error may include a suggestion when a close keyword match is found.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"Lexical Error: {self.message}"


# --------------------------------------------------------------------------- #
# Symbol Table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColumnInfo:
    """Information about a database column."""
    name: str
    table: str
    data_type: str  # e.g., "VARCHAR", "INTEGER", "DECIMAL", "DATE"
    nullable: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True)
class TableInfo:
    """Information about a database table."""
    name: str
    columns: Dict[str, ColumnInfo]  # column_name -> ColumnInfo
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class SymbolTable:
    """Tracks tables, columns, and their relationships for semantic validation."""
    
    def __init__(self):
        self._tables: Dict[str, TableInfo] = {}
        self._column_to_tables: Dict[str, List[str]] = {}  # column_name -> [table_names]
    
    def register_table(self, table: TableInfo) -> None:
        """Register a table and its columns in the symbol table."""
        self._tables[table.name.lower()] = table
        
        # Update column-to-tables mapping
        for column_name in table.columns:
            col_lower = column_name.lower()
            if col_lower not in self._column_to_tables:
                self._column_to_tables[col_lower] = []
            if table.name.lower() not in self._column_to_tables[col_lower]:
                self._column_to_tables[col_lower].append(table.name.lower())
    
    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Get table information by name (case-insensitive)."""
        return self._tables.get(table_name.lower())
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        return table_name.lower() in self._tables
    
    def get_column(self, column_name: str, table_name: Optional[str] = None) -> Optional[ColumnInfo]:
        """Get column information.
        
        If table_name is provided, returns the column from that specific table.
        If table_name is None, returns the column if it's unambiguous (exists in exactly one table).
        Raises AmbiguousColumnError if the column exists in multiple tables.
        """
        col_lower = column_name.lower()
        
        if table_name:
            table = self.get_table(table_name)
            if table:
                return table.columns.get(col_lower)
            return None
        
        # Check if column exists in any table
        tables_with_column = self._column_to_tables.get(col_lower, [])
        
        if not tables_with_column:
            return None
        
        if len(tables_with_column) == 1:
            table = self._tables[tables_with_column[0]]
            return table.columns.get(col_lower)
        
        # Ambiguous - exists in multiple tables
        return None
    
    def find_column_tables(self, column_name: str) -> List[str]:
        """Find all tables that contain a given column."""
        return self._column_to_tables.get(column_name.lower(), []).copy()
    
    def validate_column_reference(self, column_name: str, table_name: Optional[str] = None) -> ColumnInfo:
        """Validate a column reference and return its info.
        
        Raises:
            UnknownColumnError: if column doesn't exist
            AmbiguousColumnError: if column exists in multiple tables and table_name not provided
        """
        if table_name:
            if not self.table_exists(table_name):
                raise UnknownTableError(table_name)
            column = self.get_column(column_name, table_name)
            if not column:
                raise UnknownColumnError(column_name, table_name)
            return column
        
        # No table specified - check for ambiguity
        tables_with_column = self.find_column_tables(column_name)
        
        if not tables_with_column:
            raise UnknownColumnError(column_name)
        
        if len(tables_with_column) > 1:
            raise AmbiguousColumnError(column_name, tables_with_column)
        
        # Unambiguous - get from the single table
        column = self.get_column(column_name, tables_with_column[0])
        if not column:
            raise UnknownColumnError(column_name)
        return column
    
    def get_all_tables(self) -> List[str]:
        """Get list of all registered table names."""
        return list(self._tables.keys())


def create_default_symbol_table() -> SymbolTable:
    """Create a symbol table with default schema information."""
    symtab = SymbolTable()
    
    # Customers table
    customers_columns = {
        "customer_id": ColumnInfo("customer_id", "Customers", "INTEGER", False),
        "customer_name": ColumnInfo("customer_name", "Customers", "VARCHAR", False),
        "name": ColumnInfo("name", "Customers", "VARCHAR", True),
        "email": ColumnInfo("email", "Customers", "VARCHAR", True),
        "phone": ColumnInfo("phone", "Customers", "VARCHAR", True),
        "city": ColumnInfo("city", "Customers", "VARCHAR", True),
        "country": ColumnInfo("country", "Customers", "VARCHAR", True),
        "status": ColumnInfo("status", "Customers", "VARCHAR", True),
    }
    symtab.register_table(TableInfo("Customers", customers_columns))
    
    # Employees table
    employees_columns = {
        "employee_id": ColumnInfo("employee_id", "Employees", "INTEGER", False),
        "employee_name": ColumnInfo("employee_name", "Employees", "VARCHAR", False),
        "name": ColumnInfo("name", "Employees", "VARCHAR", True),
        "job_title": ColumnInfo("job_title", "Employees", "VARCHAR", True),
        "hire_date": ColumnInfo("hire_date", "Employees", "DATE", True),
        "email": ColumnInfo("email", "Employees", "VARCHAR", True),
        "city": ColumnInfo("city", "Employees", "VARCHAR", True),
    }
    symtab.register_table(TableInfo("Employees", employees_columns))
    
    # Students table
    students_columns = {
        "student_id": ColumnInfo("student_id", "Students", "INTEGER", False),
        "student_name": ColumnInfo("student_name", "Students", "VARCHAR", False),
        "name": ColumnInfo("name", "Students", "VARCHAR", True),
        "gpa": ColumnInfo("gpa", "Students", "DECIMAL", True),
        "email": ColumnInfo("email", "Students", "VARCHAR", True),
    }
    symtab.register_table(TableInfo("Students", students_columns))
    
    # Books table
    books_columns = {
        "book_id": ColumnInfo("book_id", "Books", "INTEGER", False),
        "title": ColumnInfo("title", "Books", "VARCHAR", False),
        "author": ColumnInfo("author", "Books", "VARCHAR", True),
        "product_name": ColumnInfo("product_name", "Books", "VARCHAR", True),
    }
    symtab.register_table(TableInfo("Books", books_columns))
    
    # Orders table
    orders_columns = {
        "order_id": ColumnInfo("order_id", "Orders", "INTEGER", False),
        "customer_id": ColumnInfo("customer_id", "Orders", "INTEGER", True),
        "order_date": ColumnInfo("order_date", "Orders", "DATE", True),
        "status": ColumnInfo("status", "Orders", "VARCHAR", True),
    }
    symtab.register_table(TableInfo("Orders", orders_columns))
    
    # Products table (if referenced)
    products_columns = {
        "product_id": ColumnInfo("product_id", "Products", "INTEGER", False),
        "product_name": ColumnInfo("product_name", "Products", "VARCHAR", False),
        "price": ColumnInfo("price", "Products", "DECIMAL", True),
    }
    symtab.register_table(TableInfo("Products", products_columns))
    
    return symtab


def load_schema_from_json(json_data: Union[str, Dict[str, Any]]) -> SymbolTable:
    """Load a symbol table from a JSON schema definition.
    
    JSON Format:
    {
        "tables": [
            {
                "name": "TableName",
                "columns": [
                    {
                        "name": "column_name",
                        "data_type": "VARCHAR" | "INTEGER" | "DECIMAL" | "DATE" | etc.,
                        "nullable": true | false,
                        "metadata": {}  // optional
                    }
                ],
                "metadata": {}  // optional
            }
        ]
    }
    
    Args:
        json_data: JSON string or dictionary containing schema definition
        
    Returns:
        SymbolTable populated with the schema
        
    Raises:
        ValueError: If JSON is invalid or schema structure is incorrect
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        try:
            schema = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        schema = json_data
    
    # Validate schema structure
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object")
    
    if "tables" not in schema:
        raise ValueError("Schema must contain a 'tables' array")
    
    if not isinstance(schema["tables"], list):
        raise ValueError("'tables' must be an array")
    
    # Create symbol table
    symtab = SymbolTable()
    
    # Process each table
    for table_def in schema["tables"]:
        if not isinstance(table_def, dict):
            raise ValueError("Each table definition must be an object")
        
        if "name" not in table_def:
            raise ValueError("Table definition must have a 'name' field")
        
        if "columns" not in table_def:
            raise ValueError(f"Table '{table_def['name']}' must have a 'columns' array")
        
        if not isinstance(table_def["columns"], list):
            raise ValueError(f"Table '{table_def['name']}' columns must be an array")
        
        table_name = str(table_def["name"])
        table_metadata = table_def.get("metadata", {})
        
        # Process columns
        columns: Dict[str, ColumnInfo] = {}
        for col_def in table_def["columns"]:
            if not isinstance(col_def, dict):
                raise ValueError(f"Column definition in table '{table_name}' must be an object")
            
            if "name" not in col_def:
                raise ValueError(f"Column in table '{table_name}' must have a 'name' field")
            
            if "data_type" not in col_def:
                raise ValueError(f"Column '{col_def['name']}' in table '{table_name}' must have a 'data_type' field")
            
            col_name = str(col_def["name"])
            data_type = str(col_def["data_type"])
            nullable = col_def.get("nullable", True)
            col_metadata = col_def.get("metadata", {})
            
            column_info = ColumnInfo(
                name=col_name,
                table=table_name,
                data_type=data_type,
                nullable=nullable,
                metadata=col_metadata
            )
            columns[col_name] = column_info
        
        # Register table
        table_info = TableInfo(
            name=table_name,
            columns=columns,
            metadata=table_metadata
        )
        symtab.register_table(table_info)
    
    return symtab


def export_schema_to_json(symbol_table: SymbolTable) -> str:
    """Export a symbol table to JSON format.
    
    Args:
        symbol_table: The symbol table to export
        
    Returns:
        JSON string representation of the schema
    """
    schema = {
        "tables": []
    }
    
    for table_name in symbol_table.get_all_tables():
        table_info = symbol_table.get_table(table_name)
        if not table_info:
            continue
        
        table_def = {
            "name": table_info.name,
            "columns": [],
            "metadata": table_info.metadata or {}
        }
        
        for col_name, col_info in table_info.columns.items():
            col_def = {
                "name": col_info.name,
                "data_type": col_info.data_type,
                "nullable": col_info.nullable,
                "metadata": col_info.metadata or {}
            }
            table_def["columns"].append(col_def)
        
        schema["tables"].append(table_def)
    
    return json.dumps(schema, indent=2)


def create_default_schema_json() -> str:
    """Create a JSON representation of the default schema (for reference/export)."""
    default_symtab = create_default_symbol_table()
    return export_schema_to_json(default_symtab)


def get_schema_format_example() -> str:
    """Get an example JSON schema format for documentation."""
    example = {
        "tables": [
            {
                "name": "Customers",
                "columns": [
                    {
                        "name": "customer_id",
                        "data_type": "INTEGER",
                        "nullable": False,
                        "metadata": {}
                    },
                    {
                        "name": "customer_name",
                        "data_type": "VARCHAR",
                        "nullable": False,
                        "metadata": {}
                    },
                    {
                        "name": "email",
                        "data_type": "VARCHAR",
                        "nullable": True,
                        "metadata": {}
                    },
                    {
                        "name": "city",
                        "data_type": "VARCHAR",
                        "nullable": True,
                        "metadata": {}
                    }
                ],
                "metadata": {}
            }
        ]
    }
    return json.dumps(example, indent=2)


# --------------------------------------------------------------------------- #
# Annotated AST
# --------------------------------------------------------------------------- #


@dataclass
class AnnotatedColumn:
    """An annotated column reference with type and source table information."""
    name: str
    source_table: Optional[str] = None
    column_info: Optional[ColumnInfo] = None
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.column_info and not self.data_type:
            self.data_type = self.column_info.data_type
        if self.column_info and not self.source_table:
            self.source_table = self.column_info.table


@dataclass
class AnnotatedCondition:
    """An annotated condition with type information."""
    left: AnnotatedColumn
    operator: str
    right: str  # literal value
    right_type: Optional[str] = None
    connector: Optional[str] = None
    next_condition: Optional["AnnotatedCondition"] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnnotatedSelect:
    """An annotated SELECT statement with semantic information."""
    columns: List[AnnotatedColumn]
    table: str
    table_info: Optional[TableInfo] = None
    where: Optional[AnnotatedCondition] = None
    distinct: bool = False
    order_clauses: List[tuple[AnnotatedColumn, bool]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.order_clauses is None:
            self.order_clauses = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnnotatedInsert:
    """An annotated INSERT statement with semantic information."""
    table: str
    assignments: List[tuple[AnnotatedColumn, str]]  # (column, value)
    table_info: Optional[TableInfo] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnnotatedDelete:
    """An annotated DELETE statement with semantic information."""
    table: str
    table_info: Optional[TableInfo] = None
    where: Optional[AnnotatedCondition] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnnotatedUpdate:
    """An annotated UPDATE statement with semantic information."""
    table: str
    assignments: List[tuple[AnnotatedColumn, str]]  # (column, value)
    table_info: Optional[TableInfo] = None
    where: Optional[AnnotatedCondition] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


AnnotatedStatement = AnnotatedSelect | AnnotatedInsert | AnnotatedDelete | AnnotatedUpdate


# --------------------------------------------------------------------------- #
# Semantic Analyzer
# --------------------------------------------------------------------------- #


class SemanticAnalyzer:
    """Performs semantic analysis on AST nodes using the symbol table."""
    
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
    
    def analyze(self, statement: SQLStatement) -> AnnotatedStatement:
        """Analyze a SQL statement and return an annotated AST."""
        if isinstance(statement, SQLSelect):
            return self._analyze_select(statement)
        if isinstance(statement, SQLInsert):
            return self._analyze_insert(statement)
        if isinstance(statement, SQLDelete):
            return self._analyze_delete(statement)
        if isinstance(statement, SQLUpdate):
            return self._analyze_update(statement)
        raise TypeError(f"Unsupported statement type: {type(statement).__name__}")
    
    def _analyze_select(self, select: SQLSelect) -> AnnotatedSelect:
        """Analyze a SELECT statement."""
        # Validate table exists
        if not self.symbol_table.table_exists(select.table):
            raise UnknownTableError(select.table)
        
        table_info = self.symbol_table.get_table(select.table)
        
        # Analyze columns
        annotated_columns: List[AnnotatedColumn] = []
        for col in select.columns:
            if col == "*":
                # Wildcard - create special annotation
                annotated_columns.append(AnnotatedColumn(
                    name="*",
                    source_table=select.table,
                    column_info=None,
                    data_type=None
                ))
            else:
                # Validate column exists in table
                try:
                    col_info = self.symbol_table.validate_column_reference(col, select.table)
                    annotated_columns.append(AnnotatedColumn(
                        name=col,
                        source_table=select.table,
                        column_info=col_info,
                        data_type=col_info.data_type
                    ))
                except (UnknownColumnError, AmbiguousColumnError) as e:
                    raise SemanticError(f"Invalid column '{col}' in SELECT: {e.message}")
        
        # Analyze WHERE clause
        annotated_where = None
        if select.where:
            annotated_where = self._analyze_condition(select.where, select.table)
        
        # Analyze ORDER BY clauses
        annotated_order: List[tuple[AnnotatedColumn, bool]] = []
        for col_name, is_desc in select.order_clauses:
            try:
                col_info = self.symbol_table.validate_column_reference(col_name, select.table)
                annotated_col = AnnotatedColumn(
                    name=col_name,
                    source_table=select.table,
                    column_info=col_info,
                    data_type=col_info.data_type
                )
                annotated_order.append((annotated_col, is_desc))
            except (UnknownColumnError, AmbiguousColumnError) as e:
                raise SemanticError(f"Invalid column '{col_name}' in ORDER BY: {e.message}")
        
        return AnnotatedSelect(
            columns=annotated_columns,
            table=select.table,
            table_info=table_info,
            where=annotated_where,
            distinct=select.distinct,
            order_clauses=annotated_order,
            limit=select.limit,
            offset=select.offset
        )
    
    def _analyze_insert(self, insert: SQLInsert) -> AnnotatedInsert:
        """Analyze an INSERT statement."""
        # Validate table exists
        if not self.symbol_table.table_exists(insert.table):
            raise UnknownTableError(insert.table)
        
        table_info = self.symbol_table.get_table(insert.table)
        
        # Analyze column assignments
        annotated_assignments: List[tuple[AnnotatedColumn, str]] = []
        for col, value in zip(insert.columns, insert.values):
            try:
                col_info = self.symbol_table.validate_column_reference(col, insert.table)
                annotated_col = AnnotatedColumn(
                    name=col,
                    source_table=insert.table,
                    column_info=col_info,
                    data_type=col_info.data_type
                )
                annotated_assignments.append((annotated_col, value))
            except (UnknownColumnError, AmbiguousColumnError) as e:
                raise SemanticError(f"Invalid column '{col}' in INSERT: {e.message}")
        
        return AnnotatedInsert(
            table=insert.table,
            assignments=annotated_assignments,
            table_info=table_info
        )
    
    def _analyze_delete(self, delete: SQLDelete) -> AnnotatedDelete:
        """Analyze a DELETE statement."""
        # Validate table exists
        if not self.symbol_table.table_exists(delete.table):
            raise UnknownTableError(delete.table)
        
        table_info = self.symbol_table.get_table(delete.table)
        
        # Analyze WHERE clause
        annotated_where = None
        if delete.where:
            annotated_where = self._analyze_condition(delete.where, delete.table)
        
        return AnnotatedDelete(
            table=delete.table,
            table_info=table_info,
            where=annotated_where
        )
    
    def _analyze_update(self, update: SQLUpdate) -> AnnotatedUpdate:
        """Analyze an UPDATE statement."""
        # Validate table exists
        if not self.symbol_table.table_exists(update.table):
            raise UnknownTableError(update.table)
        
        table_info = self.symbol_table.get_table(update.table)
        
        # Analyze column assignments
        annotated_assignments: List[tuple[AnnotatedColumn, str]] = []
        for col, value in update.assignments:
            try:
                col_info = self.symbol_table.validate_column_reference(col, update.table)
                annotated_col = AnnotatedColumn(
                    name=col,
                    source_table=update.table,
                    column_info=col_info,
                    data_type=col_info.data_type
                )
                annotated_assignments.append((annotated_col, value))
            except (UnknownColumnError, AmbiguousColumnError) as e:
                raise SemanticError(f"Invalid column '{col}' in UPDATE: {e.message}")
        
        # Analyze WHERE clause
        annotated_where = None
        if update.where:
            annotated_where = self._analyze_condition(update.where, update.table)
        
        return AnnotatedUpdate(
            table=update.table,
            assignments=annotated_assignments,
            table_info=table_info,
            where=annotated_where
        )
    
    def _analyze_condition(self, condition: SQLCondition, table_name: str) -> AnnotatedCondition:
        """Analyze a condition chain."""
        # Analyze left side (column)
        try:
            col_info = self.symbol_table.validate_column_reference(condition.left, table_name)
            annotated_left = AnnotatedColumn(
                name=condition.left,
                source_table=table_name,
                column_info=col_info,
                data_type=col_info.data_type
            )
        except (UnknownColumnError, AmbiguousColumnError) as e:
            raise SemanticError(f"Invalid column '{condition.left}' in condition: {e.message}")
        
        # Infer right side type (simple heuristic - could be enhanced)
        right_type = self._infer_literal_type(condition.right)
        
        annotated_cond = AnnotatedCondition(
            left=annotated_left,
            operator=condition.operator,
            right=condition.right,
            right_type=right_type,
            connector=condition.connector
        )
        
        # Recursively analyze next condition
        if condition.next_condition:
            annotated_cond.next_condition = self._analyze_condition(condition.next_condition, table_name)
        
        return annotated_cond
    
    def _infer_literal_type(self, literal: str) -> str:
        """Infer the type of a literal value."""
        # Remove quotes if present
        stripped = literal.strip()
        if stripped.startswith("'") and stripped.endswith("'"):
            return "VARCHAR"
        if stripped.startswith('"') and stripped.endswith('"'):
            return "VARCHAR"
        
        # Check if numeric
        try:
            float(stripped)
            if '.' in stripped:
                return "DECIMAL"
            return "INTEGER"
        except ValueError:
            pass
        
        # Default to VARCHAR
        return "VARCHAR"


@dataclass
class CompilerArtifacts:
    sql: str
    dsl: str
    recommendations: List[str]


@dataclass
class CompilerDebugInfo:
    """Debug information showing all compiler phases."""
    original_text: str
    enhanced_text: str
    tokens: List[Token]
    nl_ast: NLStatement
    dsl_spec: DSLStatementSpec
    dsl_script: str
    sql_ast: SQLStatement
    annotated_ast: Optional[AnnotatedStatement] = None
    semantic_errors: List[SemanticError] = None
    sql: str = ""
    recommendations: List[str] = None
    vm_output: Optional[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.semantic_errors is None:
            self.semantic_errors = []


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


class VirtualMachine:
    """A simple virtual machine that simulates how the SQL AST would be executed.

    This VM does NOT connect to a real database. Instead, it produces a
    human-readable execution trace that explains how a typical SQL engine
    would conceptually process the compiled statement.
    """

    def execute(self, ast: SQLStatement) -> str:
        steps: List[str] = []

        if isinstance(ast, SQLSelect):
            steps.append("VM: BEGIN SELECT")
            steps.append(f"  - Target table        : {ast.table}")

            cols = ast.columns or ["*"]
            steps.append(f"  - Project columns     : {', '.join(cols)}")

            if ast.where:
                steps.append("  - Apply filter (WHERE):")
                for line in self._render_condition_steps(ast.where):
                    steps.append(f"      {line}")
            else:
                steps.append("  - No WHERE filter (full table scan or index scan)")

            if getattr(ast, "order_clauses", None):
                order_desc: List[str] = []
                for col, is_desc in ast.order_clauses:
                    direction = "DESC" if is_desc else "ASC"
                    order_desc.append(f"{col} {direction}")
                if order_desc:
                    steps.append(f"  - Sort rows (ORDER BY): {', '.join(order_desc)}")
            else:
                steps.append("  - No explicit ORDER BY (engine default ordering)")

            if getattr(ast, "limit", None) is not None:
                if getattr(ast, "offset", None) is not None:
                    steps.append(
                        f"  - Limit rows          : LIMIT {ast.limit} OFFSET {ast.offset}"
                    )
                else:
                    steps.append(f"  - Limit rows          : LIMIT {ast.limit}")
            elif getattr(ast, "offset", None) is not None:
                steps.append(f"  - Offset rows         : OFFSET {ast.offset}")

            steps.append("VM: END SELECT")
            return "\n".join(steps)

        if isinstance(ast, SQLInsert):
            steps.append("VM: BEGIN INSERT")
            steps.append(f"  - Target table        : {ast.table}")
            cols = ast.columns or []
            if cols:
                steps.append(f"  - Columns             : {', '.join(cols)}")
            else:
                steps.append("  - Columns             : (all table columns in defined order)")
            steps.append(f"  - Values              : {', '.join(ast.values)}")
            steps.append("  - Action              : append new row to table")
            steps.append("VM: END INSERT")
            return "\n".join(steps)

        if isinstance(ast, SQLDelete):
            steps.append("VM: BEGIN DELETE")
            steps.append(f"  - Target table        : {ast.table}")
            if ast.where:
                steps.append("  - Delete rows where   :")
                for line in self._render_condition_steps(ast.where):
                    steps.append(f"      {line}")
            else:
                steps.append("  - Delete ALL rows from table (no WHERE clause)")
            steps.append("VM: END DELETE")
            return "\n".join(steps)

        if isinstance(ast, SQLUpdate):
            steps.append("VM: BEGIN UPDATE")
            steps.append(f"  - Target table        : {ast.table}")
            if ast.where:
                steps.append("  - Update rows where   :")
                for line in self._render_condition_steps(ast.where):
                    steps.append(f"      {line}")
            else:
                steps.append("  - Update ALL rows (no WHERE clause)")

            if ast.assignments:
                steps.append("  - Set columns         :")
                for col, val in ast.assignments:
                    steps.append(f"      {col} = {val}")
            steps.append("VM: END UPDATE")
            return "\n".join(steps)

        return f"VM: Unsupported statement type: {type(ast).__name__}"

    def _render_condition_steps(self, cond: SQLCondition) -> List[str]:
        lines: List[str] = []
        current = cond
        while current:
            lines.append(f"{current.left} {current.operator} {current.right}")
            if current.connector and current.next_condition:
                lines.append(f"[{current.connector}]")
            current = current.next_condition
        return lines


# --------------------------------------------------------------------------- #
# DSL Builder + LALR Parser
# --------------------------------------------------------------------------- #


class DSLBuilder:
    """Produces a deterministic DSL representation from interpreted NL statements."""

    def build(self, spec: Union[DSLStatementSpec, IRStatement]) -> str:
        # Accept either DSL spec objects or IR tree nodes
        if isinstance(spec, IRSelect):
            # convert IRSelect back into a DSLSelectSpec-like rendering
            dsl_spec = DSLSelectSpec(
                columns=list(spec.projections),
                table=spec.table,
                conditions=[DSLConditionSpec(column=c.column, operator=c.operator, literal=c.literal, connector=c.connector) for c in spec.conditions],
                distinct=spec.distinct,
                order_clauses=list(spec.order_clauses),
                limit=spec.limit,
                offset=spec.offset,
            )
            return self._render_select(dsl_spec)
        if isinstance(spec, IRInsert):
            dsl_spec = DSLInsertSpec(table=spec.table, columns=list(spec.columns), values=list(spec.values))
            return self._render_insert(dsl_spec)
        if isinstance(spec, IRDelete):
            conds = [DSLConditionSpec(column=c.column, operator=c.operator, literal=c.literal, connector=c.connector) for c in spec.conditions]
            dsl_spec = DSLDeleteSpec(table=spec.table, conditions=conds)
            return self._render_delete(dsl_spec)
        if isinstance(spec, IRUpdate):
            dsl_spec = DSLUpdateSpec(table=spec.table, assignments=list(spec.assignments), conditions=[DSLConditionSpec(column=c.column, operator=c.operator, literal=c.literal, connector=c.connector) for c in spec.conditions])
            return self._render_update(dsl_spec)
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

    def __init__(self, symbol_table: Optional[SymbolTable] = None) -> None:
        vocabulary = self._build_vocabulary()
        self.recommender = QueryRecommender(vocabulary)
        self.mapper = SemanticMapper(SYSTEMATIC_MAPPING_TABLE, ATTRIBUTE_PATTERNS)
        self.dsl_builder = DSLBuilder()
        self.dsl_parser = DSLParser()
        self.optimizer = None
        self.generator = CodeGenerator()
        self.symbol_table = symbol_table or create_default_symbol_table()
        self.semantic_analyzer = SemanticAnalyzer(self.symbol_table)
        self.vm = VirtualMachine()

    def compile(self, text: str) -> str:
        sql, _, _ = self._run_pipeline(text)
        return sql

    def compile_with_artifacts(self, text: str) -> CompilerArtifacts:
        sql, dsl, recommendations = self._run_pipeline(text)
        return CompilerArtifacts(sql=sql, dsl=dsl, recommendations=recommendations)
    
    def compile_with_debug(self, text: str) -> CompilerDebugInfo:
        """Compile with full debug information for all phases."""
        enhanced_text, recommendations = self.recommender.enhance(text)
        tokens = NLLexer(enhanced_text).tokenize()
        nl_ast = LL1Parser(tokens).parse()

        # Semantic mapping -> DSL spec
        dsl_spec = self.mapper.map(nl_ast)

        # Convert DSL spec to IR and optimize
        ir = self._dsl_spec_to_ir(dsl_spec)
        self._optimize_ir(ir)

        # Render DSL script from optimized IR and parse to SQL AST
        dsl_script = self.dsl_builder.build(ir)
        sql_ast = self.dsl_parser.parse(dsl_script)

        # Perform semantic analysis
        annotated_ast = None
        semantic_errors: List[SemanticError] = []
        sql = ""
        vm_output: Optional[str] = None

        try:
            annotated_ast = self.semantic_analyzer.analyze(sql_ast)
            sql = self.generator.generate(sql_ast)
        except SemanticError as e:
            semantic_errors.append(e)
            try:
                sql = self.generator.generate(sql_ast)
            except Exception:
                sql = "-- SQL generation failed due to semantic errors"
        except Exception as e:
            semantic_errors.append(SemanticError(f"Semantic analysis failed: {str(e)}"))
            try:
                sql = self.generator.generate(sql_ast)
            except Exception:
                sql = "-- SQL generation failed"
        # VM execution phase (conceptual execution trace only)
        try:
            vm_output = self.vm.execute(sql_ast)
        except Exception as e:
            vm_output = f"VM execution simulation failed: {str(e)}"

        return CompilerDebugInfo(
            original_text=text,
            enhanced_text=enhanced_text,
            tokens=tokens,
            nl_ast=nl_ast,
            dsl_spec=dsl_spec,
            dsl_script=dsl_script,
            sql_ast=sql_ast,
            annotated_ast=annotated_ast,
            semantic_errors=semantic_errors,
            sql=sql,
            recommendations=recommendations,
            vm_output=vm_output,
        )

    def _run_pipeline(self, text: str) -> Tuple[str, str, List[str]]:
        enhanced_text, recommendations = self.recommender.enhance(text)
        tokens = NLLexer(enhanced_text).tokenize()
        statement = LL1Parser(tokens).parse()
        interpretation = self.mapper.map(statement)

        # Convert interpretation (DSL spec) to IR and optimize
        ir = self._dsl_spec_to_ir(interpretation)
        self._optimize_ir(ir)

        # Build DSL script from optimized IR and parse
        dsl_script = self.dsl_builder.build(ir)
        reconstructed_ast = self.dsl_parser.parse(dsl_script)

        # Perform semantic analysis (but don't fail on errors, just validate)
        try:
            self.semantic_analyzer.analyze(reconstructed_ast)
        except SemanticError:
            pass

        sql = self.generator.generate(reconstructed_ast)
        return sql, dsl_script, recommendations

    def _build_vocabulary(self) -> Set[str]:
        vocabulary: Set[str] = set(NLLexer.KEYWORDS)
        for mapping in SYSTEMATIC_MAPPING_TABLE.values():
            vocabulary.update(mapping.keys())
            vocabulary.update(mapping.values())
        vocabulary.update({"record", "records", "table", "tables"})
        return vocabulary

    # --- DSL -> IR conversion and optimization passes -----------------
    def _dsl_spec_to_ir(self, spec: DSLStatementSpec) -> IRStatement:
        if isinstance(spec, DSLSelectSpec):
            conditions = [IRCondition(column=c.column, operator=c.operator, literal=c.literal, connector=c.connector) for c in spec.conditions]
            projections = list(spec.columns)
            ir = IRSelect(
                projections=projections,
                table=spec.table,
                conditions=conditions,
                distinct=spec.distinct,
                order_clauses=list(spec.order_clauses),
                limit=spec.limit,
                offset=spec.offset,
            )
            return ir

        if isinstance(spec, DSLInsertSpec):
            return IRInsert(table=spec.table, columns=list(spec.columns), values=list(spec.values))

        if isinstance(spec, DSLDeleteSpec):
            conditions = [IRCondition(column=c.column, operator=c.operator, literal=c.literal, connector=c.connector) for c in spec.conditions]
            return IRDelete(table=spec.table, conditions=conditions)

        if isinstance(spec, DSLUpdateSpec):
            conditions = [IRCondition(column=c.column, operator=c.operator, literal=c.literal, connector=c.connector) for c in spec.conditions]
            return IRUpdate(table=spec.table, assignments=list(spec.assignments), conditions=conditions)

        raise TypeError("IR conversion not implemented for this DSL statement type")

    def _optimize_ir(self, ir: IRStatement) -> None:
        """Run a set of simple optimization passes on the IR in-place.

        Passes implemented:
        - predicate simplification: remove duplicate conditions
        - projection pruning (conservative): remove duplicate projections
        - ORDER+LIMIT pushdown hinting: add hint if both present
        """
        if isinstance(ir, IRSelect):
            # Predicate simplification: remove exact duplicate conditions
            seen_cond: Set[Tuple[str, str, str, Optional[str]]] = set()
            new_conds: List[IRCondition] = []
            for c in ir.conditions:
                key = (c.column.lower(), c.operator, c.literal, c.connector)
                if key in seen_cond:
                    continue
                seen_cond.add(key)
                new_conds.append(c)
            ir.conditions = new_conds

            # Projection pruning (conservative): remove duplicate projections preserving order
            seen_proj: Set[str] = set()
            new_projs: List[str] = []
            for p in ir.projections:
                if p == '*':
                    new_projs = ['*']
                    break
                low = p.lower()
                if low in seen_proj:
                    continue
                seen_proj.add(low)
                new_projs.append(p)
            ir.projections = new_projs

            # ORDER+LIMIT pushdown: annotate IR with a hint (no structural change here)
            if ir.limit is not None and ir.order_clauses:
                try:
                    ir.hints = getattr(ir, 'hints', {})
                    ir.hints['order_limit_pushdown'] = True
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# Streamlit GUI
# --------------------------------------------------------------------------- #


def format_tokens(tokens: List[Token]) -> str:
    """Format tokens for display."""
    lines = []
    for token in tokens:
        if token.type == TokenType.EOF:
            lines.append(f"EOF")
        else:
            lines.append(f"{token.type.name:15} : {token.value!r}")
    return "\n".join(lines)


def format_nl_ast(ast: NLStatement) -> str:
    """Format NL AST for display."""
    import json
    from dataclasses import asdict
    
    def ast_to_dict(obj):
        """Convert AST to dictionary, handling nested structures."""
        if isinstance(obj, (NLQuery, NLInsert, NLDelete, NLUpdate)):
            result = {
                "type": obj.__class__.__name__,
            }
            for key, value in asdict(obj).items():
                if value is None:
                    continue
                if isinstance(value, NLCondition):
                    result[key] = format_condition(value)
                elif isinstance(value, list) and value and isinstance(value[0], tuple):
                    result[key] = [list(t) for t in value]
                else:
                    result[key] = value
            return result
        return str(obj)
    
    def format_condition(cond: NLCondition) -> dict:
        """Format condition chain recursively."""
        result = {
            "words": cond.words,
            "connector": cond.connector
        }
        if cond.next_condition:
            result["next"] = format_condition(cond.next_condition)
        return result
    
    ast_dict = ast_to_dict(ast)
    return json.dumps(ast_dict, indent=2)


# format_ast_tree removed — textual AST tree rendering disabled in UI


def nl_ast_to_dot(ast: NLStatement) -> str:
    raise RuntimeError("NL DOT representation removed; use nl_ast_to_graphviz_source instead.")


def format_sql_ast(ast: SQLStatement) -> str:
    """Format SQL AST for display."""
    import json
    from dataclasses import asdict
    
    def ast_to_dict(obj):
        """Convert SQL AST to dictionary."""
        if isinstance(obj, (SQLSelect, SQLInsert, SQLDelete, SQLUpdate)):
            result = {
                "type": obj.__class__.__name__,
            }
            for key, value in asdict(obj).items():
                if value is None:
                    continue
                if isinstance(value, SQLCondition):
                    result[key] = format_sql_condition(value)
                elif isinstance(value, list) and value and isinstance(value[0], tuple):
                    result[key] = [list(t) for t in value]
                else:
                    result[key] = value
            return result
        return str(obj)
    
    def format_sql_condition(cond: SQLCondition) -> dict:
        """Format SQL condition chain recursively."""
        result = {
            "left": cond.left,
            "operator": cond.operator,
            "right": cond.right,
            "connector": cond.connector
        }
        if cond.next_condition:
            result["next"] = format_sql_condition(cond.next_condition)
        return result
    
    ast_dict = ast_to_dict(ast)
    return json.dumps(ast_dict, indent=2)


def sql_ast_to_dot(ast: SQLStatement) -> str:
    raise RuntimeError("SQL DOT representation removed; use sql_ast_to_graphviz_source instead.")


def nl_ast_to_graphviz_source(ast: NLStatement) -> str:
    """Build a Graphviz Digraph source programmatically from the NL AST (JSON-first).

    Falls back to a single-node DOT if JSON conversion fails.
    """
    try:
        from graphviz import Digraph
    except Exception:
        raise

    import json

    from dataclasses import is_dataclass, fields

    def _ast_to_plain(obj: Any) -> Any:
        if obj is None:
            return None
        if is_dataclass(obj):
            result: Dict[str, Any] = {"type": obj.__class__.__name__}
            for f in fields(obj):
                v = getattr(obj, f.name)
                if v is None:
                    continue
                result[f.name] = _ast_to_plain(v)
            return result
        if isinstance(obj, dict):
            return {k: _ast_to_plain(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, (list, tuple)):
            return [_ast_to_plain(v) for v in obj]
        if hasattr(obj, "__dict__"):
            result = {"type": obj.__class__.__name__}
            for k, v in vars(obj).items():
                if v is None:
                    continue
                result[k] = _ast_to_plain(v)
            return result
        return obj

    try:
        parsed = _ast_to_plain(ast)
    except Exception:
        # fallback to simple single-node graph
        d = Digraph("NL_AST", node_attr={"shape": "box"})
        d.node("n1", label=str(ast))
        return d.source

    d = Digraph("NL_AST", node_attr={"shape": "box", "style": "rounded"})
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"n{counter}"

    def add(obj) -> str:
        nid = next_id()
        if isinstance(obj, dict):
            label = obj.get('type', 'obj')
            d.node(nid, label=label)
            for k, v in obj.items():
                if k == 'type':
                    continue
                kid = next_id()
                d.node(kid, label=str(k))
                d.edge(nid, kid)
                child_id = add(v)
                d.edge(kid, child_id)
            return nid
        if isinstance(obj, list):
            d.node(nid, label='list')
            for item in obj:
                child = add(item)
                d.edge(nid, child)
            return nid
        # primitive
        d.node(nid, label=str(obj))
        return nid

    add(parsed)
    return d.source


def sql_ast_to_graphviz_source(ast: SQLStatement) -> str:
    """Build a Graphviz Digraph source programmatically from the SQL AST (JSON-first)."""
    try:
        from graphviz import Digraph
    except Exception:
        raise

    import json

    from dataclasses import is_dataclass, fields

    def _ast_to_plain(obj: Any) -> Any:
        if obj is None:
            return None
        if is_dataclass(obj):
            result: Dict[str, Any] = {"type": obj.__class__.__name__}
            for f in fields(obj):
                v = getattr(obj, f.name)
                if v is None:
                    continue
                result[f.name] = _ast_to_plain(v)
            return result
        if isinstance(obj, dict):
            return {k: _ast_to_plain(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, (list, tuple)):
            return [_ast_to_plain(v) for v in obj]
        if hasattr(obj, "__dict__"):
            result = {"type": obj.__class__.__name__}
            for k, v in vars(obj).items():
                if v is None:
                    continue
                result[k] = _ast_to_plain(v)
            return result
        return obj

    try:
        parsed = _ast_to_plain(ast)
    except Exception:
        d = Digraph("SQL_AST", node_attr={"shape": "box"})
        d.node("s1", label=str(ast))
        return d.source

    d = Digraph("SQL_AST", node_attr={"shape": "box", "style": "rounded"})
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"s{counter}"

    def add(obj) -> str:
        nid = next_id()
        if isinstance(obj, dict):
            label = obj.get('type', 'obj')
            d.node(nid, label=label)
            for k, v in obj.items():
                if k == 'type':
                    continue
                kid = next_id()
                d.node(kid, label=str(k))
                d.edge(nid, kid)
                child_id = add(v)
                d.edge(kid, child_id)
            return nid
        if isinstance(obj, list):
            d.node(nid, label='list')
            for item in obj:
                child = add(item)
                d.edge(nid, child)
            return nid
        d.node(nid, label=str(obj))
        return nid

    add(parsed)
    return d.source


def format_dsl_spec(spec: DSLStatementSpec) -> str:
    """Format DSL spec for display."""
    import json
    from dataclasses import asdict
    
    def spec_to_dict(obj):
        """Convert DSL spec to dictionary."""
        if isinstance(obj, (DSLSelectSpec, DSLInsertSpec, DSLDeleteSpec, DSLUpdateSpec)):
            result = {
                "type": obj.__class__.__name__,
            }
            for key, value in asdict(obj).items():
                if value is None or (isinstance(value, list) and not value):
                    continue
                if isinstance(value, list) and value and isinstance(value[0], DSLConditionSpec):
                    result[key] = [asdict(c) for c in value]
                elif isinstance(value, list) and value and isinstance(value[0], tuple):
                    result[key] = [list(t) for t in value]
                else:
                    result[key] = value
            return result
        return str(obj)
    
    spec_dict = spec_to_dict(spec)
    return json.dumps(spec_dict, indent=2)


def format_annotated_ast(ast: Optional[AnnotatedStatement]) -> str:
    """Format annotated AST for display."""
    import json
    from dataclasses import asdict
    
    if ast is None:
        return "No annotated AST available"
    
    def annotated_to_dict(obj):
        """Convert annotated AST to dictionary."""
        if isinstance(obj, (AnnotatedSelect, AnnotatedInsert, AnnotatedDelete, AnnotatedUpdate)):
            result = {
                "type": obj.__class__.__name__,
            }
            for key, value in asdict(obj).items():
                if value is None:
                    continue
                if isinstance(value, AnnotatedColumn):
                    result[key] = {
                        "name": value.name,
                        "source_table": value.source_table,
                        "data_type": value.data_type,
                        "nullable": value.column_info.nullable if value.column_info else None
                    }
                elif isinstance(value, list) and value:
                    if isinstance(value[0], AnnotatedColumn):
                        result[key] = [
                            {
                                "name": col.name,
                                "source_table": col.source_table,
                                "data_type": col.data_type
                            }
                            for col in value
                        ]
                    elif isinstance(value[0], tuple) and len(value[0]) == 2 and isinstance(value[0][0], AnnotatedColumn):
                        result[key] = [
                            {
                                "column": {
                                    "name": col.name,
                                    "source_table": col.source_table,
                                    "data_type": col.data_type
                                },
                                "is_desc": is_desc
                            }
                            for col, is_desc in value
                        ]
                    elif isinstance(value[0], tuple) and len(value[0]) == 2:
                        result[key] = [
                            {
                                "column": {
                                    "name": col.name,
                                    "source_table": col.source_table,
                                    "data_type": col.data_type
                                } if isinstance(col, AnnotatedColumn) else col,
                                "value": val
                            }
                            for col, val in value
                        ]
                    else:
                        result[key] = value
                elif isinstance(value, AnnotatedCondition):
                    result[key] = format_annotated_condition(value)
                elif isinstance(value, TableInfo):
                    result[key] = {
                        "name": value.name,
                        "columns": list(value.columns.keys())
                    }
                else:
                    result[key] = value
            return result
        return str(obj)
    
    def format_annotated_condition(cond: AnnotatedCondition) -> dict:
        """Format annotated condition recursively."""
        result = {
            "left": {
                "name": cond.left.name,
                "source_table": cond.left.source_table,
                "data_type": cond.left.data_type
            },
            "operator": cond.operator,
            "right": cond.right,
            "right_type": cond.right_type,
            "connector": cond.connector
        }
        if cond.next_condition:
            result["next"] = format_annotated_condition(cond.next_condition)
        return result
    
    ast_dict = annotated_to_dict(ast)
    return json.dumps(ast_dict, indent=2)


def annotated_ast_to_dot(ast: Optional[AnnotatedStatement]) -> str:
    """Convert Annotated AST to DOT format for Graphviz by traversing dataclass/object structure."""
    from dataclasses import is_dataclass, fields

    def _ast_to_plain(obj: Any) -> Any:
        if obj is None:
            return None
        if is_dataclass(obj):
            result: Dict[str, Any] = {"type": obj.__class__.__name__}
            for f in fields(obj):
                v = getattr(obj, f.name)
                if v is None:
                    continue
                result[f.name] = _ast_to_plain(v)
            return result
        if isinstance(obj, dict):
            return {k: _ast_to_plain(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, (list, tuple)):
            return [_ast_to_plain(v) for v in obj]
        if hasattr(obj, "__dict__"):
            result = {"type": obj.__class__.__name__}
            for k, v in vars(obj).items():
                if v is None:
                    continue
                result[k] = _ast_to_plain(v)
            return result
        return obj

    try:
        parsed = _ast_to_plain(ast)
    except Exception:
        lines = ["digraph Annotated_AST {", "  node [shape=box, fontname=Helvetica]"]
        safe = str(ast).replace('"', '\\"')
        lines.append(f'  a1 [label="{safe}"]')
        lines.append('}')
        return "\n".join(lines)

    def dict_to_dot(obj: Any, lines: List[str], parent: Optional[str], counter: List[int]) -> str:
        counter[0] += 1
        nid = f"a{counter[0]}"
        if isinstance(obj, dict):
            label = obj.get('type', 'obj')
            safe = str(label).replace('"', '\\"')
            lines.append(f'  {nid} [label="{safe}"]')
            if parent:
                lines.append(f"  {parent} -> {nid}")
            for k, v in obj.items():
                if k == 'type':
                    continue
                counter[0] += 1
                kid = f"a{counter[0]}"
                ksafe = str(k).replace('"', '\\"')
                lines.append(f'  {kid} [label="{ksafe}"]')
                lines.append(f"  {nid} -> {kid}")
                if isinstance(v, (dict, list)):
                    dict_to_dot(v, lines, kid, counter)
                else:
                    counter[0] += 1
                    leaf = f"a{counter[0]}"
                    vsafe = str(v).replace('"', '\\"')
                    lines.append(f'  {leaf} [label="{vsafe}"]')
                    lines.append(f"  {kid} -> {leaf}")
            return nid
        if isinstance(obj, list):
            label = 'list'
            lines.append(f'  {nid} [label="{label}"]')
            if parent:
                lines.append(f"  {parent} -> {nid}")
            for item in obj:
                dict_to_dot(item, lines, nid, counter)
            return nid
        safe = str(obj).replace('"', '\\"')
        lines.append(f'  {nid} [label="{safe}"]')
        if parent:
            lines.append(f"  {parent} -> {nid}")
        return nid

    lines: List[str] = ["digraph Annotated_AST {", "  node [shape=box, fontname=Helvetica]"]
    dict_to_dot(parsed, lines, None, [0])
    lines.append('}')
    return "\n".join(lines)


def annotated_ast_to_graphviz_source(ast: Optional[AnnotatedStatement]) -> str:
    """Build a Graphviz Digraph source programmatically from an annotated AST."""
    try:
        from graphviz import Digraph
    except Exception:
        raise

    from dataclasses import is_dataclass, fields

    def _ast_to_plain(obj: Any) -> Any:
        if obj is None:
            return None
        if is_dataclass(obj):
            result: Dict[str, Any] = {"type": obj.__class__.__name__}
            for f in fields(obj):
                v = getattr(obj, f.name)
                if v is None:
                    continue
                result[f.name] = _ast_to_plain(v)
            return result
        if isinstance(obj, dict):
            return {k: _ast_to_plain(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, (list, tuple)):
            return [_ast_to_plain(v) for v in obj]
        if hasattr(obj, "__dict__"):
            result = {"type": obj.__class__.__name__}
            for k, v in vars(obj).items():
                if v is None:
                    continue
                result[k] = _ast_to_plain(v)
            return result
        return obj

    try:
        parsed = _ast_to_plain(ast)
    except Exception:
        d = Digraph("Annotated_AST", node_attr={"shape": "box"})
        d.node("a1", label=str(ast))
        return d.source

    d = Digraph("Annotated_AST", node_attr={"shape": "box", "style": "rounded"})
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"a{counter}"

    def add(obj) -> str:
        nid = next_id()
        if isinstance(obj, dict):
            label = obj.get('type', 'obj')
            d.node(nid, label=label)
            for k, v in obj.items():
                if k == 'type':
                    continue
                kid = next_id()
                d.node(kid, label=str(k))
                d.edge(nid, kid)
                child_id = add(v)
                d.edge(kid, child_id)
            return nid
        if isinstance(obj, list):
            d.node(nid, label='list')
            for item in obj:
                child = add(item)
                d.edge(nid, child)
            return nid
        d.node(nid, label=str(obj))
        return nid

    add(parsed)
    return d.source


def get_grammar_info() -> str:
    """Get grammar information."""
    info = []
    info.append("LL(1) Parser Grammar (Natural Language):")
    info.append("=" * 50)
    info.append("\nKeywords:")
    info.append(", ".join(sorted(NLLexer.KEYWORDS)))
    info.append("\n\nDSL Grammar (LALR):")
    info.append("=" * 50)
    info.append("\nDSL Keywords:")
    info.append(", ".join(sorted(DSLTokenizer.KEYWORDS)))
    info.append("\n\nProduction Rules:")
    info.append("Statement → SelectStmt | InsertStmt | DeleteStmt | UpdateStmt")
    info.append("SelectStmt → SELECT [DISTINCT] TABLE IDENT COLUMNS SelectColumns [WhereOpt] [OrderOpt] [LimitOpt]")
    info.append("InsertStmt → INSERT TABLE IDENT COLUMNS ColumnSeq VALUES LiteralSeq")
    info.append("DeleteStmt → DELETE TABLE IDENT [WhereOpt]")
    info.append("UpdateStmt → UPDATE TABLE IDENT SET AssignmentSeq [WhereOpt]")
    return "\n".join(info)


def main() -> None:
    st.set_page_config(
        page_title="NLP to SQL: Translation Engine Using RuleBased and Parsing Techniques",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize compiler with schema management
    if 'compiler' not in st.session_state:
        st.session_state.compiler = NLToSQLCompiler()
    if 'schema_loaded' not in st.session_state:
        st.session_state.schema_loaded = False
    if 'schema_json' not in st.session_state:
        st.session_state.schema_json = None
    
    compiler = st.session_state.compiler
    
    # Title
    st.title("🔍 NLP to SQL: Translation Engine Using RuleBased and Parsing Techniques")
    st.markdown("---")
    
    # Sidebar with instructions and examples
    with st.sidebar:
        st.header("🗄️ Schema Management")
        
        # Schema upload section
        with st.expander("📤 Load Schema from JSON", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload Schema JSON File",
                type=['json'],
                help="Upload a JSON file defining your database schema"
            )
            
            if uploaded_file is not None:
                try:
                    content_bytes = uploaded_file.getvalue()
                    current_hash = hashlib.sha256(content_bytes).hexdigest()
                    if st.session_state.get('last_uploaded_hash') != current_hash:
                        json_content = content_bytes.decode('utf-8')
                        schema = load_schema_from_json(json_content)
                        st.session_state.compiler = NLToSQLCompiler(symbol_table=schema)
                        st.session_state.schema_loaded = True
                        st.session_state.schema_json = json_content
                        st.session_state.last_uploaded_hash = current_hash
                        st.success(f"✅ Schema loaded successfully! ({len(schema.get_all_tables())} tables)")
                except Exception as e:
                    st.error(f"❌ Error loading schema: {str(e)}")
            
            st.markdown("---")
            
                        # (Schema format example and download removed)
            
            # Reset to default (load the hardcoded schema inside the code)
            if st.button("🔄 Reset to Default Schema", width='stretch', key="reset_schema_btn"):
                default_symtab = create_default_symbol_table()
                st.session_state.compiler = NLToSQLCompiler(symbol_table=default_symtab)
                st.session_state.schema_loaded = True
                try:
                    st.session_state.schema_json = export_schema_to_json(default_symtab)
                except Exception:
                    st.session_state.schema_json = None
                # Prevent an existing uploaded file from immediately reloading
                try:
                    if uploaded_file is not None:
                        uploaded_bytes = uploaded_file.getvalue()
                        st.session_state.last_uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()
                except Exception:
                    st.session_state.last_uploaded_hash = st.session_state.get('last_uploaded_hash')

                st.success("✅ Reset to hardcoded default schema")
                st.rerun()
        
        # Re-read compiler from session state in case it was updated above
        compiler = st.session_state.compiler

        # Show current schema info
        if st.session_state.schema_loaded:
            st.info(f"📊 Custom schema loaded ({len(compiler.symbol_table.get_all_tables())} tables)")
        else:
            st.info("📊 Using default schema")
        
        st.markdown("---")
        
        # Live Database Schema View
        with st.expander("📋 Database Schema View", expanded=True):
            symtab = compiler.symbol_table
            all_tables = symtab.get_all_tables()
            
            if not all_tables:
                st.warning("No tables found in schema")
            else:
                # Summary statistics
                total_columns = sum(
                    len(symtab.get_table(table).columns) if symtab.get_table(table) else 0
                    for table in all_tables
                )
                
                # Summary removed: counters hidden as requested
                
                
                # Display each table in a card-like format
                for idx, table_name in enumerate(sorted(all_tables)):
                    table_info = symtab.get_table(table_name)
                    if not table_info:
                        continue
                    
                    # Table header with expander
                    with st.expander(
                        f"📑 **{table_info.name}** - {len(table_info.columns)} column(s)",
                        expanded=False
                    ):
                        # Table info badge
                        st.markdown(f"**Table Name:** `{table_info.name}`")
                        
                        # Display columns in a nice table
                        try:
                            import pandas as pd
                            # Create DataFrame for better display
                            columns_list = []
                            for col_name, col_info in sorted(table_info.columns.items()):
                                nullable_status = "✅ Yes" if col_info.nullable else "❌ No"
                                columns_list.append({
                                    "Column": col_info.name,
                                    "Type": col_info.data_type,
                                    "Nullable": nullable_status
                                })
                            
                            df = pd.DataFrame(columns_list)
                            st.dataframe(
                                df,
                                width='stretch',
                                hide_index=True
                            )
                        except ImportError:
                            # Fallback: Use markdown table
                            st.markdown("**Columns:**")
                            table_md = "| Column | Type | Nullable |\n"
                            table_md += "|--------|------|----------|\n"
                            for col_name, col_info in sorted(table_info.columns.items()):
                                nullable_text = "Yes" if col_info.nullable else "No"
                                table_md += f"| `{col_info.name}` | `{col_info.data_type}` | {nullable_text} |\n"
                            st.markdown(table_md)
                        
                        # Show primary key candidates (columns with NOT NULL that look like IDs)
                        pk_candidates = [
                            col_info.name for col_name, col_info in table_info.columns.items()
                            if not col_info.nullable and (
                                col_info.name.lower().endswith('_id') or
                                col_info.name.lower() == 'id' or
                                col_info.name.lower().endswith('id')
                            )
                        ]
                        if pk_candidates:
                            st.caption(f"🔑 Potential Primary Keys: {', '.join([f'`{pk}`' for pk in pk_candidates])}")
                        
                        # Show metadata if available
                        if table_info.metadata and table_info.metadata != {}:
                            with st.expander("📝 Table Metadata", expanded=False):
                                st.json(table_info.metadata)
                    
                    # Add separator between tables (except last one)
                    if idx < len(all_tables) - 1:
                        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("📖 Available SQL Syntax")
        st.markdown("""
        - **SELECT**: Get data from tables
        - **INSERT**: Add new records
        - **DELETE**: Remove records
        - **UPDATE**: Modify records
        - **ORDER BY**: Sort records
        """)
        
        st.header("💡 Example Queries")
        examples = [
            "Get the names and emails of customers who live in Jakarta.",
            "Insert a new record into customers with name Sarah and status Active.",
            "Delete the records from customers who live in Jakarta.",
            "Update the customers with status Active where city is Jakarta.",
            "Sort employees by hire date from oldest to newest"
        ]
        
        # Initialize query_text in session state if not present
        if 'query_text' not in st.session_state:
            st.session_state.query_text = ''
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.query_text = example
                st.rerun()
    
    # Initialize query_text if not present
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ''
    
    # Main input area
    st.header("Enter Your Query")
    
    # Buttons
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        compile_button = st.button("🚀 Compile", type="primary", width='stretch')
    with col2:
        clear_button = st.button("🗑️ Clear", width='stretch')
    
    # Handle clear button click - must be before widget creation
    if clear_button:
        st.session_state.query_text = ""
        if 'artifacts' in st.session_state:
            del st.session_state.artifacts
        if 'debug_info' in st.session_state:
            del st.session_state.debug_info
        st.rerun()
    
    # Create text_area WITHOUT key to allow manual state management
    query = st.text_area(
        "Natural Language Query:",
        value=st.session_state.query_text,
        height=100,
        placeholder="e.g., Get the names and emails of customers who live in Jakarta."
    )
    
    # Update session state with widget value
    st.session_state.query_text = query
    
    # Process query
    if compile_button:
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
            if 'artifacts' in st.session_state:
                del st.session_state.artifacts
            if 'debug_info' in st.session_state:
                del st.session_state.debug_info
        else:
            try:
                # Get both regular artifacts and debug info
                artifacts = compiler.compile_with_artifacts(query)
                debug_info = compiler.compile_with_debug(query)
                st.session_state.artifacts = artifacts
                st.session_state.debug_info = debug_info
                st.session_state.last_query = query
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
                if 'artifacts' in st.session_state:
                    del st.session_state.artifacts
                if 'debug_info' in st.session_state:
                    del st.session_state.debug_info
    
    # Display results if available
    if 'artifacts' in st.session_state and st.session_state.artifacts:
        artifacts = st.session_state.artifacts
        
        # Output sections
        st.markdown("---")
        
        # DSL output (collapsible)
        with st.expander("🔷 DSL (Intermediate Representation)", expanded=False):
            st.code(artifacts.dsl, language="text")
        
        # Semantic errors (if any) - show prominently
        if 'debug_info' in st.session_state and st.session_state.debug_info:
            debug_info = st.session_state.debug_info
            if debug_info.semantic_errors:
                st.error(f"⚠️ **Semantic Errors Detected** ({len(debug_info.semantic_errors)})")
                with st.expander("View Semantic Errors", expanded=True):
                    for i, error in enumerate(debug_info.semantic_errors, 1):
                        error_type = error.__class__.__name__
                        st.error(f"**{i}. {error_type}**")
                        st.code(str(error), language="text")
                st.markdown("---")
        
        # SQL output (expanded by default)
        st.header("💾 Generated SQL")
        st.code(artifacts.sql, language="sql")
        
        # Recommendations (always shown)
        st.markdown("**💡 Recommendations**")
        if artifacts.recommendations:
            for rec in artifacts.recommendations:
                st.info(rec)
        else:
            st.success("No recommendations. Query processed successfully!")
        
        # Debug/Compiler Phases Section
        st.markdown("---")
        with st.expander("🔬 Compiler Phases & Debug Information", expanded=False):
            if 'debug_info' in st.session_state and st.session_state.debug_info:
                debug_info = st.session_state.debug_info
                
                # Create tabs for different phases
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                    "📝 Input/Output", "🔤 Tokens", "🌳 NL AST", 
                    "🔷 DSL Spec", "🌲 SQL AST", "🔍 Semantic Analysis", "📚 Grammar", "📊 Parser Info",
                    "🧮 VM Execution"
                ])
                
                with tab1:
                    st.subheader("Original Input")
                    st.code(debug_info.original_text, language="text")
                    
                    if debug_info.enhanced_text != debug_info.original_text:
                        st.subheader("Enhanced Text (after recommendations)")
                        st.code(debug_info.enhanced_text, language="text")
                    
                    st.subheader("Final SQL Output")
                    st.code(debug_info.sql, language="sql")
                    
                    st.subheader("Console Output")
                    console_output = f"""Compiler Execution Log:
{'='*60}
Phase 1: Lexical Analysis
  - Input: {debug_info.original_text!r}
  - Enhanced: {debug_info.enhanced_text!r}
  - Tokens generated: {len(debug_info.tokens)}

Phase 2: Parsing (LL(1))
  - AST Type: {debug_info.nl_ast.__class__.__name__}

Phase 3: Semantic Mapping
  - DSL Spec Type: {debug_info.dsl_spec.__class__.__name__}

Phase 4: DSL Generation
  - DSL Script: {debug_info.dsl_script}

Phase 5: DSL Parsing (LALR)
  - SQL AST Type: {debug_info.sql_ast.__class__.__name__}

Phase 6: Semantic Analysis
    - Semantic Errors: {len(debug_info.semantic_errors) if debug_info.semantic_errors else 0}

Phase 7: Code Generation
  - SQL Generated: {debug_info.sql}

Phase 8: Virtual Machine Execution
  - VM Trace (summary): {'available' if debug_info.vm_output else 'not available'}

Recommendations: {len(debug_info.recommendations)} suggestion(s)
"""
                    st.code(console_output, language="text")
                
                with tab2:
                    st.subheader("Tokens (Lexical Analysis)")
                    st.markdown("**Token Stream:**")
                    tokens_formatted = format_tokens(debug_info.tokens)
                    st.code(tokens_formatted, language="text")
                    
                    st.markdown("**Token Statistics:**")
                    token_counts = {}
                    for token in debug_info.tokens:
                        if token.type != TokenType.EOF:
                            token_counts[token.type.name] = token_counts.get(token.type.name, 0) + 1
                    
                    for token_type, count in sorted(token_counts.items()):
                        st.text(f"  {token_type}: {count}")
                
                with tab3:
                    st.subheader("Natural Language AST (LL(1) Parser Output)")
                    st.markdown("**Abstract Syntax Tree (JSON):**")
                    nl_ast_formatted = format_nl_ast(debug_info.nl_ast)
                    st.code(nl_ast_formatted, language="json")

                    try:
                        dot_src = nl_ast_to_graphviz_source(debug_info.nl_ast)
                        st.graphviz_chart(dot_src)
                    except Exception:
                        pass
                    
                    st.markdown("**AST Type:**")
                    st.info(f"{debug_info.nl_ast.__class__.__name__}")
                
                with tab4:
                    st.subheader("DSL Specification (Semantic Mapping Output)")
                    st.markdown("**DSL Spec:**")
                    dsl_spec_formatted = format_dsl_spec(debug_info.dsl_spec)
                    st.code(dsl_spec_formatted, language="json")
                    
                    st.markdown("**DSL Script:**")
                    st.code(debug_info.dsl_script, language="text")
                
                with tab5:
                    st.subheader("SQL AST (LALR Parser Output)")
                    st.markdown("**SQL Abstract Syntax Tree (JSON):**")
                    sql_ast_formatted = format_sql_ast(debug_info.sql_ast)
                    st.code(sql_ast_formatted, language="json")

                    try:
                        dot_src = sql_ast_to_graphviz_source(debug_info.sql_ast)
                        st.graphviz_chart(dot_src)
                    except Exception:
                        pass
                    
                    st.markdown("**AST Type:**")
                    st.info(f"{debug_info.sql_ast.__class__.__name__}")
                
                with tab6:
                    st.subheader("Semantic Analysis")
                    
                    # Display semantic errors if any
                    if debug_info.semantic_errors:
                        st.error(f"⚠️ Found {len(debug_info.semantic_errors)} semantic error(s):")
                        for i, error in enumerate(debug_info.semantic_errors, 1):
                            error_type = error.__class__.__name__
                            st.error(f"**{i}. {error_type}**")
                            st.code(str(error), language="text")
                    else:
                        st.success("✅ No semantic errors found!")
                    
                    st.markdown("---")
                    
                    # Show symbol table information
                    st.subheader("Symbol Table Information")
                    compiler = st.session_state.compiler
                    symtab = compiler.symbol_table
                    tables = symtab.get_all_tables()
                    st.info(f"**Registered Tables**: {len(tables)}")
                    for table_name in sorted(tables):
                        table_info = symtab.get_table(table_name)
                        if table_info:
                            st.text(f"  • {table_info.name}: {len(table_info.columns)} columns")
                
                with tab7:
                    st.subheader("Grammar Information")
                    grammar_info = get_grammar_info()
                    st.code(grammar_info, language="text")
                    
                    st.markdown("**Parser Types:**")
                    st.markdown("""
                    - **LL(1) Parser**: Used for parsing Natural Language queries
                      - Left-to-right, Leftmost derivation
                      - 1 token lookahead
                      - Recursive descent parsing
                    
                    - **LALR(1) Parser**: Used for parsing DSL (Domain-Specific Language)
                      - Look-Ahead LR parser
                      - More powerful than SLR
                      - Handles left recursion and ambiguity
                    """)
                
                with tab8:
                    st.subheader("Parser Information")
                    
                    st.markdown("**LL(1) Parser Details:**")
                    st.markdown(f"""
                    - **Type**: LL(1) - Left-to-right, Leftmost derivation with 1 token lookahead
                    - **Input**: Natural Language tokens
                    - **Output**: NL AST (Natural Language Abstract Syntax Tree)
                    - **Grammar**: Context-free grammar for NL queries
                    - **Method**: Recursive descent parsing
                    """)
                    
                    st.markdown("**LALR(1) Parser Details:**")
                    st.markdown(f"""
                    - **Type**: LALR(1) - Look-Ahead LR parser
                    - **Input**: DSL tokens
                    - **Output**: SQL AST (SQL Abstract Syntax Tree)
                    - **Grammar**: LALR grammar for DSL
                    - **Method**: Table-driven parsing with action/goto tables
                    """)
                    
                    st.markdown("**Parser Phases:**")
                    st.markdown("""
                    1. **Lexical Analysis**: Converts input text to tokens
                    2. **Syntax Analysis (LL(1))**: Builds NL AST from tokens
                    3. **Semantic Analysis**: Maps NL AST to DSL specification
                    4. **DSL Generation**: Converts DSL spec to DSL script
                    5. **DSL Parsing (LALR)**: Parses DSL script to SQL AST
                    6. **Code Generation**: Converts SQL AST to SQL string
                    7. **Virtual Machine Execution**: Simulates how the SQL statement would be executed
                    """)
                
                with tab9:
                    st.subheader("Virtual Machine Execution (Conceptual)")
                    st.markdown(
                        """
                        This virtual machine does not operate on real data; it represents the logical execution stages of the generated SQL.
                        The VM trace corresponds directly to the generated SQL statement and mirrors the execution stages performed internally by the database engine.
                        This execution trace represents the backend execution semantics of the compiler after SQL code generation (IR / Execution semantics / Backend).
                        """
                    )
                    if getattr(debug_info, "vm_output", None):
                        st.markdown("**VM Execution Trace:**")
                        st.code(debug_info.vm_output, language="text")
                        try:
                            import re
                            sql_text = getattr(debug_info, 'sql', '') or ''
                            table_name = None
                            m = re.search(r'from\s+([A-Za-z0-9_]+)', sql_text, re.IGNORECASE)
                            if m:
                                table_name = m.group(1)

                            cols = '*'
                            m2 = re.search(r'select\s+(.*?)\s+from', sql_text, re.IGNORECASE | re.DOTALL)
                            if m2:
                                cols = m2.group(1).strip()

                            order_clause = ''
                            m3 = re.search(r'order\s+by\s+(.+?)(?:\s+limit\b|$)', sql_text, re.IGNORECASE | re.DOTALL)
                            if m3:
                                order_clause = m3.group(1).strip()

                            result_lines = []
                            result_lines.append('VM: RESULT')
                            if table_name:
                                result_lines.append(f'  - Relation over table: {table_name}')
                            result_lines.append(f'  - Columns returned   : {cols}')
                            if order_clause:
                                result_lines.append(f'  - Ordering applied   : {order_clause}')
                            st.code('\n'.join(result_lines), language='text')
                        except Exception:
                            pass
                    else:
                        st.info("VM execution trace is not available for this query.")
            else:
                st.info("Debug information not available. Please compile a query first.")


if __name__ == "__main__":
    main()

