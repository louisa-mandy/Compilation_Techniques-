# Natural Language to SQL Compiler 🗣️➡️🗄️

Welcome to **CompTech**, a fascinating mini-compiler that transforms everyday English sentences into executable SQL queries! Ever dreamed of chatting with your database? This project makes it possible by bridging natural language and SQL through a multi-stage compilation pipeline.

## 🌟 What Makes It Cool?

CompTech isn't just another query builder—it's a **full-fledged compiler** inspired by real programming language design. It tokenizes, parses, maps semantics, builds an intermediate DSL, validates it, and generates SQL. Plus, it comes with a sleek GUI for easy interaction!

### Key Features
- **Natural Language Support**: Handles queries like "Get the names of customers in Jakarta" or "Update employees with status Active."
- **Multi-Operation SQL**: Supports SELECT, INSERT, DELETE, UPDATE with WHERE, ORDER BY, DISTINCT, and LIMIT.
- **Intelligent Mapping**: Uses patterns and tables (e.g., Customers, Employees) to interpret business terms.
- **Error Handling & Recommendations**: Suggests fixes for unrecognized words.
- **User-Friendly GUI**: Scrollable interface with input, output, and examples— no command-line hassle!
- **Extensible Design**: Built with modularity in mind; easy to add new tables, columns, or patterns.

## 🏗️ Architecture Overview

CompTech follows a classic compiler pipeline:

1. **Lexer (NLLexer)**: Breaks English text into tokens (keywords, identifiers, etc.).
2. **Parser (LL1Parser)**: Builds an Abstract Syntax Tree (AST) for NL statements.
3. **Semantic Mapper**: Translates AST into SQL-ready specs using mappings and patterns.
4. **DSL Builder**: Generates a compact intermediate DSL string.
5. **LALR Parser**: Validates and re-parses the DSL into an executable SQL AST.
6. **Code Generator**: Outputs final SQL strings.

This design ensures robustness and makes debugging a breeze!

## 🚀 How to Run

### Prerequisites
- **Python 3.8+** (tested on 3.13).
- **Tkinter** (usually included with Python; if not, install via `pip install tk` or your package manager).
- No other dependencies—pure Python magic!

### Installation & Setup
1. Clone or download the repository.
2. Navigate to the project folder: `cd Final_project`.
3. Run the script: `python CompTech.py`.

That's it! The GUI window will pop up. If it doesn't (e.g., in headless environments), check your display setup.

### Usage
- **Enter Queries**: Type natural language in the input box (e.g., "Show me all customer names.").
- **Compile**: Click "Compile" to see DSL, SQL, and recommendations.
- **Examples**: Click "Examples" for sample queries.
- **Clear**: Reset the interface.
- **Scroll**: Use the scrollbar for long outputs.

## 📝 Example Queries

Try these to see CompTech in action:

- **SELECT**: "Get the names and emails of customers who live in Jakarta."
- **INSERT**: "Insert a new record into customers with name Sarah and status Active."
- **DELETE**: "Delete the records from customers who live in Jakarta."
- **UPDATE**: "Update the customers with status Active where city is Jakarta."
- **Advanced**: "Show me the top 5 most expensive products ordered by price descending."

Output includes:
- **DSL**: Intermediate representation.
- **SQL**: Executable query (e.g., `SELECT name, email FROM customers WHERE city = 'Jakarta';`).
- **Recommendations**: Suggestions for typos or unmapped terms.

## 🧪 Limitations


- **Limitations**: No support for complex joins, subqueries, or custom tables yet. Errors occur for unsupported syntax—check recommendations!

