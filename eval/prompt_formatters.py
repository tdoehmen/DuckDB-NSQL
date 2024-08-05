"""Rajkumar prompt formatter."""

from random import shuffle
from manifest import Manifest
from schema import Table
import re


class RajkumarFormatter:
    """RajkumarFormatter class.

    From https://arxiv.org/pdf/2204.00498.pdf.
    """

    table_sep: str = "\n\n"
    shuffle_table_order: bool = True
    _cache: dict[tuple[str, str, str], list[str]] = {}
    clean_whitespace = False

    @classmethod
    def format_table(cls, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table.name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table.name}"
        return create_tbl

    @classmethod
    def format_all_tables(cls, tables: list[Table], instruction: str) -> list[str]:
        """Get all tables format."""
        table_texts = [cls.format_table(table) for table in tables]
        key = ("tables", instruction, str(tables))
        if key not in cls._cache:
            shuffle(table_texts)
            cls._cache[key] = table_texts
        else:
            table_texts = cls._cache[key]
        return table_texts

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n\n/*\nHere is additional documentation about DuckDB that could be useful.\n--------\n{context_str}\n--------\n*/"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        return f"""{table_text}\n\n\n-- Using valid DuckDB SQL, answer the following question for the tables provided above.{context_text}\n\n-- {instruction}\n"""  # noqa: E501

    @classmethod
    def format_model_output(cls, output_sql: str, prompt: str) -> str:
        """Format model output."""
        clean_sql = (output_sql
            .replace('```sql\n', '')
            .replace('```duckdb\n', '')
            .replace('```\n', '')
            .replace('```', '')).strip()

        if clean_sql.find(';') != -1:
            clean_sql[:clean_sql.find(';')].strip()
        
        if not clean_sql.endswith(";"):
            clean_sql += ";"
        
        return clean_sql

    @classmethod
    def format_gold_output(cls, output_sql: str) -> str:
        """Format gold output for demonstration."""
        return output_sql


class DuckDBFormatter(RajkumarFormatter):
    """DuckDB class."""

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        return f"""{table_text}\n\n\n-- Using valid DuckDB SQL, answer the following question for the tables provided above.{context_text}\n\n-- {instruction}\n```sql\n"""  # noqa: E501


class DuckDBInstFormatter(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}{context}\n### Question:\n{question}\n\n### Response (use duckdb shorthand if possible):\n"""
    INSTRUCTION_TEMPLATE = """Your task is to generate valid duckdb SQL to answer the following question{has_schema}"""  # noqa: E501

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        input = ""
        if table_text:
            input = """Here is the database schema that the SQL query will run on:\n{schema}\n""".format(  # noqa: E501
                schema=table_text
            )
        instruction = cls.PROMPT_TEMPLATE.format(
            instruction=cls.INSTRUCTION_TEMPLATE.format(
                has_schema="."
                if table_text == ""
                else ", given a duckdb database schema."
            ),
            context=context_text,
            input=input,
            question=instruction,
        )
        return instruction

class DuckDBInstFormatterLlamaShort(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.

Here are some DuckDB SQL syntax specifics you should be aware of:

- DuckDB uses double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity and single quotes (') to define string literals
- DuckDB can query CSV, Parquet, and JSON directly without loading them first, e.g. `SELECT * FROM 'data.csv';`
- DuckDB supports CREATE TABLE AS (CTAS): `CREATE TABLE new_table AS SELECT * FROM old_table;`
- DuckDB queries can start with FROM, and optionally omit SELECT *, e.g. `FROM my_table WHERE condition;` is equivalent to `SELECT * FROM my_table WHERE condition;`
- DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly, e.g. `SELECT 1 + 1 AS result;`
- DuckDB supports attaching multiple databases, unsing the ATTACH statement: `ATTACH 'my_database.duckdb' AS mydb;`. Tables within attached databases can be accessed using the dot notation (.), e.g. `SELECT * FROM mydb.table_name syntax`. The default databases doesn't require the do notation to access tables. The default database can be changed with the USE statement, e.g. `USE my_db;`.
- DuckDB is generally more lenient with implicit type conversions (e.g. `SELECT '42' + 1;` - Implicit cast, result is 43), but you can always be explicit using `::`, e.g. `SELECT '42'::INTEGER + 1;`
- DuckDB can extract parts of strings and lists using [start:end] or [start:end:step] syntax. Indexes start at 1. String slicing: `SELECT 'DuckDB'[1:4];`. Array/List slicing: `SELECT [1, 2, 3, 4][1:3];`
- DuckDB has a powerful way to select or transform multiple columns using patterns or functions. You can select columns matching a pattern: `SELECT COLUMNS('sales_.*') FROM sales_data;` or transform multiple columns with a function: `SELECT AVG(COLUMNS('sales_.*')) FROM sales_data;`
- DuckDB has an easy way to include/exclude or modify columns when selecting all: e.g. Exclude: `SELECT * EXCLUDE (sensitive_data) FROM users;` Replace: `SELECT * REPLACE (UPPER(name) AS name) FROM users;` 
- DuckDB has a shorthand for grouping/ordering by all non-aggregated/all columns. e.g `SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;` and `SELECT * FROM my_table ORDER BY ALL;`
- DuckDB can combine tables by matching column names, not just their positions using UNION BY NAME. E.g. `SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;`
- DuckDB has an inutitive syntax to create List/Struct/Map and Array types. Create complex types using intuitive syntax. List: `SELECT [1, 2, 3] AS my_list;`, Struct: `{{'a': 1, 'b': 'text'}} AS my_struct;`, Map: `MAP([1,2],['one','two']) as my_map;`. All types can also be nested into each other. Array types are fixed size, while list types have variable size. Compared to Structs, MAPs do not need to have the same keys present for each row, but keys can only be of type Integer or Varchar. Example: `CREATE TABLE example (my_list INTEGER[], my_struct STRUCT(a INTEGER, b TEXT), my_map MAP(INTEGER, VARCHAR),  my_array INTEGER[3], my_nested_struct STRUCT(a INTEGER, b Integer[3]));`
- DuckDB has an inutive syntax to access struct fields using dot notation (.) or brackets ([]) with the field name. Maps fields can be accessed by brackets ([]).
- DuckDB's way of converting between text and timestamps, and extract date parts. Current date as 'YYYY-MM-DD': `SELECT strftime(NOW(), '%Y-%m-%d');` String to timestamp: `SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP;`, Extract Year from date: `SELECT EXTRACT(YEAR FROM DATE '2023-07-23');`
- Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses. E.g.: `SELECT a + b AS total FROM my_table WHERE total > 10 GROUP BY total HAVING total < 20;` 
- DuckDB allows generating lists using expressions similar to Python list comprehensions. E.g. `SELECT [x*2 FOR x IN [1, 2, 3]];` Returns [2, 4, 6].
- DuckDB allows chaining multiple function calls together using the dot (.) operator. E.g.: `SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB';`
- DuckDB has a JSON data type. It supports selecting fields from the JSON with a JSON-Path expression using the arrow operator, -> (returns JSON) or ->> (returns text) with JSONPath expressions. For example: `SELECT data->'$.user.id' AS user_id, data->>'$.event_type' AS event_type FROM events;`
- DuckDB has built-in functions for regex regexp_matches(column, regex), regexp_replace(column, regex), and regexp_extract(column, regex).
- DuckDB has a way to quickly get a subset of your data with `SELECT * FROM large_table USING SAMPLE 10%;`<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Database Schema:
Here is the schema of the DuckDB database that the SQL query will run on:
```sql
{schema}
```

Question:
Here is the question or an instruction the user provided:
{question}

Task:
Write a DuckDB SQL query for the given question!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```sql
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterLlamaBasic(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Database Schema:
Here is the schema of the DuckDB database that the SQL query will run on:
```sql
{schema}
```

Question:
Here is the question or an instruction the user provided:
{question}

Task:
Write a DuckDB SQL query for the given question!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```sql
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterLlamaSyntax(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.

Here are some DuckDB SQL syntax specifics you should be aware of:

- DuckDB uses double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity and single quotes (') to define string literals
- DuckDB can query CSV, Parquet, and JSON directly without loading them first, e.g. `SELECT * FROM 'data.csv';`
- DuckDB supports CREATE TABLE AS (CTAS): `CREATE TABLE new_table AS SELECT * FROM old_table;`
- DuckDB queries can start with FROM, and optionally omit SELECT *, e.g. `FROM my_table WHERE condition;` is equivalent to `SELECT * FROM my_table WHERE condition;`
- DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly, e.g. `SELECT 1 + 1 AS result;`
- DuckDB supports attaching multiple databases, unsing the ATTACH statement: `ATTACH 'my_database.duckdb' AS mydb;`. Tables within attached databases can be accessed using the dot notation (.), e.g. `SELECT * FROM mydb.table_name syntax`. The default databases doesn't require the do notation to access tables. The default database can be changed with the USE statement, e.g. `USE my_db;`.
- DuckDB is generally more lenient with implicit type conversions (e.g. `SELECT '42' + 1;` - Implicit cast, result is 43), but you can always be explicit using `::`, e.g. `SELECT '42'::INTEGER + 1;`
- DuckDB can extract parts of strings and lists using [start:end] or [start:end:step] syntax. Indexes start at 1. String slicing: `SELECT 'DuckDB'[1:4];`. Array/List slicing: `SELECT [1, 2, 3, 4][1:3];`
- DuckDB has a powerful way to select or transform multiple columns using patterns or functions. You can select columns matching a pattern: `SELECT COLUMNS('sales_.*') FROM sales_data;` or transform multiple columns with a function: `SELECT AVG(COLUMNS('sales_.*')) FROM sales_data;`
- DuckDB has an easy way to include/exclude or modify columns when selecting all: e.g. Exclude: `SELECT * EXCLUDE (sensitive_data) FROM users;` Replace: `SELECT * REPLACE (UPPER(name) AS name) FROM users;` 
- DuckDB has a shorthand for grouping/ordering by all non-aggregated/all columns. e.g `SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;` and `SELECT * FROM my_table ORDER BY ALL;`
- DuckDB can combine tables by matching column names, not just their positions using UNION BY NAME. E.g. `SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;`
- DuckDB has an inutitive syntax to create List/Struct/Map and Array types. Create complex types using intuitive syntax. List: `SELECT [1, 2, 3] AS my_list;`, Struct: `{{'a': 1, 'b': 'text'}} AS my_struct;`, Map: `MAP([1,2],['one','two']) as my_map;`. All types can also be nested into each other. Array types are fixed size, while list types have variable size. Compared to Structs, MAPs do not need to have the same keys present for each row, but keys can only be of type Integer or Varchar. Example: `CREATE TABLE example (my_list INTEGER[], my_struct STRUCT(a INTEGER, b TEXT), my_map MAP(INTEGER, VARCHAR),  my_array INTEGER[3], my_nested_struct STRUCT(a INTEGER, b Integer[3]));`
- DuckDB has an inutive syntax to access struct fields using dot notation (.) or brackets ([]) with the field name. Maps fields can be accessed by brackets ([]).
- DuckDB's way of converting between text and timestamps, and extract date parts. Current date as 'YYYY-MM-DD': `SELECT strftime(NOW(), '%Y-%m-%d');` String to timestamp: `SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP;`, Extract Year from date: `SELECT EXTRACT(YEAR FROM DATE '2023-07-23');`
- Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses. E.g.: `SELECT a + b AS total FROM my_table WHERE total > 10 GROUP BY total HAVING total < 20;` 
- DuckDB allows generating lists using expressions similar to Python list comprehensions. E.g. `SELECT [x*2 FOR x IN [1, 2, 3]];` Returns [2, 4, 6].
- DuckDB allows chaining multiple function calls together using the dot (.) operator. E.g.: `SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB';`
- DuckDB has a JSON data type. It supports selecting fields from the JSON with a JSON-Path expression using the arrow operator, -> (returns JSON) or ->> (returns text) with JSONPath expressions. For example: `SELECT data->'$.user.id' AS user_id, data->>'$.event_type' AS event_type FROM events;`
- DuckDB has built-in functions for regex regexp_matches(column, regex), regexp_replace(column, regex), and regexp_extract(column, regex).
- DuckDB has a way to quickly get a subset of your data with `SELECT * FROM large_table USING SAMPLE 10%;`<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Database Schema:
Here is the schema of the DuckDB database that the SQL query will run on:
```sql
{schema}
```

Question:
Here is the question or an instruction the user provided:
{question}

Task:
Write a DuckDB SQL query for the given question!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```sql
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterGraniteShort(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """System:
Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.

Here are some DuckDB SQL syntax specifics you should be aware of:


- DuckDB use double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity and single quotes (') to define string literals
- DuckDB can query CSV, Parquet, and JSON directly without loading them first, e.g. `SELECT * FROM 'data.csv';`
- DuckDB supports CREATE TABLE AS (CTAS): `CREATE TABLE new_table AS SELECT * FROM old_table;`
- DuckDB queries can start with FROM, and optionally omit SELECT *, e.g. `FROM my_table WHERE condition;` is equivalent to `SELECT * FROM my_table WHERE condition;`
- DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly, e.g. `SELECT 1 + 1 AS result;`
- DuckDB supports attaching multiple databases, unsing the ATTACH statement: `ATTACH 'my_database.duckdb' AS mydb;`. Tables within attached databases can be accessed using the dot notation (.), e.g. `SELECT * FROM mydb.table_name syntax`. The default databases doesn't require the do notation to access tables. The default database can be changed with the USE statement, e.g. `USE my_db;`.
- DuckDB is generally more lenient with implicit type conversions (e.g. `SELECT '42' + 1;` - Implicit cast, result is 43), but you can always be explicit using `::`, e.g. `SELECT '42'::INTEGER + 1;`
- DuckDB can extract parts of strings and lists using [start:end] or [start:end:step] syntax. Indexes start at 1. String slicing: `SELECT 'DuckDB'[1:4];`. Array/List slicing: `SELECT [1, 2, 3, 4][1:3];`
- DuckDB has a powerful way to select or transform multiple columns using patterns or functions. You can select columns matching a pattern: `SELECT COLUMNS('sales_.*') FROM sales_data;` or transform multiple columns with a function: `SELECT AVG(COLUMNS('sales_.*')) FROM sales_data;`
- DuckDB an easy way to include/exclude or modify columns when selecting all: e.g. Exclude: `SELECT * EXCLUDE (sensitive_data) FROM users;` Replace: `SELECT * REPLACE (UPPER(name) AS name) FROM users;` 
- DuckDB has a shorthand for grouping/ordering by all non-aggregated/all columns. e.g `SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;` and `SELECT * FROM my_table ORDER BY ALL;`
- DuckDB can combine tables by matching column names, not just their positions using UNION BY NAME. E.g. `SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;`
- DuckDB has an inutitive syntax to create List/Struct/Map and Array types. Create complex types using intuitive syntax. List: `SELECT [1, 2, 3] AS my_list;`, Struct: `{{'a': 1, 'b': 'text'}} AS my_struct;`, Map: `MAP([1,2],['one','two']) as my_map;`. All types can also be nested into each other. Array types are fixed size, while list types have variable size. Compared to Structs, MAPs do not need to have the same keys present for each row, but keys can only be of type Integer or Varchar. Example: `CREATE TABLE example (my_list INTEGER[], my_struct STRUCT(a INTEGER, b TEXT), my_map MAP(INTEGER, VARCHAR),  my_array INTEGER[3], my_nested_struct STRUCT(a INTEGER, b Integer[3]));`
- DuckDB has an inutive syntax to access struct fields using dot notation (.) or brackets ([]) with the field name. Maps fields can be accessed by brackets ([]).
- DuckDB's way of converting between text and timestamps, and extract date parts. Current date as 'YYYY-MM-DD': `SELECT strftime(NOW(), '%Y-%m-%d');` String to timestamp: `SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP;`, Extract Year from date: `SELECT EXTRACT(YEAR FROM DATE '2023-07-23');`
- Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses. E.g.: `SELECT a + b AS total FROM my_table WHERE total > 10 GROUP BY total HAVING total < 20;` 
- DuckDB allows generating lists using expressions similar to Python list comprehensions. E.g. `SELECT [x*2 FOR x IN [1, 2, 3]];` Returns [2, 4, 6].
- DuckDB allows chaining multiple function calls together using the dot (.) operator. E.g.: `SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB';`
- DuckDB has a JSON data type. It supports selecting fields from the JSON with a JSON-Path expression using the arrow operator, -> (returns JSON) or ->> (returns text) with JSONPath expressions. For example: `SELECT data->'$.user.id' AS user_id, data->>'$.event_type' AS event_type FROM events;`
- DuckDB has built-in functions for regex regexp_matches(column, regex), regexp_replace(column, regex), and regexp_extract(column, regex).
- DuckDB has a way to quickly get a subset of your data with `SELECT * FROM large_table USING SAMPLE 10%;`

Here is the schema of the DuckDB database that the SQL query will run on:
{schema}

Question:
Here is the question or an instruction the user provided:
{question}

Write a DuckDB SQL query for the given question!

Answer:
```
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterLlama(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """<|begin_of_text|>

Your task is to generate valid DuckDB SQL to answer the following question, given a DuckDB database schema.

## DuckDB SQL syntax specifics you should be aware of:

### Case Insensitivity and Quoting:

Identifiers (tables, columns): Case-insensitive, but DuckDB remembers the case you use. Use double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity.
``` 
CREATE TABLE "My Table" ("column_name" VARCHAR); -- Spaces and mixed case
SELECT "column_name" FROM "My Table"; 
``` 

### String Literals: Always use single quotes (') to define string literals.
``` 
SELECT 'This is a string' AS text;
``` 

### Direct File Querying: Query CSV, Parquet, and JSON files directly without loading them first.

``` 
SELECT * FROM 'data.csv';
SELECT * FROM 'data.parquet';
SELECT * FROM 'data.json';
``` 

### CREATE TABLE AS (CTAS): Create tables from query results.

``` 
CREATE TABLE squares AS SELECT i, i * i AS square FROM generate_series(1, 10) t(i);
``` 

### FROM-First Syntax (Optional SELECT): Start queries with FROM, and optionally omit SELECT *.

``` 
FROM my_table WHERE condition;  -- Equivalent to SELECT * FROM my_table WHERE condition
``` 

### SELECT without FROM: DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly.

``` 
SELECT 1 + 1 AS result;
``` 

### GROUP BY ALL/ORDER BY ALL:  Shorthand for grouping/ordering by all non-aggregated/all columns.

``` 
SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;
SELECT * FROM my_table ORDER BY ALL;
``` 

### SELECT COLUMNS(): Powerful way to select or transform multiple columns using patterns or functions.

``` 
-- Select columns matching a pattern
SELECT COLUMNS('sales_.*') FROM sales_data; 

-- Transform multiple columns with a function
SELECT AVG(COLUMNS(*)) FROM sales_data; 
``` 

### UNION BY NAME: Combine tables by matching column names, not just their positions.

``` 
SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;
``` 

### Implicit/Explicit Casting: DuckDB is generally more lenient with implicit type conversions, but you can always be explicit using ::

``` 
SELECT '42' + 1;  -- Implicit cast, result is 43
SELECT '42'::INTEGER + 1; -- Explicit cast, result is 43
``` 

### String/List Slicing: Extract parts of strings and lists using [start:end] or [start:end:step] syntax.

``` 
SELECT 'DuckDB'[1:4];  -- Returns 'Duck'
SELECT [1, 2, 3, 4][1:3]; -- Returns [1, 2, 3]
``` 

### Simple List/Struct/Map/Array Creation: Create complex types using intuitive syntax.

In a SELECT statement:
``` 
SELECT [1, 2, 3] AS my_list, {{'a': 1, 'b': 'text'}} AS my_struct, MAP([1,2],['one','two']) as my_map;
``` 

When creating a table:
``` 
CREATE TABLE data (
    my_list INTEGER[],
    my_struct STRUCT(a INTEGER, b TEXT),
    my_map MAP(INTEGER, VARCHAR),
    my_array INTEGER[3]
);
``` 

### Timestamp Conversions and Extraction: Convert between text and timestamps, and extract date parts.

``` 
SELECT strftime(NOW(), '%Y-%m-%d');  -- Current date as 'YYYY-MM-DD'
SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP; -- String to timestamp
SELECT EXTRACT(YEAR FROM DATE '2023-07-23'); -- Extract year
``` 

### Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses.

``` 
SELECT a + b AS total
FROM my_table
WHERE total > 10
GROUP BY total
HAVING total < 20;
``` 

### List Comprehensions:  Generate lists using expressions similar to Python list comprehensions.

``` 
SELECT [x*2 FOR x IN [1, 2, 3]];  -- Returns [2, 4, 6]
``` 

### Function Chaining: Chain multiple function calls together using the dot (.) operator.

``` 
SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB'
``` 

### Regular Expressions: DuckDB has built-in functions for regex matching, replacement, and extraction.

``` 
SELECT regexp_matches('DuckDB', 'Duck'); -- Returns true
SELECT regexp_replace('DuckDB', 'Duck', 'Goose'); -- Returns 'GooseDB'
SELECT regexp_extract('DuckDB', '(\w+)(DB)', 1); -- Returns 'Duck'
``` 

### Sampling: Quickly get a subset of your data with SAMPLE or TABLESAMPLE.

``` 
SELECT * FROM large_table USING SAMPLE 10%; -- Random 10% sample
SELECT * FROM large_table TABLESAMPLE BERNOULLI(10); -- Bernoulli sampling
``` 

### ATTACH and Access: Attach external databases and reference their objects using databasename.table_name syntax.

``` 
ATTACH 'my_database.duckdb' AS mydb;
SELECT * FROM mydb.my_table;
``` 

### SUMMARIZE: Get summary statistics (min, max, unique count, average, standard deviation, quartiles, and count) of a table.

``` 
SUMMARIZE table_name;
``` 

### DESCRIBE: Get schema of a table (column_name, column_type, null, key, default, extra).

``` 
DESCRIBE table_name;
``` 

Database Schema:
Here is the schema of the DuckDB database that the SQL query will run on:
{schema}

Question:
Here is the question or an instruction the user provided:
{question}

Task:
Write a DuckDB SQL query for the given question!

Here is the valid DuckDB SQL query:
```
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterGranite(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """System:

Your task is to generate valid DuckDB SQL to answer the following question, given a DuckDB database schema.

## DuckDB SQL syntax specifics you should be aware of:

### Case Insensitivity and Quoting:

Identifiers (tables, columns): Case-insensitive, but DuckDB remembers the case you use. Use double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity.
``` 
CREATE TABLE "My Table" ("column_name" VARCHAR); -- Spaces and mixed case
SELECT "column_name" FROM "My Table"; 
``` 

### String Literals: Always use single quotes (') to define string literals.
``` 
SELECT 'This is a string' AS text;
``` 

### Direct File Querying: Query CSV, Parquet, and JSON files directly without loading them first.

``` 
SELECT * FROM 'data.csv';
SELECT * FROM 'data.parquet';
SELECT * FROM 'data.json';
``` 

### CREATE TABLE AS (CTAS): Create tables from query results.

``` 
CREATE TABLE squares AS SELECT i, i * i AS square FROM generate_series(1, 10) t(i);
``` 

### FROM-First Syntax (Optional SELECT): Start queries with FROM, and optionally omit SELECT *.

``` 
FROM my_table WHERE condition;  -- Equivalent to SELECT * FROM my_table WHERE condition
``` 

### SELECT without FROM: DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly.

``` 
SELECT 1 + 1 AS result;
``` 

### GROUP BY ALL/ORDER BY ALL:  Shorthand for grouping/ordering by all non-aggregated/all columns.

``` 
SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;
SELECT * FROM my_table ORDER BY ALL;
``` 

### SELECT COLUMNS(): Powerful way to select or transform multiple columns using patterns or functions.

``` 
-- Select columns matching a pattern
SELECT COLUMNS('sales_.*') FROM sales_data; 

-- Transform multiple columns with a function
SELECT AVG(COLUMNS(*)) FROM sales_data; 
``` 

### UNION BY NAME: Combine tables by matching column names, not just their positions.

``` 
SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;
``` 

### Implicit/Explicit Casting: DuckDB is generally more lenient with implicit type conversions, but you can always be explicit using ::

``` 
SELECT '42' + 1;  -- Implicit cast, result is 43
SELECT '42'::INTEGER + 1; -- Explicit cast, result is 43
``` 

### String/List Slicing: Extract parts of strings and lists using [start:end] or [start:end:step] syntax.

``` 
SELECT 'DuckDB'[1:4];  -- Returns 'Duck'
SELECT [1, 2, 3, 4][1:3]; -- Returns [1, 2, 3]
``` 

### Simple List/Struct/Map/Array Creation: Create complex types using intuitive syntax.

In a SELECT statement:
``` 
SELECT [1, 2, 3] AS my_list, {{'a': 1, 'b': 'text'}} AS my_struct, MAP([1,2],['one','two']) as my_map;
``` 

When creating a table:
``` 
CREATE TABLE data (
    my_list INTEGER[],
    my_struct STRUCT(a INTEGER, b TEXT),
    my_map MAP(INTEGER, VARCHAR),
    my_array INTEGER[3]
);
``` 

### Timestamp Conversions and Extraction: Convert between text and timestamps, and extract date parts.

``` 
SELECT strftime(NOW(), '%Y-%m-%d');  -- Current date as 'YYYY-MM-DD'
SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP; -- String to timestamp
SELECT EXTRACT(YEAR FROM DATE '2023-07-23'); -- Extract year
``` 

### Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses.

``` 
SELECT a + b AS total
FROM my_table
WHERE total > 10
GROUP BY total
HAVING total < 20;
``` 

### List Comprehensions:  Generate lists using expressions similar to Python list comprehensions.

``` 
SELECT [x*2 FOR x IN [1, 2, 3]];  -- Returns [2, 4, 6]
``` 

### Function Chaining: Chain multiple function calls together using the dot (.) operator.

``` 
SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB'
``` 

### Regular Expressions: DuckDB has built-in functions for regex matching, replacement, and extraction.

``` 
SELECT regexp_matches('DuckDB', 'Duck'); -- Returns true
SELECT regexp_replace('DuckDB', 'Duck', 'Goose'); -- Returns 'GooseDB'
SELECT regexp_extract('DuckDB', '(\w+)(DB)', 1); -- Returns 'Duck'
``` 

### Sampling: Quickly get a subset of your data with SAMPLE or TABLESAMPLE.

``` 
SELECT * FROM large_table USING SAMPLE 10%; -- Random 10% sample
SELECT * FROM large_table TABLESAMPLE BERNOULLI(10); -- Bernoulli sampling
``` 

### ATTACH and Access: Attach external databases and reference their objects using databasename.table_name syntax.

``` 
ATTACH 'my_database.duckdb' AS mydb;
SELECT * FROM mydb.my_table;
``` 

### SUMMARIZE: Get summary statistics (min, max, unique count, average, standard deviation, quartiles, and count) of a table.

``` 
SUMMARIZE table_name;
``` 

### DESCRIBE: Get schema of a table (column_name, column_type, null, key, default, extra).

``` 
DESCRIBE table_name;
``` 

Here is the schema of the DuckDB database that the SQL query will run on:
{schema}

Question:
Here is the question or an instruction the user provided:
{question}

Please write a DuckDB SQL query that answers the user's question or instruction. Use DuckDB-specific syntax if possible.

Answer:
```
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterPhi(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """<|endoftext|><|user|>
Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.

Here are some DuckDB SQL syntax specifics you should be aware of:


- DuckDB use double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity and single quotes (') to define string literals
- DuckDB can query CSV, Parquet, and JSON directly without loading them first, e.g. `SELECT * FROM 'data.csv';`
- DuckDB supports CREATE TABLE AS (CTAS): `CREATE TABLE new_table AS SELECT * FROM old_table;`
- DuckDB queries can start with FROM, and optionally omit SELECT *, e.g. `FROM my_table WHERE condition;` is equivalent to `SELECT * FROM my_table WHERE condition;`
- DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly, e.g. `SELECT 1 + 1 AS result;`
- DuckDB supports attaching multiple databases, unsing the ATTACH statement: `ATTACH 'my_database.duckdb' AS mydb;`. Tables within attached databases can be accessed using the dot notation (.), e.g. `SELECT * FROM mydb.table_name syntax`. The default databases doesn't require the do notation to access tables. The default database can be changed with the USE statement, e.g. `USE my_db;`.
- DuckDB is generally more lenient with implicit type conversions (e.g. `SELECT '42' + 1;` - Implicit cast, result is 43), but you can always be explicit using `::`, e.g. `SELECT '42'::INTEGER + 1;`
- DuckDB can extract parts of strings and lists using [start:end] or [start:end:step] syntax. Indexes start at 1. String slicing: `SELECT 'DuckDB'[1:4];`. Array/List slicing: `SELECT [1, 2, 3, 4][1:3];`
- DuckDB has a powerful way to select or transform multiple columns using patterns or functions. You can select columns matching a pattern: `SELECT COLUMNS('sales_.*') FROM sales_data;` or transform multiple columns with a function: `SELECT AVG(COLUMNS('sales_.*')) FROM sales_data;`
- DuckDB an easy way to include/exclude or modify columns when selecting all: e.g. Exclude: `SELECT * EXCLUDE (sensitive_data) FROM users;` Replace: `SELECT * REPLACE (UPPER(name) AS name) FROM users;` 
- DuckDB has a shorthand for grouping/ordering by all non-aggregated/all columns. e.g `SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;` and `SELECT * FROM my_table ORDER BY ALL;`
- DuckDB can combine tables by matching column names, not just their positions using UNION BY NAME. E.g. `SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;`
- DuckDB has an inutitive syntax to create List/Struct/Map and Array types. Create complex types using intuitive syntax. List: `SELECT [1, 2, 3] AS my_list;`, Struct: `{{'a': 1, 'b': 'text'}} AS my_struct;`, Map: `MAP([1,2],['one','two']) as my_map;`. All types can also be nested into each other. Array types are fixed size, while list types have variable size. Compared to Structs, MAPs do not need to have the same keys present for each row, but keys can only be of type Integer or Varchar. Example: `CREATE TABLE example (my_list INTEGER[], my_struct STRUCT(a INTEGER, b TEXT), my_map MAP(INTEGER, VARCHAR),  my_array INTEGER[3], my_nested_struct STRUCT(a INTEGER, b Integer[3]));`
- DuckDB has an inutive syntax to access struct fields using dot notation (.) or brackets ([]) with the field name. Maps fields can be accessed by brackets ([]).
- DuckDB's way of converting between text and timestamps, and extract date parts. Current date as 'YYYY-MM-DD': `SELECT strftime(NOW(), '%Y-%m-%d');` String to timestamp: `SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP;`, Extract Year from date: `SELECT EXTRACT(YEAR FROM DATE '2023-07-23');`
- Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses. E.g.: `SELECT a + b AS total FROM my_table WHERE total > 10 GROUP BY total HAVING total < 20;` 
- DuckDB allows generating lists using expressions similar to Python list comprehensions. E.g. `SELECT [x*2 FOR x IN [1, 2, 3]];` Returns [2, 4, 6].
- DuckDB allows chaining multiple function calls together using the dot (.) operator. E.g.: `SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB';`
- DuckDB has a JSON data type. It supports selecting fields from the JSON with a JSON-Path expression using the arrow operator, -> (returns JSON) or ->> (returns text) with JSONPath expressions. For example: `SELECT data->'$.user.id' AS user_id, data->>'$.event_type' AS event_type FROM events;`
- DuckDB has built-in functions for regex regexp_matches(column, regex), regexp_replace(column, regex), and regexp_extract(column, regex).
- DuckDB has a way to quickly get a subset of your data with `SELECT * FROM large_table USING SAMPLE 10%;`

Here is the schema of the DuckDB database that the SQL query will run on:
{schema}

Question:
Here is the question or an instruction the user provided:
{question}

Write a DuckDB SQL query for the given question!<|end|>
<|assistant|>
```sql
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterGPTmini(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """Schema:
```sql
{schema}
```

Question:
{question}

Write a valid DuckDB SQL query to answer the question!
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstFormatterPhiAzure(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.

Here is the schema of the DuckDB database that the SQL query will run on:
{schema}

Question:
Here is the question or an instruction the user provided:
{question}

Write a DuckDB SQL query for the given question!
"""

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        instruction = cls.PROMPT_TEMPLATE.format(
            schema=table_text,
            question=instruction
        )
        return instruction

class DuckDBInstNoShorthandFormatter(DuckDBInstFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}{context}\n### Question:\n{question}\n\n### Response:\n"""
    INSTRUCTION_TEMPLATE = """Your task is to generate valid duckdb SQL to answer the following question{has_schema}"""  # noqa: E501


class DuckDBChat:
    """DuckDB Inst class."""

    table_sep: str = "\n\n"
    shuffle_table_order: bool = True
    _cache: dict[tuple[str, str, str], list[str]] = {}
    clean_whitespace = False
    model = None

    @classmethod
    def format_table(cls, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table.name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table.name}"
        return create_tbl

    @classmethod
    def format_all_tables(cls, tables: list[Table], instruction: str) -> list[dict]:
        """Get all tables format."""
        if not cls.model:
            cls.model = Manifest(
                engine="gpt-3.5-turbo",
                client_name="openaichat",
                cache_name="sqlite",
                cache_connection=".manifest.sqlite",
            )
        table_texts = [cls.format_table(table) for table in tables]
        full_schema = cls.table_sep.join(table_texts)
        prompt = f"""SQL schema of my database:
{full_schema}
Explain in a few sentences what the data is about:
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can generate an human redable summary of database content based on the schema.",
            },
            {"role": "user", "content": prompt},
        ]
        explanation = cls.model.run(messages, temperature=0)
        messages.append({"role": "assistant", "content": explanation})
        return messages[1:]

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n\nHere is additional documentation about DuckDB that could be useful.\n--------\n{context_str}\n--------\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: list[dict],
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        prompt = f"""Now output a single SQL query without any explanation and do not add anything 
to the query that was not part of the question, also do not use markdown. Make sure to only 
use information provided in the prompt, or tables and columns from the schema above and write a query to answer the question.{context_text}\n\nMy quesiton is \n`{instruction}`\n\nGenerate the DuckDB specific SQL query:"""  # noqa: E501
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can generate DuckDB sql queries, which is a superset of Postgresql, based on the user input. You do not respond with any human readable text, only SQL code.",
            },
            *table_text,
            {"role": "user", "content": prompt},
        ]
        return messages

    @classmethod
    def format_model_output(cls, output_sql: str, prompt: str) -> str:
        """Format model output."""
        return output_sql

    @classmethod
    def format_gold_output(cls, output_sql: str) -> str:
        """Format gold output for demonstration."""
        return output_sql
