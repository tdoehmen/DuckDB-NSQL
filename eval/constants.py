"""Constants."""

from prompt_formatters import (
    DuckDBFormatter,
    MotherDuckFormatter,
    DuckDBInstFormatter,
    DuckDBInstNoShorthandFormatter,
    RajkumarFormatter,
    DuckDBChat,
    DuckDBInstFormatterLlamaShort,
    DuckDBInstFormatterGraniteShort,
    DuckDBInstFormatterLlama,
    DuckDBInstFormatterLlamaBasic,
    DuckDBInstFormatterGranite,
    DuckDBInstFormatterPhi,
    DuckDBInstFormatterGPTmini,
    DuckDBInstFormatterPhiAzure,
    DuckDBInstFormatterLlamaSyntax,
)

PROMPT_FORMATTERS = {
    "rajkumar": RajkumarFormatter,
    "duckdb": DuckDBFormatter,
    "motherduck": MotherDuckFormatter,
    "duckdbinst": DuckDBInstFormatter,
    "duckdbinstllamashort": DuckDBInstFormatterLlamaShort,
    "duckdbinstgraniteshort": DuckDBInstFormatterGraniteShort,
    "duckdbinstllama": DuckDBInstFormatterLlama,
    "duckdbinstgranite": DuckDBInstFormatterGranite,
    "duckdbinstnoshort": DuckDBInstNoShorthandFormatter,
    "duckdbchat": DuckDBChat,
    "duckdbinstphi": DuckDBInstFormatterPhi,
    "duckdbinstgptmini": DuckDBInstFormatterPhi,
    "duckdbinstphiazure": DuckDBInstFormatterPhiAzure,
    "duckdbinstllamabasic": DuckDBInstFormatterLlamaBasic,
    "duckdbinstllamasyntax": DuckDBInstFormatterLlamaSyntax
}
