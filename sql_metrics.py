"""Exact-match metrics and SQLite compilation without executing model queries."""
import re
import sqlite3

FENCE = re.compile(r"```(?:[\w-]+[ \t]*)?\r?\n?(.*?)```", re.DOTALL)
SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)
QUOTED_OR_SEMICOLON = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*"|`[^`]*`|\[[^\]]*\]|;""")


def normalize(sql):
    sql = " ".join(sql.lower().strip().split()).removesuffix(";").rstrip()
    return re.sub(r"\s*([(),])\s*", r"\1", sql.replace("'", '"'))


def extract_sql(output):
    fenced = FENCE.search(output)
    if fenced:
        return fenced[1].strip()
    select = SELECT.search(output)
    if not select:
        return output.strip()
    sql = output[select.start():].strip()
    for token in QUOTED_OR_SEMICOLON.finditer(sql):
        if token[0] == ";":
            return sql[:token.end()]
    return sql


def sql_error(context, sql):
    try:
        with sqlite3.connect(":memory:") as database:
            # The Hub source is data. Permit schema construction only, with no ATTACH or PRAGMA.
            schema_actions = {
                sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX,
                sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_READ,
                sqlite3.SQLITE_SELECT,
            }
            database.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in schema_actions else sqlite3.SQLITE_DENY)
            database.executescript(context)
            database.set_authorizer(
                lambda action, *_: sqlite3.SQLITE_OK if action in {
                    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE,
                } else sqlite3.SQLITE_DENY
            )
            database.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    except sqlite3.Error as exc:
        return str(exc)
    return None


def score(context, prediction, gold):
    extracted = extract_sql(prediction)
    return {
        "strict": prediction.strip() == gold.strip(),
        "normalized": normalize(prediction) == normalize(gold),
        "lenient": normalize(extracted) == normalize(gold),
        "valid_sql": sql_error(context, extracted) is None,
        "valid_raw_sql": sql_error(context, prediction.strip()) is None,
    }
