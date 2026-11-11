"""Pinned candidate data, formatting, scoring and token-length checks."""

from dataclasses import dataclass, field
import json
import math
import random
import re


SEED = 42
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRAIN_ROWS = 500
HELDOUT_ROWS = 100
MAX_TRAIN_LENGTH = 512

SOURCES = {
    "A": {
        "dataset": "b-mc2/sql-create-context",
        "revision": "9d80a6a118b838d9defc3798d659a54a2ac2ff37",
        "file": "sql_create_context_v4.json",
        "license": "cc-by-4.0",
        "gated": False,
        "task": "sql",
        "pool": "train",
        "context_definition": "normalized table schema",
        "category_field": None,
        "mapping": "Instruction: fixed SQL system prompt. Input: CREATE TABLE context and question. Output: SQL answer.",
    },
    "B": {
        "dataset": "google/mobile-actions",
        "revision": "e920309bc2acbc2e99a5e3201cf37df2b9fd9151",
        "file": "dataset.jsonl",
        "license": "cc-by-4.0",
        "gated": False,
        "task": "function",
        "pool": "metadata=train, exactly one tool call",
        "context_definition": "normalized available tool set",
        "category_field": "reference function name",
        "mapping": "Instruction: fixed one-call JSON system prompt with seven tool signatures. Input: date/time, weekday and user utterance. Output: reference function name and argument object.",
    },
    "C": {
        "dataset": "AmazonScience/massive",
        "revision": "ed58ac423a2f4121720918bf5301577edce4ffd3",
        "source_revision": "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
        "file": "en-US/train/0000.parquet",
        "license": "cc-by-4.0",
        "gated": False,
        "task": "function",
        "pool": "en-US train, at least one annotated slot",
        "context_definition": "normalized published intent",
        "category_field": "published intent",
        "mapping": "Instruction: fixed slot extraction JSON system prompt with allowed slot names. Input: published intent and utterance. Output: extract_slots call with exact annotated text spans grouped by slot name in arrays.",
    },
}

SQL_SYSTEM = (
    "Write one SQL query that answers the question using the supplied table schemas. "
    "Use the supplied table and column names. Return only the SQL query."
)
FUNCTION_SYSTEM = (
    "Select one of the available functions to fulfill the user's request. Use the given date "
    "and weekday for relative times. Return exactly one JSON object with a string key "
    "\"name\" and an object key \"arguments\". Use the function's exact name and argument keys. "
    "Return only JSON. All arguments are strings. Omit optional arguments when no value is given. "
    "Calendar datetime uses YYYY-MM-DDTHH:MM:SS. A map query can be a place, business, or address.\n\n"
    "Available functions (optional arguments have a ?):\n"
)
SLOT_SYSTEM = (
    "Extract the slot values from the request using the supplied intent. "
    'Return exactly one JSON object with "name": "extract_slots" and an "arguments" object. '
    "Each argument key is a slot name from the list below. Its value is an array of verbatim text spans "
    "from the request, in their original order. Use arrays even for a single span. Include only slots "
    "present in the request. Return only JSON.\n\nAllowed slots: "
)

H2_REVIEWS = {
    "A": {
        "source_indices_zero_based": [14592, 3278, 36048, 32098, 29256, 18289, 13434, 71482, 11395, 77397],
        "note": "Ten English questions and SQL answers checked for readability. Row 71482 compares a VARCHAR column to a numeric literal, which may depend on SQL dialect.",
    },
    "B": {
        "source_indices_zero_based": [8137, 912, 204, 2253, 2006, 1828, 1143, 839, 8442, 7366],
        "note": "Ten English requests and reference call names and argument objects checked for readability.",
    },
    "C": {
        "published_row_ids": ["11680", "1992", "543", "13448", "5018", "4427", "4035", "2509", "13276", "1844"],
        "note": "Ten English requests and reference slot annotations checked for readable, verbatim spans.",
    },
}

QUOTES = str.maketrans({character: "'" for character in "\"'`‘’“”"})
SQL_LITERALS = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*" """.strip())
HARD_SQL = re.compile(r"\bJOIN\b|\bGROUP\s+BY\b|\(\s*(?:SELECT|WITH)\b", re.IGNORECASE)
TABLE_NAME = re.compile(r"\bCREATE\s+TABLE\s+(\w+)", re.IGNORECASE)
SLOT_ANNOTATION = re.compile(r"\[([^:\[\]]+?)\s*:\s*([^\[\]]+?)\s*\]")


@dataclass
class Example:
    source_index: int
    system: str
    prompt: str
    answer: str
    context: str | None = None
    category: str | None = None
    copies_prompt_value: bool = False
    utterance: str | None = None
    prompt_ids: list[int] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    @property
    def messages(self):
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.answer},
        ]


@dataclass
class PreparedCandidate:
    train: list[Example]
    heldout: list[Example]
    metrics: dict
    system: str
    task: str
    max_new_tokens: int

    def score(self, prediction, answer):
        return score_answer(self.task, prediction, answer)


def normalize_prompt(text):
    return " ".join(text.lower().split())


def normalize_sql(text):
    return normalize_prompt(text.translate(QUOTES)).removesuffix(";").rstrip()


def score_sql(prediction, answer):
    import sqlglot

    try:
        statements = sqlglot.parse(prediction)
        parses = len(statements) == 1 and statements[0] is not None and not isinstance(statements[0], sqlglot.exp.Command)
    except sqlglot.errors.SqlglotError:
        parses = False
    return {"correct": normalize_sql(prediction) == normalize_sql(answer), "parses": parses}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def strict_json(value):
    def object_without_duplicates(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("Repeated JSON key")
        return result

    def reject_constant(constant):
        raise ValueError(f"Invalid JSON constant: {constant}")

    return json.loads(value, object_pairs_hook=object_without_duplicates, parse_constant=reject_constant)


def score_function(prediction, answer):
    try:
        decoded = strict_json(prediction)
    except (ValueError, RecursionError):
        return {"correct": False, "json_valid": False, "name_correct": False, "arguments_correct": False}
    gold = json.loads(answer)
    is_object = isinstance(decoded, dict)
    name = is_object and decoded.get("name") == gold["name"]
    arguments = is_object and isinstance(decoded.get("arguments"), dict)
    arguments = arguments and decoded["arguments"] == gold["arguments"]
    return {"correct": name and arguments, "json_valid": True, "name_correct": name, "arguments_correct": arguments}


def score_answer(task, prediction, answer):
    if task == "sql":
        return score_sql(prediction, answer)
    if task == "function":
        return score_function(prediction, answer)
    if task == "classification":
        return {"correct": prediction.strip().lower() == answer.strip().lower()}
    raise ValueError(f"Unsupported scoring task: {task}")


def map_sql(raw, index):
    schema = raw["context"].strip()
    question = raw["question"].strip()
    answer = raw["answer"].strip()
    copies_name = bool(set(TABLE_NAME.findall(schema.lower())) & set(re.findall(r"\w+", answer.lower())))
    return Example(
        index, SQL_SYSTEM, f"Schema:\n{schema}\n\nQuestion:\n{question}", answer,
        context=schema, copies_prompt_value=copies_name,
    )


def render_tool_signatures(tools):
    signatures = []
    for function in tools:
        parameters = function["parameters"]
        required = set(parameters.get("required", []))
        if any(field["type"] != "STRING" for field in parameters["properties"].values()):
            raise ValueError(f"Unexpected non-string field in {function['name']}")
        arguments = [key if key in required else f"{key}?" for key in sorted(parameters["properties"])]
        signatures.append(f"{function['name']}({', '.join(arguments)}): {function['description']}")
    return "\n".join(signatures)


def map_function(raw, index, system, tools):
    developer, user, assistant = raw["messages"]
    timestamp, weekday, _ = developer["content"].splitlines()
    timestamp = timestamp.split(": ", maxsplit=1)[1]
    weekday = weekday.removeprefix("Day of week is ")
    utterance = user["content"].strip()
    function = assistant["tool_calls"][0]["function"]
    if not isinstance(function["arguments"], dict):
        raise ValueError(f"Unexpected non-object arguments at source row {index}")
    copies_value = any(
        isinstance(value, str) and value and value.lower() in utterance.lower()
        for value in function["arguments"].values()
    )
    return Example(
        index, system,
        f"Current date/time: {timestamp}\nDay of week: {weekday}\n\nUser request:\n{utterance}",
        compact_json(function), context=tools, category=function["name"],
        copies_prompt_value=copies_value, utterance=utterance,
    )


def load_slots(table):
    hf_features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    intents = hf_features["intent"]["names"]
    raw = table.select(["id", "utt", "annot_utt", "intent"]).to_pylist()
    eligible = []
    slot_types = set()
    for index, row in enumerate(raw):
        slots = {}
        for slot_type, span in SLOT_ANNOTATION.findall(row["annot_utt"]):
            slots.setdefault(slot_type, []).append(span)
        if not slots:
            continue
        reconstructed = SLOT_ANNOTATION.sub(r"\2", row["annot_utt"])
        if normalize_prompt(reconstructed) != normalize_prompt(row["utt"]):
            raise ValueError(f"Unexpected slot annotation at published row {row['id']}")
        slot_types.update(slots)
        eligible.append((index, row, slots))

    system = SLOT_SYSTEM + ", ".join(sorted(slot_types))
    rows = []
    for index, row, slots in eligible:
        intent = intents[row["intent"]]
        utterance = row["utt"].strip()
        rows.append(Example(
            index, system, f"Intent: {intent}\nRequest: {utterance}",
            compact_json({"name": "extract_slots", "arguments": slots}),
            context=intent, category=intent, copies_prompt_value=True, utterance=utterance,
        ))
    return raw, rows


def load_candidate(candidate, *, source_file=None):
    """Read the pinned source file, or an identical local copy for offline screening."""
    source = SOURCES[candidate]
    if source_file is None:
        from huggingface_hub import hf_hub_download

        source_file = hf_hub_download(
            repo_id=source["dataset"], repo_type="dataset",
            filename=source["file"], revision=source["revision"],
        )
    if candidate == "A":
        with open(source_file, encoding="utf-8") as file:
            raw = json.load(file)
        rows = [map_sql(row, index) for index, row in enumerate(raw)]
    elif candidate == "B":
        with open(source_file, encoding="utf-8") as file:
            raw = [json.loads(line) for line in file]
        eligible = [
            (index, row) for index, row in enumerate(raw)
            if row["metadata"] == "train" and len(row["messages"][-1]["tool_calls"]) == 1
        ]
        tool_sets = {
            compact_json(sorted((tool["function"] for tool in row["tools"]), key=lambda item: item["name"]))
            for _, row in eligible
        }
        if len(tool_sets) != 1:
            raise ValueError("Expected the same tool set for every eligible Mobile Actions row")
        tools = tool_sets.pop()
        system = FUNCTION_SYSTEM + render_tool_signatures(json.loads(tools))
        rows = [map_function(row, index, system, tools) for index, row in eligible]
    else:
        import pyarrow.parquet as parquet

        raw, rows = load_slots(parquet.read_table(source_file))
    source_revision = source.get("source_revision", source["revision"])
    metadata = {
        **source, "source_rows": len(raw), "excluded_by_source_pool": len(raw) - len(rows),
        "source_url": f"https://huggingface.co/datasets/{source['dataset']}/tree/{source_revision}",
    }
    return rows, metadata


def is_harder_sql(answer):
    return bool(HARD_SQL.search(SQL_LITERALS.sub("", answer)))


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile / 100) - 1]


def length_stats(values):
    return {"p50": nearest_rank(values, 50), "p95": nearest_rank(values, 95), "max": max(values)}


def describe_lengths(full_lengths, answer_lengths):
    return {"full": length_stats(full_lengths), "answer": length_stats(answer_lengths)}


def batched_pool_lengths(tokenizer, rows):
    full_lengths, answer_lengths = [], []
    for start in range(0, len(rows), 512):
        batch = rows[start:start + 512]
        rendered = tokenizer.apply_chat_template([row.messages for row in batch], tokenize=False)
        full_ids = tokenizer(rendered, add_special_tokens=False, verbose=False)["input_ids"]
        answer_ids = tokenizer([row.answer for row in batch], add_special_tokens=False, verbose=False)["input_ids"]
        full_lengths.extend(map(len, full_ids))
        answer_lengths.extend(map(len, answer_ids))
    return full_lengths, answer_lengths


def encode_example(tokenizer, row):
    full = tokenizer.apply_chat_template(row.messages, tokenize=False)
    prompt = tokenizer.apply_chat_template(row.messages[:-1], tokenize=False, add_generation_prompt=True)
    row.input_ids = tokenizer.encode(full, add_special_tokens=False)
    row.prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix_len = len(row.prompt_ids)
    if row.input_ids[:prefix_len] != row.prompt_ids:
        raise ValueError(f"Template tokenization does not preserve the prompt prefix at row {row.source_index}")
    stop = len(row.input_ids) - row.input_ids[::-1].index(tokenizer.eos_token_id) - 1
    if stop <= prefix_len:
        raise ValueError(f"No answer tokens at row {row.source_index}")
    row.labels = [-100] * prefix_len + row.input_ids[prefix_len:stop + 1] + [-100] * (len(row.input_ids) - stop - 1)


def prepare_candidate(candidate, tokenizer, *, harder=False, loaded=None):
    """Normalize and deduplicate, then sample and prepare train and held-out examples."""
    rows, metadata = loaded if loaded is not None else load_candidate(candidate)
    task = metadata["task"]
    if harder and task != "sql":
        raise ValueError("The harder slice is defined only for SQL")
    scoped = [row for row in rows if is_harder_sql(row.answer)] if harder else rows
    usable = [row for row in scoped if row.prompt.strip() and row.answer.strip()]
    unique = {}
    conflicting_reference_prompts = set()
    for row in usable:
        key = normalize_prompt(row.prompt)
        first = unique.setdefault(key, row)
        if first.answer.lower() != row.answer.lower():
            conflicting_reference_prompts.add(key)
    deduped = list(unique.values())
    if len(deduped) < TRAIN_ROWS + HELDOUT_ROWS:
        raise ValueError(f"{candidate} has too few usable prompts after deduplication: {len(deduped)}")

    full_lengths, answer_lengths = batched_pool_lengths(tokenizer, deduped)
    pool_lengths = describe_lengths(full_lengths, answer_lengths)
    indices = random.Random(SEED).sample(range(len(deduped)), TRAIN_ROWS + HELDOUT_ROWS)
    train = [deduped[index] for index in indices[:TRAIN_ROWS]]
    heldout = [deduped[index] for index in indices[TRAIN_ROWS:]]
    for row in train + heldout:
        encode_example(tokenizer, row)

    train_prompts = {normalize_prompt(row.prompt) for row in train}
    prompt_overlap = sum(normalize_prompt(row.prompt) in train_prompts for row in heldout)
    if prompt_overlap:
        raise ValueError(f"Train/held-out prompt overlap: {prompt_overlap}")
    train_contexts = {normalize_prompt(row.context) for row in train if row.context is not None}
    contexts = [normalize_prompt(row.context) for row in heldout if row.context is not None]
    context_overlap = sum(context in train_contexts for context in contexts) if contexts else None
    n1_fraction = sum(5 <= length <= 100 for length in answer_lengths) / len(answer_lengths)
    copy_fraction = sum(row.copies_prompt_value for row in deduped) / len(deduped)
    near_duplicates = []
    if candidate == "B":
        seen_utterances = {}
        for row in deduped:
            key = (normalize_prompt(row.utterance), row.category)
            first = seen_utterances.setdefault(key, row.source_index)
            if first != row.source_index:
                near_duplicates.append([first, row.source_index])
    review = H2_REVIEWS.get(candidate)
    cap = 2 * pool_lengths["answer"]["p95"]

    metrics = {
        **metadata, "candidate": candidate, "harder_sql_slice": harder,
        "rows_in_slice": len(scoped), "unusable_rows": len(scoped) - len(usable),
        "duplicates_dropped": len(usable) - len(deduped), "usable_dedup_rows": len(deduped),
        "duplicate_prompts_with_conflicting_references": len(conflicting_reference_prompts),
        "near_duplicate_source_index_pairs": near_duplicates,
        "train_rows": len(train), "heldout_rows": len(heldout),
        "heldout_prompt_overlap_count": prompt_overlap,
        "heldout_context_seen_train_count": context_overlap,
        "heldout_context_seen_train_fraction": context_overlap / len(contexts) if contexts else None,
        "pool_lengths": pool_lengths,
        "sampled_lengths": describe_lengths([full_lengths[index] for index in indices], [answer_lengths[index] for index in indices]),
        "train_over_512": sum(len(row.input_ids) > MAX_TRAIN_LENGTH for row in train),
        "heldout_over_512": sum(len(row.input_ids) > MAX_TRAIN_LENGTH for row in heldout),
        "max_new_tokens": cap,
        "h2_review": review,
        "n1_answer_5_to_100_fraction": n1_fraction,
        "n2_prompt_copy_fraction": copy_fraction,
        "hard_requirements": {
            "H1": True, "H2": bool(review) if review is not None else None,
            "H3": True, "H4": len(deduped) >= 2000 and not metadata["gated"],
            "H5": pool_lengths["full"]["p95"] <= MAX_TRAIN_LENGTH, "H6": None, "H7": None,
        },
        "nice_to_have": {
            "N1": n1_fraction >= 0.9, "N2": copy_fraction >= 0.5,
            "N3": metadata["category_field"] is not None, "N4": len(usable) > len(deduped) or bool(near_duplicates),
        },
    }
    return PreparedCandidate(train, heldout, metrics, train[0].system, task, cap)
