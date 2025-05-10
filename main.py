from utils import get_database_matches
from collections import defaultdict
import json
def extract_columns(sql, column_names, table_names, aliases=None, from_table_idxs=None):
    used_tables = defaultdict(set)
    current_from_table_idxs = set()
    alias_to_table_idx = {}

    def resolve_col(col_unit):
        if col_unit is None:
            return
        _, col_idx, _ = col_unit
        if col_idx == 0:  # wildcard
            for t_idx, col_name in column_names:
                if t_idx in from_table_idxs:
                    table_name = table_names[t_idx]
                    used_tables[table_name].add(col_name)
        elif col_idx > 0:
            t_idx, col_name = column_names[col_idx]
            if t_idx >= 0:
                table_name = table_names[t_idx]
                used_tables[table_name].add(col_name)

    def resolve_val_unit(val_unit):
        if val_unit is None:
            return
        _, col_unit1, col_unit2 = val_unit
        resolve_col(col_unit1)
        resolve_col(col_unit2)

    def resolve_val(val):
        if isinstance(val, dict):  # subquery
            sub_from_idxs = get_from_table_idxs(val)
            sub_used = extract_columns(val, column_names, table_names, {}, sub_from_idxs)
            for tbl, cols in sub_used.items():
                used_tables[tbl].update(cols)
        elif isinstance(val, (tuple, list)) and len(val) == 3:
            resolve_col(tuple(val))  # Convert list to tuple

    def resolve_condition(conds):
        if not conds:
            return
        if isinstance(conds[0], list):
            for cond in conds:
                if isinstance(cond, list) and len(cond) == 5:
                    not_op, op_id, val_unit, val1, val2 = cond
                    resolve_val_unit(val_unit)
                    resolve_val(val1)
                    resolve_val(val2)
        else:
            not_op, op_id, val_unit, val1, val2 = conds
            resolve_val_unit(val_unit)
            resolve_val(val1)
            resolve_val(val2)

    def resolve_table_units(table_units):
        for unit in table_units:
            type_, val = unit
            if type_ == "table_unit":
                if isinstance(val, int):
                    current_from_table_idxs.add(val)
                    table_name = table_names[val]
                    alias_to_table_idx[str(val)] = val
                elif isinstance(val, dict):  # subquery
                    sub_from_idxs = get_from_table_idxs(val)
                    sub_used = extract_columns(val, column_names, table_names, {}, sub_from_idxs)
                    for tbl, cols in sub_used.items():
                        used_tables[tbl].update(cols)
            elif type_ == "sql":  # e.g., derived table
                sub_from_idxs = get_from_table_idxs(val)
                sub_used = extract_columns(val, column_names, table_names, {}, sub_from_idxs)
                for tbl, cols in sub_used.items():
                    used_tables[tbl].update(cols)

    def get_from_table_idxs(sql_dict):
        from_ = sql_dict.get("from", {})
        idxs = set()
        for unit in from_.get("table_units", []):
            type_, val = unit
            if type_ == "table_unit" and isinstance(val, int):
                idxs.add(val)
        return idxs

    # FROM clause
    from_ = sql.get("from", {})
    table_units = from_.get("table_units", [])
    resolve_table_units(table_units)
    resolve_condition(from_.get("conds", []))
    
    # ✅ Set this before using resolve_col
    if from_table_idxs is None:
        from_table_idxs = current_from_table_idxs
    
    # SELECT
    select_clause = sql.get("select", [False, []])
    if select_clause and isinstance(select_clause, list):
        _, select_cols = select_clause
        for agg_id, val_unit in select_cols:
            resolve_val_unit(val_unit)

    # WHERE clause
    resolve_condition(sql.get("where", []))

    # GROUP BY clause
    for col_unit in sql.get("groupBy", []):
        resolve_col(col_unit)

    # HAVING clause
    resolve_condition(sql.get("having", []))

    # ORDER BY clause
    order_by = sql.get("orderBy")
    if order_by:
        _, val_units = order_by
        for val_unit in val_units:
            resolve_val_unit(val_unit)

    # INTERSECT / UNION / EXCEPT
    for key in ["intersect", "union", "except"]:
        if sql.get(key):
            sub_from_idxs = get_from_table_idxs(sql[key])
            sub_used = extract_columns(sql[key], column_names, table_names, {}, sub_from_idxs)
            for tbl, cols in sub_used.items():
                used_tables[tbl].update(cols)

    return used_tables
def format_used_columns(used_tables):
    return " | ".join(f"{table}: {' , '.join(sorted(cols))}" for table, cols in used_tables.items())
def extract_skeleton(sql, db_schema):
    table_names_original, table_dot_column_names_original, column_names_original = [], [], []
    for table in db_schema["schema_items"]:
        table_name_original = table["table_name_original"]
        table_names_original.append(table_name_original)

        for column_name_original in ["*"]+table["column_names_original"]:
            table_dot_column_names_original.append(table_name_original+"."+column_name_original)
            column_names_original.append(column_name_original)
    
    parsed_sql = Parser(sql)
    new_sql_tokens = []
    for token in parsed_sql.tokens:
        # mask table names
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        # mask column names
        elif token.value in column_names_original \
            or token.value in table_dot_column_names_original:
            new_sql_tokens.append("_")
        # mask string values
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        # mask positive int number
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        # mask negative int number
        elif isNegativeInt(token.value):
            new_sql_tokens.append("_")
        # mask float number
        elif isFloat(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)
    
    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while("_ , _" in sql_skeleton):
        sql_skeleton = sql_skeleton.replace("_ , _", "_")
    
    # remove clauses in WHERE keywords
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
        if "where _ and _"in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    return sql_skeleton


def get_db_contents(question, table_name_original, column_names_original, db_id, db_path):
    matched_contents = []
    # extract matched contents for each column
    for column_name_original in column_names_original:
        matches = get_database_matches(
            question, 
            table_name_original, 
            column_name_original, 
            db_path + "/{}/{}.sqlite".format(db_id, db_id)
        )
        matches = sorted(matches)
        matched_contents.append(matches)
    
    return matched_contents

def get_db_schemas(all_db_infos):
    db_schemas = {}

    for db in all_db_infos:
        table_names_original = db["table_names_original"]
        table_names = db["table_names"]
        column_names_original = db["column_names_original"]
        column_names = db["column_names"]
        column_types = db["column_types"]

        db_schemas[db["db_id"]] = {}
        
        primary_keys, foreign_keys = [], []
        # record primary keys
        for pk_column_idx in db["primary_keys"]:
            pk_table_name_original = table_names_original[column_names_original[pk_column_idx][0]]
            pk_column_name_original = column_names_original[pk_column_idx][1]
            
            primary_keys.append(
                {
                    "table_name_original": pk_table_name_original.lower(), 
                    "column_name_original": pk_column_name_original.lower()
                }
            )

        db_schemas[db["db_id"]]["pk"] = primary_keys

        # record foreign keys
        for source_column_idx, target_column_idx in db["foreign_keys"]:
            fk_source_table_name_original = table_names_original[column_names_original[source_column_idx][0]]
            fk_source_column_name_original = column_names_original[source_column_idx][1]

            fk_target_table_name_original = table_names_original[column_names_original[target_column_idx][0]]
            fk_target_column_name_original = column_names_original[target_column_idx][1]
            
            foreign_keys.append(
                {
                    "source_table_name_original": fk_source_table_name_original.lower(),
                    "source_column_name_original": fk_source_column_name_original.lower(),
                    "target_table_name_original": fk_target_table_name_original.lower(),
                    "target_column_name_original": fk_target_column_name_original.lower(),
                }
            )
        db_schemas[db["db_id"]]["fk"] = foreign_keys

        db_schemas[db["db_id"]]["schema_items"] = []
        for idx, table_name_original in enumerate(table_names_original):
            column_names_original_list = []
            column_names_list = []
            column_types_list = []
            
            for column_idx, (table_idx, column_name_original) in enumerate(column_names_original):
                if idx == table_idx:
                    column_names_original_list.append(column_name_original.lower())
                    column_names_list.append(column_names[column_idx][1].lower())
                    column_types_list.append(column_types[column_idx])
            
            db_schemas[db["db_id"]]["schema_items"].append({
                "table_name_original": table_name_original.lower(),
                "table_name": table_names[idx].lower(), 
                "column_names": column_names_list, 
                "column_names_original": column_names_original_list,
                "column_types": column_types_list
            })

    return db_schemas
def normalization(sql):
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()
            
            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True
        
        return out_s
    
    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation 
    def double2single(s):
        return s.replace("\"", "'") 
    
    def add_asc(s):
        pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1,11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
        
        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)
        
        return s
    
    processing_func = lambda x : (add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    # processing_func = lambda x : remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))
    
    return processing_func(sql)

# extract the skeleton of sql and natsql
def extract_skeleton(sql, db_schema):
    table_names_original, table_dot_column_names_original, column_names_original = [], [], []
    for table in db_schema["schema_items"]:
        table_name_original = table["table_name_original"]
        table_names_original.append(table_name_original)

        for column_name_original in ["*"]+table["column_names_original"]:
            table_dot_column_names_original.append(table_name_original+"."+column_name_original)
            column_names_original.append(column_name_original)
    
    parsed_sql = Parser(sql)
    new_sql_tokens = []
    for token in parsed_sql.tokens:
        # mask table names
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        # mask column names
        elif token.value in column_names_original \
            or token.value in table_dot_column_names_original:
            new_sql_tokens.append("_")
        # mask string values
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        # mask positive int number
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        # mask negative int number
        elif isNegativeInt(token.value):
            new_sql_tokens.append("_")
        # mask float number
        elif isFloat(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)
    
    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while("_ , _" in sql_skeleton):
        sql_skeleton = sql_skeleton.replace("_ , _", "_")
    
    # remove clauses in WHERE keywords
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
        if "where _ and _"in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    return sql_skeleton


def build_input_output_sequences(tables_path, spider_path, opt):
    with open(tables_path, 'r') as f:
        all_db_infos = json.load(f)

    with open(spider_path, 'r') as f:
        dataset = json.load(f)

    db_schemas = get_db_schemas(all_db_infos)
    input_output_pairs = []

    for data in tqdm(dataset):
        question = data["question"].replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace("\u201d", "'").strip()
        db_id = data["db_id"]
        sql = data["query"].strip()

        input_sequence = question + " | "
        sql = data["sql"]
        db_id = data["db_id"]
        query = data["query"]
        table_entry = next(item for item in all_db_infos if item["db_id"] == db_id)
        column_names = table_entry["column_names_original"]
        table_names = table_entry["table_names_original"]
        aliases = {}
    
        used = extract_columns(sql, column_names, table_names, aliases)
        output_sequence = format_used_columns(used).lower()

        for table in db_schemas[db_id]["schema_items"]:
            db_contents = get_db_contents(
                question,
                table["table_name_original"],
                table["column_names_original"],
                db_id,
                opt.db_path
            )

            # Add to input
            input_sequence += table["table_name_original"] + " : "
            input_sequence += " , ".join(table["column_names_original"]) + " | "

        # Add foreign keys to input
        for fk in db_schemas[db_id]["fk"]:
            input_sequence += (
                fk["source_table_name_original"] + "." + fk["source_column_name_original"]
                + " = " +
                fk["target_table_name_original"] + "." + fk["target_column_name_original"]
                + " | "
            )

        input_output_pairs.append((input_sequence.strip(), output_sequence.strip()))
# CAN ALSO KEEP PRIMARY KEY
    return input_output_pairs
from collections import Counter
import re
def parse_schema_output(text):
    """Robustly parse 'table : col1, col2 | table2:col1,col2' into dict."""
    schema = defaultdict(set)
    parts = text.strip().split("|")

    for part in parts:
        if ":" not in part:
            continue  # skip invalid parts
        table, cols = map(str.strip, part.split(":", 1))
        col_list = re.split(r"\s*,\s*", cols.strip())
        for col in col_list:
            if col:
                schema[table].add(col)
    return schema
from tqdm import tqdm

def evaluate_predictions(pairs, predictions):
    assert len(pairs) == len(predictions), "Mismatch between predictions and ground truth pairs"

    table_correct = 0
    table_total = 0
    table_recall_total = 0

    column_correct = 0
    column_total = 0
    column_recall_total = 0

    print("\nSample Predictions (First 10):\n" + "-"*50)

    for i, ((_, ground_truth), prediction) in enumerate(tqdm(zip(pairs, predictions), total=len(pairs))):
        gt_schema = parse_schema_output(ground_truth)
        pred_schema = parse_schema_output(prediction)

        if i < 10:
            print(f"\nExample {i+1}")
            print("Ground Truth:")
            print(ground_truth)
            print("Prediction:")
            print(prediction)
            print("-" * 50)

        # Evaluate tables
        gt_tables = set(gt_schema.keys())
        pred_tables = set(pred_schema.keys())

        table_correct += len(gt_tables & pred_tables)
        table_total += len(pred_tables)
        table_recall_total += len(gt_tables)

        # Evaluate columns
        for table in gt_tables:
            gt_cols = gt_schema.get(table, set())
            pred_cols = pred_schema.get(table, set())

            column_correct += len(gt_cols & pred_cols)
            column_total += len(pred_cols)
            column_recall_total += len(gt_cols)

    table_accuracy = table_correct / table_total if table_total else 0.0
    table_recall = table_correct / table_recall_total if table_recall_total else 0.0

    column_accuracy = column_correct / column_total if column_total else 0.0
    column_recall = column_correct / column_recall_total if column_recall_total else 0.0

    print("\nFinal Evaluation Metrics\n" + "="*50)
    print(f"Table Accuracy : {table_accuracy:.4f}")
    print(f"Table Recall   : {table_recall:.4f}")
    print(f"Column Accuracy: {column_accuracy:.4f}")
    print(f"Column Recall  : {column_recall:.4f}")
import re
from sql_metadata import Parser
def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False

def isFloat(string):
    if string.startswith("-"):
        string = string[1:]
    
    s = string.split(".")
    if len(s)>2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True




tables_path = "/spider-data/tables.json"
train_path = "/spider-data/train_spider.json"
with open(tables_path, "r") as f:
    tables = {tbl["db_id"]: tbl for tbl in json.load(f)}

with open(train_path, "r") as f:
    train = json.load(f)

for i, item in enumerate(train[40:50]):
    sql = item["sql"]
    db_id = item["db_id"]
    query = item["query"]
    table_entry = tables[db_id]
    column_names = table_entry["column_names_original"]
    table_names = table_entry["table_names_original"]
    aliases = {}

    used = extract_columns(sql, column_names, table_names, aliases)
    formatted = format_used_columns(used).lower()

    print(f"\nExample {i+1} ({query}\n{db_id}):\n{formatted}")



import json
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("/kaggle/input/spider-data/train_spider.json", "r") as f:
    train_data = json.load(f)

with open("/kaggle/input/spider-data/tables.json", "r") as f:
    tables_data = json.load(f)
class Opt:
    db_path = "/kaggle/input/databased/database"

opt = Opt()
import json
from tqdm import tqdm

pairs = build_input_output_sequences("/kaggle/input/spider-data/tables.json", "/kaggle/input/spider-data/train_spider.json", opt)

for inp, out in pairs[:3]:
    print("INPUT:", inp)
    print("OUTPUT:", out)
    print("=" * 80)




input_texts, output_texts = zip(*pairs)

# Create HuggingFace Dataset
dataset = Dataset.from_dict({
    "input_text": list(input_texts),
    "output_text": list(output_texts)
})


tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(examples["output_text"], max_length=256, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "output_text"])

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: {
        key: torch.tensor([d[key] for d in x]) for key in x[0]
    }
)
# Model
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-4)

# Training
model.train()
for epoch in range(5):
    total_loss = 0
    for batch in tqdm(train_dataloader,desc="train:"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
    
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}")

# Save
model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")

"""TRAINING OF SCHEMA LINKING"""



pairs2 = build_input_output_sequences(
    "/kaggle/input/spider-data/tables.json",
    "/kaggle/input/spider-data/dev.json",
    opt
)

inputs = [p[0] for p in pairs2]
gold_outputs = [p[1] for p in pairs2]
from transformers import T5ForConditionalGeneration, T5Tokenizer

# model = T5ForConditionalGeneration.from_pretrained("path/to/your/model")
# tokenizer = T5Tokenizer.from_pretrained("t5-base")  # or your custom tokenizer

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

from tqdm import tqdm
import torch
import random

batch_size = 8  # Try increasing this if you have enough VRAM
num_beams = 6
num_return_sequences = 4

print_indices = set(random.sample(range(len(inputs)), 2))
predictions = []

print("\nGenerating predictions with beam search (batched)...\n")

for start_idx in tqdm(range(0, len(inputs), batch_size)):
    end_idx = min(start_idx + batch_size, len(inputs))
    batch_inputs = inputs[start_idx:end_idx]

    encoded = tokenizer(batch_inputs, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=256,
            num_beams=num_beams,
            length_penalty = 0.8,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )

    decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Group decoded outputs (num_return_sequences per input)
    grouped = [decoded[i:i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]

    for i, group in enumerate(grouped):
        predictions.append(group[0])  # main prediction

        global_idx = start_idx + i
        if global_idx in print_indices:
            print(f"\nBeam Search Results for Example {global_idx + 1}")
            print("-" * 50)
            print("Input:")
            print(inputs[global_idx].strip())
            print("\nBeam Outputs:")
            for j, out in enumerate(group):
                print(f"[Beam {j+1}]: {out}")
            print("-" * 50)

# Evaluation

print(predictions[:5])
evaluate_predictions(pairs2, predictions)


"""EVALUATION OF SCHEMA LINKING DONE"""
"""TRAINING OF ENCODER_DECODER TEXT2SQL"""

def build_input_output_sequences(tables_path, spider_path, opt):
    with open(tables_path, 'r') as f:
        all_db_infos = json.load(f)

    with open(spider_path, 'r') as f:
        dataset = json.load(f)

    db_schemas = get_db_schemas(all_db_infos)
    input_output_pairs = []

    for data in tqdm(dataset):
        question = data["question"].replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace("\u201d", "'").strip()
        db_id = data["db_id"]
        sql = data["query"].strip()

        input_sequence = question + " | "
        sql = data["sql"]
        db_id = data["db_id"]
        query = data["query"]
        table_entry = next(item for item in all_db_infos if item["db_id"] == db_id)
        column_names = table_entry["column_names_original"]
        table_names = table_entry["table_names_original"]
        table_names.append("t1")
        table_names.append("t2")
        table_names.append("T1")
        table_names.append("T2")
        aliases = {}
    
        used = extract_columns(sql, column_names, table_names, aliases)
        schema = format_used_columns(used).lower()

        input_sequence += schema
        input_sequence +=' | '

        # Add foreign keys to input
        # for fk in db_schemas[db_id]["fk"]:
        #     input_sequence += (
        #         fk["source_table_name_original"] + "." + fk["source_column_name_original"]
        #         + " = " +
        #         fk["target_table_name_original"] + "." + fk["target_column_name_original"]
        #         + " | "
        #     )
         
        norm_sql = normalization(query).strip()
        sql_skeleton = extract_skeleton(norm_sql, db_schemas[db_id]).strip()
        output_sequence = f"{sql_skeleton} | {norm_sql}"
  
        

        input_output_pairs.append((input_sequence.strip(), output_sequence.strip()))
# CAN ALSO KEEP PRIMARY KEY
    return input_output_pairs

pairs_train = build_input_output_sequences("/kaggle/input/spider-data/tables.json", "/kaggle/input/spider-data/train_spider.json", opt)

for inp, out in pairs_train[:3]:
    print("INPUT:", inp)
    print("OUTPUT:", out)
    print("=" * 80)





input_texts, output_texts = zip(*pairs_train)

# Create HuggingFace Dataset
dataset = Dataset.from_dict({
    "input_text": list(input_texts),
    "output_text": list(output_texts)
})


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(examples["output_text"], max_length=256, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "output_text"])

train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: {
        key: torch.tensor([d[key] for d in x]) for key in x[0]
    }
)
# Model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-4)

# Training
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(train_dataloader,desc="train:"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
    
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}")

# Save
model.save_pretrained("./encdec_output")
tokenizer.save_pretrained("./encdec_output")    





"""EVALUATION OF TEXT2SQL"""
pairs2_encdec = build_input_output_sequences(
    "/kaggle/input/spider-data/tables.json",
    "/kaggle/input/spider-data/dev.json",
    opt
)

inputs = [p[0] for p in pairs2_encdec]
gold_outputs = [p[1] for p in pairs2_encdec]
from transformers import T5ForConditionalGeneration, T5Tokenizer

# model = T5ForConditionalGeneration.from_pretrained("path/to/your/model")
# tokenizer = T5Tokenizer.from_pretrained("t5-base")  # or your custom tokenizer

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

from tqdm import tqdm
import torch
import random

batch_size = 8  # Try increasing this if you have enough VRAM
num_beams = 4
num_return_sequences = 4

print_indices = set(random.sample(range(len(inputs)), 2))
predictions = []

print("\nGenerating predictions with beam search (batched)...\n")

for start_idx in tqdm(range(0, len(inputs), batch_size)):
    end_idx = min(start_idx + batch_size, len(inputs))
    batch_inputs = inputs[start_idx:end_idx]

    encoded = tokenizer(batch_inputs, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=256,
            num_beams=num_beams,
            length_penalty = 0.8,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )

    decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Group decoded outputs (num_return_sequences per input)
    grouped = [decoded[i:i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]

    for i, group in enumerate(grouped):
        predictions.append(group[0])  # main prediction

        global_idx = start_idx + i
        if global_idx in print_indices:
            print(f"\nBeam Search Results for Example {global_idx + 1}")
            print("-" * 50)
            print("Input:")
            print(inputs[global_idx].strip())
            print("\nBeam Outputs:")
            for j, out in enumerate(group):
                print(f"[Beam {j+1}]: {out}")
            print("-" * 50)

# Evaluation

norm_sql_list = [pred.split("|", 1)[1].strip() for pred in predictions]

with open("/kaggle/input/spider-data/dev.json", "r") as f:
    val_data = json.load(f)
db_ids = []
labels=[]

for item in val_data:
    db_id = item['db_id']
    sql = item['query']
    labels.append(sql)
    db_ids.append(db_id)


prediction_entries = []
for pred, db_id in zip(norm_sql_list, db_ids):
    prediction_entries.append({
        "query": pred,
        # "question": gold,  # Not used by evaluator
        "db_id": db_id
    })

with open("full.json", "w") as f:
    json.dump(prediction_entries, f, indent=2)    
for i in range(30,50):
    print("="*50)
    print(f"Input:\n{input_texts[i]}")
    print(f"Target:\n{labels[i]}")
    print(f"Prediction:\n{norm_sql_list[i]}")


import json

# Load gold data from Spider dev.json
with open("/kaggle/input/spider-data/dev.json") as f:
    gold_data = json.load(f)

# Write gold.sql (each line: <GOLD_SQL> \t <DB_ID>)
with open("gold.sql", "w") as f:
    for example in gold_data:
        f.write(f"{example['query']}\t{example['db_id']}\n")

# Write pred.sql (each line: predicted SQL)
# `preds` must be in the same order as `gold_data`
with open("pred.sql", "w") as f:
    for pred in norm_sql_list:
        f.write(pred.strip() + "\n")


# Uncomment this to run
"""git clone https://github.com/taoyds/spider.git"""


"""
python3 ./spider/evaluation.py \
    --gold gold.sql \
    --pred pred.sql \
    --etype all \
    --db ../databased/database \
    --table ../spider-data/tables.json

"""    