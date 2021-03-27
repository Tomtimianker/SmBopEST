import json
from pathlib import Path
import os
from tqdm import tqdm

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']
syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG',
        'AGGOPS', 'CONDOPS']



def wiki_2_spider_sql(wiki_path: Path, wiki_tables_path: Path, prefix: str):
    output_json = []
    with open(wiki_tables_path, 'rb') as tables_file:
        tables = json.load(tables_file)
    tables_dict = {}
    for table in tables:
        tables_dict[table['db_id']] = table
    with open(wiki_path, 'rb') as wiki_queries:
        for line in tqdm(wiki_queries.readlines()):
            output_dict = {}
            query_dict = json.loads(line)
            output_dict['question'] = query_dict['question']
            output_dict['db_id'] = query_dict['table_id']
            output_dict['query'] = parse_sql(query_dict['sql'], query_dict['table_id'], tables_dict)
            output_dict['query_toks'] = tokenize_query(output_dict['query'])
            output_json.append(output_dict)
            # as the model isn't using `quesion_toks` or `sql` we are not saving them
    with open(wiki_path.parent / f'Processed/{prefix}_processed_queries.json', 'w') as output_file:
        json.dump(output_json, output_file, indent=4,)

def parse_sql(sql_dict, table_name, tables_dict):
    sel_index = sql_dict['sel']
    agg_index = sql_dict['agg']
    conditions = sql_dict['conds']
    if agg_index == 0 :
        rep = 'SELECT {sel} FROM {table_name} as T1'.format(
            sel=get_column_from_table(table_name, sel_index, tables_dict),
            table_name=table_name,
        )
    else:
        rep = 'SELECT {agg}({sel}) FROM {table_name} as T1'.format(
            agg=agg_ops[agg_index],
            sel=get_column_from_table(table_name, sel_index, tables_dict),
            table_name=table_name,
        )
    if conditions:
        rep += ' WHERE ' + ' AND '.join(
            ['T1.[{}] {} "{}"'.format(get_column_from_table(table_name, i, tables_dict), cond_ops[o], v) for i, o, v in conditions])
    return rep

def get_column_from_table(table_name: str, col_id: int, tables_dict: dict):
    return tables_dict[table_name]['column_names'][col_id][1]


def tokenize_query(sql_query: str):
    pass


def wiki_2_spider_tables(wiki_tables_path: Path, prefix: str):
    final_tables_json = []
    with open(wiki_tables_path, 'rb') as wiki_tables:
        for line in tqdm(wiki_tables.readlines()):
            output_dict = {}
            table_dict = json.loads(line)
            output_dict['column_names'] = [[0, column_name.lower().replace(' ', '_')] for column_name in table_dict.get('header')]
            output_dict['column_names_original'] = output_dict['column_names']
            output_dict['column_types'] = table_dict.get('types')
            output_dict['table_names'] = [table_dict.get('id')]
            output_dict['table_names_original'] = output_dict['table_names']
            output_dict['db_id'] = output_dict['table_names'][0]
            output_dict['primary_keys'] = []
            output_dict['foreign_keys'] = []
            final_tables_json.append(output_dict)
    os.makedirs(str(wiki_tables_path.parent / 'Processed' ), exist_ok=True)
    with open(wiki_tables_path.parent / f'Processed/{prefix}_processed_tables.json', 'w') as output_file:
        json.dump(final_tables_json, output_file, indent=4,)
    return wiki_tables_path.parent / f'Processed/{prefix}_processed_tables.json'


def wiki_2_spider(wiki_path: Path, wiki_tables_path: Path, prefix: str):
    output_tables_path = wiki_2_spider_tables(wiki_tables_path, prefix)
    wiki_2_spider_sql(wiki_path, output_tables_path, prefix)

if __name__ == '__main__':
    wiki_2_spider(Path('/Users/orlichter/Desktop/data/dev.jsonl'), Path('/Users/orlichter/Desktop/data/dev.tables.jsonl'), 'dev')