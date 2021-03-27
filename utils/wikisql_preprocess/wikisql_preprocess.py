import json
from pathlib import Path
import os
from tqdm import tqdm

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']
syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG',
        'AGGOPS', 'CONDOPS']



def wiki_2_spider_sql(wiki_path: Path, table_to_db_dict: Path, prefix: str):
    output_json = []
    with open(wiki_path, 'rb') as wiki_queries:
        for line in tqdm(wiki_queries.readlines()):
            output_dict = {}
            query_dict = json.loads(line)
            output_dict['question'] = query_dict['question']
            output_dict['db_id'] = table_to_db_dict[table_id_2_name(query_dict['table_id'])]
            output_dict['query'] = parse_sql(query_dict['sql'],  table_id_2_name(query_dict['table_id']))
            # output_dict['query_toks'] = tokenize_query(output_dict['query'])
            output_json.append(output_dict)
            # as the model isn't using `quesion_toks` or `sql` we are not saving them
    with open(wiki_path.parent / f'Processed/{prefix}_processed_queries.json', 'w') as output_file:
        json.dump(output_json, output_file, indent=4,)


def parse_sql(sql_dict, table_name):
    sel_index = sql_dict['sel']
    agg_index = sql_dict['agg']
    conditions = sql_dict['conds']
    if agg_index == 0 :
        rep = 'SELECT {sel} FROM {table_name}'.format(
            sel=get_column_from_table(sel_index),
            table_name=table_name,
        )
    else:
        rep = 'SELECT {agg}({sel}) FROM {table_name}'.format(
            agg=agg_ops[agg_index],
            sel=get_column_from_table(sel_index),
            table_name=table_name,
        )
    if conditions:
        rep += ' WHERE ' + ' AND '.join(
            ['{} {} "{}"'.format(get_column_from_table(i), cond_ops[o], str(v).lower()) for i, o, v in conditions])
    return rep


def get_column_from_table(col_id: int):
    return f'col{col_id}'


def tokenize_query(sql_query: str):
    pass


def table_id_2_name(table_id: str):
    return 'table_'+table_id.replace('-', '_')


def wiki_2_spider_tables(wiki_tables_path: Path, prefix: str):
    final_dbs_json = []
    table_to_db_dict = {}
    db_counter = 0
    table_counter = 0
    db_dict = {'db_id': f'main_{db_counter}',
              'primary_keys': [],
              'foreign_keys': []}
    columns = [[-1, '*']]
    column_types = []
    table_names = []
    with open(wiki_tables_path, 'rb') as wiki_tables:
        for line in tqdm(wiki_tables.readlines()):
            if table_counter != 0 and table_counter % 500 == 0:
                db_dict['column_names'] = columns
                db_dict['column_names_original'] = columns
                db_dict['column_types'] = column_types
                db_dict['table_names'] = table_names
                db_dict['table_names_original'] = table_names
                final_dbs_json.append(db_dict)

                db_counter += 1
                db_dict = {'db_id': f'main_{db_counter}',
                           'primary_keys': [],
                           'foreign_keys': []}
                columns = [[-1, '*']]
                column_types = []
                table_names = []
                table_counter = 0
            table_dict = json.loads(line)
            columns.extend([[table_counter, f'col{i}'] for i in range(len(table_dict.get('header')))])
            column_types.extend(table_dict.get('types'))
            table_names.append(table_id_2_name(table_dict.get('id')))
            table_to_db_dict[table_names[-1]] = f'main_{db_counter}'
            table_counter += 1
    db_dict['column_names'] = columns
    db_dict['column_names_original'] = columns
    db_dict['column_types'] = column_types
    db_dict['table_names'] = table_names
    db_dict['table_names_original'] = table_names
    final_dbs_json.append(db_dict)
    os.makedirs(str(wiki_tables_path.parent / 'Processed'), exist_ok=True)
    with open(wiki_tables_path.parent / f'Processed/{prefix}_processed_tables.json', 'w') as output_file:
        json.dump(final_dbs_json, output_file, indent=4,)
    return wiki_tables_path.parent / f'Processed/{prefix}_processed_tables.json', table_to_db_dict


def wiki_2_spider(wiki_path: Path, wiki_tables_path: Path, prefix: str):
    output_tables_path, table_to_db_dict = wiki_2_spider_tables(wiki_tables_path, prefix)
    wiki_2_spider_sql(wiki_path, table_to_db_dict, prefix )

if __name__ == '__main__':
    train_or_dev = 'dev'
    wiki_2_spider(Path(f'/Users/orlichter/Desktop/data/{train_or_dev}.jsonl'), Path(f'/Users/orlichter/Desktop/data/{train_or_dev}.tables.jsonl'), train_or_dev)