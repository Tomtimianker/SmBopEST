import json
from pathlib import Path
import os
from tqdm import tqdm
from typing import List
import stanza

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']
syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG',
        'AGGOPS', 'CONDOPS']

# This is the tokenize library
nlp = stanza.Pipeline(lang='en', processors='tokenize')


def parse_sql(sql_dict, table_name, tables_dict) -> str:
    """
    Parses the sql from WikiSQL format to regluar SQL. Column names are changed in order to be used in the model.
    Notice that in the original db the columns are named 'col0', 'col1' ... Since we have the actual name of the
    columns from the table.jsonl file, we will use it.

    Args:
        sql_dict: The sql dict from the train/dev/test.jsonl file
        table_name: The name of the table.
        tables_dict: a dictionary containing the {table name: table info} in order to access the table data faster

    Returns:
        SQL query.
    """
    sel_index = sql_dict['sel']
    agg_index = sql_dict['agg']
    conditions = sql_dict['conds']
    if agg_index == 0 :
        rep = 'SELECT {sel} FROM {table_name}'.format(
            sel=get_column_from_table(table_name, sel_index, tables_dict),
            table_name=table_name,
        )
    else:
        rep = 'SELECT {agg}( {sel} ) FROM {table_name}'.format(
            agg=agg_ops[agg_index],
            sel=get_column_from_table(table_name, sel_index, tables_dict),
            table_name=table_name,
        )
    if conditions:
        rep += ' WHERE ' + ' AND '.join(
            ['{} {} "{}"'.format(get_column_from_table(table_name, i, tables_dict), cond_ops[o], v) for i, o, v in conditions])
    rep = rep.replace('"', "'")
    return rep


def get_column_from_table(table_name: str, col_id: int, tables_dict: dict) -> str:
    """
    return the name of the column according to the table name and column index.

    Args:
        table_name: The name of the table
        col_id: The column ID
        tables_dict:  a dictionary containing the {table name: table info} in order to access the table data faster
    Returns:
        Column name
    """
    return tables_dict[table_name]['column_names'][col_id+1][1]


def tokenize(text: str, sql=True) -> List:
    """
    Tokenize the text using Stanza

    Args:
        text: The text to be tokenized

    Returns:
        A list of tokens.
    """
    doc = nlp(text)
    tokens = []
    long_tok=None
    for sentence in doc.sentences:
        for token in sentence.tokens:
            if sql and token.text == "'" and long_tok is None:
                long_tok = ''
                continue
            if sql and long_tok is not None:
                if token.text != "'":
                    long_tok = long_tok + ' ' + token.text
                    continue
                else:
                    tokens.append(long_tok.replace('( ', '(').replace(' )', ')'))
            tokens.append(token.text)
    return tokens


def table_id_2_name(table_id: str) -> str:
    """
    Convert the table id to table name as is in the database
    Args:
        table_id: The table ID

    Returns:
        The table name
    """
    return 'table_'+table_id.replace('-', '_')


def format_column_name(column_name, i):
    column_name = column_name.lower().replace(' ', '_').replace('no.', 'number').replace('/', '_slash_').replace('.', '_dot_').replace(')', '').replace('(', '') + '_col_' + str(i)
    column_name = column_name.replace('#','number').replace('&', 'and').replace(",", "").replace('{', '').replace('}', '')
    column_name = column_name.replace('%', 'percent').replace('?', 'question_mark').replace('°', 'degrees').replace('²', 'square').replace('-','_')
    column_name = column_name.replace('1st', 'first').replace('2nd', 'second').replace('3rd', 'third').replace('4th', 'fourth')
    # column_name = ''.join(i for i in column_name if not i.isdigit())
    column_name = column_name.replace(' _', ' ')
    return column_name


def wiki_2_spider_tables(wiki_tables_path: Path, prefix: str):
    """
    Converet the table.jsonl file to be similar to spider table.json

    Args:
        wiki_tables_path: The path for the original jsonl file.
        prefix: the prefix of the file to save.  dev or train or test.

    Returns:
        The path the tables.json file was saved in
    """
    final_tables_json = []
    with open(wiki_tables_path, 'rb') as wiki_tables:
        for line in tqdm(wiki_tables.readlines()):
            output_dict = {}
            table_dict = json.loads(line)
            output_dict['column_names'] = [[-1, '*']]
            output_dict['column_names'].extend([[0, format_column_name(column_name, i) ] for i, column_name in enumerate(table_dict.get('header'))])
            output_dict['column_names_original'] = output_dict['column_names']
            output_dict['column_types'] = table_dict.get('types')
            output_dict['table_names'] = [table_id_2_name(table_dict.get('id'))]
            output_dict['table_names_original'] = output_dict['table_names']
            output_dict['db_id'] = output_dict['table_names'][0]
            output_dict['primary_keys'] = []
            output_dict['foreign_keys'] = []
            final_tables_json.append(output_dict)
    os.makedirs(str(wiki_tables_path.parent / 'Processed' ), exist_ok=True)
    with open(wiki_tables_path.parent / f'Processed/{prefix}_processed_tables.json', 'w') as output_file:
        json.dump(final_tables_json, output_file, indent=4,)
    return wiki_tables_path.parent / f'Processed/{prefix}_processed_tables.json'


def wiki_2_spider_sql(wiki_path: Path, wiki_tables_path: Path, prefix: str):
    """
    Converts wikisql data to spider format.

    Args:
        wiki_path: The path for the wiki data
        wiki_tables_path: The path for the processed wiki tables (This code assumes that wiki_2_spider_tables has run
            before.
        prefix: the prefix of the file to save.  dev or train.

    """
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
            output_dict['db_id'] = table_id_2_name(query_dict['table_id'])
            output_dict['query'] = parse_sql(query_dict['sql'], table_id_2_name(query_dict['table_id']), tables_dict)
            output_dict['query_toks'] = tokenize(output_dict['query'], True)
            output_dict['query_toks_no_value'] = output_dict['query_toks']
            output_dict['question_toks'] = tokenize(output_dict['question'], False)
            output_json.append(output_dict)
            # as the model isn't using `quesion_toks` or `sql` we are not saving them
    with open(wiki_path.parent / f'Processed/{prefix}_processed_queries.json', 'w') as output_file:
        json.dump(output_json, output_file, indent=4, )


def wiki_2_spider(wiki_path: Path, wiki_tables_path: Path, prefix: str):
    output_tables_path = wiki_2_spider_tables(wiki_tables_path, prefix)
    wiki_2_spider_sql(wiki_path,output_tables_path, prefix)


if __name__ == '__main__':
    train_or_dev = 'dev'
    wiki_2_spider(Path(f'/home/orl/Documents/Shani/SmBopEST/wikisql_dataset/{train_or_dev}.jsonl'), Path(f'/home/orl/Documents/Shani/SmBopEST/wikisql_dataset/{train_or_dev}.tables.jsonl'), train_or_dev)