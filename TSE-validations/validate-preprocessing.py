import dataset_readers.disamb_sql as disamb_sql
import json
from utils import moz_sql_parser as msp
from utils import node_util

data_set = r'C:\Edan\NLP datasets\spider\spider\train_spider.json'
tables_file = r'C:\Edan\NLP datasets\spider\spider\tables.json'


def validateSQLtoIR():
    """
    Goes over the dataset json example by example, preforms the exact same preprocessing as the model's data-reader
    which includes translation sql->IR->relationship algebra, and then reverses the process to validate we got back to
    an acceptable output.
    """
    with open(data_set, "r") as data_file:
        json_obj = json.load(data_file)
        for total_cnt, ex in enumerate(json_obj):
            sql = None
            sql_with_values = None
            if "query_toks" in ex:
                try:
                    # Preform preprocessing on queries - remove values if needed etc.
                    ex = disamb_sql.fix_number_value(ex)
                    sql = disamb_sql.disambiguate_items(
                        ex["db_id"],
                        ex["query_toks_no_value"],
                        tables_file,
                        allow_aliases=False,
                    )
                    sql_with_values = disamb_sql.sanitize(ex["query"])
                except Exception as e:
                    # there are two examples in the train set that are wrongly formatted, skip them
                    print(f"error with {ex['query']}")
                    continue
            # convert to IR
            tree_dict = msp.parse(sql)
            tree_dict_values = msp.parse(sql_with_values)
            # convert to relationship algebra
            tree_obj = node_util.get_tree(tree_dict["query"], None)
            tree_obj_values = node_util.get_tree(tree_dict_values["query"], None)
            # convert back to sql
            recovered_sql = node_util.print_sql(tree_obj)
            recovered_sql_values = node_util.print_sql(tree_obj_values)
            # test that the query with values is identical to original
            recovered_sql_values, original_query = fix_whitespace(recovered_sql_values, ex["query"])
            if recovered_sql_values != original_query:
                print(f'Record {total_cnt} is not identical')
                print(f'Got \n {recovered_sql_values}')
                print(f'expected \n {original_query}')


def fix_whitespace(s, q):
    q = q.lower()
    s = s.lower()
    q = ' '.join(q.split())
    s = ' '.join(s.split())
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    return s, q


if __name__ == '__main__':
    validateSQLtoIR()
