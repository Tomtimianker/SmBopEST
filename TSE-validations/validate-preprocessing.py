import dataset_readers.disamb_sql as disamb_sql
import json
from utils import moz_sql_parser as msp
from utils import node_util
from eval_final.evaluation import evaluate_single
from tqdm import tqdm

data_set = r'/home/orl/Documents/Shani/SmBopEST/wikisql_dataset/Processed/train_processed_queries.json'
tables_file = r'/home/orl/Documents/Shani/SmBopEST/wikisql_dataset/Processed/train_processed_tables.json'
db_dir = '/home/orl/Documents/Shani/SmBopEST/wikisql_dataset/train.db'


def validateSQLtoIR(evaluation_function):
    """
    Goes over the dataset json example by example, preforms the exact same preprocessing as the model's data-reader
    which includes translation sql->IR->relationship algebra, and then reverses the process to validate we got back to
    an acceptable output.
    """
    # read samples from the dataset - copied from dataset_readers\smbop.py _read_examples_file
    error_counter = 0
    with open(data_set, "r") as data_file:
        json_obj = json.load(data_file)
        for total_cnt, ex in tqdm(enumerate(json_obj)):
            sql = ex['query']
            sql_with_values = ex['query']
            # if "query_toks" in ex:
            #     try:
            #         # Preform preprocessing on queries - remove values if needed etc.
            #         ex = disamb_sql.fix_number_value(ex)
            #         sql = disamb_sql.disambiguate_items(
            #             ex["db_id"],
            #             ex["query_toks_no_value"],
            #             tables_file,
            #             allow_aliases=False,
            #         )
            #         sql_with_values = disamb_sql.sanitize(ex["query"])
            #     except Exception as e:
            #         # there are two examples in the train set that are wrongly formatted, skip them
            #         print(f"error with {ex['query']}")
            #         continue
            # Convert to IR - copied from dataset_readers\smbop.py text_to_instance
            try:
                tree_dict = msp.parse(sql)
            except msp.ParseException as e:
                print(f"couldn't parse {sql}")
                error_counter +=1
                print(f'errors: {error_counter}')
                continue
                # return None
            try:
                tree_obj = node_util.get_tree(tree_dict["query"], None)

                # convert back to sql, done as in models/semantic_parsing/smbop.py method _compute_validation_outputs
                tree_res = node_util.remove_keep(tree_obj)
                sql = node_util.print_sql(tree_res)
                sql = node_util.fix_between(sql)
                sql = sql.replace("LIMIT value", "LIMIT 1")

                # print(sql_with_values)

                score = evaluation_function(g_str=sql_with_values, p_str=sql, db_id=ex["db_id"], db_dir=db_dir,
                                            table_file=tables_file)
                if score != 1:
                    print(f'Record {total_cnt} is not identical')
                    print(f'Got \n {sql}')
                    print(f'expected \n {sql_with_values}')
                    print('************************************')
            except Exception as e:
                print(f"Terrible error, could not process at all: {sql}")
                print(str(e))
                error_counter +=1
                print(f'errors: {error_counter}')


def fix_whitespace(s, q):
    q = q.lower()
    s = s.lower()
    q = ' '.join(q.split())
    s = ' '.join(s.split())
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    return s, q


def evaluate_exact_string_match(recovered_sql_values, gold_query, total_cnt):
    """
    Primitive evaluation function that looks for exact string match.
    """
    recovered_sql_values, original_query = fix_whitespace(recovered_sql_values, gold_query)
    if recovered_sql_values != original_query:
        print(f'Record {total_cnt} is not identical')
        print(f'Got \n {recovered_sql_values}')
        print(f'expected \n {original_query}')


if __name__ == '__main__':
    validateSQLtoIR(evaluate_single)
