from allennlp.data import DatasetReader, TokenIndexer
from dataset_readers.smbop import SmbopDatasetReader
from eval_final.process_sql import tokenize
from overrides import overrides
from typing import Dict
import json


@DatasetReader.register("sparc-naive")
class SparcNaiveDatasetReader(SmbopDatasetReader):
    def __init__(
            self,
            lazy: bool = True,
            question_token_indexers: Dict[str, TokenIndexer] = None,
            keep_if_unparsable: bool = True,
            tables_file: str = None,
            dataset_path: str = "dataset/database",
            cache_directory: str = "cache/train",
            include_table_name_in_column=True,
            fix_issue_16_primary_keys=False,
            qq_max_dist=2,
            cc_max_dist=2,
            tt_max_dist=2,
            max_instances=None,
            decoder_timesteps=9,
            limit_instances=-1,
            value_pred=True,
    ):
        super().__init__(
            lazy,
            question_token_indexers,
            keep_if_unparsable,
            tables_file,
            dataset_path,
            cache_directory,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            max_instances,
            decoder_timesteps,
            limit_instances,
            value_pred
        )

    @overrides
    def enumerate_json(self, json_obj):
        i = 0
        for interaction in json_obj:
            dbid = interaction['database_id']
            examples = interaction['interaction']
            for example in examples:
                example['db_id'] = dbid
                example['query_toks_no_value'] = tokenize(example['db_id'])
                example['question_toks'] = example.pop('utterance_toks')
                example['question'] = example.pop('utterance')
                yield i, example
                i += 1
