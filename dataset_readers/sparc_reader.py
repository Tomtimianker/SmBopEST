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
            cache_directory: str = "cache-big-bird/train",
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
        """
        Enumerates the sparc dataset file, such that each question contains the previous utterances also made.
        """
        i = 0
        for interaction in json_obj:
            dbid = interaction['database_id']
            examples = interaction['interaction']
            if not examples: # there are a few wierd examples in the set which are empty
                continue
            # Want to join all utterances in sequence and add history to each single question.
            # Doing naive string concatination takes a lot of time.
            # instead we do a single join per iteration and by keeping the length of the original questions can slice the exact question sequence we want.
            questions = [example['utterance'] for example in examples]
            lengths = [len(q) + 1 for q in questions]
            question_sequence = ' '.join(questions)
            for k in range(1, len(lengths)):
                lengths[k] += lengths[k - 1]
            lengths[-1] -= 1
            for j, example in enumerate(examples):
                example['db_id'] = dbid
                tokenization = tokenize(example['query'])
                # ; causes exeptions
                if tokenization[-1] == ';':
                    tokenization = tokenization[:-1]
                example['query_toks'] = tokenization
                example['query_toks_no_value'] = tokenization
                # example['question_toks'] = example.pop('utterance_toks')
                example['question'] = question_sequence[:lengths[j]] # slice the sequence to get current question and all previous.
                yield i, example
                i += 1