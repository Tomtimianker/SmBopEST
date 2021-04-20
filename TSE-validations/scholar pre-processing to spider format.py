import json
from dataset_readers.disamb_sql import tokenize

with open('Scholar files/scholar.json') as f:
    scholar_data = json.load(f)

scholar_in_spider_format_dev = []
scholar_in_spider_format_train = []
scholar_in_spider_format_test = []

for query_obj in scholar_data:
    for sent_obj in query_obj['sentences']:
        sentence = sent_obj['text']
        for sql_str in query_obj['sql']:
            query = sql_str
            for key, value in sent_obj['variables'].items():
                sentence = sentence.replace(key, value)
                query = query.replace(key, value)
            tok_sentence = tokenize(sentence)
            ques_spider_obj = {
                'db_id': 'scholar',
                'query': query,
                'query_toks': [],
                'query_toks_no_value': [],
                'question': sentence,
                'question_toks': tok_sentence,
                'sql': {} #Placeholer for a never-used object
            }

            if sent_obj['question-split'] == "dev":
                scholar_in_spider_format_dev.append(ques_spider_obj)
            elif sent_obj['question-split'] == "train":
                scholar_in_spider_format_train.append(ques_spider_obj)
            elif sent_obj['question-split'] == "test":
                scholar_in_spider_format_test.append(ques_spider_obj)

    def try_index(lst, search_term, begin):
        try:
            return lst.index(search_term, begin)
        except:
            return -1

scholar_in_spider_format_data_check = [
    {
        "query": "SELECT DISTINCT COUNT( DISTINCT WRITESalias0.PAPERID ) FROM AUTHOR AS AUTHORalias0 , KEYPHRASE AS KEYPHRASEalias0 , PAPERKEYPHRASE AS PAPERKEYPHRASEalias0 , WRITES AS WRITESalias0 WHERE AUTHORalias0.AUTHORNAME = \"authorname0\" AND KEYPHRASEalias0.KEYPHRASENAME LIKE \"keyphrasename0\" AND PAPERKEYPHRASEalias0.KEYPHRASEID = KEYPHRASEalias0.KEYPHRASEID AND WRITESalias0.PAPERID = PAPERKEYPHRASEalias0.PAPERID ;"
    }
]

def get_table_name(from_clause, index):
    return from_clause[index].split(" AS ")[1]

def get_tables_names(where_clause, index):
    phrases =  where_clause[index].split(" = ")
    if len(phrases) != 2:
        return -1
    return list(map(lambda exp: exp.split(".")[0] , phrases))

# Assume the list's length is 2
def is_two_phrases_in_list(phrase1, phrase2, lst):
    try:
        if len(lst) != 2:
            return -1
    except:
        print(f'len(lst) went wrong. lst is {lst}')
        return -1
    return ((lst[0] == phrase1 and lst[1] == phrase2) or (lst[0] == phrase2 and lst[1] == phrase1))

def get_suffix_and_where_clause(where_clause):
    where_clause = " AND ".join(where_clause).split(" ")
    group_by_index = try_index(where_clause, "GROUP", 0)
    having_count_index = try_index(where_clause, "HAVING", 0)

    if group_by_index == -1 and having_count_index == -1:
        return " ".join(where_clause).split(" AND "), ""

    elif group_by_index == -1 and having_count_index != -1:
        cut_index = having_count_index

    elif group_by_index != -1 and having_count_index == -1:
        cut_index = group_by_index

    else:
        cut_index = min(group_by_index, having_count_index)

    return (" ".join(where_clause[:cut_index])).split(" AND "), " ".join(where_clause[cut_index:])



def build_query(from_clause, where_clause):
    stop = False
    new_query = [" FROM"]
    not_used_from = from_clause.copy()
    used_from = []

    where_clause, suffix = get_suffix_and_where_clause(where_clause)

    not_used_where = where_clause.copy()
    used_where = []
    # get first couple
    for i in range(len(not_used_from)):
        i_from_name = get_table_name(not_used_from, i)
        for j in range(i + 1, len(not_used_from)):
            j_from_name = get_table_name(not_used_from, j)
            for k in range(len(not_used_where)):
                k_where_couple = get_tables_names(not_used_where, k)
                if k_where_couple == -1:
                    continue
                if is_two_phrases_in_list(i_from_name,j_from_name,k_where_couple):
                    # Creating the new query
                    new_query.append(not_used_from[i])
                    new_query.append("JOIN")
                    new_query.append(not_used_from[j])
                    new_query.append("ON")
                    where_to_query = not_used_where[k].split(" ")
                    if where_to_query[len(where_to_query) - 1] == ";":
                        where_to_query.remove(";")
                    new_query.append(" ".join(where_to_query))

                    # Adding for later
                    used_from.append(not_used_from[i])
                    used_from.append(not_used_from[j])
                    used_where.append(not_used_where[k])
                    stop = True
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    try:
        not_used_from.remove(used_from[0])
    except:
        print(f'not working: from: {from_clause}\n where: {where_clause}\n suffix: {suffix} ')
    not_used_from.remove(used_from[1])
    not_used_where.remove(used_where[0])


    while len(used_from) < len(from_clause):
        stop = False
        from_to_add = ""
        where_to_add = ""
        for i in range(len(not_used_from)):
            i_from_name = get_table_name(not_used_from, i)
            from_to_add = not_used_from[i]
            for j in range(len(used_from)):
                j_from_name = get_table_name(used_from, j)
                for k in range(len(not_used_where)):
                    k_where_couple = get_tables_names(not_used_where, k)
                    where_to_add = not_used_where[k]
                    if k_where_couple == -1:
                        continue
                    if is_two_phrases_in_list(i_from_name, j_from_name, k_where_couple):
                        # Creating the new query
                        new_query.append("JOIN")
                        new_query.append(not_used_from[i])
                        new_query.append("ON")
                        where_to_query = not_used_where[k].split(" ")
                        if where_to_query[len(where_to_query) - 1] == ";":
                            where_to_query.remove(";")
                        new_query.append(" ".join(where_to_query))

                        # Adding for later
                        used_from.append(not_used_from[i])
                        used_where.append(not_used_where[k])
                        stop = True
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if len(not_used_from) > 0:
            not_used_from.remove(from_to_add)
        if where_to_add != '' and len(not_used_where) > 0:
            not_used_where.remove(where_to_add)

    if len(not_used_where) > 0:
        where_clause = " AND ".join(not_used_where)
        new_query.append("WHERE")
        new_query.append(where_clause)
    new_query.append(suffix)
    return " ".join(new_query)

def remove_as_from_query(query):
    as_indices = [i for i, w in enumerate(query) if w == "AS"]
    print(query)
    if len(as_indices) == 0:
        return " ".join(query)

    for as_index in as_indices:
        alias = query[as_index + 1]
        name = query[as_index - 1]
        query_no_alias = []
        for w in query:
            if w == alias:
                query_no_alias.append(name)
            elif w.split(".")[0] == alias:
                tmp = name + "." + w.split(".")[1]
                query_no_alias.append(tmp)
            else:
                query_no_alias.append(w)
        query = query_no_alias

    as_indices_to_remove = []
    for as_index in as_indices:
        as_indices_to_remove.append(as_index)
        as_indices_to_remove.append(as_index + 1)

    query = [w for i, w in enumerate(query) if i not in as_indices_to_remove]

    return " ".join(query)

# Adding corrections for scholar_in_spider_format_data:

def process_data(scholar_in_spider_format_data):
    for qq_obj in scholar_in_spider_format_data:
        tmp_query = qq_obj['query'].split(" ")
        from_index = try_index(tmp_query, "FROM", 0)
        if from_index < 0:
            print(f"from_index < 0  for {qq_obj['query']}")
            break
        where_after_from_index = try_index(tmp_query, "WHERE", from_index)
        from_clause = " ".join(tmp_query[from_index+1:where_after_from_index]).split(" , ")
        if len(from_clause) == 1:
            continue
        where_clause = " ".join(tmp_query[where_after_from_index+1:len(tmp_query)]).split(" AND ")
        check = build_query(from_clause, where_clause)
        new_query = " ".join(tmp_query[:from_index]) + build_query(from_clause, where_clause)
        qq_obj['query'] = new_query

    for qq_obj in scholar_in_spider_format_data:
        no_as_query = remove_as_from_query(qq_obj['query'].split(" "))
        new_query = []
        query_for_tokenize = []

        for word in no_as_query.split(" "):
            #             if word == "AS":
            #                 as_before = True
            #                 continue
            #             if as_before:
            #                 as_before = False
            #                 continue
            if "." in word:
                new_query.append(word)
                query_for_tokenize.append(word.split(".")[0])
                query_for_tokenize.append(".")
                query_for_tokenize.append(word.split(".")[1])
                continue
            query_for_tokenize.append(word)
            new_query.append(word)
        new_query = " ".join(new_query)
        query_for_tokenize = " ".join(query_for_tokenize)
        qq_obj['query']  = new_query
        tok_query = tokenize(query_for_tokenize)
        tok_query = ["value" if token.startswith("\"") else token for token in tok_query]
        if tok_query[len(tok_query) - 1] == ";":
            tok_query.pop()
        qq_obj['query_toks']  =  tok_query
        qq_obj['query_toks_no_value']  =  tok_query


process_data(scholar_in_spider_format_dev)
process_data(scholar_in_spider_format_train)
process_data(scholar_in_spider_format_test)

with open('scholar_in_spider_format_dev3.json', 'w') as outfile:
    json.dump(scholar_in_spider_format_dev, outfile)

with open('scholar_in_spider_format_train3.json', 'w') as outfile:
    json.dump(scholar_in_spider_format_train, outfile)

with open('scholar_in_spider_format_test3.json', 'w') as outfile:
    json.dump(scholar_in_spider_format_test, outfile)