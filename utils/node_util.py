import anytree
from functools import reduce
from utils.moz_sql_parser import parse
from tqdm import tqdm
from itertools import *
from anytree import NodeMixin, RenderTree, PostOrderIter, Node
import torch
from hashlib import blake2b
import struct
import copy
from anytree.search import *

import re 

def is_number(x):
    '''
        Takes a word and checks if Number (Integer or Float).
    '''
    try:
        # only integers and float converts safely
        num = float(x)
        return True
    except: # not convertable to float
        return False

is_field = (
    lambda node: hasattr(node, "val")
    and node.name == "Value" and not is_number(node.val)
)
else_dict = {
    "Selection": " WHERE ",
    "Groupby": " GROUP BY ",
    "Limit": " LIMIT ",
    "Having": " HAVING ",
}

pred_dict = {
    "eq": " = ",
    "like": " LIKE ",
    "nin": " NOT IN ",
    "lte": " <= ",
    "lt": "<",
    "neq": " != ",
    "in": " IN ",
    "gte": " >= ",
    "gt": " > ",
    "And": " AND ",
    "Or": " OR ",
    "except": " EXCEPT ",
    "union": " UNION ",
    "intersect": " INTERSECT ",
    "Val_list": " , ",
    "Product": " , ",
}

createTable = lambda x: {"table": x}
reduce_and = lambda l: reduce(lambda a, b: {"and": [a, b]}, l)
reduce_or = lambda l: reduce(lambda a, b: {"or": [a, b]}, l)
wrap_and = lambda x: [Node("And", children=x)] if len(x) > 1 else x

remove_semicolon = lambda s: s[:-1] if s.endswith(";") else s
string_in = lambda s, x: x in s.lower()


# def dethash(key):
#     digest = blake2b(key=key.encode(), digest_size=4).digest()
#     return struct.unpack("I", digest)[0]


def transform(x):
    x = str(x).lower().replace("%","").replace(" ,","").replace(",","").strip()
    if len(x)>1 and x[-1]=='s':
        x=x[:-1]
    return x

def dethash(key):
    my_hash = blake2b(key="hi".encode(), digest_size=4)
    my_hash.update(transform(key).encode())
    digest = my_hash.digest()
    return struct.unpack("I", digest)[0]

class HashableNode(Node):
    def __hash__(self):
        nary_self = travel_tree(copy.deepcopy(self))
        return hash(self.tuplize(nary_self))

    def __eq__(self, other):
        nary_self = travel_tree(copy.deepcopy(self))
        nary_other = travel_tree(copy.deepcopy(other))
        return self.tuplize(nary_self) == self.tuplize(nary_other)

    @classmethod
    def tuplize(cls, x):
        res = x.name
        if hasattr(x, "val"):
            sec = str(x.val)
        else:
            if x.name in ["And", "Or", "Val_list", "Product", "eq", "neq", "union"]:
                applier = frozenset
            else:
                applier = tuple
            sec = applier(cls.tuplize(child) for child in x.children)
        return tuple([res, sec])

    @classmethod
    def trust_equal(cls, nary_obj, nary_other):
        # nary_obj = travel_tree(copy.deepcopy(obj))
        # nary_other = travel_tree(copy.deepcopy(other))
        return cls.tuplize(nary_obj) == cls.tuplize(nary_other)

    @classmethod
    def trust_hash(cls, nary_obj):
        # nary_obj = travel_tree(copy.deepcopy(obj))
        return hash(cls.tuplize(nary_obj))

    def __len__(self):
        return 1 + sum(len(x) for x in self.children)

    def get_desc(self):
        desc = [self]
        #         desc=[]
        for child in self.children:
            gran = child.get_desc()
            #             desc.append(child)
            desc.extend(gran)
        return desc

    def is_leaf(self):
        return len(self.children) == 0


def pad_with_keep(node, i):
    root = node
    prev = None
    for k in range(i):
        curr = Node("keep", parent=prev, max_depth=node.max_depth)
        if k == 0:
            root = curr
        prev = curr
    node.parent = prev
    return root


def add_max_depth_att(node):
    if not node.children:
        node.max_depth = node.depth
    else:
        node.children = [add_max_depth_att(child) for child in node.children]
        node.max_depth = max([child.max_depth for child in node.children])
    assert hasattr(node, "max_depth")
    return node


def tree2maxdepth(tree):
    if tree.parent:
        tree = pad_with_keep(tree, tree.parent.max_depth - tree.max_depth)
    if tree.children:
        tree.children = [tree2maxdepth(child) for child in tree.children]
    return tree

    # def append_child(self, child):
    #     prev_child = self.children
    #     if not isinstance(child, ActionTree):
    #         child = ActionTree(child)
    #     #         child.id = len(prev_child)
    #     self.children = list(prev_child) + [child]
    #
    # def insert_child(self, child):
    #     prev_child = self.children
    #     if not isinstance(child, ActionTree):
    #         child = ActionTree(child)
    #     #         child.id = len(prev_child)
    #     self.children = [child] + list(prev_child)

    # @classmethod
    # def dict2tree(cls, in_dict):
    #     if isinstance(in_dict, dict):
    #         key = list(in_dict.keys())[0]
    #         children = []
    #         for child in in_dict[key]:
    #             res = cls.dict2tree(child)
    #             if isinstance(res, list):
    #                 children.extend(res)
    #             else:
    #                 children.append(res)
    #
    #         return ActionTree(key, children)
    #     elif isinstance(in_dict, list):
    #         if len(in_dict) == 2:
    #             return [ActionTree(in_dict[0]), ActionTree(in_dict[1])]
    #         else:
    #             return ActionTree(in_dict[0])
    #     else:
    #         return ActionTree(in_dict)


class Hasher:
    def __init__(self, device):
        self.device = device
        self.tensor1 = torch.LongTensor([402653189]).to(device)
        self.tensor2 = torch.LongTensor([3644798167]).to(device)
        self.tensor3 = torch.LongTensor([28]).to(device)
        self.tensor4 = torch.LongTensor([1]).to(device)
        self.tensor5 = torch.LongTensor([56]).to(device)

    def set_hash(self, h_list, _h=None, _hash=None):
        flag = False
        if _hash is None or _h is None:
            flag = True
            _hash = torch.tensor([1], dtype=torch.long).to(self.device)
            _h = torch.tensor([1], dtype=torch.long).to(self.device)
            h_list = torch.tensor(h_list, dtype=torch.long).to(self.device)

        if len(h_list) == 3:
            parent, a, b = h_list
        else:
            parent, a = h_list
            b = a
        _hash.copy_(a)
        _h.copy_(b)
        _hash <<= 28
        _h >>= 1
        _hash = _hash.add_(_h)
        parent <<= 56
        _hash = _hash.add_(parent)
        _hash *= self.tensor2
        _hash = _hash.fmod(self.tensor1)
        if flag:
            return int(_hash[0])
        return _hash

    # TODO: change name to hashify, add as a instance method of the new Node class
    def add_hash_att(self, node, type_dict):
        try:
            if not node.children:
                if isinstance(node.val, dict):
                    node.hash = self.set_hash([type_dict[node.name], dethash("value")])
                else:
                    node.hash = self.set_hash([type_dict[node.name], dethash(str(node.val))])
            else:
                node.children = [
                    self.add_hash_att(child, type_dict) for child in node.children
                ]
                if node.name == "keep":
                    node.hash = node.children[0].hash
                else:
                    node.hash = self.set_hash(
                        [type_dict[node.name]] + [child.hash for child in node.children]
                    )
        except Exception as e:
            print(print_tree(node))
            raise Exception
        assert hasattr(node, "hash")
        return node

def reconstruct_tuple(op_names,binary_op_count,batch_el,idx,items,cnt):
    type_data = int(items[cnt].curr_type[batch_el][idx])
    tuple_el = [op_names[type_data]]
    if cnt>0:
        if type_data<binary_op_count:
            l_idx = items[cnt].l_child_idx[batch_el][idx]
            r_idx = items[cnt].r_child_idx[batch_el][idx]

            l_child = helper(batch_el,l_idx,items,cnt-1)
            r_child = helper(batch_el,r_idx,items,cnt-1)
            tuple_el.append([l_child,r_child])
        else:
            idx = items[cnt].l_child_idx[batch_el][idx]
            child = helper(batch_el,idx,items,cnt-1)
            tuple_el.append([child])
            
    else:
        tuple_el.append([items[cnt].l_child_idx[batch_el][idx]])
    return tuple_el

def fix_between(inp):
    inp = re.sub(r"([\s|\S]+) >= (\d*) AND \1 <= (\d*)", r"\1 BETWEEN \2 and \3", inp)
    inp = re.sub(r"LIKE '([\s|\S]+?)'", r"LIKE '%\1%'", inp)
    return inp

def reconstruct_tree(op_names, binary_op_count, batch_el, idx, items, cnt, num_schema_leafs, chosen_leaf_mask = None):
    type_data = int(items[cnt].curr_type[batch_el][idx])
    # tuple_el = [op_names[type_data]]
    tuple_el = Node(op_names[type_data])
    if cnt>0:
        if type_data<binary_op_count:
            l_idx = items[cnt].l_child_idx[batch_el][idx]
            r_idx = items[cnt].r_child_idx[batch_el][idx]

            l_child = reconstruct_tree(op_names,binary_op_count,batch_el,l_idx,items,cnt-1,num_schema_leafs, chosen_leaf_mask)
            r_child = reconstruct_tree(op_names,binary_op_count,batch_el,r_idx,items,cnt-1,num_schema_leafs, chosen_leaf_mask)
            # tuple_el.append([l_child,r_child])
            tuple_el.children = [l_child,r_child]
        else:
            idx = items[cnt].l_child_idx[batch_el][idx]
            child = reconstruct_tree(op_names,binary_op_count,batch_el,idx,items,cnt-1,num_schema_leafs, chosen_leaf_mask)
            # tuple_el.append([child])
            tuple_el.children = [child]
    else: 
        if idx < num_schema_leafs:
            entities = items[cnt].entities[batch_el]
            entity_idx = items[cnt].final_leaf_indices[batch_el][idx]
            #TREECOPY
            # Mark that the specific schema leaf was chosen.
            if chosen_leaf_mask is not None:
                chosen_leaf_mask[entity_idx] = 1
            tuple_el.val = entities[entity_idx]
        else:
            span_idx = idx-num_schema_leafs
            enc_tokens = items[cnt].enc["tokens"]['token_ids'][batch_el][1:].tolist()
            start_id = items[cnt].span_start_indices[batch_el][span_idx]
            end_id = items[cnt].span_end_indices[batch_el][span_idx]
            tuple_el.val = items[cnt].tokenizer.decode(enc_tokens[start_id:end_id+1]).strip()
    return tuple_el

# TODO: class method of the new node class
def tuple2tree(in_dict):
    if not in_dict is None:
        if in_dict[0] not in ["Table", "Value"]:
            # if in_dict[1]
            key = in_dict[0]
            children = []

            for child in in_dict[1]:
                res = tuple2tree(child)
                children.append(res)
            return Node(key, children=children)
        else:
            return Node(in_dict[0], val=in_dict[1])
    else:
        return Node("None")


# TODO: class method of the new node class
def subtrees(tree):
    desc = [copy.deepcopy(tree)]
    original_children = tree.children
    for node_set in powerset(original_children):
        if 1 < len(node_set) < len(original_children):
            tree.children = node_set
            desc.append(copy.deepcopy(tree))
    for child in original_children:
        desc.extend(subtrees(child))
    tree.children = original_children
    return desc


# TODO: move to iter utils
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# TODO: move to a method of the new Node
def get_leafs(tree):
    res = []
    # for y in findall(tree, filter_=lambda x: len(x.children) == 0):
    for y in findall(tree, filter_=lambda x: hasattr(x, "val")):
        if isinstance(y.val, dict):
            if "value" in y.val:
                s = y.val["value"]
            elif "literal" in y.val:
                s = y.val["literal"]
            else:
                print(y.val)
                print(y)
                print(print_tree(tree))
                raise Exception
        elif isinstance(y.val, str):
            s = y.val
        elif isinstance(y.val, int) or isinstance(y.val, float):
            s = y.val  # TODO: fixme
            # s = str(y.val)
        else:
            print(y.val)
            print(y)
            print(y.parent.name)
            # print
            # y.parent.name
            raise Exception
        res.append(s)

    return res


# TODO: move to a __repr__ or str of the new node class
def print_tree(root, print_hash=True):
    tree = [
        f"{pre}{node.name} {node.val if hasattr(node, 'val') else ''} {node.hash if hasattr(node, 'hash') and print_hash  else ''}"
        for pre, fill, node in anytree.RenderTree(root)
    ]
    return "\n".join(tree)
 

def remove_keep(node):
    node.children = [remove_keep(child) for child in node.children]
    child_list = []
    for child in node.children:
        if child.name == "keep":
            child_list.append(child.children[0])
        else:
            child_list.append(child)
    node.children = child_list
    return node


def promote(node, root=False):
    #     if node.name in ['Project', 'Selection', 'Orderby_desc', 'Orderby_asc','Groupby','Limit','Having']:
    children = node.children
    if node.name in ["Having"]:
        while True:
            if not node.is_root and node.parent.name not in [
                "union",
                "intersect",
                "Subquery",
                "except",
            ]:
                prev_parent = node.parent
                grandparent = (
                    prev_parent.parent if not prev_parent.is_root else prev_parent
                )
                #         if not grandparent.is_root:
                node.parent = grandparent
            else:
                break
        node.siblings[0].parent = node
    for child in children:
        promote(child)


def parse_list_from_sql(sql_list):
    parse_list = []
    for i, s in tqdm(enumerate(sql_list)):
        # for i,s in tqdm(enumerate(df.gold[:30])):
        s = s.replace('"', "'")
        if s.endswith(";"):
            s = s[:-1]
        try:
            res = parse(s)
        except Exception as e:
            res = ""
        parse_list.append(res)
    return parse_list


def tree_list_from_parse_list(parse_list):
    tree_list = []

    for sql_tree in tqdm(parse_list):
        if isinstance(sql_tree, dict):
            p = []
            try:
                p = sql_tree["query"]
                p = get_tree(p, None)
                build_sql(p)
            except Exception as e:
                pass

            tree_list.append(p)

        else:
            tree_list.append([])
    return tree_list


def proccess_from(inp, args):
    tables = []
    on_list = []
    if isinstance(inp, str) or isinstance(inp, dict):
        tables, on_list = [createTable(inp)], []
    else:

        for i in inp:
            if isinstance(i, dict):
                if i.get("join"):
                    tables.append(createTable(i.get("join")))
                    # i.get('on')
                    # print(i)
                    if not i.get("on"):
                        pass
                    elif i.get("on").get("and"):
                        on_list += i.get("on").get("and")
                    else:
                        on_list.append(i.get("on"))

                else:
                    tables.append(createTable(i))

            else:
                tables.append(createTable(i))
    # TODO: add parameter to change into list
    tables = table_recurse(reduce(lambda a, b: {"product": [a, b]}, tables), args)
    return tables, on_list


def proccess_select(in_dict, args):
    if isinstance(in_dict, str) and in_dict == "*":
        return Node("Value", val="*", n_type="Value")
    if isinstance(in_dict, dict):
        #         return Node('Value',val=in_dict['value'])
        return agg_check(in_dict["value"])

    if isinstance(in_dict, list):
        # TODO: add parameter to change into list
        in_dict = reduce(lambda a, b: {"val_list": [a, b]}, in_dict)
        return select_recurse(in_dict)


def proccess_where(where_list, on_list, having_list, args):
    if on_list:
        on_list = reduce_and(on_list)
    where_list = cnf_recurse(where_list)
    having_list = cnf_recurse(having_list)
    res = []

    res += [on_list] if on_list else []
    res += [having_list] if having_list else []
    res += [where_list] if where_list else []

    if res:
        # TODO: add parameter to change into list
        res = reduce_and(res)
        res = cnf_recurse(res)
        res = node_recurse(res, args)
    return res


def table_recurse(in_dict, args):
    if in_dict.get("product"):
        parent = Node("Product", n_type="Table")
        table_recurse(in_dict["product"][0], args).parent = parent
        table_recurse(in_dict["product"][1], args).parent = parent
        return parent
    else:
        parent = Node("Table", n_type="Table")
        if (
            isinstance(in_dict["table"], dict)
            and isinstance(in_dict["table"].get("value"), dict)
            and in_dict["table"]["value"].get("query")
        ):
            res = get_tree(in_dict["table"]["value"]["query"], args)
            res.parent = Node("Subquery", parent=parent, n_type="Table")
        else:
            res = in_dict["table"]
            parent.val = res

        return parent


def select_recurse(in_dict):
    if in_dict.get("val_list"):
        parent = Node("Val_list", n_type="Value")
        select_recurse(in_dict["val_list"][0]).parent = parent
        select_recurse(in_dict["val_list"][1]).parent = parent
        return parent
    else:
        return agg_check(in_dict["value"])


def cnf_recurse(in_dict):
    if isinstance(in_dict, dict):
        if in_dict.get("and"):
            # TODO: add parameter to change into list
            return reduce_and([cnf_recurse(el) for el in in_dict["and"]])
        elif in_dict.get("or"):
            return reduce_or([cnf_recurse(el) for el in in_dict["or"]])
        return in_dict
    else:
        return in_dict


def node_recurse(in_dict, args):
    if in_dict.get("and"):
        parent = Node("And", n_type="Predicate")
        node_recurse(in_dict["and"][0], args).parent = parent
        node_recurse(in_dict["and"][1], args).parent = parent
        return parent
    elif in_dict.get("or"):
        parent = Node("Or", n_type="Predicate")
        node_recurse(in_dict["or"][0], args).parent = parent
        node_recurse(in_dict["or"][1], args).parent = parent
        return parent
    predicate_type = list(in_dict.keys())[0]
    predicate_node = Node(predicate_type, n_type="Predicate")
    if len(in_dict[predicate_type]) == 2:
        val1, val2 = in_dict[predicate_type]
        handle_subquery(val1, args).parent = predicate_node
        handle_subquery(val2, args).parent = predicate_node
    else:
        assert predicate_type == "between"
        val0, val1, val2 = in_dict[predicate_type]
        # print(in_dict[predicate_type])
        predicate_node = Node("And", n_type="Predicate")
        pred1 = Node("gte", parent=predicate_node, n_type="Predicate")
        pred2 = Node("lte", parent=predicate_node, n_type="Predicate")
        agg_check(val0).parent = pred1
        handle_subquery(val1, args).parent = pred1
        agg_check(val0).parent = pred2
        handle_subquery(val2, args).parent = pred2
    return predicate_node


def handle_subquery(in_dict, args):
    if isinstance(in_dict, dict) and in_dict.get("query"):
        query = get_tree(in_dict["query"], args)
        curr = Node("Subquery", n_type="Table")
        query.parent = curr
        return curr
    else:
        return agg_check(in_dict)


def agg_check(in_dict):
    node = Node("Value", n_type="Value")
    if (
        isinstance(in_dict, str)
        or isinstance(in_dict, int)
        or isinstance(in_dict, float)
    ):
        node.val = in_dict
        return node
    if isinstance(in_dict, dict):
        agg_type = list(in_dict.keys())[0]
        agg_type_node = Node(agg_type, n_type="Agg")

        if isinstance(in_dict[agg_type], dict):
            sec_agg_type = list(in_dict[agg_type].keys())[0]
            if sec_agg_type in ["distinct", "count"]:
                distinct_type_node = Node(
                    sec_agg_type, parent=agg_type_node, n_type="Agg"
                )
                node.val = in_dict[agg_type][sec_agg_type]
                node.parent = distinct_type_node
            elif sec_agg_type in ["add", "sub", "div", "mul", "eq", "like", "nlike", "nin", "lte", "lt", "neq", "in", "gte", "gt"]:
                sec_agg_node = Node(sec_agg_type, parent=agg_type_node, n_type="Agg")
                val1, val2 = in_dict[agg_type][sec_agg_type]
                agg_check(val1).parent = sec_agg_node
                agg_check(val2).parent = sec_agg_node
            else:
                raise Exception
        else:
            node.val = in_dict[agg_type]
            node.parent = agg_type_node

        return agg_type_node
    else:
        print(in_dict)


def travel_tree(in_node):
    if in_node.name in ["And", "Or", "Val_list", "Product"]:
        return new_travel_tree(in_node, in_node.name, True)
    else:
        children_list = []
        for child in in_node.children:
            child.parent = None
            child = travel_tree(child)
            children_list.append(child)
        in_node.children = children_list
        return in_node


def new_travel_tree(in_node, n_type, is_root=False):
    other_op = "And" if n_type == "Or" else "Or"
    if in_node.name == n_type:
        res = []
        for child in in_node.children:
            child.parent = None
            res += new_travel_tree(child, n_type)
        if is_root:
            in_node.children = res
            return in_node
        else:
            return res
    elif in_node.name == other_op:
        return [new_travel_tree(in_node, other_op, True)]
    else:
        if not is_root:
            children_list = []
            for child in in_node.children:
                child.parent = None
                child = travel_tree(child)
                children_list.append(child)
            in_node.children = children_list
        return [in_node]

def remove_aliases(tree):
    rename_dict ={}
    for node in findall(tree, filter_=lambda x: x.name == "Table"):
        if not hasattr(node,"val"):
            continue
        if isinstance(node.val,dict):
            d = node.val
            rename_dict[d['name']] =d['value']
            node.val=d['value']
    for node in findall(tree, filter_=lambda x: hasattr(x,"val")):
        for alias,table_name in rename_dict.items():
            if isinstance(node.val,str):
                node.val = node.val.replace(alias,table_name)
    return tree

# def is_nonstar_nondot_column(leaf):
#     # assert hasattr(leaf,"val")
#     if isinstance(leaf.val,int) or isinstance(leaf.val,float):
#         return False
#     elif leaf.parent is not None and leaf.parent.name=="literal":
#         return False
#     elif leaf.val=="*":
#         return False
#     elif "." in leaf.val:
#         return False
#     elif leaf.name=="Table":
#         return False
#     return True

# def add_back_tablenames(tree):
#     tree_tables = list(findall(tree, filter_=lambda x: x.name == "Table"))
#     if len(tree_tables)==1:
#         table_name = tree_tables[0].val
#         for leaf in findall(tree, filter_=lambda x: hasattr(x,"val")):
#             if is_nonstar_nondot_column(leaf):
#                 leaf.val = f"{table_name}.{leaf.val}"
#     return tree

#ir_to_ra
def get_tree(p, args=None):
    if p.get("op"):
        res1 = get_tree(p["op"]["query1"], args)
        
        res2 = get_tree(p["op"]["query2"], args)
        c = Node(p["op"].get("type"), n_type="Op")
        parent1 = Node("Subquery", parent=c, n_type="Table")
        parent2 = Node("Subquery", parent=c, n_type="Table")
        res1.parent = parent1
        res2.parent = parent2
        return c

    root = Node("Project", n_type="Table")
    select_node = proccess_select(p["select"], args)
    select_node.parent = root
    node = root

    tables, on_list = proccess_from(p["from"], args)
    where_list = p.get("where")
    having_list = p.get("having")
    condition = proccess_where(where_list, on_list, having_list, args)
    if condition:
        node = Node("Selection", parent=node, n_type="Table")
        condition.parent = node

    tables.parent = node
    node = tables
    if p.get("groupby"):
        curr = Node("Groupby", n_type="Table")
        proccess_select(p["groupby"], args).parent = curr
        root.parent = curr
        root = root.parent
    if p.get("orderby"):
        if isinstance(p["orderby"], dict) and p["orderby"].get("sort"):
            sort = "Orderby_" + p["orderby"]["sort"]
        else:
            sort = "Orderby_asc"
        curr = Node(sort, n_type="Table")
        proccess_select(p["orderby"], args).parent = curr
        root.parent = curr
        root = root.parent

    if p.get("limit"):
        curr = Node("Limit", n_type="Table")
        val = p["limit"]
        if isinstance(val, dict):
            val = val["literal"]
        Node("Value", val=val, n_type="Value").parent = curr
        root.parent = curr
        root = root.parent
    # root = remove_aliases(root)
    # root = add_back_tablenames(root)
    return root


#ra_to_sql
def build_sql(tree, peren=True):
    if len(tree.children) == 0:
        # assert tree.name in ['Value', 'Table']
        if tree.name == "Table" and isinstance(tree.val, dict):
            return tree.val["value"] + " AS " + tree.val["name"]
        if hasattr(tree, "val"):
            return str(tree.val)
        else:
            print(tree)
            return ""
    if len(tree.children) == 1:
        # assert tree.name in ['Predicate', 'Subquery', 'Table', 'min', 'count', 'literal', 'max', 'avg', 'sum', 'Agg',
        #                      'distinct', 'Where', 'Join', 'Selection', 'Val_list', 'Value'], tree.name
        if tree.name in [
            "min",
            "count",
            "max",
            "avg",
            "sum",
        ]:
            return "".join([tree.name.upper(), "( ", build_sql(tree.children[0]), " )"])
        elif tree.name == "distinct":
            return "DISTINCT " + build_sql(tree.children[0])
        elif tree.name == "literal":
            return """\'""" + str(build_sql(tree.children[0])) + """\'"""
        elif tree.name == "Subquery":
            if peren:
                return "".join(["(", build_sql(tree.children[0]), ")"])
            else:
                return build_sql(tree.children[0])
        elif tree.name == "Join_on":
            tree = tree.children[0]
            if tree.name == "eq":
                first_table_name = tree.children[0].val.split(".")[0]
                second_table_name = tree.children[1].val.split(".")[0]
                return f"{first_table_name} JOIN {second_table_name} ON {tree.children[0].val} = {tree.children[1].val}"
            else:
                # print(tree)
                if len(tree.children) > 0:
                    # try:
                    # t_Res = [child.val for child in tree.children]
                    # except:
                    # print([child for child in tree.children])
                    t_Res = ", ".join([child.val for child in tree.children])
                    return t_Res
                else:
                    return tree.val
        else:  # Predicate or Table or 'literal' or Agg
            return build_sql(tree.children[0])
    else:
        # assert tree.name in ['Project', 'Selection', 'Val_list', 'Orderby_desc', 'Orderby_asc', 'Product', 'Limit',
        #                      'Groupby', 'except', 'union',
        #                      'intersect', 'Or', 'And', 'eq', 'like', 'nin', 'lte', 'lt', 'neq', 'in', 'gte', 'gt',
        #                      'Having'], tree.name
        if tree.name in [
            "eq",
            "like",
            "nin",
            "lte",
            "lt",
            "neq",
            "in",
            "gte",
            "gt",
            "And",
            "Or",
            "except",
            "union",
            "intersect",
            "Product",
            "Val_list",
        ]:
            # return build_sql(tree.children[0])+pred_dict[tree.name].upper()+build_sql(tree.children[1])
            pren_t = tree.name in [
                "eq",
                "like",
                "nin",
                "lte",
                "lt",
                "neq",
                "in",
                "gte",
                "gt",
            ]
            return (
                pred_dict[tree.name]
                .upper()
                .join([build_sql(child, pren_t) for child in tree.children])
            )
        elif tree.name == "Orderby_desc":
            return (
                build_sql(tree.children[1])
                + " ORDER BY "
                + build_sql(tree.children[0])
                + " DESC"
            )
        elif tree.name == "Orderby_asc":
            return (
                build_sql(tree.children[1])
                + " ORDER BY "
                + build_sql(tree.children[0])
                + " ASC"
            )
        elif tree.name == "Project":
            return (
                "SELECT "
                + build_sql(tree.children[0])
                + " FROM "
                + build_sql(tree.children[1])
            )
        # elif tree.name == 'Join_on':
        #     tree = tree.children[0]
        #     first_table_name = tree.children[0].val.split('.')[0]
        #     second_table_name = tree.children[1].val.split('.')[0]
        #     return f"{first_table_name} JOIN {second_table_name} ON {tree.children[0].val} = {tree.children[1].val}"
        elif tree.name == "Join_on":
            # tree
            table_name = lambda x: x.val.split(".")[0]
            table_tups = [
                (table_name(child.children[0]), table_name(child.children[1]))
                for child in tree.children
            ]
            res = table_tups[0][0]
            seen_tables = set(res)
            for (first, sec), child in zip(table_tups, tree.children):
                tab = first if sec in seen_tables else sec
                res += (
                    f" JOIN {tab} ON {child.children[0].val} = {child.children[1].val}"
                )
                seen_tables.add(tab)
            # print(res)

            return res
        elif tree.name == "Selection":
            if len(tree.children) == 1:
                return build_sql(tree.children[0])
            return build_sql(tree.children[1]) + " WHERE " + build_sql(tree.children[0])
        else:  # 'Selection'/'Groupby'/'Limit'/Having'
            return (
                build_sql(tree.children[1])
                + else_dict[tree.name]
                + build_sql(tree.children[0])
            )


def print_sql(tree):
    if tree:
        flat_tree = travel_tree(copy.deepcopy(tree))
        for node in findall(flat_tree, filter_=lambda x: x.name == "Selection"):
            table_node = node.children[1]
            join_list = []
            where_list = []
            having_list = []
            if node.children[0].name == "And":
                for predicate in node.children[0].children:
                    if (
                        all(is_field(child) for child in predicate.children)
                        and predicate.name == "eq"
                    ):
                        join_list.append(predicate)
                    else:
                        if predicate.name == "Or" or all(
                            child.name in ["literal", "Subquery", "Value", "Or"]
                            for child in predicate.children
                        ):
                            where_list.append(predicate)
                        else:
                            having_list.append(predicate)
                    predicate.parent = None
            else:
                if node.children[0].name == "eq" and all(
                    is_field(child) for child in node.children[0].children
                ):
                    join_list = [node.children[0]]
                elif node.children[0].name == "Or":
                    where_list = [node.children[0]]
                else:
                    if all(
                        child.name in ["literal", "Subquery", "Value", "Or"]
                        for child in node.children[0].children
                    ):
                        where_list = [node.children[0]]
                    else:
                        having_list = [node.children[0]]
                node.children[0].parent = None
            having_node = (
                [Node("Having", children=wrap_and(having_list))] if having_list else []
            )
            join_on = Node("Join_on", children=join_list)
            if len(join_on.children) == 0:
                join_on.children = [table_node]
            node.children = having_node + wrap_and(where_list) + [join_on]
        flat_tree = Node("Subquery", children=[flat_tree])
        promote(flat_tree)
        return build_sql(flat_tree.children[0])
    else:
        return ""
RULES_novalues = """[["And", ["And", "like"]], ["And", ["eq", "gt"]], ["And", ["And", "neq"]], ["And", ["eq", "nin"]], ["And", ["eq", "gte"]], ["And", ["eq", "in"]], ["And", ["lt", "neq"]], ["And", ["eq", "eq"]], ["And", ["eq", "And"]], ["And", ["gt", "eq"]], ["And", ["gte", "gte"]], ["And", ["eq", "neq"]], ["And", ["And", "gte"]], ["And", ["like", "neq"]], ["And", ["gt", "lt"]], ["And", ["eq", "like"]], ["And", ["gt", "nin"]], ["And", ["gt", "lte"]], ["And", ["And", "gt"]], ["And", ["gt", "gt"]], ["And", ["eq", "Or"]], ["And", ["gte", "gt"]], ["And", ["eq", "lte"]], ["And", ["lt", "eq"]], ["And", ["gt", "in"]], ["And", ["eq", "lt"]], ["And", ["And", "Or"]], ["And", ["in", "in"]], ["And", ["gt", "gte"]], ["And", ["gte", "lte"]], ["And", ["gt", "neq"]], ["And", ["And", "And"]], ["And", ["And", "eq"]], ["And", ["gte", "eq"]], ["And", ["And", "lt"]], ["Groupby", ["Val_list", "Project"]], ["Groupby", ["Value", "Project"]], ["Limit", ["Value", "Orderby_desc"]], ["Limit", ["Value", "Orderby_asc"]], ["Or", ["neq", "neq"]], ["Or", ["gt", "gt"]], ["Or", ["eq", "lt"]], ["Or", ["like", "like"]], ["Or", ["gt", "eq"]], ["Or", ["eq", "eq"]], ["Or", ["gte", "gte"]], ["Or", ["lt", "gt"]], ["Or", ["eq", "gt"]], ["Or", ["gt", "lt"]], ["Orderby_asc", ["sum", "Groupby"]], ["Orderby_asc", ["Value", "Project"]], ["Orderby_asc", ["Value", "Groupby"]], ["Orderby_asc", ["Val_list", "Project"]], ["Orderby_asc", ["avg", "Groupby"]], ["Orderby_asc", ["count", "Groupby"]], ["Orderby_desc", ["avg", "Groupby"]], ["Orderby_desc", ["Value", "Project"]], ["Orderby_desc", ["count", "Groupby"]], ["Orderby_desc", ["sum", "Groupby"]], ["Orderby_desc", ["max", "Groupby"]], ["Orderby_desc", ["Value", "Groupby"]], ["Product", ["Product", "Table"]], ["Product", ["Table", "Table"]], ["Project", ["Val_list", "Product"]], ["Project", ["distinct", "Table"]], ["Project", ["min", "Table"]], ["Project", ["Val_list", "Table"]], ["Project", ["min", "Selection"]], ["Project", ["Val_list", "Selection"]], ["Project", ["count", "Selection"]], ["Project", ["sum", "Table"]], ["Project", ["avg", "Table"]], ["Project", ["Value", "Selection"]], ["Project", ["max", "Table"]], ["Project", ["max", "Selection"]], ["Project", ["Value", "Table"]], ["Project", ["count", "Table"]], ["Project", ["avg", "Selection"]], ["Project", ["sum", "Selection"]], ["Project", ["distinct", "Selection"]], ["Selection", ["And", "Product"]], ["Selection", ["Or", "Table"]], ["Selection", ["lte", "Table"]], ["Selection", ["neq", "Table"]], ["Selection", ["lt", "Product"]], ["Selection", ["gte", "Table"]], ["Selection", ["nin", "Table"]], ["Selection", ["eq", "Table"]], ["Selection", ["gt", "Table"]], ["Selection", ["lt", "Table"]], ["Selection", ["in", "Table"]], ["Selection", ["And", "Table"]], ["Selection", ["eq", "Product"]], ["Selection", ["like", "Table"]], ["Selection", ["nlike", "Table"]], ["Subquery", ["Limit"]], ["Subquery", ["Groupby"]], ["Subquery", ["except"]], ["Subquery", ["Project"]], ["Subquery", ["union"]], ["Subquery", ["intersect"]], ["Table", []], ["Table", ["Subquery"]], ["Val_list", ["max", "min"]], ["Val_list", ["sum", "sum"]], ["Val_list", ["Val_list", "avg"]], ["Val_list", ["Value", "Value"]], ["Val_list", ["Value", "sum"]], ["Val_list", ["avg", "min"]], ["Val_list", ["avg", "count"]], ["Val_list", ["Val_list", "max"]], ["Val_list", ["avg", "Value"]], ["Val_list", ["sum", "max"]], ["Val_list", ["avg", "sum"]], ["Val_list", ["count", "sum"]], ["Val_list", ["max", "max"]], ["Val_list", ["Value", "max"]], ["Val_list", ["sum", "avg"]], ["Val_list", ["max", "Value"]], ["Val_list", ["max", "sum"]], ["Val_list", ["Val_list", "count"]], ["Val_list", ["count", "max"]], ["Val_list", ["count", "Value"]], ["Val_list", ["distinct", "Value"]], ["Val_list", ["sum", "min"]], ["Val_list", ["min", "min"]], ["Val_list", ["count", "count"]], ["Val_list", ["count", "avg"]], ["Val_list", ["sum", "Value"]], ["Val_list", ["avg", "max"]], ["Val_list", ["min", "Value"]], ["Val_list", ["Val_list", "min"]], ["Val_list", ["Val_list", "sum"]], ["Val_list", ["min", "avg"]], ["Val_list", ["Value", "avg"]], ["Val_list", ["max", "avg"]], ["Val_list", ["Value", "count"]], ["Val_list", ["avg", "avg"]], ["Val_list", ["Val_list", "Value"]], ["Val_list", ["min", "max"]], ["Value", []], ["avg", ["Value"]], ["count", ["distinct"]], ["count", ["Value"]], ["distinct", ["Value"]], ["eq", ["Value", "Subquery"]], ["eq", ["Value", "Value"]], ["eq", ["Value", "literal"]], ["eq", ["count", "literal"]], ["except", ["Subquery", "Subquery"]], ["gt", ["avg", "Subquery"]], ["gt", ["count", "Subquery"]], ["gt", ["avg", "literal"]], ["gt", ["Value", "Subquery"]], ["gt", ["max", "literal"]], ["gt", ["Value", "literal"]], ["gt", ["Value", "Value"]], ["gt", ["count", "literal"]], ["gt", ["sum", "literal"]], ["gte", ["Value", "Subquery"]], ["gte", ["sum", "literal"]], ["gte", ["Value", "literal", "literal"]], ["gte", ["count", "literal"]], ["gte", ["Value", "literal"]], ["gte", ["Value", "Subquery", "literal"]], ["gte", ["avg", "literal"]], ["gte", ["count", "literal", "literal"]], ["in", ["Value", "Subquery"]], ["intersect", ["Subquery", "Subquery"]], ["like", ["Value", "literal"]], ["literal", ["Value"]], ["lt", ["Value", "Subquery"]], ["lt", ["min", "literal"]], ["lt", ["Value", "Value"]], ["lt", ["count", "literal"]], ["lt", ["avg", "literal"]], ["lt", ["Value", "literal"]], ["lte", ["Value"]], ["lte", ["count"]], ["lte", ["Value", "literal"]], ["lte", ["Value", "Subquery"]], ["lte", ["sum", "literal"]], ["lte", ["count", "literal"]], ["max", ["Value"]], ["min", ["Value"]], ["neq", ["Value", "literal"]], ["neq", ["Value", "Subquery"]], ["neq", ["Value", "Value"]], ["nin", ["Value", "Subquery"]], ["nlike", ["Value", "literal"]], ["sum", ["Value"]], ["union", ["Subquery", "Subquery"]]]"""
RULES_values = """[["Or", ["neq", "neq"]], ["Orderby_desc", ["max", "Groupby"]], ["Or", ["eq", "lt"]], ["And", ["lt", "neq"]], ["Selection", ["like", "Table"]], ["And", ["gte", "lte"]], ["And", ["eq", "And"]], ["Val_list", ["sum", "Value"]], ["Project", ["min", "Table"]], ["Or", ["lt", "gt"]], ["Selection", ["gte", "Table"]], ["Selection", ["lt", "Product"]], ["And", ["gte", "gte"]], ["lte", ["Value", "literal"]], ["Project", ["distinct", "Table"]], ["Subquery", ["intersect"]], ["And", ["And", "And"]], ["count", ["Value"]], ["Orderby_desc", ["Value", "Project"]], ["And", ["eq", "neq"]], ["Or", ["like", "like"]], ["Limit", ["Value", "Orderby_asc"]], ["gt", ["Value", "Subquery"]], ["Val_list", ["max", "max"]], ["Or", ["eq", "gt"]], ["Val_list", ["min", "min"]], ["Val_list", ["Val_list", "Value"]], ["sum", ["Value"]], ["Selection", ["eq", "Product"]], ["Project", ["sum", "Selection"]], ["Val_list", ["count", "Value"]], ["neq", ["Value", "literal"]], ["Orderby_asc", ["avg", "Groupby"]], ["Val_list", ["min", "Value"]], ["min", ["Value"]], ["Or", ["gt", "lt"]], ["eq", ["Value", "Subquery"]], ["lt", ["Value", "Subquery"]], ["Val_list", ["count", "max"]], ["Selection", ["And", "Product"]], ["gte", ["avg", "Value"]], ["Val_list", ["Val_list", "count"]], ["Project", ["Val_list", "Selection"]], ["lte", ["Value", "Value"]], ["Val_list", ["sum", "min"]], ["Or", ["gt", "gt"]], ["Val_list", ["max", "min"]], ["gt", ["count", "Value"]], ["Product", ["Table", "Table"]], ["neq", ["Value", "Value"]], ["And", ["lt", "eq"]], ["And", ["eq", "nin"]], ["Orderby_asc", ["Val_list", "Project"]], ["Groupby", ["Val_list", "Project"]], ["Val_list", ["Val_list", "min"]], ["gte", ["Value", "literal"]], ["gt", ["avg", "Value"]], ["eq", ["count", "Value"]], ["Project", ["avg", "Table"]], ["lt", ["count", "Value"]], ["Orderby_desc", ["avg", "Groupby"]], ["Val_list", ["count", "sum"]], ["And", ["eq", "eq"]], ["lt", ["min", "Value"]], ["Selection", ["Or", "Product"]], ["And", ["gt", "in"]], ["Or", ["gt", "eq"]], ["Val_list", ["sum", "avg"]], ["lt", ["avg", "Value"]], ["Project", ["max", "Selection"]], ["Val_list", ["sum", "sum"]], ["And", ["And", "lt"]], ["Limit", ["Value", "Orderby_desc"]], ["Selection", ["eq", "Table"]], ["gt", ["max", "Value"]], ["Orderby_asc", ["Value", "Groupby"]], ["Project", ["max", "Table"]], ["And", ["eq", "gt"]], ["literal", ["Value"]], ["Val_list", ["avg", "Value"]], ["gt", ["Value", "literal"]], ["gte", ["Value", "Value"]], ["Selection", ["lte", "Table"]], ["Selection", ["And", "Table"]], ["Project", ["count", "Selection"]], ["Val_list", ["Val_list", "sum"]], ["And", ["gte", "gt"]], ["And", ["gt", "lt"]], ["And", ["in", "in"]], ["Val_list", ["Value", "max"]], ["in", ["Value", "Subquery"]], ["lte", ["sum", "Value"]], ["Selection", ["neq", "Table"]], ["lt", ["Value", "literal"]], ["And", ["And", "lte"]], ["Val_list", ["avg", "count"]], ["Project", ["avg", "Selection"]], ["Val_list", ["Value", "count"]], ["Val_list", ["max", "Value"]], ["union", ["Subquery", "Subquery"]], ["Selection", ["gt", "Table"]], ["Val_list", ["sum", "max"]], ["except", ["Subquery", "Subquery"]], ["Subquery", ["Project"]], ["And", ["gt", "gt"]], ["Project", ["count", "Table"]], ["Val_list", ["Value", "avg"]], ["gt", ["Value", "Value"]], ["And", ["eq", "Or"]], ["Project", ["Value", "Table"]], ["like", ["Value", "literal"]], ["Orderby_desc", ["Value", "Groupby"]], ["And", ["gt", "lte"]], ["Val_list", ["distinct", "Value"]], ["Val_list", ["Value", "sum"]], ["Selection", ["lt", "Table"]], ["And", ["eq", "lt"]], ["And", ["gt", "gte"]], ["Orderby_asc", ["Value", "Project"]], ["Val_list", ["avg", "min"]], ["eq", ["Value", "Value"]], ["And", ["And", "Or"]], ["Val_list", ["avg", "max"]], ["Subquery", ["union"]], ["Orderby_asc", ["count", "Groupby"]], ["lt", ["Value", "Value"]], ["Subquery", ["Groupby"]], ["Project", ["Val_list", "Product"]], ["Val_list", ["min", "max"]], ["Selection", ["in", "Table"]], ["And", ["like", "neq"]], ["And", ["gte", "eq"]], ["count", ["distinct"]], ["Project", ["distinct", "Selection"]], ["lte", ["Value", "Subquery"]], ["Subquery", ["Limit"]], ["Or", ["gte", "gte"]], ["Val_list", ["Value", "Value"]], ["Orderby_asc", ["sum", "Groupby"]], ["And", ["eq", "lte"]], ["max", ["Value"]], ["Selection", ["nlike", "Table"]], ["Or", ["eq", "eq"]], ["gte", ["sum", "Value"]], ["And", ["eq", "gte"]], ["Product", ["Product", "Table"]], ["Val_list", ["min", "avg"]], ["eq", ["Value", "literal"]], ["nlike", ["Value", "literal"]], ["Selection", ["nin", "Table"]], ["Val_list", ["count", "count"]], ["neq", ["Value", "Subquery"]], ["Val_list", ["avg", "avg"]], ["gt", ["avg", "Subquery"]], ["Project", ["Value", "Selection"]], ["Val_list", ["avg", "sum"]], ["And", ["And", "gte"]], ["And", ["eq", "like"]], ["Orderby_desc", ["count", "Groupby"]], ["distinct", ["Value"]], ["gte", ["count", "Value"]], ["lte", ["count", "Value"]], ["And", ["And", "neq"]], ["And", ["And", "like"]], ["And", ["And", "eq"]], ["Val_list", ["Val_list", "max"]], ["gt", ["sum", "Value"]], ["Val_list", ["max", "avg"]], ["Orderby_desc", ["sum", "Groupby"]], ["Project", ["sum", "Table"]], ["Groupby", ["Value", "Project"]], ["Selection", ["Or", "Table"]], ["Val_list", ["max", "sum"]], ["Table", ["Subquery"]], ["avg", ["Value"]], ["intersect", ["Subquery", "Subquery"]], ["gte", ["Value", "Subquery"]], ["And", ["gt", "neq"]], ["nin", ["Value", "Subquery"]], ["Val_list", ["Val_list", "avg"]], ["And", ["gt", "eq"]], ["And", ["And", "gt"]], ["Project", ["Val_list", "Table"]], ["Val_list", ["count", "avg"]], ["Project", ["min", "Selection"]]]"""
