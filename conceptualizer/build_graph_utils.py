from nltk.corpus import wordnet as wn
import copy
from collections import defaultdict
import math

ul_magic = 'Wozment'
names = ['Alex', 'Bob', 'Charlie'] + ['bob', 'dan', 'jeff', 'tim', 'tom']

person_pron = ['one', 'everyone', 'anyone', 'someone', 'us', 'thee',
               'you', 'nobody', 'they', 'them', 'ye', 'anybody', 'somebody', 'me', 'him',
               'oneself', 'himself', 'herself', 'themselves', 'nobody', 'thyself'] + names
unspecified_person_pron = ['one', 'someone', 'somebody', 'anyone', 'anybody']
unspecified_thing_pron = ['it', 'something']
thing_pron = ['it', 'something', 'itself', 'anything', 'nothing', 'everything']

sp_nodes = person_pron + thing_pron + [ul_magic]
wildcard_nodes = unspecified_person_pron + unspecified_thing_pron + [ul_magic]

wn_cache = {}
def wn_query(x):
    if x not in wn_cache:
        wn_cache[x] = wn.synsets(x, pos='n')[:3]
    return wn_cache[x]

wn_hypernym_cache = {}
def wn_query_hypernym(x, noun_only=True):
    if x in wn_hypernym_cache:
        return wn_hypernym_cache[x]
    synsets = wn_query(x)
    hn_texts = set()
    for s in synsets:
        if s._pos != 'n' and noun_only:
            continue
        hns = s.hypernyms()
        for h_syn in hns:
            hn_texts = hn_texts.union([s.lower().replace('_', ' ') for s in h_syn._lemma_names])
    wn_hypernym_cache[x] = hn_texts
    return hn_texts

def pb_query_abstract_prim(pb, instance):
    self = pb
    pb = pb._probase
    if instance not in pb.instance2idx:
        return {}
    instance_freq = self.inst_freq_map[instance]
    instance_idx = pb.instance2idx[instance]
    concept_list = pb.instance_inverted_list[instance_idx]
    rst_list = []
    for concept_idx, co_occurrence in concept_list:
        concept = pb.idx2concept[concept_idx]
        pmi = co_occurrence / self.concept_freq_map[concept] / instance_freq
        mi = co_occurrence * math.log(pmi * self.co_occurrence_sum)
        likelihood = co_occurrence / instance_freq
        inst_cnts = self.inst_cnt_map[concept]
        rst_list.append((concept, {'type': 'pb', 'co_occurrence': co_occurrence, 'pmi': pmi, 'likelihood': likelihood,
                                   'mi': mi, 'inst_cnt': inst_cnts}))
    return dict(rst_list)

pb_query_abstract_cache = {}
def pb_query_abstract(pb, x):
    if x not in pb_query_abstract_cache:
        pb_query_abstract_cache[x] = pb_query_abstract_prim(pb, x)
    return pb_query_abstract_cache[x]

def pb_query_instance_prim(pb, concept):
    self = pb
    pb = pb._probase
    if concept not in pb.concept2idx:
        return {}
    concept_idx = pb.concept2idx[concept]
    concept_freq = self.concept_freq_map[concept]
    inst_list = pb.concept_inverted_list[concept_idx]
    rst_list = []
    for inst_idx, co_occurrence in inst_list:
        inst = pb.idx2instance[inst_idx]
        pmi = co_occurrence / self.inst_freq_map[inst] / concept_freq
        mi = co_occurrence * math.log(pmi * self.co_occurrence_sum)
        likelihood = co_occurrence / concept_freq
        rst_list.append((inst, {'type': 'pb', 'co_occurrence': co_occurrence, 'pmi': pmi, 'likelihood': likelihood,
                                   'mi': mi}))
    return dict(rst_list)

pb_query_instance_cache = {}
def pb_query_instance(pb, x):
    if x not in pb_query_instance_cache:
        pb_query_instance_cache[x] = pb_query_instance_prim(pb, x)
    return pb_query_instance_cache[x]


is_wn_person_cache = {}
is_wn_typical_person_cache = {}

def is_wn_person(x):
    if x in is_wn_person_cache:
        return is_wn_person_cache[x]
    synsets = wn_query(x)
    is_wn_person_cache[x] = False
    for i, sn in enumerate(synsets[:3]):
        if sn._lemma_names[0] == x.capitalize(): # Avoid names
            continue
        if sn._lexname == 'noun.person':
            is_wn_person_cache[x] = True
            break
    return is_wn_person_cache[x]

def is_wn_typical_person_prim(x):
    synsets = wn_query(x)[:3]
    synsets = [sn for sn in synsets if sn._lexname.startswith('noun')]
    if len(synsets) == 0:
        return False
    for i, sn in enumerate(synsets):
        if sn._lexname != 'noun.person':
            return False
    return True

def is_wn_typical_person(x):
    if x not in is_wn_typical_person_cache:
        is_wn_typical_person_cache[x] = is_wn_typical_person_prim(x)
    return is_wn_typical_person_cache[x]

find_abstract_cache = {}
def find_abstraction(pb, x):
    if x in find_abstract_cache:
        return find_abstract_cache[x]
    hypernyms = {}
    if not (x in names or x in person_pron or x in thing_pron or x == ul_magic):
        if pb is not None:
            hypernyms = dict(copy.copy(pb_query_abstract(pb, x)))
        wn_hypernyms = wn_query_hypernym(x)
        for w in wn_hypernyms:
            if w not in hypernyms:
                hypernyms[w] = {'type': 'wn'}
    if x != ul_magic:
        hypernyms[ul_magic] = {'type': 'ul_magic'}
    if not (x in names or x in person_pron or is_wn_typical_person(x) or x == ul_magic):
        for y in unspecified_thing_pron:
            if y not in hypernyms:
                hypernyms[y] = {'type': 'thing'}
    if x in names or is_wn_person(x) or x in person_pron:
        for y in unspecified_person_pron + ['person']:
            if y not in hypernyms:
                hypernyms[y] = {'type': 'person'}
    if x in pb.ps:
        for a in pb.ps[x]:
            if a not in hypernyms:
                hypernyms[a] = {'type': 'modifier'}
    if x in hypernyms:
        del hypernyms[x]
    find_abstract_cache[x] = hypernyms
    return hypernyms

def find_edge(pb, x, y):
    ys = find_abstraction(pb, x)
    if y in ys:
        return ys[y]
    return None

def get_node_height(g_map, nodes):
    node_to_id = dict([(x, i) for (i, x) in enumerate(nodes)])
    edges = defaultdict(set)
    for (x, y), (edge, conn) in g_map.items():
        # if edge['type'] == 'pb' and edge['co_occurrence'] < 30:
        #     continue
        edges[node_to_id[y]].add(node_to_id[x])
    color = [i for i in range(len(nodes))]

    def find(x):
        if color[x] == x:
            return x
        color[x] = find(color[x])
        return color[x]

    def get_reachable(src):
        Q = [src]
        visited = [False for _ in nodes]
        visited[src] = True
        while len(Q) > 0:
            u, Q = Q[0], Q[1:]
            for v in edges[u]:
                if not visited[v]:
                    visited[v] = True
                    Q.append(v)
        return visited

    reachable = [get_reachable(x) for x in range(len(nodes))]
    for x in range(len(nodes)):  # Slow and dirty way to get BCCs, using DSU
        for y in range(x + 1, len(nodes)):
            if find(x) != find(y):
                if reachable[x][y] and reachable[y][x]:
                    fx = find(x)
                    fy = find(y)
                    color[fx] = fy
    for x in range(len(nodes)):
        edges[x] = set([find(y) for y in edges[x]])
    ds = [set() for _ in nodes]
    for x in range(len(nodes)):
        edges[find(x)] = edges[find(x)].union(edges[x]).difference([find(x)])
        ds[find(x)].add(x)

    def find_height(src):
        Q = [src]
        visited = [-1 for _ in nodes]
        visited[src] = 0
        max_height = 0
        while len(Q) > 0:
            u, Q = Q[0], Q[1:]
            for v in edges[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    max_height = max(max_height, visited[v])
                    Q.append(v)
        return max_height

    height = {}
    for x in range(len(nodes)):
        if len(ds[x]) > 0:
            h = find_height(x)
            for y in ds[x]:
                height[y] = h

    height = list(height.items())
    height.sort(key=lambda x: x[0])
    height = [(nodes[x], d) for x, d in height]
    return height

def deg_counts(G):
    degs = [(k, len(v)) for k, v in G.items()]
    return degs

