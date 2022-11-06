# Various utility functions & classes for parsing. Many of them are not really used now.

# One important thing: we use only nodes in PB with some occurrences, as we find that
# the quality of conceptualizations for rare nodes is not good.

from nltk.corpus import wordnet as wn
from collections import namedtuple, defaultdict
from threading import Lock
from lemminflect import getInflection
import json
import editdistance
import copy, os

Token = namedtuple('Token', ['text', 'ws', 'pos', 'tag', 'dep', 'head', 'lemma', 'modifier', 'predicate', 'prep',
                         'nominal', 'i', 'idx'])

wn_cache = {}
def wn_query(x, pos=None):
    key = (x, pos)
    if key not in wn_cache:
        wn_cache[key] = wn.synsets(x, pos)
    return [c for c in wn_cache[key]]

wn_hypernym_cache = {}
def wn_query_hypernym(x):
    if x in wn_hypernym_cache:
        return [c for c in wn_hypernym_cache[x]]
    synsets = wn_query(x)
    hn_texts = set()
    for s in synsets:
        hns = s.hypernyms()
        for h_syn in hns:
            hn_texts = hn_texts.union([s.lower().replace('_', ' ') for s in h_syn._lemma_names])
    wn_hypernym_cache[x] = hn_texts
    return copy.copy(hn_texts)

top_concepts = open(os.path.join('data', 'top_concepts.txt')).read().splitlines()
class ProbaseClient:
    def __init__(self):
        from multiprocessing.connection import Client
        address = ('localhost', 9233) # Use a fixed one. Hope it's not occupied.
        print('Connecting Probase...')
        self.conn = Client(address)
        print('Connected')
        self.cache = {}
        self._lock = Lock()

    def query(self, x, sort_method='mi', truncate=50, target='abs', do_filter=True):
        x = x.lower()
        if (x, sort_method, truncate, target) in self.cache:
            return copy.copy(self.cache[(x, sort_method, truncate, target)])
        with self._lock:
            self.conn.send(json.dumps([x, sort_method, truncate, target]))
            res = json.loads(self.conn.recv())
            key_remove = []
            if do_filter:
                for key in res: # Some concepts are unwanted
                    if key.split(' ')[-1] in ['word', 'phrase', 'noun', 'adjective', 'verb', 'pronoun', 'term', 'aux'] or key in top_concepts:
                        key_remove.append(key)
            for k in key_remove:
                del res[k]
            self.cache[(x, sort_method, truncate, target)] = copy.copy(res)
        return res

def wn_nounify(word=None, pos=None, exclude_person=True, synset_name=None, max_lem=1000, base_lemma=None):
    if synset_name is None:
        verb_synsets = wn_query(word, pos)
    else:
        verb_synsets = [wn.synset(synset_name)]
        if verb_synsets[0].pos() == 'n':
            if base_lemma:
                lems = [l for l in verb_synsets[0].lemmas() if l.key() == base_lemma.key()]
            else:
                lems = verb_synsets[0].lemmas()
            return lems

    # Word not found
    if not verb_synsets:
        return []

    # Get all verb lemmas of the word
    verb_lemmas = [l for s in verb_synsets for l in s.lemmas()[:max_lem] if (
        (base_lemma is None and word is None) or (word is not None and l.name() == word)
        or (base_lemma is not None and l.key() == base_lemma.key()))]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in verb_lemmas]

    # filter only the nouns
    related_noun_lemmas = []
    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == 'n':
                related_noun_lemmas.append(l)
    related_noun_lemmas = list(set(related_noun_lemmas))
    # We exclude them by default, as we find that they often contain unwanted results like person names
    if exclude_person:
        related_noun_lemmas = [lem for lem in related_noun_lemmas if lem._synset._lexname
                               not in ['noun.person', 'noun.substance', 'noun.animal', 'noun.artifact', 'noun.group']]
    # return all the possibilities sorted by probability
    return related_noun_lemmas

def is_possibly_light(tok):
    return tok.pos == 'VERB' and tok.lemma in ['do', 'get', 'have', 'give', 'make', 'take', 'undergo', 'come', 'put',
                                               'provide', 'offer', 'show', 'deliver', 'bring', 'let']

def is_possibly_mod(tok):
    return (tok.pos == 'VERB' and tok.lemma in ['finish', 'start', 'complete', 'continue',
                                                'end', 'stop', 'begin', 'cease', 'try', 'attempt'])

# Some mis-lemmatizations that spacy gives. They are not necessarily wrong, but give suboptimal conceptualizations.
def correct_mislemmas(tok):
    mislemmas = {'taxes': 'tax', 'lenses': 'lens', 'goods': 'goods', 'waters': 'waters', 'ashes': 'ash',
                 'fries': 'fries', 'politics': 'politics', 'glasses': 'glasses', 'clothes': 'clothes',
                 'scissors': 'scissors', 'shorts': 'shorts', 'thanks': 'thanks',
                 'media': 'media', 'woods': 'woods', 'data': 'data', 'belongings': 'belongings'}
    if tok.text not in mislemmas:
        return tok.lemma
    return mislemmas[tok.text]

def get_children(doc, tok):
    return [t for t in doc if t.head == tok.i and t.i != tok.i]

def get_left_edge(doc, tok):
    while True:
        c = get_children(doc, tok)
        if len(c) == 0 or c[0].i > tok.i:
            return tok
        else:
            tok = c[0]

def get_right_edge(doc, tok, exclude_conj=False):
    c = get_children(doc, tok)
    if exclude_conj:
        for k in range(len(c)):
            if c[k].dep in ['cc', 'conj']:
                c = c[:k]
                break

    while True:
        if len(c) == 0 or c[-1].i < tok.i:
            return tok
        else:
            tok = c[-1]
        c = get_children(doc, tok)

def get_text(doc):
    text = ''
    for t in doc:
        text += t.text + t.ws
    return text.strip()

def is_pb_close(lemma_resp, base_resp):
    lem_total = sum([s['co_occurrence'] for s in lemma_resp.values()])
    base_total = sum([s['co_occurrence'] for s in base_resp.values()])
    diffs = set(base_resp.keys()).difference(lemma_resp.keys())
    diff_sum = sum([base_resp[k]['co_occurrence'] for k in diffs])
    return diff_sum < 0.1 * max(lem_total, base_total) and diff_sum < 100

def merge_pb_response(a, b):
    if a is None:
        keys = set(b.keys())
        a = {}
    else:
        keys = set(a.keys()).union(b.keys())
    response = dict([(k, a.get(k, 0) + (b[k]['co_occurrence'] if k in b else 0)) for k in keys])
    return response

def to_gerund(l):
    if ' ' in l:
        l = l.split(' ')
        l[0] = getInflection(l[0], 'VBG')[0]
        l = ' '.join(l)
    else:
        l = getInflection(l, 'VBG')[0]
    return l

def filter_cand_lems(pb, lems):
    syns = set()
    for l in lems:
        syns.add(l.synset())
    results = []
    for s in syns:
        for l in s.lemmas():
            if l in lems:
                wnn = l.name().lower().replace('_', ' ')
                if len(pb.query(wnn)) >= 10:
                    results.append(wnn)
                    break
    results = list(set(results))
    return results

# Get PB alignments from synset
pb_align_cache = {}
def get_pb_align(pb, nl, s, lem, restrict_lem=True, expand_synset=False):
    if s.startswith('PB'):
        return [s[3:]]
    key = (s, lem.name(), restrict_lem, expand_synset)
    if key in pb_align_cache:
        return [c for c in pb_align_cache[key]]
    if restrict_lem:
        wn_nouns = wn_nounify(synset_name=s, base_lemma=lem)
    else:
        wn_nouns = wn_nounify(synset_name=s)
    if not restrict_lem and expand_synset:
        lems = set()
        for lem in wn_nouns:
            for rel_lem in lem.synset().lemmas():
                lems.add(rel_lem)
        wn_nouns = list(lems)
    pbn = filter_cand_lems(pb, wn_nouns)
    pbn.sort()
    pb_align_cache[key] = pbn
    return copy.copy(pbn)

# Get the desired linking/alignment to PB nodes from the text and synsets.
# Direct alignments in the previous stage might not be optimal.
# So we tried to used some additional heuristics, filtering, and
# external data (like nomlex) to improve the results as discussed in the paper.
pb_align_cache_text = {}
def get_pb_align_by_text(pb, nl, text, is_nominal, is_predicate, syns):
    if (text, is_nominal, is_predicate) in pb_align_cache_text:
        return [c for c in pb_align_cache_text[(text, is_nominal, is_predicate)]]
    pbn = []
    if is_predicate and text in nl:
        for t in nl[text]:
            if len(pb.query(t)) >= 10 and t not in pbn:
                pbn.append(t)
    if is_predicate and len(pbn) == 0 and not text.endswith('ing') and \
            [s for s, score in syns if s[:3] != 'PB:' and s[:6] != 'Idiom:' and s.split('.')[1] == 'v' and score > 0.1]:
        l = to_gerund(text)
        if l not in pbn and len(pb.query(l)) >= 10:
            pbn.append(l)
    if is_nominal:
        if not is_predicate or text.endswith('ing'):
            if len(pb.query(text)) >= 10 and text not in pbn:
                pbn.append(text)
    pb_align_cache_text[(text, is_nominal, is_predicate)] = pbn
    return copy.copy(pbn)

def find_wn_lemma(text, sid):
    synset = wn.synset(sid)
    lemmas = synset.lemmas()
    min_ev = 100
    result = None
    for lem in lemmas:
        ev = editdistance.eval(text, lem._name)
        if ev < min_ev:
            min_ev = ev
            result = lem
    return result
