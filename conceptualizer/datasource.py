import os, json, sys
import threading
from atomic_parse_utils import wn_query, Token, get_left_edge, get_right_edge, ProbaseClient, merge_pb_response, get_pb_align, get_pb_align_by_text, find_wn_lemma
import random
import logging
import copy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

output_path = r'parse'

# DataSource class to hold the data: parsing results, Probase client, etc.
# Used in annotation and building Abstract ATOMIC
class DataSource():
    def __init__(self):
        docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
        linked_concepts = [json.loads(l) for l in open(
            os.path.join(output_path, 'linked_concepts.1.jsonl')).read().splitlines()]
        components = [json.loads(l) for l in open(os.path.join(output_path, 'components.jsonl')).read().splitlines()]
        exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
        self._nl = json.load(open(os.path.join('data', 'nomlex.nounify.reduced.json')))

        for i, doc in enumerate(docs):
            if i in exclusions:
                continue
            doc = [Token(**t) for t in doc]
            docs[i] = doc
        self._rand = random.Random()
        self._rand.seed(1)
        self._lock = threading.Lock()
        try:
            self._pb = ProbaseClient()
        except:
            import traceback as tb
            tb.print_exc()
            self._pb = None
        self._docs = docs
        self._components = components
        self._concepts = linked_concepts

    def get_sent(self, d_i):
        return self._docs[d_i]

    def get_sent_concept(self, d_i, c_i):
        l, r, comp_i = self.get_component(d_i, c_i)
        sent = self.get_sent(d_i)
        prev = post = ''
        for tok in sent[:l]:
            prev = prev + tok.text + tok.ws
        base_text = prev + '['
        for i, tok in enumerate(sent[l: r + 1]):
            base_text += tok.text + (tok.ws if tok.i < r else ']' + tok.ws)
        for tok in sent[r + 1:]:
            post = post + tok.text + tok.ws
        base_text = base_text + post
        return prev, post, base_text

    def get_component(self, d_i, c_i):
        comp_i = self._components[d_i][c_i]
        le = get_left_edge(self._docs[d_i], self._docs[d_i][comp_i])
        re = get_right_edge(self._docs[d_i], self._docs[d_i][comp_i], exclude_conj=True)
        return le.i, re.i, comp_i

    def get_variations(self, d_i, c_i):
        results = []
        for mods, syn, sub_text, l_i, r_i in self._concepts[d_i][c_i]:
            results.append((sub_text, l_i, r_i))
        return results

    # Get a list of possible alignments/linkings from a constituent to PB.
    # Note that the ones in self._concepts (collected in atomic_parse.py) might only have WN alignment,
    # and the Probase alignments might not be optimal (e.g. without nounification).
    def get_pb_alignments(self, d_i, c_i, v_i, collect_extra=True):
        mods, syn, sub_text, l_i, r_i, = self._concepts[d_i][c_i][v_i]

        tok_i = self._components[d_i][c_i]

        pb_aligns = get_pb_align_by_text(self._pb, self._nl, sub_text, self._docs[d_i][tok_i].nominal,
                                         self._docs[d_i][tok_i].predicate, syn)
        base_pb_aligns = copy.copy(pb_aligns)
        used_syn = 0
        for k, (sid, score) in enumerate(syn):
            if sid.startswith('PB') and sid[3:] not in pb_aligns:
                pb_aligns.append(sid[3:])
            elif sid.startswith('Idiom:'):
                continue
            elif (score > 0.5 or used_syn == 0) and len(base_pb_aligns) == 0:
                used_syn += 1
                lem = find_wn_lemma(sub_text, sid)
                if self._docs[d_i][tok_i].predicate and ' ' in sub_text:
                    restrict = False
                else:
                    restrict = True
                options = get_pb_align(self._pb, self._nl, sid, lem, restrict_lem=restrict)
                if len(options) == 0 and restrict:
                    options = get_pb_align(self._pb, self._nl, sid, lem, restrict_lem=False)
                if len(options) == 0:
                    options = get_pb_align(self._pb, self._nl, sid, lem, restrict_lem=False, expand_synset=True)
                options = [o for o in options if o not in pb_aligns]
                if len(options) > 3:
                    options.sort(key=lambda x: len(self._pb.query(x, truncate=500)), reverse=True)
                    options = options[:3]
                for o in options:
                    pb_aligns.append(o)
        return pb_aligns

    def get_abstractions(self, node, d_i=-1, c_i=-1, do_shuffle=True, n_samples=10):
        if d_i != -1 and c_i != -1:
            expired_abs = self.get_expired_abstractions(d_i, c_i, node)
        else:
            expired_abs = []
        s = self._pb.query(node, truncate=n_samples + 5 + len(expired_abs))
        pbs = list(s.items())
        pbs = [x for x in pbs if x[0] not in expired_abs]
        pbs.sort(key=lambda x: -x[1]['mi'])
        pbs = pbs[:n_samples]
        if do_shuffle:
            self._rand.shuffle(pbs)
        return [p[0] for p in pbs]

    def pb_query(self, text):
        s = self._pb.query(text)
        return s