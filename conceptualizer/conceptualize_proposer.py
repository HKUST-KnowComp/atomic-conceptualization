from pbconcept import ProbaseConcept
from server_utils import EntityMentionE
import json, random, tqdm, os
import copy
from typing import List, Callable, Tuple, NamedTuple, Any
import numpy as np
import inflect

expr = 'atomic'
_infl = inflect.engine()

probase_path = r'/home/mhear/data/probase/data-concept-instance-relations.txt'
if not os.path.exists(probase_path):
    probase_path = r'/data/mhear/data/probase/data-concept-instance-relations.txt'
if not os.path.exists(probase_path):
    probase_path = r'C:\Users\dy.octa\Store\data-concept\data-concept-instance-relations.txt'
if not os.path.exists(probase_path):
    probase_path = r'C:\Users\t-muh\Store\kg\data-concept\data-concept-instance-relations.txt'

top_k_candidates = 5


class Substitution(NamedTuple):
    alt_text: str
    idx: int
    base_slice: str
    alt_slice: str
    entity_mention_id: int
    weight: float


def remove_det(x: str):
    if x.startswith('a '):
        return x[2:]
    elif x.startswith('an '):
        return x[3:]
    elif x.startswith('the '):
        return x[4:]
    else:
        return x

def substitute(pre_text, cand, post_text, det_idx, det_text, alt_dets):
    if not alt_dets:
        if det_text in ['a', 'an'] and expr != 'aser':
            possible_dets = [_infl.an(pre_text[det_idx + len(det_text) + 1:] + cand + post_text).split(' ')[0]]
        else:
            possible_dets = [det_text]
    else:
        possible_dets = ['a', 'an', 'the', '']
    for d in possible_dets:
        if det_text == '' and d != '':
            d += ' '
        _pre_text = pre_text[:det_idx] + d + pre_text[det_idx + len(det_text):]
        yield _pre_text + cand + post_text

class Proposer:
    _probase = None

    def __init__(self):
        if self._probase is not None:
            return
        Proposer._probase = ProbaseConcept(probase_path)
        all_entities = set(self._probase.concept2idx.keys()).union(set(self._probase.instance2idx.keys()))
        entity_freqs = []
        concept_freqs = []
        inst_freqs = []
        inst_cnts = []
        co_occurrence_sum = 0
        for entity in tqdm.tqdm(all_entities):
            c_freq = self._probase.get_concept_freq(entity)
            i_freq = self._probase.get_instance_freq(entity)
            e_freq = c_freq + i_freq
            co_occurrence_sum += c_freq

            concept_freqs.append((entity, c_freq))
            inst_freqs.append((entity, i_freq))
            entity_freqs.append((entity, e_freq))

            if c_freq:
                i_c = len(self._probase.concept_inverted_list[self._probase.concept2idx[entity]])
            else:
                i_c = 0
            inst_cnts.append((entity, i_c))

        concept_freqs.sort(key=lambda x: x[-1], reverse=True)
        inst_freqs.sort(key=lambda x: x[-1], reverse=True)
        entity_freqs.sort(key=lambda x: x[-1], reverse=True)
        inst_cnts.sort(key=lambda x: x[-1], reverse=True)

        Proposer.co_occurrence_sum = co_occurrence_sum
        Proposer.concept_freqs = concept_freqs
        Proposer.inst_freqs = inst_freqs
        Proposer.entity_freqs = entity_freqs
        Proposer.inst_cnts = inst_cnts

        Proposer.concept_freq_map = dict(concept_freqs)
        Proposer.inst_freq_map = dict(inst_freqs)
        Proposer.entity_freq_map = dict(entity_freqs)
        Proposer.inst_cnt_map = dict(inst_cnts)

    def is_a(self, x: str, y: str):
        if x not in self._probase.instance2idx or y not in self._probase.concept2idx:
            return False
        x_id = self._probase.instance2idx[x]
        y_id = self._probase.concept2idx[y]
        concept_inverted_list = self._probase.concept_inverted_list[y_id]
        return any([t[0] == x_id for t in concept_inverted_list])

    def collect_substitution(self, text: str, concepts: List[EntityMentionE],
                             candidate_gen: Callable[[Any], List[Tuple[str, float]]]
                             , try_alt_dets) -> List[Substitution]:
        results = []
        for i, mention in enumerate(concepts):
            candidates = candidate_gen(mention.base_concept)
            pre_text = text[:mention.l]
            post_text = text[mention.r:]
            for cand, w in candidates:
                cand = remove_det(cand)
                if cand == mention.base_concept:
                    continue
                for _text in substitute(pre_text, cand, post_text, mention.det_idx, mention.det_text, try_alt_dets):
                    results.append(Substitution(_text, mention.l, mention.base_concept, cand, i, w))
        return results

    def entity_abstract(self, instance, score_method="likelihood", threshold=3):
        pb = self._probase
        if instance not in pb.instance2idx:
            return []
        instance_idx = pb.instance2idx[instance]
        instance_freq = self.inst_freq_map[instance]
        concept_list = pb.instance_inverted_list[instance_idx]
        rst_list = []
        for concept_idx, co_occurrence in concept_list:
            if co_occurrence <= threshold:
                continue
            concept = pb.idx2concept[concept_idx]
            if score_method == "pmi":
                score = co_occurrence / self.concept_freq_map[concept] / instance_freq
            elif score_method == "likelihood":
                score = co_occurrence / instance_freq
            elif score_method == "co_occurrence":
                score = co_occurrence
            else:
                raise NotImplementedError(score_method)
            rst_list.append((concept, score))
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def entity_instantiate(self, concept, score_method='likelihood', threshold=3):
        pb = self._probase
        if concept not in pb.concept2idx:
            return []
        concept_idx = pb.concept2idx[concept]
        concept_freq = self.concept_freq_map[concept]
        inst_list = pb.concept_inverted_list[concept_idx]
        rst_list = []
        for inst_idx, co_occurrence in inst_list:
            if co_occurrence <= threshold:
                continue
            inst = pb.idx2instance[inst_idx]
            if score_method == "pmi":
                score = co_occurrence / self.inst_freq_map[inst] / concept_freq
            elif score_method == "likelihood":
                score = co_occurrence / concept_freq
            elif score_method == "co_occurrence":
                score = co_occurrence
            else:
                raise NotImplementedError(score_method)
            rst_list.append((inst, score))
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def conceptualize(self, text, concepts, try_alt_dets, score_method='likelihood',
                      top_k=40, abs_rate=0.5, max_entity_word_inc=2):
        def produce(entity):
            n_words = len(entity.split(' '))
            n_abs = int(top_k * abs_rate + 0.5)
            abs = self.entity_abstract(entity, score_method=score_method)
            abs = abs[:n_abs]
            abs = [(c, s) for c, s in abs if len(c.split(' ')) <= n_words + max_entity_word_inc]
            inst = self.entity_instantiate(entity, score_method=score_method)
            inst = inst[:top_k - n_abs]
            inst = [(c, s) for c, s in inst if len(c.split(' ')) <= n_words + max_entity_word_inc]
            return abs + inst
        return self.collect_substitution(text, concepts, produce, try_alt_dets)

    def random_substitution(self, text, concepts, mode: str, top_k=10000, n_pre_roll=100,
                            weighted=True, max_entity_word_inc=2):
        candidates = []
        if mode == 'entity':
            candidates = self.entity_freqs[:top_k]
        elif mode == 'concept':
            candidates = self.concept_freqs[:top_k]
        elif mode == 'instance':
            candidates = self.inst_freqs[:top_k]
        else:
            raise ValueError("Unsupported mode: " + str(mode))
        n_pre_roll = min(n_pre_roll, top_k)
        if weighted:
            weights = [c[1] for c in candidates]
            weights = np.asarray(weights) / sum(weights)
            indices = list(range(len(candidates)))
            indices = np.random.choice(indices, size=n_pre_roll, replace=False, p=weights).tolist()
        else:
            indices = np.random.choice(candidates, size=n_pre_roll, replace=False).tolist()
        candidates = [candidates[c] for c in indices]
        n_words = len(text.split(' '))
        candidates = [(c, s) for c, s in abs if len(c.split(' ')) <= n_words + max_entity_word_inc]
        results = self.collect_substitution(text, concepts, lambda _: candidates, False)
        return results

if __name__ == '__main__':
    proposer = Proposer()
