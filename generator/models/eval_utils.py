import json
import pandas as pd
import numpy as np
from typing import List
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from collections import defaultdict

def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def get_gen(strs):
    strs = strs.split()
    st = 0
    ed = 0
    for i in range(len(strs)):
        if strs[i] == "[GEN]":
            st = i
        if strs[i] == "[EOS]":
            ed = i
            break
    return " ".join(strs[st+1:ed])

def get_prefix_scores(refs, hyps):
    results = [0, 0, 0, 0]
    for k in refs:
        if k not in refs or not refs[k]:
            continue
        best_bleu = -1
        best_bleus = None
        keys = ['a ', 'an ', 'the ', "'", '"', '.', '�', '“', '”']
        hyp = hyps[k][0]
        hyp = hyp.split('.')[0]
        for n in keys:
            hyp = hyp.replace(n, '')
        hyp = hyp.strip()
        best_bleus = Bleu(4).compute_score({'_': refs[k]}, {'_': [hyp]})[0]
        # hyp = hyp.strip().split(' ')
        # # for i in range(1, len(hyp) + 1):
        # #     t = ' '.join(hyp[:i])
        # #     t_bleu = Bleu(4).compute_score({'_': refs[k]}, {'_': [t]})
        # #     if t_bleu[0][1] > best_bleu:
        # #         best_bleu = t_bleu[0][1]
        # #         best_bleus = t_bleu[0]
        for t in range(4):
            results[t] += best_bleus[t]
    results = [t / len(refs) for t in results]
    return results

def get_generation_eval_scores(write_path, selected_ground, check_prefix=False):
    pred = read_jsonl_lines(write_path)
    if isinstance(selected_ground, str):
        selected_ground = pd.read_csv(selected_ground, sep="\t")

    hyps = defaultdict(list)
    refs = defaultdict(list)
    for p in pred:
        p['source'] = p['source'].replace("<c> ", "<c>").replace(" </c>", "</c>")\
            .replace(" <c>-", "<c>-").replace("-</c> ", "-</c>")
        gen_tail = p['generations'][0][len(p['source']):]
        if 'GEN]' in gen_tail:
            gen_tail = gen_tail.split('GEN]')[1]
        gen_tail = gen_tail.split('<|endoftext|>')[0].split("[EOS]")[0].strip()
        hyps[p['source']].append(gen_tail)
    for i in range(len(selected_ground)):
        if not isinstance(selected_ground.loc[i, 'tail_event'], str) and np.isnan(selected_ground.loc[i, 'tail_event']):
            selected_ground.loc[i, 'tail_event'] = "nan"
        key = selected_ground.loc[i, "head_event"]
        key = key.replace(" ,", ",")
        if key in hyps:
            tail = selected_ground.loc[i, 'tail_event'].replace(" [EOS]", "")
            refs[key].append(tail)
        else:
            print("Missing hyp for eval:", key)
    for k in hyps:
        if k not in refs:
            print("Missing reference in eval:", k)
            print(hyps[k])

    if check_prefix:
        bleus = get_prefix_scores(refs, hyps)
    else:
        bleus = Bleu(4).compute_score(refs, hyps)[0]
    bleus = [str(round(score, 5)) for score in bleus]
    cider = Cider().compute_score(refs, hyps)[0]
    meteor = Meteor().compute_score(refs, hyps)[0]
    rouge = Rouge().compute_score(refs, hyps)[0]
    metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    scores = bleus + [str(round(meteor, 3)), str(round(rouge, 3)), str(round(cider, 3))]
    return metrics, scores


def get_reference_sentences(filename):
    result = []
    with open(filename) as file:
        for line in file:
            result.append([x.strip() for x in line.split('\t')[1].split('|')])
    return result

def postprocess(sentence):
    return sentence

def get_heads_and_relations(filename):
    result = []
    with open(filename) as file:
        for line in file:
            line = line.split('\t')[0]
            head_event = line.split('@@')[0].strip()
            relation = line.split('@@')[1].strip()
            to_add = {
                'head': head_event,
                'relation': relation
            }
            result.append(to_add)
    return result