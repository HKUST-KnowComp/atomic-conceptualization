import copy
import os, json
import pickle
import random
from collections import defaultdict
import re
import traceback

try:
    from build_exp_data import sequence_sample
    import datasource
    from build_exp_data import stage2_data, stage3_data, prepare_stage2_nl, convert_item, relation_mapping
    source = datasource.DataSource()
except:
    traceback.print_exc()
import tqdm

def export_cand_heads():
    cands = json.load(open(os.path.join(datasource.output_path, 'all_valid_cands.json')))
    annotated_cands = set([t['sent'] for t in stage2_data])
    results = []
    for d_i, c_i in tqdm.tqdm(cands):
        _, _, text = source.get_sent_concept(d_i, c_i)
        is_annotated = text in annotated_cands
        concepts = source._concepts[d_i][c_i]
        all_aligns = []
        for v_i in range(len(concepts)):
            aligns = source.get_pb_alignments(d_i, c_i, v_i, False)
            all_aligns.append(aligns)
        if d_i == 236 and c_i == 1:
            print(all_aligns)
        results.append({"sent": text, "d_i": d_i, "c_i": c_i, "aligns": all_aligns, "in_anno": is_annotated})
    print("Total %d heads annotated, %d unannotated" % (len(annotated_cands), len(results)))
    json.dump(results, open(os.path.join('data', 'extend_cand_head.json'), 'w'), indent=1)

def prepare_head_gen():
    heads = json.load(open(os.path.join('data', 'extend_cand_head.json')))
    results = []
    for h in heads:
        d_i = h['d_i']
        c_i = h['c_i']
        comp_i = source._components[d_i][c_i]
        comp = source._docs[d_i][comp_i]
        results.append({'sent_a': h['sent'], 'sent_b': '', 'label': 1,
                        'info': {'dep': comp.dep, 'in_anno': h['in_anno'],
                                 'd_i': h['d_i'], 'c_i': h['c_i'], 'aligns': h['aligns']}})

    json.dump(results, open(os.path.join('data', 'exp', 'extend_cand_head.json'), 'w'), indent=1)

def prepare_extend_concepts(extend_num=10):
    heads = json.load(open(os.path.join('data', 'exp', 'extend_cand_head.json')))
    extensions = []
    for h in tqdm.tqdm(heads):
        concepts = {'sub': [], 'parent': [], 'sibling': []}
        for v_i, v_aligns in enumerate(h['info']['aligns']):
            for align in v_aligns:
                concepts['sub'].append({'concept': align, 'var_i': v_i})
                parents = source._pb.query(align, truncate=extend_num, target='abs')
                for p, info in parents.items():
                    concepts['parent'].append({'concept': p, 'var_i': v_i, 'base': align, 'score': info['mi']})
        extensions.append(concepts)

    json.dump(extensions, open(os.path.join('data', 'concept_extensions.json'), 'w'), indent=1)

def prepare_extended_head_abs():
    heads = json.load(open(os.path.join('data', 'exp', 'extend_cand_head.json')))
    extensions = json.load(open(os.path.join('data', 'concept_extensions.json')))

    results = []
    n_sub = 0
    n_parent = 0
    n_sibling = 0
    for i, h in enumerate(tqdm.tqdm(heads)):
        ext = extensions[i]
        n_sub += len(ext['sub'])
        n_parent += len(ext['parent'])
        n_sibling += len(ext['sibling'])
        concepts = set([t['concept'] for t in ext['sub'] + ext['parent']])
        for c in concepts:
            t = copy.copy(h)
            t['sent_b'] = c
            results.append(t)
    print("Total %d extended heads from %d original, %.3f per head (%.3f sub, %.3f parent, %.3f sibling)" % (
        len(results), len(heads), len(results) / len(heads),
        n_sub / len(heads), n_parent / len(heads), n_sibling / len(heads)))

    json.dump(results, open(os.path.join('data', 'exp', 'extended_heads.json'), 'w'), indent=1)

def prepare_extended_head_from_gen(base_path, source_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    all_gens = {}
    n_heads = 0
    for split in ['trn', 'tst', 'dev']:
        n_fail = 0
        lines = open(os.path.join(base_path, split, split + '.tsv_pred_generations.jsonl')).read().splitlines()
        sources = open(os.path.join(source_path, split + '.json')).read().splitlines()
        sources = [json.loads(t) for t in sources]
        heads = dict([(s['sent_a'], copy.copy(s)) for s in sources]) # sources contain some duplicates
        n_heads += len(heads)
        source_i = 0
        used_source = {}
        all_info = defaultdict(list)
        for gen_i in range(len(lines)):
            while sources[source_i]['sent_a'] in used_source:
                all_info[sources[source_i]['sent_a']].append(sources[source_i]['info'])
                source_i += 1
            sent_a = sources[source_i]['sent_a']
            all_info[sent_a].append(sources[source_i]['info'])
            used_source[sent_a] = source_i

            gen = json.loads(lines[gen_i])
            gen_head = gen['source'].replace("<c>", "[").replace("</c>", "]").replace("[GEN]", "").split('is an instance of')[0].replace(' ', '').split('.')[0]
            assert gen_head == sources[source_i]['sent_a'].replace(' ', '')

            tails = []
            for tail in gen['generations']:
                try:
                    tail = tail.split('[GEN]')[1].split('[EOS]')[0].strip()
                    tails.append(tail)
                except:
                    n_fail += 1
            heads[sent_a]['sent_b'] = list(set(tails))
            all_gens[sent_a] = heads[sent_a]
            source_i += 1
        print("Total %d sources, %d failed" % (len(used_source), n_fail))

        fw = open(os.path.join(target_path, split + '.json'), 'w')
        for k, triple in heads.items():
            tails = triple['sent_b']
            triple['info']['all_info'] = [copy.copy(t) for t in all_info[triple['sent_a']]]
            for t in tails:
                triple['sent_b'] = t
                fw.write(json.dumps(triple) + '\n')

def summarize_valid_head(base_paths, data_paths, out_path, ex_out_path=None, thres=0.8, filter_target=True):
    os.makedirs(out_path, exist_ok=True)
    if ex_out_path:
        os.makedirs(ex_out_path, exist_ok=True)
    for split in ['trn', 'dev', 'tst']:
        results = defaultdict(lambda : defaultdict(set))
        scores = {}
        for base_path, data_path in zip(base_paths, data_paths):
            data = open(os.path.join(data_path, split + '.json')).read().splitlines()
            data = [json.loads(l) for l in data]
            score = pickle.load(open(os.path.join(base_path, split, 'test_outputs.pickle'), 'rb'))
            score = list(score[0])
            n_triple = 0
            n_cand = 0
            n_invalid = 0
            appeared_triples = set()
            appeared_bases = set()
            for d, s in zip(data, score):
                n_cand += 1
                if filter_target and not source._pb.query(d['sent_b'], target='inst'):
                    n_invalid += 1
                    if (n_invalid < 1000 and n_invalid % 100 == 0) or n_invalid < 10:
                        print("Invalid found:", d['sent_a'], d['sent_b'])
                    continue

                scores[(d['sent_a'], d['sent_b'])] = s.item()
                if s > thres:
                    results[d['sent_a']][d['sent_b']].add((d['info']['d_i'], d['info']['c_i']))
                    if 'all_info' in d['info']:
                        for info in d['info']['all_info']:
                            results[d['sent_a']][d['sent_b']].add((info['d_i'], info['c_i']))
                    n_triple += 1
                    if n_triple < 20 or (n_triple < 1000 and n_triple % 100 == 0):
                        print(d['sent_a'], '->', d['sent_b'], s)
                    appeared_triples.add((d['sent_a'], d['sent_b']))
                    appeared_bases.add(d['sent_a'])
            print(split, base_path, '%d tuple valid, out of %d examined, rate %.3f, %d unique tuples, %d invalid' %
                  (n_triple, n_cand, n_triple / n_cand, len(appeared_triples), n_invalid))
            print("%d sent_a, %.2f branch per sent_a" %
                  (len(appeared_bases), len(appeared_triples) / len(appeared_bases)))
            print("Now total %d sent_a, %d total tuple" % (len(results), sum([len(t) for t in results.values()])))
            pass
        n_abs = n_multi = n_branch = n_nonroot = 0
        for head in results:
            for tail in list(results[head]):
                results[head][tail] = list(results[head][tail])
                n_abs += 1
                if len(results[head][tail]) > 1:
                    n_multi += 1
                if not (head[0] == '[' and head[-1] == ']'):
                    n_nonroot += 1
                n_branch += len(results[head][tail])
        print("Total %d abstractions; %d with multiple origins; %.3f origins in average; %d not root" %
              (n_abs, n_multi, n_branch / n_abs, n_nonroot))
        json.dump(results, open(os.path.join(out_path, split + '.json'), 'w'))

        if ex_out_path is not None:
            scores = list(scores.items())
            scores.sort(key=lambda x: -x[-1])
            fw = open(os.path.join(ex_out_path, split + '.json'), 'w')
            for (head, tail), s in scores:
                fw.write(json.dumps({"sent_a": head, "sent_b": tail, "info": {"score": s}}) + '\n')


def prepare_triple_gen(base_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    for split in ['dev', 'tst', 'trn']:
        data = json.load(open(os.path.join(base_path, split + '.json')))
        heads = {}
        fw = open(os.path.join(out_path, split + '.json'), 'w')
        n_sub = 0
        for base, ents in data.items():
            for ent in ents:
                n_sub += 1
                sub_head = re.sub(r"\[.*\]", '[' + ent + ']', base)
                for relation in relation_mapping:
                    heads[(sub_head, relation)] = {'sent_a': sub_head, 'sent_b': '', 'relation': relation, 'label': 1,
                                       'info': {'d_i': ents[ent][0][0], 'c_i': ents[ent][0][1], 'source': 'c'}}
        print(split, "%d heads built from %d subs on %d bases" % (len(heads), n_sub, len(data)))
        for v in heads.values():
            v = convert_item(v)
            fw.write(json.dumps(v) + '\n')

def prepare_triple_scoring(atomic_path, base_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    base_data = json.load(open(atomic_path))
    base_triples = defaultdict(set)
    for t in base_data:
        base_triples[t['info']['d_i']].add((t['relation'], t['sent_b']))
    for split in ['dev', 'tst', 'trn']:
        data = json.load(open(os.path.join(base_path, split + '.json')))
        results = defaultdict(list)
        fw = open(os.path.join(out_path, split + '.json'), 'w')
        n_sub = 0
        n_triple = 0
        n_source = n_multi = 0
        for base, ents in data.items():
            for ent in ents:
                n_sub += 1
                sub_head = re.sub(r"\[.*\]", '[' + ent + ']', base)
                for d_i, c_i in ents[ent]:
                    if d_i not in base_triples:
                        print(d_i)
                        continue
                    for relation, sent_b in base_triples[d_i]:
                        results[(sub_head, relation, sent_b)].append((d_i, c_i))
                        n_triple += 1
        print(split, "%d triples built from %d subs making %d triples on %d bases" % (len(results), n_sub, n_triple, len(data)))

        for (sent_a, relation, sent_b), infos in results.items():
            v = {'sent_a': sent_a, 'relation': relation, 'sent_b': sent_b, 'label': 1,
                 'info': {'d_i': infos[0][0], 'c_i': infos[0][1], 'all_info': infos, 'source': 'c'}}
            if len(infos) > 1:
                n_multi += 1
            n_source += len(infos)
            v = convert_item(v)
            fw.write(json.dumps(v) + '\n')
        print("Avg %.3f sources, %d multi among %d" % (n_source / len(results), n_multi, len(results)))

def summarize_valid_triples(base_path, data_path, out_path, thres=0.5):
    os.makedirs(out_path, exist_ok=True)
    for split in ['dev', 'tst', 'trn']:
        results = defaultdict(lambda : defaultdict(list))
        data = open(os.path.join(data_path, split + '.json')).read().splitlines()
        data = [json.loads(l) for l in data]
        score = pickle.load(open(os.path.join(base_path, split, 'test_outputs.pickle'), 'rb'))
        score = list(score[0])
        scores = {}
        n_triple = 0
        n_cand = 0
        appeared_triples = set()
        appeared_bases = set()
        for d, s in tqdm.tqdm(zip(data, score)):
            n_cand += 1
            s = s.item()
            if len(results) < 200:
                print(d['sent_a'], '->', d['sent_b'], s)
            if s > thres:
                results[d['sent_a']][d['sent_b']].extend(d['info']['all_info'])
                assert (d['sent_a'], d['sent_b']) not in scores or abs(scores[(d['sent_a'], d['sent_b'])] - s) < 1e-6
                scores[(d['sent_a'], d['sent_b'])] = s
                n_triple += 1
                appeared_triples.add((d['sent_a'], d['sent_b']))
                appeared_bases.add(d['sent_a'])

        print(split, base_path, n_triple, 'triple out of', n_cand, 'rate %.3f, %d unique' % (n_triple / n_cand, len(appeared_triples)), 'now total', len(results))
        print("Total %d bases, %.2f branch per base" % (len(appeared_bases), len(appeared_triples) / len(appeared_bases)))
        fw = open(os.path.join(out_path, split + '.json'), 'w')
        outputs = []
        for head in results:
            for tail in list(results[head]):
                all_info = results[head][tail]
                all_info = list(set([tuple(t) for t in all_info]))
                v = {'sent_a': head, 'sent_b': tail, 'info': {'d_i': all_info[0][0], 'c_i': all_info[0][1],
                                                              'all_info': all_info, 'score': scores[(head, tail)]}}
                outputs.append(v)
        outputs.sort(key=lambda x: -x['info']['score'])
        for v in outputs:
            fw.write(json.dumps(v) + '\n')

if __name__ == '__main__':
    pass
    # Part 1: Data preparation
    os.makedirs(os.path.join('data', 'exp'), exist_ok=True)
    export_cand_heads()
    prepare_head_gen()
    prepare_stage2_nl(os.path.join('data', 'exp', 'extend_cand_head.json'), os.path.join('data', 'exp', 'extend_cand_head.nl.json'))
    prepare_extend_concepts()
    prepare_extended_head_abs()

    # Now the extended heads (ready for scoring and generation) are saved to data/exp
    # Run data conversion for concept generation, see README

    # Part 2: Collect generated candidates
    prepare_extended_head_from_gen(r"../generator/population/extend_cand_head.nl",
                                   r"../discriminator/split_atomic/extend_cand_head",
                                   r"../discriminator/split_atomic/generated_heads")


    # Part 3: Collect results from conceptualization verification, and prepare for inference verification.
    summarize_valid_head([r"../discriminator/population/head_gen/",
                          r"../discriminator/population/head/"],
                         [r"../discriminator/split_atomic/generated_heads/",
                          r"../discriminator/split_atomic/extended_heads/"],
                         r"../discriminator/population/selected_heads",
                         ex_out_path=r"../discriminator/population/all_head_scores")

    prepare_triple_scoring(r"../discriminator/data/atomic_all.json",
                           r"../discriminator/population/selected_heads",
                           r"../discriminator/split_atomic/extended_triples")

    # Part 4: Collect results from inference verification

    summarize_valid_triples(r"../discriminator/population/triple",
                            r"../discriminator/split_atomic/extended_triples",
                            r"../discriminator/population/selected_triples_90", thres=0.9)
