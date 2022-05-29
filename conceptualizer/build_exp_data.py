import copy
import glob
import json, random, os
import shutil


import tqdm
from collections import defaultdict
import re, traceback

def sequence_sample(data, n):
    print("Actual sample rate:", n / len(data), ';', len(data), '->', n)
    results = []
    while n > len(data):
        results.extend(data)
        n -= len(data)
    random.seed(0)
    random.shuffle(data)
    results.extend(data[:n])
    return results

import pandas

try:
    from atomic_parse_utils import ProbaseClient

    stage2_data = json.load(open(os.path.join('data', 'stage2_combined.json')))
    name = 'stage3_combined.neg2.json'
    print("Current stage3 data:", name)
    stage3_data = json.load(open(os.path.join('data', name)))
except:
    traceback.print_exc()


def build_stage2_full(out_file='stage2_all.json', require_stage=None):
    data = []
    n_pos = n_neg = 0
    for item in tqdm.tqdm(stage2_data):
        for sub in item['subs']:
            if sub['score'] >= 4:
                label = 1
            elif sub['score'] <= 1:
                label = 0
            else:
                continue
            if require_stage == 2 and sub['is_stage1']:
                continue
            if require_stage == 1 and not sub['is_stage1']:
                continue
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1
            data.append({'sent_a': item['sent'], 'sent_b': sub['sub'],
                         'label': label, 'info': {'d_i': item['d_i'], 'c_i': item['c_i']}})
    print("Total %d, %d positive, %d negative" % (len(data), n_pos, n_neg))
    json.dump(data, open(os.path.join('data', 'exp', out_file), 'w'), indent=1)

def downsampling(in_file, out_file, n_samples):
    random.seed(0)
    data = json.load(open(os.path.join('data', 'exp', in_file)))
    random.shuffle(data)
    data = data[:n_samples]
    print("in_file: %s, out_file: %s, %d samples" % (in_file, out_file, n_samples))
    json.dump(data, open(os.path.join('data', 'exp', out_file), 'w'), indent=1)

def get_concept_negative_sample(positives, base, ent, n_neg=5, n_related_bound=5, include_rel=('parent')):
    global pb
    if pb is None:
        pb = ProbaseClient()
    cands = set()
    parent = None
    if 'parent' in include_rel or 'sibling' in include_rel:
        abs = pb.query(ent, 'mi', target='abs')
        abs = list(abs.items())
        abs.sort(key=lambda x: -x[1]['mi'])
        if 'parent' in include_rel:
            cands = cands.union([a[0] for a in abs[:n_related_bound]])
        if len(abs) > 0:
            parent = abs[0][0]
    if 'child' in include_rel:
        inst = pb.query(ent, 'mi', target='inst')
        inst = list(inst.items())
        inst.sort(key=lambda x: -x[1]['mi'])
        cands = cands.union([a[0] for a in inst[:n_related_bound]])
    if 'sibling' in include_rel and parent:
        inst = pb.query(parent, 'mi', target='inst')
        inst = list(inst.items())
        inst.sort(key=lambda x: -x[1]['mi'])
        cands = cands.union([a[0] for a in inst[:n_related_bound]])
    for c in list(cands):
        if c in positives[base]:
            cands.remove(c)
    cands = list(cands)
    random.shuffle(cands)
    cands = cands[:n_neg]
    return cands

def get_all_negative_samples(bases, positives, es_rand, all_ents, n_es_neg, n_cs_neg, n_related_bound, include_rel):
    data = []
    cs_samples = []
    branch_cnt = defaultdict(int)
    for a, item in tqdm.tqdm(bases):
        data.append({'sent_a': item['sent'], 'sent_b': a,
                     'label': 1, 'info': {'d_i': item['d_i'], 'c_i': item['c_i']}})
        es = es_rand.choices(all_ents, k=n_es_neg * 2)
        es = [e for e in es if e not in positives[item['sent']]][:n_es_neg]
        for e in es:
            data.append({'sent_a': item['sent'], 'sent_b': e,
                         'label': 0, 'info': {'d_i': item['d_i'], 'c_i': item['c_i'], 'type': 'es'}})
        css = get_concept_negative_sample(positives, item['sent'], a, n_cs_neg * 2, n_related_bound, include_rel)
        cs_samples.append((item, css))
        branch_cnt[len(css)] += 1
    n_rem = n_cs_neg * len(bases)
    current_i = 0
    print("Preparing collecting CS samples, %d targets, %d bases, branches: %s" % (n_rem, len(bases), str(branch_cnt)))
    random.seed(0)
    while n_rem > 0:
        random.shuffle(cs_samples)
        for item, c in cs_samples:
            if current_i < len(c):
                data.append({'sent_a': item['sent'], 'sent_b': c[current_i],
                             'label': 0, 'info': {'d_i': item['d_i'], 'c_i': item['c_i'], 'type': 'cs'}})
                n_rem -= 1
                if n_rem == 0:
                    break
        current_i += 1
    random.seed(0)
    random.shuffle(data)
    return data

def build_stage2_from_gt_rule_ns(out_file, n_es_neg=3, n_cs_neg=3, n_related_bound=6, include_rel=('parent', 'child', 'sibling')):
    random.seed(0)
    es_rand = random.Random(0)
    positives = defaultdict(set)
    all_ents = set()
    for item in stage2_data:
        for a in item['aligns']:
            positives[item['sent']].add(a)
            all_ents.add(a)
    all_ents = list(all_ents)
    bases = []
    for item in tqdm.tqdm(stage2_data):
        for a in item['aligns']:
            bases.append((a, item))
    data = get_all_negative_samples(bases, positives, es_rand, all_ents, n_es_neg, n_cs_neg,
                                    n_related_bound, include_rel)
    json.dump(data, open(os.path.join('data', 'exp', out_file), 'w'), indent=1)


def build_stage2_from_positive_ns(out_file, n_es_neg=3, n_cs_neg=3, n_related_bound=6, include_rel=('parent', 'child', 'sibling'), stage1_only=False):
    random.seed(0)
    es_rand = random.Random(0)
    positives = defaultdict(set)
    all_ents = set()
    for item in stage2_data:
        for a in item['subs']:
            if stage1_only and not a['is_stage1']:
                continue
            if a['score'] >= 4:
                positives[item['sent']].add(a['sub'])
            all_ents.add(a['sub'])
    all_ents = list(all_ents)
    bases = []
    for item in tqdm.tqdm(stage2_data):
        for a in item['subs']:
            if stage1_only and not a['is_stage1']:
                continue
            if a['score'] >= 4:
                bases.append((a['sub'], item))

    data = get_all_negative_samples(bases, positives, es_rand, all_ents, n_es_neg, n_cs_neg,
                                    n_related_bound, include_rel)

    json.dump(data, open(os.path.join('data', 'exp', out_file), 'w'), indent=1)

def build_stage2_by_dep():
    data = defaultdict(list)
    dep_pos = defaultdict(int)
    dep_neg = defaultdict(int)
    n_pos = n_neg = 0
    for item in tqdm.tqdm(stage2_data):
        for sub in item['subs']:
            if sub['score'] >= 4:
                label = 1
            elif sub['score'] <= 1:
                label = 0
            else:
                continue

            if label == 1:
                n_pos += 1
                dep_pos[item['dep']] += 1
            else:
                n_neg += 1
                dep_neg[item['dep']] += 1
            data[item['dep']].append({'sent_a': item['sent'], 'sent_b': sub['sub'],
                         'label': label, 'info': {'d_i': item['d_i'], 'c_i': item['c_i']}})
    n = n_pos + n_neg
    for k in data:
        print("%s: total %d, %d pos, %d neg, ratio=%.3f (%.3f)" % (k, len(data[k]), dep_pos[k], dep_neg[k], dep_pos[k] / dep_neg[k], len(data[k]) / n))
        json.dump(data[k], open(os.path.join('data', 'exp', 'stage2.%s.json' % k), 'w'), indent=1)

def collect_atomic_split():
    df = pandas.read_csv(r"C:\Users\dy.octa\Store\atomic\v4_atomic_all_agg.csv")
    head_to_split = {}
    for i, r in df.iterrows():
        head_to_split[i] = r['split']
    json.dump(head_to_split, open(os.path.join('data', 'exp', 'head_to_split.json'), 'w'), indent=1)

relation_mapping = {'xIntent': "PersonX's intention is", "xNeed": "Before that, PersonX needs",
               "xAttr": "PersonX will be described as", "xEffect": "Effects on PersonX will be",
               "xWant": "After that, PersonX wants", "xReact": "After that, PersonX feels",
               "oReact": "After that, others feel", "oWant": "After that, others want",
               "oEffect": "Effects on others will be"
               }
def relation_to_text(relation):
    return relation_mapping[relation]

def convert_item(item):
    info = copy.copy(item['info'])
    # info['head'] = item['sent_a']
    # info['relation'] = item['relation']
    # info['tail'] = item['sent_b']
    return {"sent_a": item['sent_a'], "sent_b": relation_to_text(item['relation']) + ': ' + item['sent_b'],
            "label": item['label'], 'info': info}

def build_base_atomic():
    atomic_data = pandas.read_csv(r"data\v4_atomic_all_agg.csv")
    atomic_events = open(r"parse\heads.5.txt").read().splitlines()
    atomic_exclude_heads = json.load(open(r"parse\exclusions.json"))
    result = []
    n_heads = 0
    for i, r in atomic_data.iterrows():
        if i in atomic_exclude_heads:
            continue
        if '_' in atomic_events[i]:
            continue
        n_heads += 1
        for key in list(atomic_data):
            if key in ['event', 'prefix', 'split']:
                continue
            tails = set(json.loads(r[key]))
            for t in tails:
                if t.lower() in ['', 'none', 'n/a']:
                    continue
                result.append({'sent_a': atomic_events[i], 'relation': key, 'sent_b': t, 'label': 1,
                               'info': {'d_i': i}})
    print("Original %d heads, now %d, %d triplets" % (len(atomic_data), n_heads, len(result)))
    json.dump(result, open(os.path.join('data', 'stage3', 'atomic_all.json'), 'w'), indent=1)

def downsample_by_events(data, n_events, use_head_text=False):
    random.seed(0)
    if use_head_text:
        events = set([t['sent_a'] for t in data])
    else:
        events = set([t['info']['d_i'] for t in data])
    events = list(events)
    rand_ds = sequence_sample(events, n_events)
    bins = defaultdict(list)
    for t in data:
        if use_head_text:
            bins[t['sent_a']].append(t)
        else:
            bins[t['info']['d_i']].append(t)
    results = []
    for e in rand_ds:
        results.extend(bins[e])
    return results

def downsample_by_triple(data, n_samples):
    return sequence_sample(data, n_samples)

def build_downsampled_atomic(n_event=2756):
    atomic_data = json.load(open(os.path.join('data', 'stage3', 'atomic_all.json')))
    rand_ds_data = downsample_by_events(atomic_data, n_event)
    print("Random downsampled, %d heads, %d triples" % (n_event, len(rand_ds_data)))
    json.dump(rand_ds_data, open(os.path.join('data', 'stage3', 'atomic.ds.rand.json'), 'w'), indent=1)

    stage3_d = set([t['info']['d_i'] for t in stage3_data])
    stage3_ds_data = [t for t in atomic_data if t['info']['d_i'] in stage3_d]
    print("Keep pre-conceptualize S3 heads only in Atomic, %d heads, %d triples" % (len(stage3_d), len(stage3_ds_data)))
    json.dump(stage3_ds_data, open(os.path.join('data', 'stage3', 'atomic.ds.s3.json'), 'w'), indent=1)

def build_stage3_mix(atomic_in, out_file, n_c, n_a, ds_mode, ds_limit, positive_only, stage_mode='mix12',
                     match_original_head=False):
    if n_c + n_a == ds_limit:
        n_s3 = n_c
        n_atomic = n_a
    else:
        n_s3 = int(ds_limit * (n_c / (n_c + n_a)))
        n_atomic = int(ds_limit * (n_a / (n_c + n_a)))

    atomic_data = json.load(open(atomic_in))
    s3_data = [s for s in stage3_data]
    if stage_mode == 's1':
        s3_data = [s for s in s3_data if s['info']['is_stage1']]
    elif stage_mode == 's2':
        s3_data = [s for s in s3_data if not s['info']['is_stage1']]
    if positive_only:
        s3_data = [s for s in s3_data if s['label'] == 1]


    if ds_mode == 'triple':
        s3_data = downsample_by_triple(s3_data, n_s3)
    elif ds_mode == 'head':
        s3_data = downsample_by_events(s3_data, n_s3, use_head_text=True)
    elif ds_mode == 'original':
        s3_data = downsample_by_events(s3_data, n_s3, use_head_text=False)
    else:
        raise ValueError("Invalid mode: " + str(ds_mode))
    if match_original_head:
        base_events = set([t['info']['d_i'] for t in s3_data])
        atomic_data = [t for t in atomic_data if t['info']['d_i'] in base_events]

    if ds_mode == 'triple':
        atomic_ds = downsample_by_triple(atomic_data, n_atomic)
    elif ds_mode == 'head':
        atomic_ds = downsample_by_events(atomic_data, n_atomic)
    elif ds_mode == 'original':
        atomic_ds = downsample_by_events(atomic_data, n_atomic)

    for s in atomic_ds:
        s['info']['source'] = 'a'
    for s in s3_data:
        s['info']['source'] = 'c'
    n_s3_heads = len(set([t['sent_a'] for t in s3_data]))
    n_original_heads = len(set([t['info']['d_i'] for t in s3_data]))
    n_atomic_heads = len(set([t['info']['d_i'] for t in atomic_ds]))
    n_neg = len([t for t in s3_data if t['label'] == 0])
    n_atomic_neg = len([t for t in atomic_ds if t['label'] == 0])

    results = atomic_ds + s3_data
    random.seed(0)
    random.shuffle(results)
    print("Saving to %s, %d triples, %d head/%d triples from atomic, %d neg; %d head/%d triples/%d o-head from stage3, %d neg" %
          (out_file, len(results), n_atomic_heads, len(atomic_ds), n_atomic_neg, n_s3_heads, len(s3_data), n_original_heads, n_neg))

    json.dump(results, open(os.path.join('data', 'stage3', 'mix', out_file), 'w'), indent=1)

def prepare_stage3_mixtures(ds_triples=20000):
    # Stage3: 71036 triples, 33723 for stage 1; 2756 heads, 1348 for stage 1; 688 original heads, 676 for stage 1
    # Atomic mode: random collected events (total 2756 heads, 81410 triples), or use the original heads only
    # Mix mode: different ratios, use the same number of heads/triples/original heads; whether use stage 1 only; whether include negative samples

    # Current scheme: all downsample to 20K triples; for atomic.s3, ensure that they are using a same set of heads

    # Stage3 data
    count_map = {'s1': (1348, 33723, 676), 's2': (1461, 37313, 509), 'mix12': (2756, 71036, 688)}
    for stage_mode in ['s1', 's2', 'mix12']:
        n_head, n_triple, n_original_head = count_map[stage_mode]
        # ds_modes = [('head', n_head), ('triple', n_triple), ('original', n_original_head)]
        ds_modes = [('triple', min(n_triple, ds_triples))]
        for positive_only in [True]:
            # Atomic data
            for atomic_mode in ['s3', 'rand']:
                # Mix method
                for ds_mode, limit in ds_modes:
                    for s3_part in [1, 2, 3]:
                        atomic_part = 4 - s3_part
                        name = "s3_%s_%d.atomic_%s_%d.%s%s.json" % (
                            stage_mode, s3_part, atomic_mode, atomic_part, ds_mode,
                            '.ineg' if not positive_only else "")
                        am_path = os.path.join('data', 'stage3', 'atomic.ds.%s.json' % atomic_mode)
                        build_stage3_mix(am_path,
                                         name, s3_part, atomic_part, ds_mode=ds_mode, ds_limit=limit,
                                         positive_only=positive_only, stage_mode=stage_mode,
                                         match_original_head=atomic_mode == 's3')
            print()
            # No atomic data
            name = "s3_%s_%d.%s%s.json" % (
                stage_mode, 4, ds_mode,
                '.ineg' if not positive_only else "")
            build_stage3_mix(os.path.join('data', 'stage3', 'atomic.ds.rand.json'),
                             name, 4, 0, ds_mode=ds_mode, ds_limit=limit,
                             positive_only=positive_only, stage_mode=stage_mode)
            print()


    # No s3 data
    print()
    for atomic_mode in ['rand', 's3']:
        # Mix method
        for ds_mode, limit in ds_modes:
            name = "atomic_%s_%d.%s.json" % (
                atomic_mode, 4, ds_mode)
            build_stage3_mix(os.path.join('data', 'stage3', 'atomic.ds.%s.json' % atomic_mode),
                             name, 0, 4, ds_mode=ds_mode, ds_limit=limit,
                             positive_only=positive_only, stage_mode='mix12')

def prepare_stage3_ineg_mixtures(ds_triples=20000):
    # Current scheme: Use fix number of 32K S3 triples, in combination with 32K atomic triples;
    # vs. all atomic (63K), 32K S3 only

    # Stage3 data
    for stage_mode in ['s1', 's2', 'mix12']:
        # ds_modes = [('head', n_head), ('triple', n_triple), ('original', n_original_head)]
        ds_modes = [('triple', ds_triples)]
        for positive_only in [False, True]:
            # Atomic data
            for atomic_mode in ['rand']:
                # Mix method
                for ds_mode, limit in ds_modes:
                    for s3_part in [1, 2, 3, 4]:
                        atomic_part = 8 - s3_part
                        name = "20k.s3_%s_%d.atomic_ns_%s_%d.%s%s.json" % (
                            stage_mode, s3_part, atomic_mode, atomic_part, ds_mode,
                            '.ineg' if not positive_only else "")
                        am_path = os.path.join('data', 'stage3', 'ns.atomic.ds.%s.json' % atomic_mode)
                        build_stage3_mix(am_path,
                                         name, s3_part, atomic_part, ds_mode=ds_mode, ds_limit=limit,
                                         positive_only=positive_only, stage_mode=stage_mode,
                                         match_original_head=atomic_mode == 's3')
            print()
            # # No atomic data
            # name = "33k.s3_%s_%d.%s%s.json" % (
            #     stage_mode, 4, ds_mode,
            #     '.ineg' if not positive_only else "")
            # build_stage3_mix(os.path.join('data', 'stage3', 'atomic.ds.rand.json'),
            #                  name, 4, 0, ds_mode=ds_mode, ds_limit=limit // 2,
            #                  positive_only=positive_only, stage_mode=stage_mode)
            # print()


    # No s3 data
    print()
    for atomic_mode in ['rand']:
        # Mix method
        for ds_mode, limit in ds_modes:
            name = "20k.atomic_ns_%s_%d.%s.json" % (
                atomic_mode, 4, ds_mode)
            build_stage3_mix(os.path.join('data', 'stage3', 'ns.atomic.ds.%s.json' % atomic_mode),
                             name, 0, 4, ds_mode=ds_mode, ds_limit=limit,
                             positive_only=positive_only, stage_mode='mix12')

def prepare_stage3_positive_mixtures(ds_triples=74910):
    # Current scheme: Use fix number of 10K S3 triples, in combination with 10K atomic triples;
    # vs. all atomic (20K)

    build_stage3_mix(os.path.join('data', 'stage3', 'atomic.ds.rand.json'),
                     '75k.pos.atomic_rand.triple.json', 0, 4, ds_mode='triple', ds_limit=ds_triples,
                     positive_only=True, stage_mode='s1',
                     match_original_head=False)

    # Stage3 data
    for stage_mode in ['mix12']:
        # ds_modes = [('head', n_head), ('triple', n_triple), ('original', n_original_head)]
        ds_modes = [('triple', ds_triples)]
        for positive_only in [True]:
            # Atomic data
            for atomic_mode in ['rand']:
                # Mix method
                for ds_mode, limit in ds_modes:
                    for s3_part in [2]:
                        atomic_part = 4 - s3_part
                        name = "75k.pos.s3_%s_%d.atomic_%s_%d.%s%s.json" % (
                            stage_mode, s3_part, atomic_mode, atomic_part, ds_mode,
                            '.ineg' if not positive_only else "")
                        am_path = os.path.join('data', 'stage3', 'atomic.ds.%s.json' % atomic_mode)
                        build_stage3_mix(am_path,
                                         name, s3_part, atomic_part, ds_mode=ds_mode, ds_limit=limit,
                                         positive_only=positive_only, stage_mode=stage_mode,
                                         match_original_head=atomic_mode == 's3')
            print()
            # # No atomic data
            # name = "75k.s3_%s_%d.%s%s.json" % (
            #     stage_mode, 4, 'triple',
            #     '.ineg' if not positive_only else "")
            # build_stage3_mix(os.path.join('data', 'stage3', 'atomic.ds.rand.json'),
            #                  name, 4, 0, ds_mode='triple', ds_limit=limit // 2,
            #                  positive_only=positive_only, stage_mode=stage_mode)
            # print()



def build_stage3_ns(in_file, out_file, n_neg=3):
    data = json.load(open(in_file))
    data = [d for d in data if d['label'] == 1]
    positives = set([(r['sent_a'], r['relation'], r['sent_b']) for r in data])
    all_ents = list(set([r['sent_b'] for r in data]))
    random.seed(0)
    results = []

    for d in tqdm.tqdm(data):
        results.append({"sent_a": d['sent_a'], "relation": d['relation'], "sent_b": d['sent_b'], 'label': 1,
                        'info': d['info']})
        es = random.choices(all_ents, k=n_neg * 2)
        es = [e for e in es if (d['sent_a'], d['relation'], e) not in positives][:n_neg]
        for e in es:
            results.append({'sent_a': d['sent_a'], "relation": d['relation'], 'sent_b': e, 'label': 0,
                            'info': d['info']})
    random.seed(0)
    random.shuffle(results)
    print("Total %d triples, from %s, to %s, n_neg=%d" % (len(data), in_file, out_file, n_neg))

    json.dump(results, open(out_file, 'w'), indent=1)

def export_stage3_data(in_file):
    data = json.load(open(in_file))
    result = []
    n_pos = n_neg = 0
    heads = set()
    original_heads = set()
    for item in tqdm.tqdm(data):
        if 'atomic_all' in in_file or 'atomic.ds.rand' in in_file:
            item['info']['source'] = 'a'
        elif 'stage3_all' in in_file:
            item['info']['source'] = 'c'
        result.append(convert_item(item))
        if item['label'] == 1:
            n_pos += 1
        else:
            n_neg += 1
        heads.add(item['sent_a'])
        original_heads.add(item['info']['d_i'])
    out_file = os.path.join('data', 'exp', os.path.split(in_file)[-1])
    print("Total %d triplets, to %s; %d pos, %d neg; %d heads, %d original heads" %
          (len(result), out_file, n_pos, n_neg, len(heads), len(original_heads)))
    json.dump(result, open(out_file, 'w'), indent=1)

def prepare_stage2_nl(in_file, out_file):
    import re
    data = json.load(open(in_file))
    result = []
    for item in tqdm.tqdm(data):
        concept = re.findall("\[(.*?)\]", item['sent_a'])[0]
        item['sent_b'] = '[%s] is an instance of [%s]' % (concept, item['sent_b'])
        result.append(item)
    json.dump(result, open(out_file, 'w'), indent=1)

if __name__ == '__main__':
    pass
    # collect_atomic_split()

    # Stage 3 Processing
    # build_base_atomic()
    # build_downsampled_atomic()
    # prepare_stage3_mixtures()
    #
    # for f in glob.iglob(os.path.join("data", "stage3", "mix", "*.json")):
    #     base, fname = os.path.split(f)
    #     if 'ns.' in fname:
    #         continue
    #     build_stage3_ns(f, os.path.join(base, 'ns.' + fname))

    # shutil.copy(os.path.join('data', 'stage3_combined.json'), os.path.join('data', 'stage3', 'stage3_all.json'))
    # shutil.copy(os.path.join('data', 'stage3_combined.neg2.json'), os.path.join('data', 'exp', 'stage3_all.neg2.json'))
    # export_stage3_data(os.path.join('data', 'exp', 'stage3_all.neg2.json'))

    # shutil.copy(os.path.join('data', 'stage3_combined.pos3.json'), os.path.join('data', 'exp', 'stage3_all.pos3.json'))
    # export_stage3_data(os.path.join('data', 'exp', 'stage3_all.pos3.json'))

    # build_stage3_ns("data\\stage3\\atomic.ds.rand.json", "data\\stage3\\ns.atomic.ds.rand.json")
    # export_stage3_data(os.path.join('data', 'stage3', 'ns.atomic.ds.rand.json'))
    # build_stage3_ns("data\\stage3\\atomic_all.json", "data\\stage3\\ns.atomic_all.json")
    # export_stage3_data(os.path.join('data', 'stage3', 'ns.atomic_all.json'))

    # build_stage3_mix("data\\stage3\\atomic_all.json", 's3_4.atomic_rand_4.head.json',
    #                  4, 4, 'head', 2756, True, 'mix12', False)
    # build_stage3_ns("data\\stage3\\mix\s3_4.atomic_rand_4.head.json", "data\\stage3\\mix\\ns.s3_4.atomic_rand_4.head.json")
    # export_stage3_data(os.path.join('data', 'stage3', 'mix', 'ns.s3_4.atomic_rand_4.head.json'))

    # Stage 3 variations, for classification experiments, supervised/NS, possibly in combination with atomic

    n_atomic_triple = 553727
    n_atomic_head = 18481
    n_stage3_triple = 81197
    n_stage3_pos = 65900
    n_stage3_head = 2756

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2.atomic_plus1.triple.json',
    #                  4, 4, 'triple', 81197 * 2, False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_pos.atomic_plus1.triple.json',
    #                  4, 4, 'triple', 65900 * 2, True, 'mix12', False)
    # build_stage3_ns("data\\stage3\\mix\\stage3_pos.atomic_plus1.triple.json", "data\\stage3\\mix\\ns.stage3_pos.atomic_plus1.triple.json")
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2.atomic_plus1.head.json',
    #                  4, 4, 'head', 2756 * 2, False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_pos.atomic_plus1.head.json',
    #                  4, 4, 'head', 2756 * 2, True, 'mix12', False)
    # build_stage3_ns("data\\stage3\\mix\\stage3_pos.atomic_plus1.head.json", "data\\stage3\\mix\\ns.stage3_pos.atomic_plus1.head.json")

    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2.atomic_plus1.triple.json")
    # export_stage3_data("data\\stage3\\mix\\ns.stage3_pos.atomic_plus1.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2.atomic_plus1.head.json")
    # export_stage3_data("data\\stage3\\mix\\ns.stage3_pos.atomic_plus1.head.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_1.atomic_3.triple.json',
    #                  n_atomic_triple // 3, n_atomic_triple, 'triple',
    #                  n_atomic_triple // 3 + n_atomic_triple, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all_1.atomic_3.triple.json")
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_1.atomic_7.triple.json',
    #                  n_atomic_triple // 7, n_atomic_triple, 'triple',
    #                  n_atomic_triple // 7 + n_atomic_triple, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all_1.atomic_7.triple.json")
    #
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_1.atomic_1.triple.json',
    #                  n_atomic_triple, n_atomic_triple, 'triple',
    #                  n_atomic_triple + n_atomic_triple, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all_1.atomic_1.triple.json")


    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all.atomic_plus2.json',
    #                  n_stage3_pos, n_stage3_pos * 2, 'triple',
    #                  n_stage3_pos * 3, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all.atomic_plus2.json")
    #
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_us1.5.atomic_plus3.json',
    #                  1, 2, 'triple',
    #                  n_stage3_pos * 4.5, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all_us1.5.atomic_plus3.json")
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_us2.atomic_plus4.json',
    #                  1, 2, 'triple',
    #                  n_stage3_pos * 6, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all_us2.atomic_plus4.json")


    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all.atomic_plus3.json',
    #                  n_stage3_pos, n_stage3_pos * 3, 'triple',
    #                  n_stage3_pos * 4, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all.atomic_plus3.json")


    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all.atomic_plus4.json',
    #                  n_stage3_pos, n_stage3_pos * 4, 'triple',
    #                  n_stage3_pos * 5, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all.atomic_plus4.json")
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all.atomic_plus5.json',
    #                  n_stage3_pos, n_stage3_pos * 5, 'triple',
    #                  n_stage3_pos * 6, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all.atomic_plus5.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", '264k.pos.s3_mix12_1.atomic_rand_3.triple.json',
    #                  n_stage3_pos, n_stage3_pos * 3, 'triple',
    #                  n_stage3_pos * 4, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\264k.pos.s3_mix12_1.atomic_rand_3.triple.json")
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", '264k.pos.atomic_rand.triple.json',
    #                  0, n_stage3_pos * 4, 'triple',
    #                  n_stage3_pos * 4, True, 'mix12', False)
    # export_stage3_data("data\\stage3\\mix\\264k.pos.atomic_rand.triple.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2.atomic_plus2.triple.json',
    #                  4, 8, 'triple', 81197 * 3, False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_pos.atomic_plus2.triple.json',
    #                  4, 8, 'triple', 65900 * 3, True, 'mix12', False)
    # build_stage3_ns("data\\stage3\\mix\\stage3_pos.atomic_plus2.triple.json", "data\\stage3\\mix\\ns.stage3_pos.atomic_plus2.triple.json")
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2.atomic_plus2.head.json',
    #                  4, 8, 'head', 2756 * 3, False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_pos.atomic_plus2.head.json',
    #                  4, 8, 'head', 2756 * 3, True, 'mix12', False)
    # build_stage3_ns("data\\stage3\\mix\\stage3_pos.atomic_plus2.head.json", "data\\stage3\\mix\\ns.stage3_pos.atomic_plus2.head.json")
    #
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2.atomic_plus2.triple.json")
    # export_stage3_data("data\\stage3\\mix\\ns.stage3_pos.atomic_plus2.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2.atomic_plus2.head.json")
    # export_stage3_data("data\\stage3\\mix\\ns.stage3_pos.atomic_plus2.head.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2.atomic_plus3.triple.json',
    #                  4, 12, 'triple', 81197 * 4, False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2.atomic_plus3.head.json',
    #                  4, 12, 'head', 2756 * 4, False, 'mix12', False)
    #
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2.atomic_plus3.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2.atomic_plus3.head.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us1.5.atomic_plus3.triple.json',
    #                  4, 8, 'triple', int(81197 * 4.5), False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us1.5.atomic_plus3.head.json',
    #                  4, 8, 'head', int(2756 * 4.5), False, 'mix12', False)
    #
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us1.5.atomic_plus3.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us1.5.atomic_plus3.head.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us2.atomic_plus4.triple.json',
    #                  4, 8, 'triple', int(81197 * 6), False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us2.atomic_plus4.head.json',
    #                  4, 8, 'head', int(2756 * 6), False, 'mix12', False)
    #
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us2.atomic_plus4.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us2.atomic_plus4.head.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us3.atomic_plus6.triple.json',
    #                  4, 8, 'triple', int(81197 * 9), False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us3.atomic_plus6.head.json',
    #                  4, 8, 'head', int(2756 * 9), False, 'mix12', False)
    #
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us3.atomic_plus6.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us3.atomic_plus6.head.json")

    # n_atomic_head = 18751
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us_1.atomic_all_ds_2.head.json',
    #                  n_atomic_head // 2, n_atomic_head, 'head', n_atomic_head // 2 + n_atomic_head, False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us_1.atomic_all_ds_2.triple.json',
    #                  n_atomic_triple // 2, n_atomic_triple, 'triple', n_atomic_triple // 2 + n_atomic_triple, False, 'mix12', False)
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us_1.atomic_all_ds_1.triple.json',
    #                  n_atomic_triple, n_atomic_triple, 'triple', int(n_atomic_triple * 2), False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us_1.atomic_all_ds_1.head.json',
    #                  n_atomic_head, n_atomic_head, 'head', int(n_atomic_head * 2), False, 'mix12', False)
    #
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us_2.atomic_all_ds_1.triple.json',
    #                  n_atomic_triple * 2, n_atomic_triple, 'triple', int(n_atomic_triple * 3), False, 'mix12', False)
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all_neg2_us_2.atomic_all_ds_1.head.json',
    #                  n_atomic_head * 2, n_atomic_head, 'head', int(n_atomic_head * 3), False, 'mix12', False)
    #
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us_1.atomic_all_ds_2.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us_1.atomic_all_ds_2.head.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us_1.atomic_all_ds_1.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us_1.atomic_all_ds_1.head.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us_2.atomic_all_ds_1.triple.json")
    # export_stage3_data("data\\stage3\\mix\\stage3_all_neg2_us_2.atomic_all_ds_1.head.json")

    # prepare_stage2_nl("data\\exp\\stage2_all.json", "data\\exp\\stage2_all.nl.json")

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'atomic_all.plus_stage3_pos.json',
    #                  n_stage3_pos, n_atomic_triple, 'triple', n_stage3_pos + n_atomic_triple, True, 'mix12')
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'atomic_all.sub_stage3_pos.triple.json',
    #                  n_stage3_pos, n_atomic_triple - n_stage3_pos, 'triple', n_atomic_triple, True, 'mix12')
    # build_stage3_mix("data\\stage3\\atomic_all.json", 'atomic_all.sub_stage3_pos.head.json',
    #                  n_stage3_head, n_atomic_head - n_stage3_head, 'head', n_atomic_head, True, 'mix12')

    # build_stage3_mix("data\\stage3\\atomic_all.json", 'stage3_all.all.json',
    #                  n_stage3_head, 0, 'head', n_stage3_head, positive_only=False)
    # export_stage3_data("data\\stage3\\mix\\stage3_all.all.json")

    # export_stage3_data("data\\stage3\\mix\\atomic_all.plus_stage3_pos.json")
    # export_stage3_data("data\\stage3\\mix\\atomic_all.sub_stage3_pos.triple.json")
    # export_stage3_data("data\\stage3\\mix\\atomic_all.sub_stage3_pos.head.json")

    # prepare_stage3_ineg_mixtures()
    # prepare_stage3_positive_mixtures()
    # for f in glob.iglob(os.path.join("data", "stage3", "mix", "*.json")):
    #     base, fname = os.path.split(f)
    #     if not '20k.pos' in fname:
    #         continue
    #     build_stage3_ns(f, os.path.join(base, 'ns.' + fname))
    #
    # for f in glob.iglob(os.path.join("data", "stage3", "**", "*.json"), recursive=True):
    #     base, fname = os.path.split(f)
    #     if 'mix.1226' in f:
    #         continue
    #     if 'mix' not in f:
    #         continue
    #     if '75k.pos' not in f:
    #         continue
    #     export_stage3_data(f)

    # Stage 2 Processing
    # build_stage2_full()
    # build_stage2_full('stage2.stage2_only.json', 2)
    # build_stage2_full('stage2.stage1_only.json', 1)
    # downsampling('stage2_all.json', 'stage2_all.ds.stage1_size.json', 26778)
    # downsampling('stage2_all.json', 'stage2_all.ds.stage2_size.json', 65457)
    # build_stage2_by_dep()

    # build_stage2_from_gt_rule_ns('stage2.gt.ns.6.0.json', 6, 0, include_rel=())

    # build_stage2_from_gt_rule_ns('stage2.gt.ns.3.3.parent.json', 3, 3, include_rel=('parent', ))
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.3.3.child.json', 3, 3, include_rel=('child'))
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.3.3.sibling.json', 3, 3, include_rel=('sibling'))

    # build_stage2_from_gt_rule_ns('stage2.gt.ns.0.6.parent.json', 0, 6, include_rel=('parent', ))
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.0.6.child.json', 0, 6, include_rel=('child'))
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.0.6.sibling.json', 0, 6, include_rel=('sibling'))


    # build_stage2_from_gt_rule_ns('stage2.gt.ns.0.6.child_sibling_parent.json', 0, 6,
    #                              include_rel=('child', 'sibling', 'parent'))
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.3.3.child_sibling_parent.json', 3, 3,
    #                              include_rel=('child', 'sibling', 'parent'))
    #
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.2.4.child.json', 2, 4,
    #                              include_rel=('child', ))
    # build_stage2_from_gt_rule_ns('stage2.gt.ns.4.2.child.json', 4, 2,
    #                              include_rel=('child', ))

    # build_stage2_from_positive_ns('stage2.pos.s1.ns.6.0.json', 6, 0, include_rel=(), stage1_only=True)
    #
    # build_stage2_from_positive_ns('stage2.pos.s1.ns.3.3.parent.json', 3, 3, include_rel=('parent', ), stage1_only=True)
    # build_stage2_from_positive_ns('stage2.pos.s1.ns.3.3.child.json', 3, 3, include_rel=('child'), stage1_only=True)
    # build_stage2_from_positive_ns('stage2.pos.s1.ns.3.3.sibling.json', 3, 3, include_rel=('sibling'), stage1_only=True)
    #
    # build_stage2_from_positive_ns('stage2.pos.ns.6.0.json', 6, 0, include_rel=())
    #
    # build_stage2_from_positive_ns('stage2.pos.ns.3.3.parent.json', 3, 3, include_rel=('parent', ))
    # build_stage2_from_positive_ns('stage2.pos.ns.3.3.child.json', 3, 3, include_rel=('child'))
    # build_stage2_from_positive_ns('stage2.pos.ns.3.3.sibling.json', 3, 3, include_rel=('sibling'))