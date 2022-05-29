import os, glob, json
import random
from utils import ProbaseClient
from collections import defaultdict
import tqdm


def save(samples, f):
    for s in samples:
        f.write(json.dumps(s) + '\n')

def split_data_by_atomic():
    os.makedirs('split_atomic', exist_ok=True)
    split = json.load(open(os.path.join('data', 'head_to_split.json')))
    for f in glob.iglob(os.path.join('data', '*.json')):
        if 'head_to_split' in f:
            continue
        name = os.path.split(f)[-1][:-5]
        if os.path.exists(os.path.join('split_atomic', name)):
            continue
        print(name)
        data = json.load(open(f))

        bins = defaultdict(list)
        for d in data:
            bins[split[str(d['info']['d_i'])]].append(d)
        os.makedirs(os.path.join('split_atomic', name), exist_ok=True)
        for key in bins:
            print(key, len(bins[key]))
            save(bins[key], open(os.path.join('split_atomic', name, key + '.json'), 'w'))

def collect_positive(base_path, out_path):
    data = json.load(open(base_path))
    data = [d for d in data if d[2] == 1]
    json.dump(data, open(out_path, 'w'), indent=1)


def is_target_exp(name):
    return name in ['extend_cand_head.nl', 'stage2_all.nl']

def text_to_relation(text):
    mapping = {'xIntent': "PersonX's intention is", "xNeed": "Before that, PersonX needs",
               "xAttr": "PersonX will be described as", "xEffect": "Effects on PersonX will be",
               "xWant": "After that, PersonX wants", "xReact": "After that, PersonX feels",
               "oReact": "After that, others feel", "oWant": "After that, others want",
               "oEffect": "Effects on others will be", "InstanceOf": "InstanceOf"
               }
    mapping = dict([(v, k) for k, v in mapping.items()])
    return mapping[text]

def prepare_gpt_data():
    for f in glob.iglob(os.path.join('split_atomic', '*')):
        name = os.path.split(f)[-1]
        if not is_target_exp(name):
            continue
        cnt = defaultdict(int)
        os.makedirs(os.path.join('gen', name), exist_ok=True)
        for key in ['trn', 'dev', 'tst']:
            fw = open(os.path.join('gen', name, key + '.tsv'), 'w', encoding='utf-8')
            fw.write("\t".join(['head_event', 'relation', 'tail_event']) + '\n')
            for line in open(os.path.join(f, key + '.json')):
                line = json.loads(line)
                # if line['label'] != 1:
                #     continue
                cnt[key] += 1
                sent_a = line['sent_a']
                if 'stage2' in name or 'uncovered_heads' in name or 'extend_cand_head' in name:
                    if 'nl' in name:
                        sent_a += '. ' + line['sent_b'][:line['sent_b'].find(']') + 1]
                        sent_b = line['sent_b'][line['sent_b'].find('[', 1) + 1: -1]
                    else:
                        sent_b = line['sent_b']
                    relation = 'InstanceOf'
                else:
                    relation = line['sent_b'].split(":")[0]
                    sent_b = line['sent_b'][len(relation) + 1:].strip()
                fw.write('\t'.join([sent_a, text_to_relation(relation), sent_b.replace("\t", " ")]) + '\n')
        print("%s: %d trn, %d dev, %d tst" % (name, cnt['trn'], cnt['dev'], cnt['tst']))

if __name__ == '__main__':
    split_data_by_atomic()
    prepare_gpt_data()
