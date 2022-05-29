import pandas as pd
import os, random
import tqdm
import copy
import unidecode
import re, json
import spacy, pickle
from collections import defaultdict, OrderedDict
from lemminflect import getInflection
from nltk.corpus import wordnet as wn

random.seed(0)

def printout(doc, i=None):
    if i is None:
        print(doc)
    else:
        print(i, doc)
    for token in doc:
        if token.i == len(doc) - 1:
            break
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.head.text)
    print()


def printout_dict(doc):
    if len(doc) > 0 and isinstance(doc[0], Token):
        doc = [t._asdict() for t in doc]
    for token in doc:
        print(token['text'], token['lemma'], token['pos'], token['tag'], token['dep'], doc[token['head']]['text'],
              ('P' if token['predicate'] else '') + ('N' if token['nominal'] else '')
              + ('M' if token['modifier'] else ''))
    print()

nlp = spacy.load("en_core_web_lg")

base_path = r'data/v4_atomic_all_agg.csv'
output_path = r'parse'
concept_path = r'~/data/probase/data-concept-instance-relations.txt'

if not os.path.exists(output_path):
    os.makedirs(output_path)

names = ['Alice', 'Bob', 'Charlie'] + ['bob', 'dan', 'jeff', 'tim', 'tom', 'umi', 'fred']
ul_magic = 'Placeholder'


# Step 0: collect heads into txt, and then manually fix those need to be fixed
def create_source():
    df = pd.read_csv(base_path)
    fw = open(os.path.join(output_path, 'heads.0.txt'), 'w')
    all_texts = []
    for idx, line in tqdm.tqdm(df.iterrows()):
        text = line[0]
        text = unidecode.unidecode_expect_ascii(text)
        fw.write(text + '\n')
        text = text.replace('PersonX', 'Alice').replace('PersonY', 'Bob').replace('PersonZ', 'Charlie')\
            .replace('___', 'entity')
        all_texts.append(text)
    docs = pickle.dumps([nlp(t) for t in all_texts])
    open(os.path.join(output_path, 'parse.0.bin'), 'wb').write(docs)


def head_normalize(text: str):
    seen_persons = ['X', 'Y', 'Z']
    place = 0
    for t in seen_persons:
        if 'Person' + t in text:
            place += 1
    for t in ['bob', 'dan', 'jeff', 'tim', 'tom']:
        if ' ' + t + ' ' in text + ' ':
            text = (text + ' ').replace(' ' + t + ' ', ' Person' + seen_persons[place] + ' ').strip()
            place += 1

    text = text.replace('PersonX', 'Alice').replace('PersonY', 'Bob').replace('PersonZ', 'Charlie')
    text = text.replace("mcdonald 's", "mcdonalds").replace('solipsist', 'solipsistic')
    text = text.replace("allnighter", "all nighter").replace("all-nighter", "all nighter").\
        replace('his friends', 'her friends').replace(' 1st ', ' first ')
    text = text.replace("Alice 'd ", "Alice should ")
    text = text.replace("shoulds", "should").replace('wills', 'will').replace('hath', 'has')\
        .replace(' scars ', ' scares ').replace('goeth', 'goes').replace("doeth", "does").replace(' shalt ', ' shall ')\
        .replace(' breaketh', ' breaks').replace(' bringeth', ' brings').replace(' divideth', ' divides')\
        .replace(' enlighteneth', ' enlightens').replace(' giveth', ' gives').replace(' loveth', ' loves')\
        .replace(' putteth', ' puts').replace(' taketh', ' takes').replace(' worketh', ' works')\
        .replace(' fees ', ' feeds ').replace(' wrappeds ', ' wraps ').replace(' spake ', ' spoke ')\
        .replace(' unring ', ' unrings ').replace(' takest ', ' takes ')
    text = text.replace('not does ', 'does not do ').replace('not gives ', 'does not give ')\
        .replace('not has ', 'does not have ').replace('not is ', 'is not').replace('not knows ', 'does not know ')\
        .replace('not sees ', 'does not see ').replace('says never', 'never says').replace('can it', 'can do it')\
        .replace('best PersonY could', 'best PersonY could do')
    text = text.replace("ca n't", "can not").replace(" n't ", " not ").replace(" '", "'").replace('didnt', 'did not')\
        .replace("wouldnt", "would not").replace(" dont ", " did not ").replace(' thee ', ' you ').\
        replace(' whats ', ' what is ').replace("understands 's", "understands what is")
    text = text.replace("Alice really like ", "Alice really likes ").replace('Alice rids ', 'Alice rides ')
    text = text.replace("Alice names the films Bob", "Alice names the films that Bob")
    text = text.replace("Alice turns his", "Alice turns her")
    text = text.replace("Alice allergic ", "Alice is allergic ")
    text = text.replace("Alice calls's", "Alice calls Bob's")
    text = text.replace("Alice about to", "Alice is about to")
    text = text.replace('Alice practices', 'Alice practises')
    text = text.replace('Alice bases ', 'Alice based ___ ')
    text = text.replace('Alice stills', 'Alice still').replace('Alice evens', 'Alice even').\
        replace('and grind', 'and grinded').replace('Alice orders', 'Alice ordered').\
        replace('Alice arches', 'Alice arched').replace('Alice waves', 'Alice waved').\
        replace('Alice hands', 'Alice handed').replace('Alice dyes', 'Alice dyed').\
        replace('slips and falls', 'slips and fells').replace('stays home and rest', 'stays home and rests').\
        replace(' off their feet', ' off her feet')
    text = text.replace('tells Bob Bob', 'tells Bob that Charlie').replace('Bob and Bob', 'Bob and Charlie').\
        replace('Bob to marry Bob', 'Bob to marry Charlie').replace("Bob's friend Bob", "Bob's friend Charlie").\
        replace("asserts Bob's right", "asserts Bob's rights")
    text = text.replace('comes live', 'comes to live').replace('comes get', 'comes to get').\
        replace('comes help', 'comes to help').replace('comes see', 'comes to see').replace('comes ___', 'comes to ___')
    text = text.replace("french kisses", "french-kissed").replace('fans dance', 'fan-danced').\
        replace('jets-set', 'jet-sets').replace('jets set', 'jet-sets').\
        replace('lines dance', 'line-danced').replace('spots check', 'spot-checked').\
        replace('markets and sell', 'markets and sells').replace("doubles check", "double-checked").\
        replace('fakes being', 'faked being').replace('wonders would', 'wonders what would').\
        replace('Alice up all night', 'Alice stays up all night').replace('up the ante', 'raises the ante').\
        replace('tells ___ Bob', 'tells ___ that Bob').replace('wads imagining', 'was imagining').\
        replace('wad imagining', 'was imagining').replace('Alice unring ', 'Alice unrings ').\
        replace('laughs , cried', 'laughs and cries').replace('gathers round ', 'gathers around ').\
        replace('sprays paint', 'spray-painted').replace('potty train', 'potty-trained').\
        replace('is done', 'has finished').replace('hikes ,', 'hiked and').replace('opposite Bob', 'opposite to Bob').\
        replace('hunt with the hounds', 'hunts with the hounds').replace(' necked', ' neck').\
        replace('dragons tail', "dragon's tail").replace("Bob breath", "Bob's breath").replace('go karts', 'Go-Kart').\
        replace('the sun rise', 'the sunrise').replace('slips , fell', 'slips and fells').\
        replace('rats arse', "rat's arse").replace("tells Bob's going", "tells Bob what is going").\
        replace('gets Bob thinking', 'gets Bob to think').replace("___ Alice's point", "___ to Alice's point").\
        replace("___ a point", "___ to a point").replace('bores Bob a son', 'bears Bob a son').\
        replace('tells Bob believed', 'tells what Bob believed').replace('every thing', 'everything').\
        replace('says thank you', 'says thankyou').replace('Alice couples with', 'Alice is coupled with').\
        replace('pope the pope', 'pope').replace('high-tail it it', 'it').replace('the on button', 'the ON button').\
        replace('pokemon go', 'Pokemon-Go').replace('whithersoever', 'to wherever it').\
        replace('comes and help', 'comes and helps').replace('Alex rents and managing', 'Alex rents and manages').\
        replace('triple a', 'triple A')
    text = text.replace('forgives men', 'forgives').replace('forgives ___', 'forgives')
    vs = ['reduces to a minimum', 'makes possible', 'confers upon Bob', 'impresses upon Bob', 'imposes upon Bob',
          'reads a time that day', 'practices hard', 'studies hard', 'acknowledges gratefully',
           'acknowledges with gratitude', 'acknowledges with thanks', 'reads aloud']
    vst = []
    dets = ['the ', 'every ', 'another ', 'some ', 'what for ', '']
    matched = False
    for v in vs + vst:
        if v not in text:
            continue
        for d in dets:
            lat = ' %s___' % (d)
            if text.startswith('Alice ' + v) and text.endswith(lat):
                if v in vst and text == 'Alice %s %s___' % (v, d):
                    pass
                elif v in vst:
                    st = 'Alice %s ' % v
                    text = 'Alice %s%s to ' % (v, lat) + text[len(st):-len(lat)]
                else:
                    vb = v.split(' ')[0]
                    text = 'Alice %s%s%s' % (vb, lat, v[len(vb):])
                matched = True
                break
        if matched:
            break
        if v in vst and text.startswith('Alice ' + v + ' Bob '):
            mid_text = text[len('Alice ' + v + ' Bob '):]
            if 'his' in mid_text:
                mid_text = mid_text.replace('his', "Bob's")
            text = 'Alice %s %s to Bob' % (v, mid_text)
    # text = text.replace('himself', 'herself')
    text = text.replace("Alice's", "her")
    if not (' his tracks' in text or ' his perch' in text or ' his den' in text or ' his due' in text):
        text = text.replace(' his ', ' her ')
    if text.count('Bob') > 1:
        text = text.replace("Bob's", "his")
    if ' his ' not in text and text.count("Charlie") > 1:
        text = text.replace("Charlie's", "his")
    text = text.replace('___', 'Placeholder')
    if text[-2] != ' ':
        text = text + '.'
    text = text.replace('Alice still another', 'Alice is still another').\
        replace('Alice thoroughly clean.', 'Alice thoroughly cleaned.').\
        replace('will definitely.', 'will definitely do.').replace('would always.', 'would always do.').\
        replace('would ever.', 'would ever do.').replace('would not even.', 'would not even do.').\
        replace('would definitely.', 'would definitely do.').replace('would eventually.', 'would eventually do.').\
        replace('would rather.', 'would rather do.').replace('goes trick or treating.', 'goes trick-or-treating.').\
        replace('tricks or treat.', 'goes trick-or-treating.').\
        replace('tricks or treating.', 'goes trick-or-treating.').\
        replace('would never.', 'would never do.').replace('would no longer.', 'would no longer do.').\
        replace('should always.', 'should always do.').replace('may never.', 'may never do.').\
        replace('gets rid.', 'gets rid of it.').replace('things her way.', 'things in her way.').\
        replace('pays her way.', 'pays in her way.').replace('talks her way.', 'talks in her way.').\
        replace('Alice gets her', 'Alice got her').replace('according.', 'according to it.').\
        replace("Bob's calm down", "Bob to calm down").replace('everyday.', 'every day.').\
        replace('using.', 'using it.').replace('trying.', 'trying it.').replace("straight's.", "straight A.").\
        replace("Placeholder Bob's way.", "Placeholder in Bob's way.").replace("straight a's.", "straight A.").\
        replace("things Bob's way.", "things in Bob's way.").replace("fights Bob's way.", "fights in Bob's way.").\
        replace("works Bob's way.", "works in Bob's way.").replace("kisses Bob's way.", "kisses in Bob's way.").\
        replace('even the score', 'evens the score').replace("hair herself.", "hair by herself.").\
        replace('hither.', 'to this place.').replace('Placeholder much.', 'Placeholder a lot.')

    mids = ['baby', 'kids', 'daughter', 'girlfriend', 'husband', 'mother', 'wife', 'mom', 'sister', 'dog', 'hand',
            'kitten', 'man', 'woman', 'girl', 'cat', 'boy']
    for mid in mids:
        if text.startswith('Alice gives') and text.endswith(mid + ' Placeholder.'):
            text = text.replace('Placeholder.', 'the Placeholder.')
    return text


# Step 1: do basic auto normalizations that doesn't need parsing results
def preprocess_heads():
    lines = open(os.path.join(output_path, 'heads.0.txt')).read().splitlines()
    fw = open(os.path.join(output_path, 'heads.1.txt'), 'w')
    all_texts = []
    for i, head in enumerate(tqdm.tqdm(lines)):
        text = head_normalize(head)
        all_texts.append(text)
        fw.write(text + '\n')
    docs = pickle.dumps([nlp(t) for t in all_texts])
    open(os.path.join(output_path, 'parse.1.bin'), 'wb').write(docs)


# Step 2: fix the wildcards
def has_determiner(doc, tok):
    for t in tok.children:
        if t.dep_ in ['det', 'poss', 'nummod']:
            return True
    if doc[tok.i - 1].dep_ in ['det', 'poss', 'nummod'] or doc[tok.i - 1].text == "'s":
        return True
    return False


# replacements: List[(l.i, r.i, source, sub, type)], replacements from heads.1, guaranteed no intersection
def add_replacement(replacement, text, pos, source, sub, type):
    updated = copy.deepcopy(replacement)
    offset = len(sub) - len(source)
    text_ = text[:pos] + sub + text[pos + len(source):]
    r_ = pos + len(sub) - 1
    r_source = pos + len(source) - 1
    for i, (l, r, source_i, sub_i, type_i) in enumerate(replacement):
        if (l <= pos <= r or l <= r_source <= r) and not (len(source) == 0 and pos in [l, r+1]):
            raise ValueError("Intersection: %s, %s -> %s and %s -> %s" % (text, source_i, sub_i, source, sub))
        if l >= r_source:
            updated[i] = (l + offset, r + offset, source_i, sub_i, type_i)
    updated.append((pos, r_, source, sub, type))
    return updated, text_

def find_alt_text(text, idx, all_texts):
    for l in all_texts:
        if l.startswith(text[:idx]) and l.endswith(text[idx + len('Placeholder'):]) and 'Placeholder' not in l:
            return l[idx: len(l) - (len(text) - idx - len('Placeholder'))]
    return None

placeholder_cands = ['thing', 'person', 'dollars', 'entity']
def fix_wildcards():
    lines = open(os.path.join(output_path, 'heads.1.txt')).read().splitlines()
    docs = pickle.load(open(os.path.join(output_path, 'parse.1.bin'), 'rb'))
    fw = open(os.path.join(output_path, 'heads.2.txt'), 'w')
    fw_r = open(os.path.join(output_path, 'replacements.2.jsonl'), 'w')
    all_texts = []
    sub_cnt = sub_match_cnt = fail_cnt = 0
    for i, text in enumerate(tqdm.tqdm(lines)):
        doc = docs[i]
        text_ = text
        r_ = []
        if 'Placeholder' in text:
            for t in doc:
                if t.text == 'Placeholder':
                    pos_x = t.idx
                    posi_x = t.i
                    if doc[t.i - 1].text not in ['Alice', 'Bob', 'Charlie'] and not has_determiner(doc, t):
                        r_, text_ = add_replacement(r_, text, pos_x, '', 'the ',
                                                    'placeholder_det')
                        pos_x += 4
                        posi_x += 1
                    success = False
                    for alt in placeholder_cands:
                        r__, text__ = add_replacement(r_, text_, pos_x, 'Placeholder', alt,
                                                    'placeholder')  # Re-do the done substitution
                        doc__ = nlp(text__)
                        t_ = doc__[posi_x]
                        if any([tc.dep_ in ['compound'] for tc in t_.children]):
                            # print('Fail for child')
                            # printout(doc, i)
                            continue
                        if t_.pos_ in ['NOUN', 'PROPN'] and t_.dep_ in [
                            'nsubj', 'dobj', 'iobj', 'attr', 'conj', 'dative', 'agent', 'pobj', 'appos', 'npadvmod',
                            'ccomp', 'oprd']:
                            success = True
                            break
                    r_ = r__
                    text_ = text__
                    if not success:
                        # print('Wildcard substitution failed')
                        # printout(doc__, i)
                        fail_cnt += 1
        all_texts.append(text_)
        fw.write(text_ + '\n')
        fw_r.write(json.dumps(r_) + '\n')
    print('Wildcard count: ', sub_cnt, sub_match_cnt, fail_cnt)
    docs = pickle.dumps([nlp(t) for t in all_texts])
    open(os.path.join(output_path, 'parse.2.bin'), 'wb').write(docs)

# Step 3: fix verbs by inflection to past tense
def fix_verbs():
    lines = open(os.path.join(output_path, 'heads.2.txt')).read().splitlines()
    docs = pickle.load(open(os.path.join(output_path, 'parse.2.bin'), 'rb'))
    replacements = [json.loads(l) for l in open(os.path.join(output_path, 'replacements.2.jsonl'))]
    fw = open(os.path.join(output_path, 'heads.3.txt'), 'w')
    fw_r = open(os.path.join(output_path, 'replacements.3.jsonl'), 'w')
    all_texts = []
    for i, text in tqdm.tqdm(enumerate(lines)):
        doc = docs[i]
        text_ = text
        r_ = replacements[i]
        if doc[1].pos_ not in ['VERB', 'ADV', 'AUX', 'NOUN', 'PROPN']:
            pass
        t = None
        for t in doc:
            if t.dep_ == 'ROOT':
                break
        used_VBD = False
        if (doc[1].pos_ in ['NOUN', 'PROPN'] or doc[0].dep_ == 'compound') and doc[1].dep_ not in ['npadvmod']:
            if (doc[0].head.i == 1 and doc[0].dep_ in ['compound', 'nsubj']) or (doc[1].dep_ == 'ROOT') \
                    or (doc[0].head.i == doc[1].head.i and doc[0].dep_ == doc[1].dep_
                        and doc[0].dep_ in ['compound', 'nsubj']):
                t = doc[1].lemma_
                if t == doc[1].text and t[-1] == 's':
                    t = t[:-1]
                t = getInflection(t, tag='VBD')[0]
                used_VBD = True
                r_, text_ = add_replacement(r_, text_, doc[1].idx, doc[1].text, t, 'nns')
                doc = nlp(text_)
        for t in doc:
            if t.dep_ == 'ROOT':
                for ct in t.children:
                    if ct.dep_ == 'conj' and ct.pos_ in ['VERB', 'AUX'] and ct.tag_ not in ['VBD', 'VBN']:
                        # printout(doc, i)
                        t_ = getInflection(ct.lemma_, tag='VBD')[0]
                        r_, text_ = add_replacement(r_, text_, ct.idx, ct.text, t_, 'verb_conj')
                break
        all_texts.append(text_)
        fw.write(text_ + '\n')
        fw_r.write(json.dumps(r_) + '\n')
    docs = pickle.dumps([nlp(t) for t in all_texts])
    open(os.path.join(output_path, 'parse.3.bin'), 'wb').write(docs)

# Step 4: fix the possessions of PersonY or determiners of iobj
def fix_poss_iobj():
    lines = open(os.path.join(output_path, 'heads.3.txt')).read().splitlines()
    docs = pickle.load(open(os.path.join(output_path, 'parse.3.bin'), 'rb'))
    replacements = [json.loads(l) for l in open(os.path.join(output_path, 'replacements.3.jsonl'))]
    fw = open(os.path.join(output_path, 'heads.4.txt'), 'w')
    fw_r = open(os.path.join(output_path, 'replacements.4.jsonl'), 'w')
    all_texts = []
    from lm_scorer import LMScorer
    scorer = LMScorer()
    for i, text in enumerate(tqdm.tqdm(lines)):
        doc = docs[i]
        text_ = text
        r_ = replacements[i]
        while True:
            run_flag = False
            if doc[0].dep_ in ['compound', 'npadvmod']:
                r_, text_ = add_replacement(r_, text_, 0, 'Alice', 'She', 'Alice_subj')
                doc = nlp(text_)

            for t in doc[1:]:
                if t.text in ['Bob'] and t.dep_ in ['nmod', 'npadvmod']:
                    r_, text_ = add_replacement(r_, text_, t.idx, t.text, 'him', t.text + '_obj')
                    doc = nlp(text_)
                    run_flag = True
                    break
                if t.text in ['Alice', 'Bob', 'Charlie'] and doc[t.i - 1].dep_ == 'compound':
                    # printout(doc, i)
                    source = t.text
                    if doc[t.i + 1].text == "'s":
                        sub = 'his'
                        source += "'s"
                        type = t.text + '_pron_poss'
                    else:
                        sub = 'him'
                        type = t.text + '_obj'
                    r_, text_ = add_replacement(r_, text_, t.idx, source, sub, type)
                    doc = nlp(text_)
                    # printout(doc, i)
                    run_flag = True
                    break
                if t.text in ['Alice', 'Bob', 'Charlie'] + placeholder_cands \
                        and (t.dep_ not in ['nsubj', 'dobj', 'iobj', 'attr', 'poss', 'pobj', 'nsubjpass', 'dative']
                             or (t.dep_ in ['nsubj', 'nsubjpass'] and t.head.pos_ in ['NOUN', 'PRON', 'PROPN'])):
                    if t.dep_ not in ['compound', 'nsubj']:
                        # print('Outlier:', t)
                        # printout(doc, i)
                        continue
                    # if t.dep_ in ['nsubj', 'nsubjpass'] and t.head.pos_ in ['NOUN', 'PRON', 'PROPN']:
                    #     printout(doc, i)
                    if t.i == len(doc) - 2:
                        continue
                    cands = []
                    if doc[t.i + 1].dep_ not in ['det', 'poss', 'nummod', 'aux', 'prep']:
                        cands.append((t.idx + len(t), '', ' a', t.text + '_extradet'))
                        cands.append((t.idx + len(t), '', ' the', t.text + '_extradet'))
                        cands.append((t.idx + len(t), '', "'s", t.text + '_extraposs'))
                        cands.append((t.idx + len(t), '', " to", t.text + '_extracomp'))
                        if t.text in ['Alice', 'Bob', 'Charlie']:
                            cands.append((t.idx, t.text, 'him', t.text + '_pron_obj'))
                            cands.append((t.idx, t.text, 'him a', t.text + '_extradet_obj'))
                            cands.append((t.idx, t.text, 'him the', t.text + '_extradet_obj'))
                            cands.append((t.idx, t.text, 'his', t.text + '_pron_poss'))
                    elif t.dep_ == 'compound':
                        if t.text in ['Alice', 'Bob', 'Charlie']:
                            cands.append((t.idx, t.text, 'him', t.text + '_pron_obj'))
                        else:
                            printout(doc, i)
                    if len(cands) == 0:
                        break
                    rlcs = []
                    for pos, src, sub, type in cands:
                        rlcs.append(add_replacement(r_, text_, pos, src, sub, type))
                    scores = [scorer.evaluate(s) for r, s in rlcs]
                    maxs = -1e9
                    for k, s in enumerate(scores):
                        if s > maxs:
                            maxs = s
                            r_, text_ = rlcs[k]
                    doc = nlp(text_)
                    run_flag = True
                    break
            if not run_flag:
                break
        all_texts.append(text_)
        fw.write(text_ + '\n')
        fw_r.write(json.dumps(r_) + '\n')
    docs = pickle.dumps([nlp(t) for t in all_texts])
    open(os.path.join(output_path, 'parse.4.bin'), 'wb').write(docs)

verbal_constituent = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl']
nominal_constituent = ['nsubj', 'dobj', 'iobj', 'attr', 'conj', 'dative', 'agent', 'oprd']
prepositional_constituent = ['prep']
verbal_additional = ['advmod', 'aux', 'neg', 'mark', 'punct', 'acomp', 'npadvmod', 'cc', 'prt', 'oprd', 'intj', 'expl']
nominal_additional = ['nmod', 'nummod', 'compound', 'acl', 'amod', 'det', 'poss', 'case', 'cc', 'advmod', 'predet',
                      'punct', 'mark', 'prt', 'appos', 'nsubj', 'quantmod']
prep_additional = ['advmod', 'amod', 'aux', 'cc']

def lemmatize(tok):
    mislemmas = {'taxes': 'tax', 'lenses': 'lens', 'goods': 'goods', 'waters': 'waters', 'ashes': 'ash',
                 'fries': 'fries', 'politics': 'politics', 'glasses': 'glasses', 'clothes': 'clothes',
                 'scissors': 'scissors', 'shorts': 'shorts', 'thanks': 'thanks',
                 'media': 'media', 'woods': 'woods',
                 'partied': 'party', 'babysitting': 'baby-sit', 'gimmed': 'gimme', 'wheaties': 'wheaties',
                 'diabetes': 'diabetes', 'armied': 'army', 'babysits': 'baby-sit', 'texted': 'text',
                 'attentioned': 'attention'}
    if tok.pos_ in ['NUM', 'PRON'] or tok.lemma_ == '-PRON-':
        lem = tok.text
    elif tok.text in mislemmas:
        lem = mislemmas[tok.text]
    else:
        lem = tok.lemma_
    return lem.lower()

def is_possibly_predicate(tok):
    if tok.dep_ in ['csubj', 'advcl', 'acl', 'relcl', 'ROOT'] and tok.pos_ in ['VERB', 'ADJ', 'ADV', 'AUX', 'NOUN']:
        return True
    if tok.dep_ in ['acomp', 'ccomp', 'xcomp', 'oprd', 'pcomp'] and tok.pos_ in ['VERB', 'ADJ', 'ADV', 'AUX']:
        return True
    if tok.dep_ == 'pobj' and tok.pos_ == 'VERB' and tok.head.pos_ == 'PART':
        return True
    if tok.dep_ in ['conj'] and (tok.pos_ == tok.head.pos_ or tok.pos_ in ['VERB', 'ADJ', 'ADV', 'AUX']) \
            and tok.head.i != tok.i and is_possibly_predicate(tok.head):
        return True
    return False

def is_possibly_nominal(tok):
    if tok.dep_ in ['nsubj', 'iobj', 'dobj', 'nsubjpass', 'attr', 'intj']:
        return True
    if tok.dep_ == 'pobj' and not (tok.pos_ == 'VERB' and tok.head.pos_ == 'PART'):
        return True
    if tok.dep_ in ['dative', 'agent', 'pcomp', 'ccomp', 'xcomp', 'acomp', 'npadvmod', 'oprd'] \
            and (tok.pos_ in ['NOUN', 'PROPN', 'PRON', 'NUM'] or tok.tag_ in ['VBG']):
        return True
    if tok.dep_ in ['conj', 'appos'] and tok.head.i != tok.i and is_possibly_nominal(tok.head)\
            and (tok.pos_ == tok.head.pos_ or tok.pos_ in ['NOUN', 'PROPN', 'PRON', 'NUM'] or tok.tag_ == 'VBG') :
        return True
    return False

def is_possibly_prep(tok):
    if tok.dep_ in ['prep'] and tok.pos_ in ['ADP', 'SCONJ', 'PART', 'ADV', 'VERB']:
        return True
    if tok.dep_ in ['dative', 'agent', 'ccomp', 'xcomp', 'advmod', 'pcomp', 'prt'] and tok.pos_ in ['ADP', 'SCONJ']:
        return True
    if tok.dep_ in ['advmod', 'prt'] and tok.pos_ in ['ADV']:
        return True
    if tok.dep_ in ['conj'] and tok.head.i != tok.i and tok.pos_ in ['ADP', 'SCONJ'] and is_possibly_prep(tok.head):
        return True
    return False

def is_modifier(t):
    modifier_deps = ['advmod', 'neg', 'mark', 'punct', 'prt', 'predet', 'nmod', 'cc', 'npadvmod', 'dep',
                     'nummod', 'amod', 'case', 'det', 'compound', 'poss', 'expl', 'quantmod', 'aux', 'auxpass']
    sub_deps = [t_.dep_ in modifier_deps or (t_.dep_ == 'conj' and t_.head.dep_ in modifier_deps) or
                (t.dep_ == 'appos' and t.text == 'all') for t_ in t.subtree]
    return all([c for c in sub_deps])

def match_noun(tok):
    match = defaultdict(list)
    for t in tok.children:
        c = t.dep_
        if is_modifier(t):
            pass
        elif is_possibly_predicate(t):
            match[c].append((t.i, match_clause(t)))
        elif is_possibly_nominal(t):
            match[c].append((t.i, match_noun(t)))
        elif is_possibly_prep(t):
            match[c].append((t.i, match_prep(t)))
        else:
            raise ValueError("Invalid nominal dep: %s as %s %s" % (str(t), t.pos_, c))
    for t in tok.children:
        if (t.text in placeholder_cands + names or t.pos_ == 'PRON') and t.dep_ in ['compound', 'npadvmod']:
            raise ValueError("Invalid nominal dep match: %s" % (t.dep_))
        if (tok.text in placeholder_cands + names or tok.pos_ == 'PRON') and t.dep_ in ['compound', 'npadvmod']:
            raise ValueError("Invalid nominal dep match: %s" % (t.dep_))
        if t.dep_ not in list(match.keys()) and not is_modifier(t):
            raise ValueError("Invalid nominal dep match: %s" % (t.dep_))
    return match


def match_prep(tok):
    match = defaultdict(list)
    for t in tok.children:
        c = t.dep_
        if is_modifier(t):
            pass
        elif is_possibly_predicate(t):
            match[c].append((t.i, match_clause(t)))
        elif is_possibly_nominal(t):
            match[c].append((t.i, match_noun(t)))
        elif is_possibly_prep(t):
            match[c].append((t.i, match_prep(t)))
        else:
            raise ValueError("Unmatched prep children dep: %s as %s of %s" % (str(t), c, str(tok)))
    return match

def match_clause(tok):
    match = defaultdict(list)
    tag_match = {'acomp': ['JJ', 'VBG', 'JJR', 'JJS', 'CD', 'VBN', 'RBR', 'NN', 'RB'],
                 'advmod': ['RB', 'WRB', 'RBR', 'RP', 'IN', 'JJ', 'JJR', 'JJS'],
                 'npadvmod': ['NN', 'NNS', 'RB', 'JJ'], 'cc': ['CC']}
    mapping = {'nsubjpass': 'nsubj', 'auxpass': 'aux'}
    for t in tok.children:
        c = t.dep_
        if c in mapping:
            c = mapping[c]
        if is_modifier(t):
            pass
        elif is_possibly_predicate(t):
            match[c].append((t.i, match_clause(t)))
        elif is_possibly_nominal(t):
            match[c].append((t.i, match_noun(t)))
        elif is_possibly_prep(t):
            match[c].append((t.i, match_prep(t)))
        elif c in verbal_constituent:
            raise ValueError("Unknown verbal dep: %s as %s %s" % (str(t), c, t.pos_))
        elif c in nominal_constituent:
            raise ValueError("Unknown nominal dep: %s as %s %s" % (str(t), c, t.pos_))
        elif c in prepositional_constituent:
            raise ValueError("Unknown prep dep: %s as %s %s" % (str(t), c, t.pos_))
        else:
            raise ValueError("Unknown dep: %s as %s" % (str(t), c))
    return match

def is_root_nsubj(tok, root):
    while tok.i != root.i and tok.head.i != tok.i:
        if tok.dep_ in ['nsubj', 'auxpass', 'aux', 'nsubjpass']:
            tok = tok.head
        else:
            break
    return tok.i == root.i

ent_exclusion = ['csubj', 'advcl', 'acl', 'advmod', 'aux', 'neg', 'mark', 'punct', 'acomp', 'npadvmod', 'cc', 'prt',
                 'nmod', 'nummod', 'compound', 'amod', 'det', 'case', 'predet', 'quantmod']
ent_child_exclusion = ['csubj', 'advcl', 'punct', 'npadvmod', 'prt', 'compound']
def match_dependencies():
    docs = pickle.load(open(os.path.join(output_path, 'parse.4.bin'), 'rb'))
    matches = []
    exclusions = []
    n_fail = n_spec_fail = 0
    error_types = defaultdict(int)
    for i, doc in enumerate(docs):
        try:
            roots = [t for t in doc if t.dep_ == 'ROOT']
            if len(roots) != 1:
                raise ValueError("More than one roots")
            if roots[0].pos_ not in ['VERB', 'AUX']:
                raise ValueError("ROOT is not VERB")
            if not is_root_nsubj(doc[0], roots[0]):
                raise ValueError("Invalid subject")
            if len(wn.synsets(lemmatize(roots[0]))) == 0:
                raise ValueError("ROOT not found in WN")
            for tok in doc:
                if is_possibly_nominal(tok):
                    for t in tok.children:
                        if (t.text in placeholder_cands + names or t.pos_ == 'PRON') \
                                and t.dep_ in ent_exclusion:
                            raise ValueError("Invalid ent dep: %s %s" % (t.text, t.dep_))
                        if (tok.text in placeholder_cands + names or tok.pos_ == 'PRON') \
                                and t.dep_ in ent_child_exclusion:
                            raise ValueError("Invalid ent child dep: %s %s" % (t.text, t.dep_))
                if (tok.text in placeholder_cands + names or tok.pos_ == 'PRON') \
                        and tok.dep_ in ent_exclusion:
                        raise ValueError("Invalid ent dep: %s %s" % (tok.text, tok.dep_))
                if tok.text in placeholder_cands and tok.pos_ != 'NOUN':
                    raise ValueError("Invalid ent pos: %s %s" % (tok.text, tok.pos_))
                if tok.text in names and tok.pos_ != 'PROPN':
                    raise ValueError("Invalid ent pos: %s %s" % (tok.text, tok.pos_))
                if tok.dep_ == 'compound' and ((tok.head.pos_ not in ['NOUN', 'PROPN', 'NUM']
                                                and tok.head.tag_ != 'VBG') or tok.head.text in ['home']):
                    raise ValueError("Invalid compound: %s" % (tok.text))
            match = match_clause(roots[0])
        except ValueError as v:
            n_fail += 1
            e_type = str(v).split(':')[0]
            if e_type == 'Invalid compound':
                print(str(v))
                printout(doc, i)
            error_types[e_type] += 1
            matches.append(None)
            exclusions.append(i)
            continue
        matches.append(match)
    print(n_fail, len(docs), n_fail / len(docs))
    for k, v in error_types.items():
        print(k, v)
    json.dump(exclusions, open(os.path.join(output_path, 'exclusions.json'), 'w'))


def print_trees():
    docs = pickle.load(open(os.path.join(output_path, 'parse.4.bin'), 'rb'))
    replacements = [json.loads(l) for l in open(os.path.join(output_path, 'replacements.4.jsonl'))]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))

    ind = list(range(len(docs)))
    random.seed(0)
    random.shuffle(ind)
    dep_samples = defaultdict(list)
    dep_pos = defaultdict(set)
    dep_tag = defaultdict(set)
    for i in ind:
        if i in exclusions:
            continue
    for i in range(len(docs)):
        printout(docs[i], i)
        print(replacements[i])
        subs = []
        for tok in docs[i]:
            if is_possibly_predicate(tok) or is_possibly_nominal(tok):
                text = str(docs[i][tok.left_edge.i: tok.right_edge.i + 1])
                subs.append((tok.text, text))
                dep_samples[tok.dep_].append((i, text))
                dep_pos[tok.dep_].add(tok.pos_)
                dep_tag[tok.tag_].add(tok.tag_)
        print(subs)
        print('==========================\n')

    for k in dep_samples:
        samples = dep_samples[k][:5]
        print(k, len(dep_samples[k]))
        for i, text in samples:
            print(text)
            printout(docs[i], i)
        print('==========================\n')


def find_i_by_idx(doc, char_idx):
    for i, token in enumerate(doc):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return token
        if char_idx < token.idx:
            return doc[i - 1]
    raise ValueError()

# replacements: List[(l.i, r.i, source, sub, type)], replacements from heads.1, guaranteed no intersection
# type

def rebuild_texts():
    docs = pickle.load(open(os.path.join(output_path, 'parse.4.bin'), 'rb'))
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
    replacements = [json.loads(l) for l in open(os.path.join(output_path, 'replacements.4.jsonl'))]
    fw = open(os.path.join(output_path, 'heads.5.txt'), 'w')
    rebuilt_docs = []

    persons = {'Alice': 'PersonX', 'Bob': 'PersonY', 'Charlie': 'PersonZ'}
    print('Len', len(replacements), len(docs))
    for i, doc in enumerate(docs):
        if i in exclusions:
            fw.write('\n')
            rebuilt_docs.append(None)
            continue
        tokens = []
        rebuilt_docs.append(tokens)
        replacement = replacements[i]
        for tok in doc:
            if tok.i == len(doc) - 1 and tok.dep_ == 'punct':
                continue
            tok_dict = {'dep': tok.dep_, 'text': tok.text, 'ws': tok.whitespace_, 'pos': tok.pos_, 'tag': tok.tag_,
                        'lemma': lemmatize(tok), 'modifier': is_modifier(tok), 'predicate': is_possibly_predicate(tok),
                        'nominal': is_possibly_nominal(tok), 'prep': is_possibly_prep(tok)}
            tokens.append(tok_dict)
        for k, t in enumerate(tokens):
            t['head'] = tokens[doc[k].head.i]

        extra_tokens = []
        for l_i, r_i, source, sub, type in replacement:
            if sub.startswith(' '):
                l_i += 1
            l_pos = find_i_by_idx(doc, l_i).i
            assert doc[l_pos].idx == l_i
            if type == 'placeholder_det' or type.endswith('extradet') or type.endswith('extraposs') \
                    or type.endswith('extracomp'):
                continue
            elif type == 'placeholder':
                tokens[l_pos]['text'] = tokens[l_pos]['lemma'] = '_'
            elif type in ['nns', 'verb_conj']:
                assert tokens[l_pos]['pos'] in ['VERB', 'AUX'] and tokens[l_pos]['tag'] in ['VBD', 'VBN', 'VB']\
                    , str(tokens[l_pos])
                vbz = getInflection(lemmatize(doc[l_pos]), 'VBZ')[0]
                tokens[l_pos]['text'] = vbz
                tokens[l_pos]['tag'] = 'VBZ'
                tokens[l_pos]['lemma'] = lemmatize(doc[l_pos])
            elif type.endswith('_obj') or type.endswith('_subj'):
                assert tokens[l_pos]['text'] in ['him', 'She'], tokens[l_pos]['text']
                original = type.split('_')[0]
                tokens[l_pos]['text'] = original
                tokens[l_pos]['pos'] = 'PROPN'
                tokens[l_pos]['tag'] = 'NNP'
                tokens[l_pos]['lemma'] = original
            elif type.endswith('_poss'):
                assert tokens[l_pos]['text'] == 'his', tokens[l_pos]['text']
                original = type.split('_')[0]
                extra_tokens.append(({'text': "'s", 'ws': tokens[l_pos]['ws'], 'pos': 'PART', 'tag': 'POS',
                                      'dep': 'case', 'head': tokens[l_pos], 'lemma': "'s", 'modifier': True,
                                      'predicate': False, 'prep': False, 'nominal': False}, l_pos + 1))
                tokens[l_pos]['text'] = original
                tokens[l_pos]['pos'] = 'PROPN'
                tokens[l_pos]['tag'] = 'NNP'
                tokens[l_pos]['ws'] = ''
                tokens[l_pos]['lemma'] = original
            else:
                raise ValueError()

        for k, tok_dict in enumerate(tokens):
            if tok_dict['text'] in ['his', 'her', 'their']:
                if i in [946, 8184]:
                    tok_dict['text'] = tok_dict['lemma'] = 'his'
                    continue
                original = 'Alice' if tok_dict['text'] == 'her' else 'Bob'
                extra_tokens.append(({'text': "'s", 'ws': tok_dict['ws'], 'pos': 'PART', 'tag': 'POS',
                                      'dep': 'case', 'head': tok_dict, 'lemma': "'s", 'modifier': True,
                                      'predicate': False, 'prep': False, 'nominal': False}, k + 1))
                tok_dict['text'] = original
                tok_dict['pos'] = 'PROPN'
                tok_dict['tag'] = 'NNP'
                tok_dict['ws'] = ''
                tok_dict['lemma'] = original

        extra_tokens.sort(key=lambda x: x[-1], reverse=True)
        for t, pos in extra_tokens:
            tokens.insert(pos, t)
        tokens = [t for t in tokens if t is not None]

        text = ''
        for k, tok_dict in enumerate(tokens):
            if tok_dict['text'] in persons:
                tok_dict['text'] = tok_dict['lemma'] = persons[tok_dict['text']]
            tok_dict['i'] = k
            tok_dict['idx'] = len(text)
            text += tok_dict['text'] + tok_dict['ws']

        for t in tokens:
            t['head'] = t['head']['i']

        # if len([r for r in replacement]):
        #     printout(doc)
        #     print(text)
        #     printout_dict(tokens)
        fw.write(text + '\n')

    fw = open(os.path.join(output_path, 'docs.jsonl'), 'w')
    for d in rebuilt_docs:
        fw.write(json.dumps(d) + '\n')

from atomic_parse_utils import Token, get_left_edge, get_right_edge, get_children, get_text

sample_patterns = []
def tree_explorer(doc, tok):
    whole_children = defaultdict(list)
    pattern_children = defaultdict(list)
    tok_children = get_children(doc, tok)
    deps = [t.dep for t in tok_children]
    for i, t in enumerate(tok_children):
        c = deps[i]
        next = t
        ct = get_children(doc, t)
        if len(ct) == 1 and t.dep == ct[0].dep:
            next = ct[0]
        if next.modifier is True:
            continue
        whole_tree, pattern_tree = tree_explorer(doc, next)
        whole_children[c].append(whole_tree)
        pattern_children[c].append(pattern_tree)
    keys = list(whole_children.keys())
    keys.sort()
    ordered_child = OrderedDict()
    ordered_pattern = OrderedDict()
    for k in keys:
        ordered_child[k] = whole_children[k]
        ordered_pattern[k] = pattern_children[k]
    if tok.nominal or tok.predicate:
        sample_patterns.append((ordered_pattern, tok))
        ordered_pattern = OrderedDict()
    return ordered_child, ordered_pattern

def get_entity(doc, tok, accept_deps=['compound'], return_range=False):
    l_i = r_i = tok.i
    ch = get_children(doc, tok)
    while any([t.dep in accept_deps and t.i < l_i for t in ch]):
        for t in ch:
            if t.i >= l_i:
                break
            if t.dep in accept_deps:
                l_i = t.i
                ch = get_children(doc, doc[l_i])
                break

    ch = get_children(doc, tok)
    while any([t.dep in accept_deps and t.i > r_i for t in ch]):
        for t in reversed(ch):
            if t.i <= r_i:
                break
            if t.dep in accept_deps:
                r_i = t.i
                ch = get_children(doc, doc[r_i])
                break
    if return_range:
        return (l_i, r_i)
    text = get_text(doc)
    pre_text = text[doc[l_i].idx: tok.idx]
    post_text = text[tok.idx + len(tok.text): doc[r_i].idx + len(doc[r_i].text)]
    return pre_text + tok.lemma + post_text

def collect_components():
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
    sample_components = []
    for i, doc in tqdm.tqdm(enumerate(docs)):
        sample_components.append([])
        if i in exclusions:
            continue
        doc = [Token(**t) for t in doc]
        sample_patterns.clear()
        roots = [t for t in doc if t.dep == 'ROOT']
        tree, _ = tree_explorer(doc, roots[0])
        sample_comp = sample_components[-1]

        for tree, tok in sample_patterns:
            sample_comp.append(tok.i)

    fw = open(os.path.join(output_path, 'components.jsonl'), 'w')
    for s in sample_components:
        fw.write(json.dumps(s) + '\n')


def stat_components():
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    components = [json.loads(l) for l in open(os.path.join(output_path, 'components.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))

    comp_poses = defaultdict(list)
    comp_deps = defaultdict(list)
    comp_tags = defaultdict(list)

    for i, doc in tqdm.tqdm(enumerate(docs)):
        if i in exclusions:
            continue
        doc = [Token(**t) for t in doc]
        for c_i in components[i]:
            comp_poses[doc[c_i].pos].append((doc, c_i))
            comp_deps[doc[c_i].dep].append((doc, c_i))
            comp_tags[doc[c_i].tag].append((doc, c_i))

    print('===========COMP POS')
    for k, samples in comp_poses.items():
        print(k, len(samples))
    for k, samples in comp_poses.items():
        print("POS = ", k)
        for doc, c_i in samples:
            print(c_i, doc[c_i].text, 'P' if doc[c_i].predicate else '',
                  'M' if doc[c_i].modifier else '', 'N' if doc[c_i].nominal else '')
            printout_dict(doc)

    print('===========COMP DEP')
    for k, samples in comp_deps.items():
        print(k, len(samples))
    for k, samples in comp_deps.items():
        print("DEP = ", k)
        for doc, c_i in samples:
            print(c_i, doc[c_i].text, 'P' if doc[c_i].predicate else '',
                  'M' if doc[c_i].modifier else '', 'N' if doc[c_i].nominal else '')
            printout_dict(doc)

    print('===========COMP TAG')
    for k, samples in comp_tags.items():
        print(k, len(samples))
    for k, samples in comp_tags.items():
        print("TAG = ", k)
        for doc, c_i in samples:
            print(c_i, doc[c_i].text, 'P' if doc[c_i].predicate else '',
                  'M' if doc[c_i].modifier else '', 'N' if doc[c_i].nominal else '')
            printout_dict(doc)

from atomic_parse_utils import wn_query
def find_phrasal_verb(doc, c_i):
    tok = doc[c_i]
    # match by raw text
    r_i = c_i + 1
    result = None
    text = doc[c_i].lemma + doc[c_i].ws
    while r_i < len(doc):
        lem = doc[r_i].lemma
        if lem in ['PersonX', 'PersonY', 'PersonZ']:
            lem = 'one'
        elif lem == '_':
            lem = 'it'
        text += lem + doc[r_i].ws
        syn = wn_query(text.strip().replace(' ', '_'))
        if syn:
            result = (syn, text.strip(), c_i, r_i)
        r_i += 1
    if result is not None:
        return result

    children = get_children(doc, tok)
    adjuncts = []
    for ch in children:
        if ch.dep in ['prep', 'prt', 'advmod', 'dobj', 'ccomp', 'xcomp']:
            adjuncts.append((ch.i, 0, ch.lemma))
            if ch.lemma != ch.text:
                adjuncts.append((ch.i, 1, ch.text))
    for adj_i, _, adj_text in adjuncts:
        children = get_children(doc, doc[adj_i])
        for ch in children:
            if ch.dep in ['prep', 'prt', 'advmod', 'dobj', 'ccomp', 'xcomp']:
                adjuncts.append((ch.i, 0, ch.lemma))
                if ch.lemma != ch.text:
                    adjuncts.append((ch.i, 1, ch.text))
    adjuncts = list(set(adjuncts))
    adjuncts.sort()
    adjuncts = [(a[0], a[2]) for a in adjuncts]

    for cent in [(tok.i, tok.lemma), (tok.i, tok.text)]:
        for i in range(len(adjuncts)):
            for j in range(i+1, len(adjuncts)):
                words = [cent, adjuncts[i], adjuncts[j]]
                words.sort()
                text = '_'.join([k[1] for k in words])
                syn = wn_query(text)
                if syn:
                    return (syn, text.replace('_', ' '), min([w[0] for w in words]), max([w[0] for w in words]))

            words = [cent, adjuncts[i]]
            words.sort()
            text = '_'.join([k[1] for k in words])
            syn = wn_query(text)
            if syn:
                return (syn, text.replace('_', ' '), min([w[0] for w in words]), max([w[0] for w in words]))
    return None

def find_mods(doc, tok):
    mods = []
    children = get_children(doc, tok)
    for ch in children:
        if ch.modifier and ch.pos != 'TO':
            mods.append(ch.i)
    return mods

from atomic_parse_utils import is_possibly_light
def merge_nominal_pred_match(nom_sub, pred_sub):
    nom_texts = [n[2] for n in nom_sub]
    results = nom_sub
    for p in pred_sub:
        if p[2] not in nom_texts:
            results.append(p)
    return results

is_covered = []
def collect_predicate_group(pb, doc, c_i, mods):
    mods = copy.copy(mods)
    tok = doc[c_i]
    results = []
    children = get_children(doc, tok)
    # Itself is the core
    mods.extend(find_mods(doc, tok))
    phrase = find_phrasal_verb(doc, c_i)
    if phrase:
        syn, text, l, r = phrase
        # mods = [m for m in mods if m not in text]
        results.append((mods, syn, text, l, r))
    # In these special constructions we ignore the multi-word case and leave it to annotators
    for ch in children:
        if ch.dep == 'acomp' and ch.text in ['certain', 'meant', 'able', 'due', 'obliged', 'about', 'forced',
                                             'set', 'allowed', 'going', 'supposed', 'bound', 'likely', 'sure']:
            gch = get_children(doc, ch)
            ex_mods = mods + [ch.i]
            for gc in gch:
                if gc.dep in ['ccomp', 'xcomp'] and gc.predicate:
                    if not any([(t.text == 'to' and t.tag == 'TO') for t in get_children(doc, gc)]):
                        continue
                    is_covered.append(gc.i)
                    is_covered.append(ch.i)
                    results.extend(collect_predicate_group(pb, doc, gc.i, ex_mods))

                    for ggch in get_children(doc, gc):
                        if ggch.dep == 'conj' and ggch.predicate:
                            results.extend(collect_predicate_group(pb, doc, ggch.i, ex_mods))

                    return results
        elif ch.dep == 'acomp':
            is_covered.append(ch.i)
            ex_mods = mods + [tok.i]
            results.extend(collect_predicate_group(pb, doc, ch.i, ex_mods))

            for gch in get_children(doc, ch):
                if gch.dep == 'conj' and gch.predicate:
                    results.extend(collect_predicate_group(pb, doc, gch.i, ex_mods))

            return results
    # Light verb, core determined by dobj (noun) or verbal comp
    if is_possibly_light(tok):
        for ch in children:
            if ch.dep in ['dobj', 'ccomp', 'xcomp', 'acomp'] and (ch.nominal or ch.predicate):
                nominal_sub = collect_nominal_group(pb, doc, ch.i, mods) if ch.nominal and ch.tag != 'VBG' else []
                pred_sub = collect_predicate_group(pb, doc, ch.i, mods) if ch.predicate else []
                results.extend(merge_nominal_pred_match(nominal_sub, pred_sub))

                for gch in get_children(doc, ch):
                    if gch.dep == 'conj' and (gch.predicate or gch.nominal):
                        nominal_sub = collect_nominal_group(pb, doc, gch.i, mods) if gch.nominal and gch.tag != 'VBG' else []
                        pred_sub = collect_predicate_group(pb, doc, gch.i, mods) if gch.predicate else []
                        results.extend(merge_nominal_pred_match(nominal_sub, pred_sub))

                if ch.tag == 'VBG' or (ch.text.endswith('ing') and wn_query(ch.text, 'v')):
                    return results
    else:
        # Itself is a modal, core determined by xcomp/ccomp
        for ch in children:
            if ch.dep in ['ccomp', 'xcomp'] and ch.predicate:
                # if not any([(t.text == 'to' and t.tag == 'TO') for t in get_children(doc, ch)]):
                #     continue
                ex_mods = mods + [tok.i]
                is_covered.append(ch.i)
                results.extend(collect_predicate_group(pb, doc, ch.i, ex_mods))

                for gch in get_children(doc, ch):
                    if gch.dep == 'conj' and gch.predicate:
                        results.extend(collect_predicate_group(pb, doc, gch.i, ex_mods))

                # Presumed modal verbs, e.g. raising-to-subject
                if tok.lemma in ['seem', 'appear', 'be', 'have', 'used', 'ought', 'get']:
                    return results

    syn = wn_query(tok.text, 'v')
    if not syn:
        syn = wn_query(tok.lemma, 'v')
    if tok.pos not in ['AUX', 'VERB'] or not syn or tok.text.endswith('ing'):
        syn_full = wn_query(tok.text)
        if not syn_full:
            syn_full = wn_query(tok.lemma)
        syn = list(set(syn).union(syn_full))
    if syn:
        results.append((mods, syn, tok.lemma, c_i, c_i))
    return results

from atomic_parse_utils import is_pb_close, correct_mislemmas
def collect_by_substr(doc, tok, c_i, text, pb, mods, marks=None):
    marks[c_i] = True
    results = []
    le = get_left_edge(doc, tok).i
    re = get_right_edge(doc, tok).i
    for l in range(le, c_i + 1):
        if marks is not None and not marks[l]:
            continue
        # if doc[l].text == 'the' or doc[l].dep == 'poss':
        if doc[l].dep in ['det', 'poss']:
            continue
        for r in range(c_i, re + 1):
            if marks is not None and not marks[r]:
                continue
            cand_subs = []
            lem = correct_mislemmas(doc[c_i])
            for mid_text in [lem, doc[c_i].text]:
                sub_text = text[doc[l].idx: doc[c_i].idx] + mid_text + \
                           text[doc[c_i].idx + len(doc[c_i].text): doc[r].idx + len(doc[r].text)]
                cand_subs.append(sub_text)
            syn = wn_query(cand_subs[1], 'n')
            if not syn:
                syn = wn_query(cand_subs[0], 'n')
            if tok.pos not in ['NOUN', 'PROPN'] or not syn:
                syn_full = wn_query(cand_subs[1])
                if not syn_full:
                    syn_full = wn_query(cand_subs[0])
                syn = list(set(syn).union(syn_full))

            pb_resp = pb.query(cand_subs[0], 'mi', 100, do_filter=False)
            if pb_resp or syn:
                results.append((mods, syn, cand_subs[0], l, r))
            elif lem != doc[c_i].text:
                pbd_resp = pb.query(cand_subs[1], 'mi', 100, do_filter=False)
                if pbd_resp:
                    results.append((mods, syn, cand_subs[1], l, r))

    return results

def collect_modifier_marks(doc, tok, c_i):
    le = get_left_edge(doc, tok).i
    re = get_right_edge(doc, tok).i
    mark = [False for _ in doc]
    for i in range(le, re + 1):
        if i == c_i or not doc[i].modifier:
            continue
        tok_i = i
        while doc[tok_i].modifier and tok_i != c_i:
            tok_i = doc[tok_i].head
        if tok_i == c_i:
            mark[i] = True
    return mark

transparent_noun_prep = json.load(open(os.path.join('data', 'nomlex.transparent.json')))
cc = set()
def collect_nominal_group(pb, doc, c_i, mods):
    mods = copy.copy(mods)
    if doc[c_i].tag == 'VBG':
        return collect_predicate_group(pb, doc, c_i, mods)
    if doc[c_i].text in ['_', 'PersonX', 'PersonY', 'PersonZ']:
        ex_mods = mods + find_mods(doc, doc[c_i])
        return [(ex_mods, [], doc[c_i].text, c_i, c_i)]
    tok = doc[c_i]
    text = get_text(doc)
    results = []
    children = get_children(doc, tok)

    transparent_preps = transparent_noun_prep.get(tok.lemma, [])
    if doc[c_i].pos in ['DET', 'PRON', 'NUM']:
        transparent_preps.append('of')
    for ch in children:
        ex_mods = mods
        if ch.text == 'of':
            if doc[c_i].pos in ['DET', 'PRON'] and doc[c_i].text.startswith('no'):
                ex_mods = mods + [c_i]

            grandchildren = get_children(doc, ch)
            for gch in grandchildren:
                if gch.nominal:
                    results.extend(collect_nominal_group(pb, doc, gch.i, ex_mods))
                    for ggch in get_children(doc, gch):
                        if ggch.dep == 'conj' and ggch.nominal:
                            results.extend(collect_nominal_group(pb, doc, ggch.i, ex_mods))
                    if transparent_preps:
                        return results

    if doc[c_i].pos not in ['PRON', 'DET']:
        marks = collect_modifier_marks(doc, tok, c_i)
        mods.extend(find_mods(doc, tok))
        collected = collect_by_substr(doc, tok, c_i, text, pb, mods, marks)
        results.extend(collected)
    if not results:
        results = [(mods, [], doc[c_i].text, c_i, c_i)]
    return results

def collect_candidate(pb, doc, c_i):
    nominal_sub = collect_nominal_group(pb, doc, c_i, []) if doc[c_i].nominal else []
    pred_sub = collect_predicate_group(pb, doc, c_i, []) if doc[c_i].predicate else []
    results = merge_nominal_pred_match(nominal_sub, pred_sub)

    results = list(reversed(results))
    for i in range(len(results)):
        results[i] = (results[i][0], [s.name() for s in results[i][1]], results[i][2], results[i][3], results[i][4])
        results[i][1].sort()
    return results

def align_concepts():
    from atomic_parse_utils import ProbaseClient
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    components = [json.loads(l) for l in open(os.path.join(output_path, 'components.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
    all_linked_concepts = []
    pb = ProbaseClient()
    # pb = None
    appeared_mods = set()

    for i, doc in tqdm.tqdm(enumerate(docs)):
        all_linked_concepts.append([])
        linked_concepts = all_linked_concepts[-1]
        if i in exclusions:
            continue
        doc = [Token(**t) for t in doc]
        is_covered.clear()
        for c_i in components[i]:
            variations = collect_candidate(pb, doc, c_i)
            linked_concepts.append(variations)

        # print(get_text(doc))
        printout_dict(doc)
        all_texts = defaultdict(set)
        for k, variations in enumerate(linked_concepts):
            c_i = components[i][k]
            if doc[c_i].text in ['_', 'PersonX', 'PersonY', 'PersonZ']:
                continue
            # print(doc[c_i].text, ':')
            for mods, syn, text, l_i, r_i in variations:
                syn_def = [wn.synset(s).definition() for s in syn]
                is_pb = pb.query(text, do_filter=False)
                # print(text, 'PB' if is_pb else '', [doc[m_i].lemma for m_i in mods], syn_def, 'Ds' if not (l_i <= c_i <= r_i) else '')
                all_texts[text].add(c_i)
            # print()
        covered_set = set(is_covered)
        # if len(covered_set) > 0:
        #     print('Covered: ', '|'.join(['%d %s' % (c_i, doc[c_i].text) for c_i in covered_set]))
        #     for k, c_i in enumerate(components[i]):
        #         if c_i in covered_set:
        #             for mods, syn, text, l_i, r_i in linked_concepts[k]:
        #                 if len(all_texts[text]) == 1:
        #                     print('Special: ', text)

        # print('============')
    open(os.path.join(output_path, 'linked_concepts.jsonl'), 'w').writelines(
        [json.dumps(l) + '\n' for l in all_linked_concepts])

def build_glossbert_inputs():
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    linked_concepts = [json.loads(l) for l in open(
        os.path.join(output_path, 'linked_concepts.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
    from nltk.corpus import wordnet as wn

    dest_path = r'parse/glossbert.csv'
    fw = open(dest_path, 'w')
    fw.write('target_id	label	sentence	gloss	sense_key\n')

    for i, doc in tqdm.tqdm(enumerate(docs)):
        if i in exclusions:
            continue
        doc = [Token(**t) for t in doc]
        for j, variations in enumerate(linked_concepts[i]):
            for k, (mods, syn, sub_text, l_i, r_i) in enumerate(variations):
                sent = ' '.join([t.text for t in doc[:l_i]] + ['"'] + [t.text for t in doc[l_i: r_i + 1]] + ['"']
                                 + [t.text for t in doc[r_i + 1:]]) + ' .'
                sent = sent.replace('PersonX', 'Alice').replace('PersonY', 'Bob').replace('PersonZ', 'Charlie') \
                    .replace('_', 'X')
                for l, s in enumerate(syn):
                    id = '%d|%d|%d|%d' % (i, j, k, l)
                    gloss = sub_text + ' : ' + wn.synset(s).definition()
                    fw.write('\t'.join([id, '0', sent, gloss, s]) + '\n')

def sort_by_glossbert_results():
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    linked_concepts = [json.loads(l) for l in open(
        os.path.join(output_path, 'linked_concepts.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))

    scores = open(os.path.join(r'parse', 'glossbert_results.txt')).read().splitlines()
    scores = [float(s.split(' ')[-1]) for s in scores]
    loc = 0

    for i, doc in tqdm.tqdm(enumerate(docs)):
        if i in exclusions:
            continue
        for j, variations in enumerate(linked_concepts[i]):
            for k, (mods, syn, sub_text, l_i, r_i) in enumerate(variations):
                for l, s in enumerate(syn):
                    if loc == len(scores):
                        print('Error at', loc)
                        continue
                    syn[l] = (syn[l], scores[loc])
                    loc += 1
                syn.sort(key=lambda x: -x[-1])
    open(os.path.join(output_path, 'linked_concepts.1.jsonl'), 'w').writelines(
        [json.dumps(l) + '\n' for l in linked_concepts])

def get_depth(doc, tok):
    dep = 0
    while tok.dep != 'ROOT':
        dep += 1
        tok = doc[tok.head]
    return dep

def inject_idioms():
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    linked_concepts = [json.loads(l) for l in open(
        os.path.join(output_path, 'linked_concepts.1.jsonl')).read().splitlines()]
    components = [json.loads(l) for l in open(
        os.path.join(output_path, 'components.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
    idioms = json.load(open(os.path.join('data', 'idiom_samples.json'))) # We managed to found more idioms this time to reduce annotation scale.
    idioms = dict([(int(k), v) for k, v in idioms.items()])

    for i, doc in tqdm.tqdm(enumerate(docs)):
        if i in exclusions:
            continue
        if i not in idioms:
            continue
        doc = [Token(**t) for t in doc]
        for j, variations in enumerate(linked_concepts[i]):
            c_i = components[i][j]
            if doc[c_i].dep != 'ROOT':
                continue
            exist = False
            for k, (mods, syn, sub_text, l_i, r_i) in enumerate(variations):
                if sub_text == idioms[i]:
                    variations[k] = (mods, syn + [('Idiom:' + idioms[i], 1.0)], sub_text, 0, -1)
                    exist = True
                    break
            if not exist:
                linked_concepts[i][j].append(([], [('Idiom:' + idioms[i], 1.0)], idioms[i], 0, -1))

    open(os.path.join(output_path, 'linked_concepts.2.jsonl'), 'w').writelines(
        [json.dumps(l) + '\n' for l in linked_concepts])


def prepare_all_annotate_cands():
    components = [json.loads(l) for l in open(os.path.join(output_path, 'components.jsonl')).read().splitlines()]
    docs = [json.loads(l) for l in open(os.path.join(output_path, 'docs.jsonl')).read().splitlines()]
    linked_concepts = [json.loads(l) for l in open(
        os.path.join(output_path, 'linked_concepts.jsonl')).read().splitlines()]
    exclusions = json.load(open(os.path.join(output_path, 'exclusions.json')))
    idioms = json.load(open(os.path.join(output_path, 'idiom_scanned.json')))

    filtered = []
    cnt = 0

    for i, doc in tqdm.tqdm(enumerate(docs)):
        if i in exclusions:
            continue
        doc = [Token(**t) for t in doc]
        if str(i) in idioms:
            print(i, "Idiom:", get_text(doc))
            continue
        text = get_text(doc)
        if '_' in text:
            continue
        print(i, get_text(doc))
        printout_dict(doc)
        for j, variations in enumerate(linked_concepts[i]):
            cnt += 1
            c_i = components[i][j]
            if doc[c_i].text in ['_', 'PersonX', 'PersonY', 'PersonZ', 'about']:
                # print(doc[c_i].text)
                continue
            if doc[c_i].dep == 'npadvmod':
                # print(doc[c_i].text)
                continue
            children = get_children(doc, doc[c_i])
            if doc[c_i].pos in ['PRON', 'DET']:
                transparent = False
                for ch in children:
                    if ch.text == 'of':
                        transparent = True
                        break
                if not transparent:
                    continue
            if doc[c_i].text == 'one' and len(variations) == 1:
                continue
            if doc[c_i].dep == 'conj' and doc[doc[c_i].head].text == doc[c_i].text:
                continue
            if any([c.dep == 'conj' and c.text == doc[c_i].text for c in children]):
                continue
            print(doc[c_i].text)
            filtered.append((i, j))
        print('\n============')

    print('Total %d samples of' % len(filtered), cnt)

    json.dump(filtered, open(os.path.join(output_path, 'all_valid_cands.json'), 'w'))


if __name__ == '__main__':
    # Run the following parts one by one
    # Part 1: fix errors and ambiguities in ATOMIC to get correct parsing results with best efforts
    print("Create sources...")
    create_source()
    print("Preprocess heads...")
    preprocess_heads()
    print("Fix wildcards...")
    fix_wildcards()
    print("Fix verbs...")
    fix_verbs()
    print("Fix possession/iobj...")
    fix_poss_iobj()
    print("Match dependencies...")
    match_dependencies()
    print("Rebuild texts...")
    rebuild_texts()
    print("ATOMIC cleaned")

    # # Part 2: identification and concept linking
    collect_components()
    stat_components()
    align_concepts()
    build_glossbert_inputs() # Saved to parse/glossbert.csv

    # # After that, run GlossBERT
    # # Part 3:
    sort_by_glossbert_results()
    inject_idioms()
    prepare_all_annotate_cands()
    pass