# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from typing import List

# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb
import logging

from torch import cuda
import sys
sys.path.append(os.getcwd())
from split.utils import write_items

from optparse import OptionParser

device = 'cuda' if cuda.is_available() else 'cpu'

from mosaic.infra.modeling import train, validate, beam_generations
from mosaic.datasets.KGDataset import KGDataset

DEBUG = False
NUM_INST = 100

CS_RELATIONS_2NL = {
    "AtLocation": "located or found at or in or on",
    "CapableOf": "is or are capable of",
    "Causes" : "causes",
    "CausesDesire": "makes someone want",
    "CreatedBy": " is created by",
    "Desires": "desires",
    "HasA": "has, possesses, or contains",
    "HasFirstSubevent": "begins with the event or action",
    "HasLastSubevent": "ends with the event or action",
    "HasPrerequisite": "to do this, one requires",
    "HasProperty": "can be characterized by being or having",
    "HasSubEvent" : "includes the event or action",
    "HinderedBy" : "can be hindered by",
    "InstanceOf" : "is an instance of",
    "isAfter" : "happens after",
    "isBefore" : "happens before",
    "isFilledBy" : "blank can be filled by",
    "MadeOf": "is made of",
    "MadeUpOf": "made up of",
    "MotivatedByGoal": "is a step towards accomplishing the goal",
    "NotDesires": "do not desire",
    "ObjectUse": "used for",
    "UsedFor": "used for",
    "oEffect" : "as a result, PersonY or others will",
    "oReact" : "as a result, PersonY or others feel",
    "oWant" : "as a result, PersonY or others want to",
    "PartOf" : "is a part of",
    "ReceivesAction" : "can receive or be affected by the action",
    "xAttr" : "PersonX is seen as",
    "xEffect" : "as a result, PersonX will",
    "xReact" : "as a result, PersonX feels",
    "xWant" : "as a result, PersonX wants to",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "xReason" : "because",
    "general Effect" : "as a result, other people or things will",
    "general Want" : "as a result, other people or things want to",
    "general React" : "as a result, other people or things feel",
    # inversed
    "AtLocation inversed": "can find or include", # "located or found at or in or on"
    "CapableOf inversed": "is a skill of", # "is or are capable of"
    "Causes inversed" : "because", # causes
    "CausesDesire inversed": "because", # "makes someone want",
    "CreatedBy inversed": "create", # "is created by",
    "Desires inversed": "is desired by", # "desires",
    "HasA inversed": "is possessed by",# "has, possesses, or contains",
    "HasFirstSubevent inversed": "is the beginning of", # "begins with the event or action",
    "HasLastSubevent inversed": "is the end of", # "ends with the event or action",
    "HasPrerequisite inversed": "is the prerequisite of",# "to do this, one requires",
    "HasProperty inversed": "is the property of", # "can be characterized by being or having",
    "HasSubEvent inversed" : "is included by",# "includes the event or action",
    "HinderedBy inversed" : "hinder", #"can be hindered by",
    "InstanceOf inversed" : "include", #" is an example or instance of", not sure about this.
    "isAfter inversed" : "happens before", # "happens after",
    "isBefore inversed" : "happens after", # "happens before",
    "isFilledBy inversed" : "can fill",# "blank can be filled by",
    "MadeOf inversed": "make up of", # "is made of", 
    "MadeUpOf inversed": "is made of", # "made up of",
    "MotivatedByGoal inversed": "motivate", # "is a step towards accomplishing the goal",
    "NotDesires inversed": "is not desired by", # "do not desire",
    "ObjectUse inversed": "could make use of", # "used for",
    "UsedFor inversed": "could make use of", # "used for",
    "oEffect inversed" : "because", #"as a result, PersonY or others will",
    "oReact inversed" : "because", #"as a result, PersonY or others feel",
    "oWant inversed" : "because", # "as a result, PersonY or others want to",
    "PartOf inversed" : "include", # "is a part of",
    "ReceivesAction inversed" : "affect", # "can receive or be affected by the action",
    "xAttr inversed" : "", # "PersonX is seen as",
    "xEffect inversed" : "because", # "as a result, PersonX will",
    "xReact inversed" : "because", # "as a result, PersonX feels",
    "xWant inversed" : "because",# "as a result, PersonX wants to",
    "xNeed inversed" : "as a result, ",# "but before, PersonX needed",
    "xIntent inversed" : "as a result, ", # "because PersonX wanted",
    "xReason inversed" : "as a result, ",# "because",
    "general Effect inversed" : "because", # "as a result, other people or things will",
    "general Want inversed" : "because", # "as a result, other people or things want to",
    "general React inversed" : "because", # "as a result, other people or things feel",
}

def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

# 1. CSKB_POP COMET: CUDA_VISIBLE_DEVICES=2 EVAL_EVERY=100 TRAIN_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False TRAIN_DATA_PATH=data/kg/pop_cskb/trn.tsv DEV_DATA_PATH=data/kg/pop_cskb/dev.tsv OUT_DIR=data/models/gpt2large-comet-pop TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large
# 2. COMET with NL relations: CUDA_VISIBLE_DEVICES=3 EVAL_EVERY=100 USE_NL_RELATION=True TRAIN_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False OUT_DIR=data/models/gpt2large-comet-nl-rel TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large
# 3. COMET, NL, with inversed, PersonX/Y replaced:
# CUDA_VISIBLE_DEVICES=2 USE_NL_RELATION=True TRAIN_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False OUT_DIR=data/models/gpt2large-comet-inv-xy-rev-nl-rel TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large TRAIN_DATA_PATH=data/kg/atomic2020_data-feb2021/train_inversed_xy_reversed.tsv python models/comet_atomic2020_gpt2/comet_gpt2.py
# CUDA_VISIBLE_DEVICES=1 USE_NL_RELATION=True TRAIN_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False OUT_DIR=data/models/gpt2large-comet-inv-nl-rel TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large TRAIN_DATA_PATH=data/kg/atomic2020_data-feb2021/train_inversed.tsv python models/comet_atomic2020_gpt2/comet_gpt2.py
# 4.1 COMET, inversed_simple, 
# CUDA_VISIBLE_DEVICES=3 USE_NL_RELATION=False TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False EVAL_EVERY=10000 TRAIN_EPOCHS=5 OUT_DIR=data/models/gpt2large-comet-inv-rel-simple TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large TRAIN_DATA_PATH=data/kg/atomic2020_data-feb2021/train_inversed_simple.tsv python models/comet_atomic2020_gpt2/comet_gpt2.py
# 4.2 COMET, inversed_processed train_inversed_processed.tsv
# CUDA_VISIBLE_DEVICES=2 USE_NL_RELATION=False TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False EVAL_EVERY=10000 TRAIN_EPOCHS=5 OUT_DIR=data/models/gpt2large-comet-inv-rel-processed TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large TRAIN_DATA_PATH=data/kg/atomic2020_data-feb2021/train_inversed_processed.tsv python models/comet_atomic2020_gpt2/comet_gpt2.py
# 4.3 COMET, inversed_processed_xy_rev
# CUDA_VISIBLE_DEVICES=2 USE_NL_RELATION=False TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False EVAL_EVERY=10000 TRAIN_EPOCHS=5 OUT_DIR=data/models/gpt2large-comet-inv-rel-processed-xy-rev TOKENIZER=gpt2-large GPT2_MODEL=gpt2-large TRAIN_DATA_PATH=data/kg/atomic2020_data-feb2021/train_inversed_processed_xy_rev.tsv python models/comet_atomic2020_gpt2/comet_gpt2.py

def main():
    # wandb.init(project="gpt2_comet_atomic")

    config = wandb.config
    config.TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 2)) # 32 for the official release
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 2)) 
    config.TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 3))
    config.VAL_EPOCHS = int(os.environ.get("VAL_EPOCHS", 1))
    config.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5")) # 5e-5 for the official release
    config.SEED = int(os.environ.get("SEED", 42))
    config.IN_LEN = int(os.environ.get("IN_LEN", 16))
    config.OUT_LEN = int(os.environ.get("OUT_LEN", 34))
    config.SUMMARY_LEN = 0 # Used for t5
    config.OUT_DIR = os.environ.get("OUT_DIR", "decodings") # dir to save the model, logs, and potentially decodings.
    config.DO_TRAIN =  os.environ.get("DO_TRAIN", "False") == "True"
    config.DO_PRED = os.environ.get("DO_PRED", "False") == "True"
    config.PRED_FILE = str(os.environ.get("PRED_FILE", "data/kg/atomic2020_data-feb2021/test.tsv"))
    config.TOP_K = int(os.environ.get("TOP_K", 40)) # Num of top_k word considered in each step of beam search
    config.PRED_BATCH = 64
    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl") # "data/models/gpt2xl-comet-atomic-2020/tokenizer/"
    config.NUM_GEN = int(os.environ.get("NUM_GEN", 1))
    config.MODEL_NAME = os.environ.get('GPT2_MODEL', "gpt2-xl")
    config.SAVE_EVERY = int(os.environ.get('SAVE_EVERY', -1))
    config.EVAL_EVERY = int(os.environ.get('EVAL_EVERY', -1))
    config.USE_NL_RELATION = bool(os.environ.get('USE_NL_RELATION', "False")=="True") # whether to use natural language descriptions for the relations.

    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    model_name = config.MODEL_NAME
    # model_name = "data/models/gpt2xl-comet-atomic-2020/"
    if not os.path.exists(config.OUT_DIR):
        os.mkdir(config.OUT_DIR)

    # logger
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    formatter = logging.Formatter('%(asctime)s || %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(config.OUT_DIR, "log.txt"), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)


    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER)

    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'additional_special_tokens': [
            'LocationOfAction',
            'HinderedBy',
            'HasFirstSubevent',
            'NotHasProperty',
            'NotHasA',
            'HasA',
            'AtLocation',
            'NotCapableOf',
            'CausesDesire',
            'HasPainCharacter',
            'NotDesires',
            'MadeUpOf',
            'InstanceOf',
            'SymbolOf',
            'xReason',
            'isAfter',
            'HasPrerequisite',
            'UsedFor',
            'MadeOf',
            'MotivatedByGoal',
            'Causes',
            'oEffect',
            'CreatedBy',
            'ReceivesAction',
            'NotMadeOf',
            'xWant',
            'PartOf',
            'DesireOf',
            'HasPainIntensity',
            'xAttr',
            'DefinedAs',
            'oReact',
            'xIntent',
            'HasSubevent',
            'oWant',
            'HasProperty',
            'IsA',
            'HasSubEvent',
            'LocatedNear',
            'Desires',
            'isFilledBy',
            'isBefore',
            'InheritsFrom',
            'xNeed',
            'xEffect',
            'xReact',
            'HasLastSubevent',
            'RelatedTo',
            'CapableOf',
            'NotIsA',
            'ObjectUse',
            'Inversed',
            '[GEN]'
        ]
    })
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_dataset = pd.read_csv(
        os.environ.get('TRAIN_DATA_PATH', "data/kg/atomic2020_data-feb2021/train.tsv"),
        encoding='latin-1', sep="\t")
    if DEBUG:
        train_dataset = train_dataset.head(NUM_INST)
    # train_dataset = train_dataset[['head_event', 'tail_event', 'relation']]
    if config.USE_NL_RELATION:
        train_dataset["head_event"] = train_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL[r], train_dataset["relation"])) \
            + " [GEN]"
    else:  
        train_dataset["head_event"] = train_dataset["head_event"] + ' ' + train_dataset["relation"] \
                               + " [GEN]"
    train_dataset["tail_event"] = train_dataset["tail_event"] + ' [EOS]'
    logger.info(train_dataset.head())
    logger.info(train_dataset["tail_event"])

    val_dataset = pd.read_csv(os.environ.get('DEV_DATA_PATH', "data/kg/atomic2020_data-feb2021/dev.tsv"), encoding='latin-1', sep="\t")
    if DEBUG:
        val_dataset = val_dataset.head(NUM_INST)
    val_dataset = val_dataset[['head_event', 'tail_event', 'relation']]
    if config.USE_NL_RELATION:
        val_dataset["head_event"] = val_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL[r], val_dataset["relation"])) + " [GEN]"
    else:
        val_dataset["head_event"] = val_dataset["head_event"] + ' ' + val_dataset["relation"] + " [GEN]"
    val_dataset["tail_event"] = val_dataset["tail_event"] + ' [EOS]'
    logger.info(val_dataset["tail_event"])
    logger.info(val_dataset.head())

    test_dataset = pd.read_csv(os.environ.get('TEST_DATA_PATH', "data/kg/atomic2020_data-feb2021/test.tsv"), encoding='latin-1', sep="\t")
    if DEBUG:
        test_dataset = test_dataset.head(NUM_INST)
    test_dataset = test_dataset[['head_event', 'tail_event', 'relation']]
    if config.USE_NL_RELATION:
        test_dataset["head_event"] = test_dataset["head_event"] + ' ' + \
        pd.Series(map(lambda r:CS_RELATIONS_2NL[r], test_dataset["relation"])) + " [GEN]"
    else:
        test_dataset["head_event"] = test_dataset["head_event"] + ' ' + test_dataset["relation"] \
                              + " [GEN]"
    test_dataset["tail_event"] = test_dataset["tail_event"] + ' [EOS]'
    logger.info(test_dataset["tail_event"])
    logger.info(test_dataset.head())

    logger.info("TRAIN Dataset tuple count: {}".format(train_dataset.shape))
    logger.info("DEV Dataset tuple_count: {}".format(val_dataset.shape))

    training_set = KGDataset(train_dataset, tokenizer, config.OUT_LEN, config.SUMMARY_LEN, model="gpt2")
    val_set = KGDataset(val_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=False)
    # is_eval controls whether the dataset use batches.
    # Even is_eval is false, we use batches to calculate validation loss 
    test_set = KGDataset(test_dataset, tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
    # for test set, is_eval is True. Generate outputs for one sentence each time.

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
    val_loader = DataLoader(val_set, **val_params, drop_last=True)
    test_loader = DataLoader(test_set, **test_params, drop_last=True)
    
    logging.info("Loading model from {}".format(model_name))
    model = GPT2LMHeadModel.from_pretrained(model_name) #, use_cdn=False
    logging.info("Move model to device {}".format(device))
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # wandb.watch(model, log="all")

    if config.DO_TRAIN:
        logger.info('Initiating Fine-Tuning for the model on our dataset')

        for epoch in range(config.TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader, 
                model_class="gpt2", save_dir=config.OUT_DIR, save_every=config.SAVE_EVERY, eval_every=config.EVAL_EVERY,
                logger=logger)
            model.save_pretrained('{}/checkpoint_{}_{}'.format(config.OUT_DIR, model_name, epoch))
            tokenizer.save_pretrained('{}/tokenizer_checkpoint_{}_{}'.format(config.OUT_DIR, model_name, epoch))
        model.save_pretrained('{}/final_model_{}.pt'.format(config.OUT_DIR, model_name))
        tokenizer.save_pretrained('{}/final_tokenizer_{}.pt'.format(config.OUT_DIR, model_name))

    if config.DO_PRED:

        if config.PRED_FILE.endswith("jsonl"):
            records = read_jsonl_lines(config.PRED_FILE)
            pred_dataset = pd.DataFrame.from_records(records)
            pred_dataset = pred_dataset.rename(columns={"head": "head_event", "tails": "tail_event"})
            pred_dataset = pred_dataset.explode('tail_event')
        else:
            pred_dataset = pd.read_csv(config.PRED_FILE, encoding='latin-1', sep="\t")

        if DEBUG:
            pred_dataset = pred_dataset.head(NUM_INST)

        pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)
        if config.USE_NL_RELATION:
            pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + \
                pd.Series(map(lambda r:CS_RELATIONS_2NL[r], pred_dataset["relation"])) + " [GEN]"
        else:
            pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + pred_dataset["relation"] + " [GEN]"
        pred_dataset["tail_event"] = pred_dataset["tail_event"] + ' [EOS]'
        logger.info(pred_dataset["tail_event"])
        logger.info(pred_dataset.head())

        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TOP_K, num_gen=config.NUM_GEN)
        write_items(os.path.join(config.OUT_DIR, os.path.basename(config.PRED_FILE)+"_pred_generations.jsonl"),
                    [json.dumps(r) for r in pred_generations])

        # Resave the model to keep generations and model associated
        # model.save_pretrained('/models')
        # tokenizer.save_pretrained('/models')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--test_install",
                      action="store_true", default=False,
                      help="Test install, without running any modeling code.")

    (options, args) = parser.parse_args()
    if not options.test_install:
        main()
