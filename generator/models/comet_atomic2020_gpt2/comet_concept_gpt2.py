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
from models.eval_utils import read_jsonl_lines, get_generation_eval_scores

from models.comet_atomic2020_gpt2.comet_gpt2 import CS_RELATIONS_2NL

DEBUG = False
NUM_INST = 100


def main():
    # wandb.init(project="gpt2_comet_atomic")

    config = wandb.config
    config.TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 2)) # 32 for the official release
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 2)) 
    config.TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 3))
    config.VAL_EPOCHS = int(os.environ.get("VAL_EPOCHS", 1))
    config.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5")) # 5e-5 for the official release
    config.SEED = int(os.environ.get("SEED", 42))
    config.IN_LEN = int(os.environ.get("IN_LEN", 24)) # 40 for Stage2 NL
    config.OUT_LEN = int(os.environ.get("OUT_LEN", 52)) # 52 for Stage2 NL
    config.SUMMARY_LEN = 0 # Used for t5
    config.OUT_DIR = os.environ.get("OUT_DIR", "decodings") # dir to save the model, logs, and potentially decodings.
    config.DO_TRAIN =  os.environ.get("DO_TRAIN", "False") == "True"
    config.DO_EVAL = os.environ.get("DO_EVAL", "False") == "True"
    config.DO_PRED = os.environ.get("DO_PRED", "False") == "True"
    config.DO_PRED_EVERY_EPOCH = os.environ.get("DO_PRED_EVERY_EPOCH", "False") == "True"
    config.PRED_FILE = str(os.environ.get("PRED_FILE", "data/kg/atomic2020_data-feb2021/test.tsv"))
    config.TOP_K = int(os.environ.get("TOP_K", 40)) # Num of top_k word considered in each step of beam search
    config.PRED_BATCH = 64
    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-large") # "data/models/gpt2xl-comet-atomic-2020/tokenizer/"
    config.NUM_GEN = int(os.environ.get("NUM_GEN", 1))
    config.MODEL_NAME = os.environ.get('GPT2_MODEL', "gpt2-large")
    config.SAVE_EVERY = int(os.environ.get('SAVE_EVERY', -1))
    config.EVAL_EVERY = int(os.environ.get('EVAL_EVERY', -1))
    config.TRAIN_SHOW_STEP = int(os.environ.get('EVAL_EVERY', 1000))
    config.SAVE_EVERY_EPOCH =  os.environ.get("SAVE_EVERY_EPOCH", "False") == "True"
    config.USE_NL_RELATION = bool(os.environ.get('USE_NL_RELATION', "False")=="True") # whether to use natural language descriptions for the relations.
    config.REMOVE_BRACKET = bool(os.environ.get('REMOVE_BRACKET', "False")=="True")
    config.REPLACE_BRACKET = bool(os.environ.get('REPLACE_BRACKET', "False")=="True")
    config.WEIGHTED_LOSS = float(os.environ.get('WEIGHTED_LOSS', "-1"))
    config.BASE_RATIO_CONTROL = float(os.environ.get('BASE_RATIO_CONTROL', "-1"))
    config.ZERO_SHOT =  os.environ.get("ZERO_SHOT", "False") == "True"

    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    model_name = config.MODEL_NAME
    # model_name = "data/models/gpt2xl-comet-atomic-2020/"
    # if not os.path.exists(config.OUT_DIR):
    os.makedirs(config.OUT_DIR, exist_ok=True)

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

    if not config.ZERO_SHOT:
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
            '[GEN]',
            '<c>', # concept start
            '</c>', # concept end
        ]
        })
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    train_dataset = pd.read_csv(
        os.environ.get('TRAIN_DATA_PATH', "data/kg/atomic2020_data-feb2021/train.tsv"),
        sep="\t", keep_default_na=False)
    if DEBUG:
        train_dataset = train_dataset.head(NUM_INST)
    if config.REMOVE_BRACKET:
        train_dataset["head_event"] = train_dataset["head_event"].str.replace('[', '').str.replace(']', '')
    if config.REPLACE_BRACKET:
        train_dataset["head_event"] = train_dataset["head_event"].str.replace('[', '<c>').str.replace(']', '</c>')
        train_dataset["tail_event"] = train_dataset["tail_event"].str.replace('[', '<c>').str.replace(']', '</c>')

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

    val_dataset = pd.read_csv(os.environ.get('DEV_DATA_PATH', "data/kg/atomic2020_data-feb2021/dev.tsv"), sep="\t"
                              , keep_default_na=False)
    if DEBUG:
        val_dataset = val_dataset.head(NUM_INST)
    val_dataset = val_dataset[['head_event', 'tail_event', 'relation']]
    if config.REMOVE_BRACKET:
        val_dataset["head_event"] = val_dataset["head_event"].str.replace('[', '').str.replace(']', '')
    if config.REPLACE_BRACKET:
        val_dataset["head_event"] = val_dataset["head_event"].str.replace('[', '<c>').str.replace(']', '</c>')
        val_dataset["tail_event"] = val_dataset["tail_event"].str.replace('[', '<c>').str.replace(']', '</c>')
    if config.USE_NL_RELATION:
        val_dataset["head_event"] = val_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL[r], val_dataset["relation"])) + " [GEN]"
    else:
        val_dataset["head_event"] = val_dataset["head_event"] + ' ' + val_dataset["relation"] + " [GEN]"
    val_dataset["tail_event"] = val_dataset["tail_event"] + ' [EOS]'
    logger.info(val_dataset["tail_event"])
    logger.info(val_dataset.head())

    test_dataset = pd.read_csv(os.environ.get('TEST_DATA_PATH', "data/kg/atomic2020_data-feb2021/test.tsv"), sep="\t",
                               keep_default_na=False)
    if DEBUG:
        test_dataset = test_dataset.head(NUM_INST)
    test_dataset = test_dataset[['head_event', 'tail_event', 'relation']]
    if config.REMOVE_BRACKET:
        test_dataset["head_event"] = test_dataset["head_event"].str.replace('[', '').str.replace(']', '')
    if config.REPLACE_BRACKET:
        test_dataset["head_event"] = test_dataset["head_event"].str.replace('[', '<c>').str.replace(']', '</c>')
        test_dataset["tail_event"] = test_dataset["tail_event"].str.replace('[', '<c>').str.replace(']', '</c>')
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

    samples = [training_set.__getitem__(i)['source_ids'].numpy() for i in range(5)]
    for s in samples:
        print(' '.join([tokenizer.decode([t]) for t in s]))

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

    if config.BASE_RATIO_CONTROL != -1:
        from torch.utils.data import WeightedRandomSampler
        is_abs = [int('[' in training_set.text[i].split('[GEN]')[0]) for i in range(len(training_set))]
        n_abs = sum(is_abs)
        n_bases = len(training_set) - n_abs
        base_weight = config.BASE_RATIO_CONTROL * n_abs / n_bases

        weights = [1 if is_abs[i] > 0 else base_weight for i in range(len(training_set))]
        sampler = WeightedRandomSampler(weights, replacement=True, num_samples=len(weights))
        train_params['sampler'] = sampler
        del train_params['shuffle']

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
    logger.info('Start...save to ' + config.OUT_DIR)

    if config.DO_TRAIN:
        logger.info('Initiating Fine-Tuning for the model on our dataset')

        for epoch in range(config.TRAIN_EPOCHS):
            #### DO_PRED every epoch: ####

            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader, 
                model_class="gpt2", save_dir=config.OUT_DIR, save_every=config.SAVE_EVERY, 
                eval_every=config.EVAL_EVERY, train_show_step=config.TRAIN_SHOW_STEP, logger=logger,
                  weighted_loss=config.WEIGHTED_LOSS)
            if config.SAVE_EVERY_EPOCH:
                model.save_pretrained('{}/checkpoint_{}_{}'.format(config.OUT_DIR, model_name, epoch))
                tokenizer.save_pretrained('{}/tokenizer_checkpoint_{}_{}'.format(config.OUT_DIR, model_name, epoch))

            # if config.DO_PRED_EVERY_EPOCH:
            #     logger.info(f"Epoch {epoch}, testing...")
            #     pred_dataset = pd.read_csv(config.PRED_FILE, sep="\t")
            #     if config.REMOVE_BRACKET:
            #         pred_dataset["head_event"] = pred_dataset["head_event"].str.replace('[', '').str.replace(']', '')
            #     if config.REPLACE_BRACKET:
            #         pred_dataset["head_event"] = pred_dataset["head_event"].str.replace('[', '<c>').str.replace(']', '</c>')
            #         pred_dataset["tail_event"] = pred_dataset["tail_event"].str.replace('[', '<c>').str.replace(']', '</c>')
            #     if config.USE_NL_RELATION:
            #         pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + \
            #             pd.Series(map(lambda r:CS_RELATIONS_2NL[r], pred_dataset["relation"])) + " [GEN]"
            #     else:
            #         pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + pred_dataset["relation"] + " [GEN]"
            #     pred_dataset["tail_event"] = pred_dataset["tail_event"] + ' [EOS]'
            #     # logger.info(pred_dataset["tail_event"])
            #     # logger.info(pred_dataset.head())
            #
            #     pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
            #     pred_loader = DataLoader(pred_set, **test_params, drop_last=False)
            #
            #     pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TOP_K, num_gen=config.NUM_GEN)
            #     write_path = os.path.join(config.OUT_DIR, os.path.basename(config.PRED_FILE)+f"_pred_generations_epoch{epoch}.jsonl")
            #     write_items(write_path,
            #                 [json.dumps(r) for r in pred_generations])
            #     # Check generation quality
            #     metric_names, eval_scores = get_generation_eval_scores(write_path, pred_dataset)
            #     logger.info(metric_names)
            #     logger.info(" & ".join(eval_scores))

        model.save_pretrained('{}/final_model_{}.pt'.format(config.OUT_DIR, model_name))
        tokenizer.save_pretrained('{}/final_tokenizer_{}.pt'.format(config.OUT_DIR, model_name))

    if config.DO_EVAL:
        from mosaic.infra.logging import log_eval
        import pickle
        val_loader = DataLoader(val_set, shuffle=False, batch_size=1, drop_last=False)
        eval_loss = log_eval(0, tokenizer, model, device, val_loader, model_class="gpt2", return_losses=True)
        pickle.dump(eval_loss, open(os.path.join(config.OUT_DIR, 'eval_outputs.pickle'), 'wb'))

    if config.DO_PRED:

        if config.PRED_FILE.endswith("jsonl"):
            records = read_jsonl_lines(config.PRED_FILE)
            pred_dataset = pd.DataFrame.from_records(records)
            pred_dataset = pred_dataset.rename(columns={"head": "head_event", "tails": "tail_event"})
            pred_dataset = pred_dataset.explode('tail_event')
        else:
            pred_dataset = pd.read_csv(config.PRED_FILE, sep="\t", keep_default_na=False)
        pred_dataset.fillna('', inplace=True)

        if DEBUG:
            pred_dataset = pred_dataset.head(NUM_INST)

        if config.REMOVE_BRACKET:
            pred_dataset["head_event"] = pred_dataset["head_event"].str.replace('[', '').str.replace(']', '')
        if config.REPLACE_BRACKET:
            pred_dataset["head_event"] = pred_dataset["head_event"].str.replace('[', '<c>').str.replace(']', '</c>')
            pred_dataset["tail_event"] = pred_dataset["tail_event"].str.replace('[', '<c>').str.replace(']', '</c>')
        if not config.ZERO_SHOT:
            if config.USE_NL_RELATION:
                pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + \
                                             pd.Series(map(lambda r: CS_RELATIONS_2NL[r], pred_dataset["relation"])) + \
                                                 " [GEN]"
            else:
                pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + pred_dataset["relation"] + " [GEN]"
            pred_dataset["tail_event"] = pred_dataset["tail_event"] + ' [EOS]'
        else:
            pred_dataset["head_event"] = pred_dataset["head_event"].str.replace("PersonX", "Anderson").str.replace("PersonY", "Brown").str.replace("PersonZ", "Carter")
            pred_dataset["head_event"] = pred_dataset["head_event"].str.replace('[', "“").str.replace(']', "”")
            pred_dataset["head_event"] = pred_dataset["head_event"] + ' ' + \
                                         pd.Series(map(lambda r: CS_RELATIONS_2NL[r], pred_dataset["relation"]))
        pred_dataset_full = pred_dataset.copy()
        pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)
        print("\n\n\nSamples", pred_dataset['head_event'][0], pred_dataset['tail_event'][0])
        # return

        logger.info(pred_dataset["tail_event"])
        logger.info(pred_dataset.head())

        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2",
                             is_eval=True, batch_gen=True)
        val_params['batch_size'] = config.VALID_BATCH_SIZE
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TOP_K,
                                            num_gen=config.NUM_GEN, out_len=config.OUT_LEN)
        write_items(os.path.join(config.OUT_DIR, os.path.basename(config.PRED_FILE)+"_pred_generations.jsonl"),
                    [json.dumps(r) for r in pred_generations])

        # get_eval_results(os.path.join(config.OUT_DIR, os.path.basename(config.PRED_FILE) + "_pred_generations.jsonl"),
        #                  pred_dataset_full, logger, check_prefix=config.ZERO_SHOT)
        if not all([t == ' [EOS]' for t in pred_dataset_full['tail_event']]):
            get_eval_results(os.path.join(config.OUT_DIR, os.path.basename(config.PRED_FILE)+"_pred_generations.jsonl"),
                             pred_dataset_full, logger, check_prefix=config.ZERO_SHOT)
        else:
            print("No tails generated")


        # Resave the model to keep generations and model associated
        # model.save_pretrained('/models')
        # tokenizer.save_pretrained('/models')

def get_eval_results(gen_file, gen_ref, logger=None, check_prefix=False):
    metric_names, eval_scores = get_generation_eval_scores(gen_file, gen_ref, check_prefix=check_prefix)
    if logger is not None:
        logger.info(metric_names)
        logger.info(" & ".join(eval_scores))
    else:
        print(metric_names)
        print(" & ".join(eval_scores))
    json.dump((metric_names, eval_scores), open(os.path.join(os.path.split(gen_file)[0], "pred_metrics.json"), 'w'))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--test_install",
                      action="store_true", default=False,
                      help="Test install, without running any modeling code.")

    (options, args) = parser.parse_args()
    if not options.test_install:
        main()
