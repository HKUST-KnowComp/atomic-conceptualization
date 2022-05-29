import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import pickle
from datasets import load_dataset, load_metric
from sklearn import metrics as skmetrics

# Before run: install ruamel_yaml==0.11.14, transformers==4.11.0, datasets; uninstall apex

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version
# from transformers.utils.versions import require_version
import torch


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.11.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=48,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    dataset: str = field(
        default=None
    )
    train_dataset: str = field(
        default=None
    )
    eval_dataset: str = field(
        default=None
    )
    test_dataset: str = field(
        default=None
    )

    do_final_evaluations: Optional[bool] = field(
        default=False, metadata={"help": "Whether do evaluations after training."}
    )

    remove_bracket: bool = field(
        default=False,
        metadata={"help": "Whether to remove brackets (concept indicator) in text"},
    )

    replace_bracket: bool = field(
        default=False,
        metadata={"help": "Use special token for bracket"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_conceptmax: bool = field(
        default=False, metadata={"help": "Use ConceptMax."}
    )
    abs_samples: int = field(
        default=4, metadata={"help": "Number of abstractions used in ConceptMax for training."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if training_args.do_train:
        ds = data_args.train_dataset if data_args.train_dataset else data_args.dataset
        train_dataset = load_dataset('json', split='train', data_files=os.path.join(ds, 'trn.json'))

    if training_args.do_eval:
        ds = data_args.eval_dataset if data_args.eval_dataset else data_args.dataset
        eval_dataset = load_dataset('json', split='train', data_files=os.path.join(ds, 'dev.json'))

    if training_args.do_predict:
        ds = data_args.test_dataset if data_args.test_dataset else data_args.dataset
        if os.path.exists(os.path.join(ds, 'tst.json')):
            ds = os.path.join(ds, 'tst.json')
        predict_dataset = load_dataset('json', split='train', data_files=ds)

    # Labels
    num_labels = 2

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    if model_args.use_conceptmax:
        from models import ConceptMaxModel
        model = ConceptMaxModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.replace_bracket:
        tokenizer.add_tokens(["<c>", "</c>"])
        model.resize_token_embeddings(len(tokenizer))

    from collections import defaultdict
    lengths = defaultdict(list)
    sample_data = []
    def preprocess_function(examples, is_eval=False):
        # Tokenize the texts
        if data_args.remove_bracket:
            for i in range(len(examples['sent_a'])):
                examples['sent_a'][i] = examples['sent_a'][i].replace('[', '').replace(']', '')
        if data_args.replace_bracket:
            for i in range(len(examples['sent_a'])):
                examples['sent_a'][i] = examples['sent_a'][i].replace('[', '<c>').replace(']', '</c>')
                examples['sent_b'][i] = examples['sent_b'][i].replace('[', '<c>').replace(']', '</c>')
        if model_args.use_conceptmax:
            import random
            r = random.Random(0)
            sents_a = []
            sents_b = []
            n_branches = 0
            n = len(examples['sent_a'])
            for i, (s_a, s_b) in enumerate(zip(examples["sent_a"], examples["sent_b"])):
                abstractions = examples["abs"][i]
                if not is_eval:
                    r.shuffle(abstractions)
                    abstractions = abstractions[:model_args.abs_samples]
                examples["sent_b"][i] = [s_b] + abstractions
                n_branches = max(n_branches, len(abstractions) + 1)
            sample_mask = np.zeros([n, n_branches], dtype=bool)
            for i in range(n):
                for j in range(n_branches):
                    if j < len(examples["sent_b"][i]):
                        sents_a.append(examples["sent_a"][i])
                        sents_b.append(examples["sent_b"][i][j])
                        sample_mask[i, j] = True
                    else:
                        sents_a.append("")
                        sents_b.append("")

            result = tokenizer(
                sents_a,
                sents_b,
                padding=padding,
                max_length=data_args.max_seq_length,
                truncation=True,
            )
            result.data['input_ids'] = np.asarray(result.data['input_ids']).reshape([n, n_branches, data_args.max_seq_length])
            result.data['attention_mask'] = np.asarray(result.data['attention_mask']).reshape([n, n_branches, data_args.max_seq_length])
            result.data['sample_mask'] = sample_mask
            return result
        for sa, sb in zip(examples['sent_a'], examples['sent_b']):
            tokens = tokenizer(sa, sb).data['input_ids']
            tokens = [tokenizer.decode(t) for t in tokens]
            if len(sample_data) < 20:
                sample_data.append(' '.join(tokens))
            lengths[len(tokens)].append(tokens)
        return tokenizer(
            examples["sent_a"],
            examples["sent_b"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
                fn_kwargs={"is_eval": True}
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    logger.info("Sample data:\n" + "\n".join(sample_data))
    # Get the metric function
    metric_fns = [('accuracy', skmetrics.accuracy_score), ('f1', skmetrics.f1_score),
                  ('precision', skmetrics.precision_score), ('recall', skmetrics.recall_score)]

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    refresh_threshold = True
    threshold = 0.5
    probs_last = None
    labels_last = None
    def compute_metrics(p: EvalPrediction):
        nonlocal threshold, refresh_threshold, probs_last, labels_last
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if model_args.use_conceptmax:
            preds = np.max(preds[:, :, 1], axis=1)
            probs = torch.sigmoid(torch.tensor(preds))
        else:
            probs = torch.softmax(torch.tensor(preds), dim=-1)[::, 1]
        probs = probs.detach().cpu().numpy()
        probs_last = probs
        labels_last = p.label_ids
        probs = 1 - probs
        labels = 1 - p.label_ids
        if refresh_threshold:
            # refresh_threshold = False
            max_acc = 0
            for thres in [0] + probs.tolist():
                preds_ = probs > thres
                # acc = skmetrics.f1_score(labels, preds_)
                acc = (preds_ == labels).mean()
                if acc > max_acc:
                    max_acc = acc
                    threshold = thres
            logger.info("Updated threshold to %.10f" % threshold)
        else:
            logger.info("Using threshold %.10f" % threshold)

        preds = probs > threshold
        results = {}
        for name, fn in metric_fns:
            results[name] = fn(labels, preds)
        results['conf_t_p'] = (np.logical_and(labels == 1, preds == 1)).mean()
        results['conf_t_n'] = (np.logical_and(labels == 1, preds == 0)).mean()
        results['conf_f_p'] = (np.logical_and(labels == 0, preds == 1)).mean()
        results['conf_f_n'] = (np.logical_and(labels == 0, preds == 0)).mean()
        try:
            results['auc'] = skmetrics.roc_auc_score(labels, probs)
        except:
            pass
        return results

    data_collator = DataCollatorWithPadding(tokenizer, 'max_length' if data_args.pad_to_max_length else 'longest', pad_to_multiple_of=8)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if not data_args.do_final_evaluations:
        logger.info("do_final_evaluations not set, exit...")
        return
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        refresh_threshold = True
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        pickle.dump((probs_last, labels_last), open(os.path.join(trainer.args.output_dir, 'eval_outputs.pickle'), 'wb'))

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_predict:
        logger.info("*** Test ***")
        refresh_threshold = False
        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        pickle.dump((probs_last, labels_last), open(os.path.join(trainer.args.output_dir, 'test_outputs.pickle'), 'wb'))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # # Prediction
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
    #
    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    #
    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)
    #
    #     predictions = np.argmax(predictions, axis=1)
    #     output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
    #     if trainer.is_world_process_zero():
    #         with open(output_predict_file, "w") as writer:
    #             writer.write("index\tprediction\n")
    #             for index, item in enumerate(predictions):
    #                 item = label_list[item]
    #                 writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()