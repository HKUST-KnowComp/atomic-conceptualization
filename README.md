# Acquiring and Modelling Abstract Commonsense Knowledge via Conceptualization
This repository contains the code and data in the [paper](), with each part introduced below. `requirements.txt` for
each part of the repository is given respectively.

# Identification & Heuristic-based Concept Linking

Download the [Probase](http://concept.research.microsoft.com/Home/Download) to
`~/data/probase/data-concept-instance-relations.txt`, and run `probase_server.py`
in the backend. Other data are included in this repository, including ATOMIC, processed NOMBANK, etc.

Then run the `conceptualizer/atomic_parse.py` following the instructions inside. Particularly, after running Part 1 and Part 2, 
`glossbert.csv` is built. Use it as the input to the [GlossBERT model](https://github.com/HSLCY/GlossBERT) following
their _test_ instructions using `run_classifier_WSD_sent`, with `--eval_data_dir` replaced to the `glossbert.csv` 
produced. After that, copy the `results.txt` to `parse` as `glossbert_results.txt`, and run Part 3.

As a result, following files are produced as outputs:
<ul>
<li>heads.5.txt: Head events after preprocessing. Empty lines are for the excluded ATOMIC events.</li>
<li>docs.jsonl: Parsed head events. Each line is a json string, a list of dicts. Each dict correspond to a token.
Properties are derived from the spacy parsing results, plus markers of `predicate` (whether it is a predicative
candidate), `nominal` (whether it is a nominal candidate), `prep` (whether it is prepositional), `modifier` (whether
it is some other modifier).
</li>
<li>components.jsonl: Conceptualization candidates in each event. Each line contains a list, and each element of the
list is the index of the head token for the candidate.</li>
<li>linked_concepts.2.jsonl: Events with linked WordNet/Probase concepts. </li>
<li>exclusion.json: Events excluded. </li>
</ul>

`linked_concepts.2.jsonl` consists of multiple lines in the original ATOMIC order, each corresponds to an event, 
represented as List[Candidate]. Each candidate is a List[Variation], and each _variation_ corresponds to linked concepts
based on a specific head word. For example, for the candidate "the city of new york", two variations will be detected,
based on the constituents "city" and "new york" respectively. This implement the various cases when concept linking
of a specific constituents depend on its subtrees. Each variation is a List[Tuple[Modifiers, Senses, Text, Left, Right]],
with each element a concept linked. Modifiers is List[Int] for the indices of children that considered to be the modifer
of the concept. Senses is List[[Sense, Score]], with Sense the WordNet synset name 
(or a found idiom, starting with "Idiom:"), and Score from GlossBERT.
Text is the textual concept, and Left/Right is the span of the concept in the original text, 
represented by token indices.

# Dataset & Pretrained Models

Our annotated data are given in `conceptualizer/data`, including `stage2_combined.json` for event conceptualizations
and `stage3_combined.neg2.json` for abstract triples.
Intermediate ATOMIC parsing results are also given in `conceptualizer/parse`, as mentioned above.
Experiment data built upon ATOMIC and our data are given in `discriminator/data` or `discriminator/gen`.
Pretrained models and Abstract ATOMIC are available [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/mhear_connect_ust_hk/Eo1sMdC6DalKtOllhrDXohABihrryFFd1MqJ9a_KPoqj6w?e=KWwaGy).

# Model Training

## Concept Generator
The code is available in `generator/`, based on the official COMET implementation. 
To train the model, run under the directory

`
IN_LEN=40 OUT_LEN=46 OUT_DIR=OUTPUT_PATH SAVE_EVERY_EPOCH=False DO_PRED_EVERY_EPOCH=False EVAL_EVERY=500 USE_NL_RELATION=True TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False TRAIN_EPOCHS=30 TOKENIZER=gpt2 GPT2_MODEL=gpt2 TRAIN_DATA_PATH=../discriminator/gen/stage2_all.nl/trn.tsv DEV_DATA_PATH=../discriminator/gen/stage2_all.nl/dev.tsv TEST_DATA_PATH=../discriminator/gen/stage2_all.nl/tst.tsv PRED_FILE=../discriminator/gen/stage2_all.nl/dev.tsv USE_NL_RELATION=True REPLACE_BRACKET=True python models/comet_atomic2020_gpt2/comet_concept_gpt2.py
`

, with OUTPUT_PATH substituted to your own one. Or you may use our pretrained model. It will only save the model when
the minimum eval loss is updated, so the best model will be the best_STEP_model with the largest STEP.
Then you may evaluate it with

`
IN_LEN=40 OUT_LEN=52 SAVE_EVERY_EPOCH=False DO_PRED_EVERY_EPOCH=False EVAL_EVERY=500 USE_NL_RELATION=True TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=False DO_PRED=True OUT_DIR=RESULT_PATH TOKENIZER=gpt2 GPT2_MODEL=MODEL_PATH PRED_FILE=../discriminator/gen/stage2_all.nl/dev.tsv TRAIN_DATA_PATH=../discriminator/gen/stage2_all.nl/trn.tsv DEV_DATA_PATH=../discriminator/gen/stage2_all.nl/dev.tsv TEST_DATA_PATH=../discriminator/gen/stage2_all.nl/tst.tsv REPLACE_BRACKET=True python models/comet_atomic2020_gpt2/comet_concept_gpt2.py
`
, with MODEL_PATH (like `OUTPUT_PATH/best-model-3000`) and RESULT_PATH substituted to your own one.

## Conceptualization Verifier
The code is available in `discriminator/`. To train the model, run under the directory

`
python main.py --model_name_or_path roberta-base --output_dir OUTPUT_PATH --do_train --do_eval --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/stage2_all --overwrite_output_dir True --save_strategy steps --save_steps 500 --evaluation_strategy steps --logging_steps 500 --per_device_eval_batch_size 16 --eval_accumulation_steps 4`

, with OUTPUT_PATH substituted to your own one. Or you may use our pretrained model. Then you may evaluate it with

`
python main.py --model_name_or_path MODEL_PATH --output_dir RESULT_PATH --do_eval --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/stage2_all --overwrite_output_dir True --save_strategy steps --save_steps 500 --evaluation_strategy steps --logging_steps 500 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --do_final_evaluations
`
, with MODEL_PATH (like `OUTPUT_PATH/checkpoint-3000`) and RESULT_PATH substituted to your own one.
Before that you may run `prepare_data.py` to split the data.

## Inference Verifier

Similar to the conceptualization verifier, but different in the dataset used. To train, use

`
python main.py --model_name_or_path roberta-base --output_dir OUTPUT_PATH --do_train --do_eval --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/stage3_all.neg2 --overwrite_output_dir True --save_strategy steps --save_steps 500 --evaluation_strategy steps --logging_steps 500 --per_device_eval_batch_size 16 --eval_accumulation_steps 4
`

To evaluate, use 

`
python main.py --model_name_or_path MODEL_PATH --output_dir RESULT_PATH --do_eval --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/stage3_all.neg2 --overwrite_output_dir True --save_strategy steps --save_steps 500 --evaluation_strategy steps --logging_steps 500 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --do_final_evaluations
`

# Abstract ATOMIC
We've built the Abstract ATOMIC used in our experiments, shared on OneDrive as mentioned above.
As for event conceptualizations, the file (like `selected_heads\trn.json`) contains a single dict. The key is the 
original event with a concept identified. Each value is a dict as well, with its keys the verified conceptualizations,
and each value is a list of all sources of this conceptualization, taken in the form of List[[Doc_i, Constituent_i]]. 
Doc_i is the ATOMIC index, and Constituent_i is the index of candidate constituent in `components.jsonl`.
While for abstract triples the files (like `selected_triples_90`) are more self-explanatory. Additional information
is given in the `info` property, with `all_info` a similar list of List[[Doc_i, Constituent_i]], and `d_i`/`c_i` one
of the sources.

To re-construct Abstract ATOMIC on your own, use  `conceptualizer/extend_kg.py` as follow:
1. Run the Part 1 to prepare data.
2. Copy `conceptualizer/data/exp/*` to `discriminator/data`, and run `discriminator/prepare_data.py`.
3. For SPLIT in {trn, dev, tst}, run: 
`
IN_LEN=40 OUT_LEN=60 SAVE_EVERY_EPOCH=False DO_PRED_EVERY_EPOCH=False EVAL_EVERY=500 USE_NL_RELATION=True TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=False DO_EVAL=False DO_PRED=True OUT_DIR=population/extend_cand_head.nl/SPLIT TOKENIZER=gpt2 GPT2_MODEL=MODEL_PATH PRED_FILE=../discriminator/gen/extend_cand_head.nl/SPLIT.tsv TRAIN_DATA_PATH=../discriminator/gen/extend_cand_head.nl/trn.tsv DEV_DATA_PATH=../discriminator/gen/extend_cand_head.nl/tst.tsv TEST_DATA_PATH=../discriminator/gen/extend_cand_head.nl/tst.tsv REPLACE_BRACKET=True NUM_GEN=10 python models/comet_atomic2020_gpt2/comet_concept_gpt2.py
`, with MODEL_PATH the pretrained concept generator path.
4. Now the concept generator produced the abstract events to `generator/population/extend_cand_head.nl`. Run the Part 2.
5. Run conceptualization verification on all produced data. For candidate abstract events from neural concept generator,
run `
python main.py --model_name_or_path MODEL_PATH --output_dir population/head_gen/SPLIT --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/generated_heads/SPLIT.json --overwrite_output_dir True --logging_steps 500 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --do_final_evaluations
`. For candidate abstract events from heuristic concept linking & conceptualization, run `
python main.py --model_name_or_path MODEL_PATH --output_dir population/head/SPLIT --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/extended_heads/SPLIT.json --overwrite_output_dir True --logging_steps 500 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --do_final_evaluations
`.
6. Run the Part 3.
7. Run inference verification:
`
python main.py --model_name_or_path MODEL_PATH --output_dir population/triple/SPLIT --do_predict --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --dataset split_atomic/extended_triples/SPLIT.json --overwrite_output_dir True --logging_steps 500 --per_device_eval_batch_size 128 --eval_accumulation_steps 4 --do_final_evaluations
`
8. Collect the results by running the Part 4.