# Importing stock libraries
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb
import logging
from tqdm import tqdm

# logger = logging.getLogger("modeling")
from mosaic.infra.logging import log_eval

best_val_loss = float("+inf")

def per_sample_loss(lm_logits, labels):
    from torch.nn import CrossEntropyLoss
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_logits.size(0), shift_logits.size(1))
    return loss.mean(-1)

def train(epoch, tokenizer, model, device, loader, optimizer, val_loader=None, model_class="t5",
          save_dir="/models", save_every=-1, eval_every=-1, train_show_step=100, logger=None,
          weighted_loss=-1):
    global best_val_loss
    model.train()
    batch_count = len(loader)
    last_best_save = None
    abs_loss = None
    for iteration, data in tqdm(enumerate(loader, 0), total=batch_count, desc="Epoch %d" % epoch):
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        y = data['target_ids'].to(device, dtype=torch.long)
        y_mask = data['target_mask'].to(device, dtype=torch.long)
        # varibles for t5
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        if model_class == "t5":
            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids,
                            lm_labels=lm_labels)
        elif model_class == "bart":
            outputs = model(input_ids=ids,
                            attention_mask=mask,
                            decoder_input_ids=y,
                            decoder_attention_mask=y_mask,
                            labels=y)
        else: # gpt2
            labels = data['labels'].to(device, dtype=torch.long)
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            if weighted_loss is not None and weighted_loss > 0:
                is_abs = data['is_abs'].to(device)
                losses = per_sample_loss(outputs[1], labels)
                abs_loss = (losses * is_abs).sum() / is_abs.sum()
                nonabs_loss = (losses * (1 - is_abs)).sum() / (1 - is_abs).sum()
                weights = weighted_loss * is_abs + 1 - is_abs
                outputs = ((weights.to(device) * losses).mean(), ) + outputs[1:]
            # for train and dev phase, 'ids' is of text + ctext = head + relation + tail
            # so this is the loss for the whole sentence
            # but, we only want loss on tail
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration = batch_count * epoch + iteration

        if iteration > 0:
            if iteration % train_show_step == 0:
                logger.info(
                    f'\nEpoch: {epoch}, Iteration: {iteration}, Training Loss:  {loss.item()}')
                if abs_loss is not None:
                    logger.info(f', Abs loss: {abs_loss.item()}, Non-abs loss: {nonabs_loss.item()}, '
                                f'abs ratio: {((is_abs).sum() / len(is_abs)).item()}, weight: {weighted_loss}')

            if eval_every > 0 and iteration % eval_every == 0 and val_loader != None:
                eval_loss = log_eval(epoch, tokenizer, model, device, val_loader,
                                     model_class=model_class, sample_limit=3505)

                logger.info(f'\nEpoch: {epoch}, Step: {iteration}, Eval Loss: {eval_loss}')
                if eval_loss < best_val_loss:
                    logger.info(f'Saving model with updated best val loss from {best_val_loss}')
                    best_val_loss = eval_loss
                    model.save_pretrained(save_dir + "/best_{}_model".format(iteration))
                    tokenizer.save_pretrained(save_dir + "/best_{}_tokenizer".format(iteration))
                    if last_best_save is not None:
                        shutil.rmtree(last_best_save + '_model')
                        shutil.rmtree(last_best_save + '_tokenizer')
                    last_best_save = save_dir + "/best_{}".format(iteration)
                model.train()
            if save_every > 0 and iteration % save_every == 0:
                logger.info(f'\nEpoch: {epoch}, Loss:  {loss.item()}, saving to {save_dir + "/iter_{}_model".format(iteration)}')
                model.save_pretrained(save_dir + "/iter_{}_model".format(iteration))
                tokenizer.save_pretrained(save_dir + "/iter_{}_tokenizer".format(iteration))


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=True,
                max_length=int(os.environ['OUT_LEN']),
                num_beams=5,
                top_k=50,
                top_p=0.95
            )

            preds = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                g in generated_ids]
            target = [
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                t in y]
            source = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                s in ids]

            if _ % 100 == 0:
                logger.info(f'Completed {_}')

            sources.extend(source)
            predictions.extend(preds)
            actuals.extend(target)
    return sources, predictions, actuals

def cleanup_frontpad(pad_token_id, ids):
    ids = list(ids)
    for i in range(len(ids)):
        p = 0
        while ids[i][p] == pad_token_id and p < len(ids[i]):
            p += 1
        ids[i] = ids[i][p:]
    return ids

def beam_generations(tokenizer, model, device, loader, top_k=40, num_gen=10, out_len=34, logger=None):
    # This method assumes batch size of 1
    model.eval()
    predictions = []
    actuals = []
    sources = []
    records = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0), total=len(loader)):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            batch_size = len(ids)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                temperature=1.0,
                do_sample=False,
                max_length=out_len,
                top_p=0.9,
                top_k=top_k,
                repetition_penalty=1.0,
                # num_return_sequences=10 if top_k > 1 else 1,
                num_return_sequences=num_gen,
                num_beams=10,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_ids = list(generated_ids.reshape([batch_size, num_gen, -1]).detach().cpu().numpy())
            try:
                target = [tokenizer.decode(t, clean_up_tokenization_spaces=True) for t in data['target_ids']]
            except:
                target = [None] * batch_size
            source = data['source_ids'].detach().cpu().numpy()
            source = cleanup_frontpad(tokenizer.pad_token_id, source)
            source = [tokenizer.decode(s, clean_up_tokenization_spaces=True) for s in source]

            for gen, tgt, src in zip(generated_ids, target, source):
                gen = cleanup_frontpad(tokenizer.pad_token_id, gen)
                preds = [tokenizer.decode(g, clean_up_tokenization_spaces=True) for g in gen]
                records.append({
                    'source': src,
                    'target': tgt,
                    'generations': preds
                })

            if _ % 100 == 0 and logger is not None:
                logger.info(f'Completed {_}')

    return records
#
# def batch_greedy_generate(tokenizer, model, dataloader, device, max_num_tokens_to_produce=20):
#
#     model.eval()
#     with torch.no_grad():
#         for _, data in enumerate(dataloader, 0):
#             input_ids = data['source_ids'].to(device, dtype = torch.long)
#             attn_mask = data['source_mask'].to(device, dtype = torch.long)
#
#             pad_token_id = tokenizer.pad_token_id
#             eos_token_id = tokenizer.eos_token_id
#             eos_not_in_sents = torch.ones(input_ids.shape[0]).long()
#
#             last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
#
#             start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size).unsqueeze(1)
#             past = None
#             seq_len = input_ids.size(1)
#             position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
#             for i, position_ids_slice in enumerate(position_ids):
#                 position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]
#
#             for step in range(max_num_tokens_to_produce):
#                 outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids)
#
#                 if step == 0:
#                     next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
#                 else:
#                     next_token_logits = outputs[0][:, -1, :]
#
#                 next_tokens = torch.argmax(next_token_logits, dim=-1)
#
#                 # this updates which sentences have not seen an <EOS> token so far
#                 # if one <EOS> token was seen the sentence is finished
#                 eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())
#
#                 # either append a padding token here if <EOS> has been seen or append next token
#                 tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)
#
#                 # Update input_ids, attn_mask and position_ids
#                 input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
#                 attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long()], dim=1)
#                 position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
#
