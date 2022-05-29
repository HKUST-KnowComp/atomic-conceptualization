import torch
from torch import nn
from transformers import RobertaForSequenceClassification, GPT2LMHeadModel
import numpy as np

class ConceptMaxModel(RobertaForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super(ConceptMaxModel, self).__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask, sample_mask, labels=None, **kwargs):
        batch_size, n_branches, seq_length = input_ids.shape
        result = super(ConceptMaxModel, self).forward(input_ids=input_ids.view([batch_size * n_branches, seq_length]),
                                                      attention_mask=attention_mask.view([batch_size * n_branches, seq_length]))
        logits = result['logits'].view([batch_size, n_branches, self.num_labels]) # [B, nB]
        result['logits'] = logits = torch.where(sample_mask.unsqueeze(-1), logits,
                             torch.tensor(-float("inf"), dtype=torch.float32, device=logits.device))
        if labels is not None:
            logits = torch.logsumexp(logits[:, :, 1], dim=1)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            result['loss'] = loss
        return result

class GeneratorModel(GPT2LMHeadModel):
    def __init__(self, *args, **kwargs):
        super(GeneratorModel, self).__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask, token_type_ids, past_key_values=None, label=None, labels=None, **kwargs):
        result = super(GeneratorModel, self).forward(input_ids=input_ids, past_key_values=past_key_values,
                                                      attention_mask=attention_mask)

        shift_logits = result['logits'][..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_logits.size(0), shift_logits.size(1))
        loss = (loss * token_type_ids[..., 1:]).sum() / (token_type_ids.sum())
        result['loss'] = loss
        return result