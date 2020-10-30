import torch
import torch.nn as nn
import torch.nn.functional as F

from project.modeling.modeling_helpers import LMGCMOutput, LMGCMOutput
from transformers import DistilBertPreTrainedModel, DistilBertModel

class DistilBertForLMGCM(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        wsd_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ) -> LMGCMOutput:
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # (bs, seq_len, dim)
        
        mlm_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        mlm_logits = F.gelu(mlm_logits)  # (bs, seq_length, dim)
        mlm_logits = self.vocab_layer_norm(mlm_logits)  # (bs, seq_length, dim)
        mlm_logits = self.vocab_projector(mlm_logits)  # (bs, seq_length, vocab_size)

        pooled_output = hidden_states[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        total_loss = None
        if labels is not None and wsd_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))
            wsd_loss = loss_fct(logits.view(-1, self.num_labels), wsd_label.view(-1))
            total_loss = mlm_loss + wsd_loss

        if not return_dict:
            output = (mlm_logits, logits,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LMGCMOutput(
            loss=total_loss,
            mlm_logits=mlm_logits,
            prediction_logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
