import torch
import torch.nn as nn

from transformers.modeling_utils import SequenceSummary
from project.modeling.modeling_helpers import LMGCMOutput

from transformers.modeling_xlnet import XLNetModel, XLNetPreTrainedModel


class XLNetForLMGCM(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.attn_type = config.attn_type
        self.same_length = config.same_length

        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_loss

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # Add dummy token at the end (no attention on this one)

        effective_batch_size = input_ids.shape[0]
        dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)

        # At every pass, the attention values for the new token and the two last generated tokens
        # are computed, the rest is reloaded from the `past` cache. A purely auto-regressive model would have
        # offset = 1; offset = 2 seems to have slightly better computation.
        offset = 2

        if past:
            input_ids = torch.cat([input_ids[:, -offset:], dummy_token], dim=1)
        else:
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # Build permutation mask so that previous tokens don't see last token
        sequence_length = input_ids.shape[1]
        perm_mask = torch.zeros(
            (effective_batch_size, sequence_length, sequence_length), dtype=torch.float, device=input_ids.device
        )
        perm_mask[:, :, -1] = 1.0

        # We'll only predict the last token
        target_mapping = torch.zeros(
            (effective_batch_size, 1, sequence_length), dtype=torch.float, device=input_ids.device
        )
        target_mapping[:, 0, -1] = 1.0

        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_cache": kwargs["use_cache"],
        }

        # if past is defined in model kwargs then use it for faster decoding
        if past:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past)

        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        wsd_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_predict)`, `optional`):
                Labels for masked language modeling.
                `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.
                The labels should correspond to the masked input words that should be predicted and depends on `target_mapping`. Note in order to perform standard auto-regressive language modeling a `<mask>` token has to be added to the `input_ids` (see `prepare_inputs_for_generation` fn and examples below)
                Indices are selected in ``[-100, 0, ..., config.vocab_size]``
                All labels set to ``-100`` are ignored, the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

        Return:

        Examples::

            >>> from transformers import XLNetTokenizer, XLNetLMHeadModel
            >>> import torch

            >>> tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            >>> model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased', return_dict=True)

            >>> # We show how to setup inputs to predict a next token using a bi-directional context.
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            >>> target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            >>> next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

            >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
            >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
            >>> assert labels.shape[0] == 1, 'only one word will be predicted'
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
            >>> target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
            >>> loss = outputs.loss
            >>> next_token_logits = outputs.logits  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = self.training or (use_cache if use_cache is not None else self.config.use_cache)

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mlm_logits = self.lm_loss(transformer_outputs[0])
        seq_output = self.sequence_summary(transformer_outputs[0])
        prediction_logits = self.logits_proj(seq_output)

        total_loss = None
        if labels is not None and wsd_label is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))
            wsd_loss = loss_fct(prediction_logits.view(-1, self.num_labels), wsd_label.view(-1))
            total_loss = mlm_loss + wsd_loss

        if not return_dict:
            output = (mlm_logits, prediction_logits) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LMGCMOutput(
            loss=total_loss,
            mlm_logits=mlm_logits,
            prediction_logits=prediction_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )