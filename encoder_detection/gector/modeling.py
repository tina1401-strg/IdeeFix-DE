from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from .configuration import GECToRConfig
from typing import List, Union, Optional

@dataclass
class GECToROutput:
    loss: torch.Tensor = None
    loss_d: torch.Tensor = None
    loss_labels: torch.Tensor = None
    logits_d: torch.Tensor = None
    logits_labels: torch.Tensor = None
    accuracy: torch.Tensor = None
    accuracy_d: torch.Tensor = None

# need
@dataclass
class GECToRPredictionOutput:
    probability_labels: torch.Tensor = None
    probability_d: torch.Tensor = None
    pred_labels: List[List[str]] = None
    pred_label_ids: torch.Tensor = None
    max_error_probability: torch.Tensor = None

class GECToR(PreTrainedModel):
    config_class = GECToRConfig
    def __init__(
        self,
        config: GECToRConfig
    ):
        super().__init__(config)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id
        )
        self.bert = AutoModel.from_pretrained(
            self.config.model_id,
            add_pooling_layer=False
        )
        # +1 is for $START token
        # Note: In transformers > 4.49.0, `self.bert` is loaded
        #   as a meta tensors, so the model does not hold actual values. 
        #   Therefore, the default setting `mean_resizing=True` causes an error 
        #   because it cannot compute statistics for the existing embeddings.
        #   For now, we avoid it by setting it to False.
        self.bert.resize_token_embeddings(
            self.bert.config.vocab_size + 1,
            mean_resizing=False
        )
        self.label_proj_layer = nn.Linear(
            self.bert.config.hidden_size,
            self.config.num_labels - 1
        )  # -1 is for <PAD>
        self.d_proj_layer = nn.Linear(
            self.bert.config.hidden_size,
            self.config.d_num_labels - 1
        )
        self.dropout = nn.Dropout(self.config.p_dropout)
        self.loss_fn = CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        self.post_init()
        self.tune_bert(False)

    def tune_bert(self, tune=True):
        # If tune=False, only classifier layers will be tuned.
        for param in self.bert.parameters():
            param.requires_grad = tune
        return
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        d_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        word_masks: Optional[torch.Tensor] = None,
    ) -> GECToROutput:
        bert_logits = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        logits_d = self.d_proj_layer(bert_logits)
        logits_labels = self.label_proj_layer(self.dropout(bert_logits))
        loss_d, loss_labels, loss = None, None, None
        accuracy, accuracy_d = None, None
        if d_labels is not None and labels is not None:
            pad_id = self.config.label2id[self.config.label_pad_token]
            # -100 is the default ignore_idx of CrossEntropyLoss
            labels[labels == pad_id] = -100
            d_labels[labels == -100] = -100
            loss_d = self.loss_fn(
                logits_d.view(-1, self.config.d_num_labels - 1),  # -1 for <PAD>
                d_labels.view(-1)
            )
            loss_labels = self.loss_fn(
                logits_labels.view(-1, self.config.num_labels - 1),
                labels.view(-1)
            )
            loss = loss_d + loss_labels

            pred_labels = torch.argmax(logits_labels, dim=-1)
            accuracy = torch.sum(
                (labels == pred_labels) * word_masks
            ) / torch.sum(word_masks)
            pred_d = torch.argmax(logits_d, dim=-1)
            accuracy_d = torch.sum(
                (d_labels == pred_d) * word_masks
            ) / torch.sum(word_masks)

        return GECToROutput(
            loss=loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            accuracy=accuracy,
            accuracy_d=accuracy_d
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_masks: torch.Tensor,
        keep_confidence: float=0,
        min_error_prob: float=0
    ): 
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask
            )
            # (batch, seq_len, num_labels)
            probability_labels = F.softmax(outputs.logits_labels, dim=-1)
            # (batch, seq_len, num_labels)
            probability_d = F.softmax(outputs.logits_d, dim=-1)

            # Apply the bias of $KEEP.
            keep_index = self.config.label2id[self.config.keep_label]
            probability_labels[:, :, keep_index] += keep_confidence
            # Get predcition tags. (batch, seq_len, num_labels) -> (batch, seq_len)
            pred_label_ids = torch.argmax(probability_labels, dim=-1)

            # Apply the minimum error probability threshold
            incor_idx = self.config.d_label2id[self.config.incorrect_label]
            # (batch_size, seq_len, num_labels) -> (batch_size, seq_len)
            probability_d_incor = probability_d[:, :, incor_idx]
            # (batch_size, seq_len) -> (batch_size)
            max_error_probability = torch.max(probability_d_incor * word_masks, dim=-1)[0]
            # Sentence-level threshold.
            #   Set the $KEEP tag to all tokens in the sentences 
            #   that have lower maximum error prob. than thredhold.
            pred_label_ids[
                max_error_probability < min_error_prob, :
            ] = keep_index
            # Token-level threshold.
            #   Set infinity to tokens that have lower probability than thredhold.
            #   Note that the probaility is not detection's, but tag's one.
            pred_label_ids[
                torch.max(probability_labels, dim=-1)[0] < min_error_prob
            ] = keep_index

            def convert_ids_to_labels(ids, id2label):
                labels = []
                for id in ids.tolist():
                    labels.append(id2label[id])
                return labels

            pred_labels = []
            for ids in pred_label_ids:
                labels = convert_ids_to_labels(
                    ids,
                    self.config.id2label
                )
                pred_labels.append(labels)
        return GECToRPredictionOutput(
            probability_labels=probability_labels,
            probability_d=probability_d,
            pred_labels=pred_labels,
            pred_label_ids=pred_label_ids,
            max_error_probability=max_error_probability
        )
 