import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier

class JointBERTMultiIntentWoISeq(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super().__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        # load pretrain bert
        self.bert = BertModel(config=config)

        # self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.multi_intent_classifier = MultiIntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        # self.intent_token_classifier = IntentTokenClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        # (len_seq, batch_size, hidden_dim), (batch_size, hidden_dim)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        
        # print('Shape of pooled_output is {}'.format(pooled_output.shape))
        
        # (batch_size, num_intents)
        intent_logits = self.multi_intent_classifier(pooled_output)
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output)
        # (batch_size, seq_len, num_intents)
        # intent_token_logits = self.intent_token_classifier(sequence_output)
        
        # print('Shape of intent logits is {}'.format(intent_logits.shape))
        # print('Shape of slot logits is {}'.format(slot_logits.shape))

        total_loss = 0

        if intent_label_ids.type() != torch.cuda.FloatTensor:
            intent_label_ids = intent_label_ids.type(torch.cuda.FloatTensor)

        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1, self.num_intent_labels))
            else:
                # intent_loss_fct = nn.CrossEntropyLoss()
                # default reduction is mean
                intent_loss_fct = nn.BCELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1, self.num_intent_labels))
            # Question: do we need to add weight here
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += self.args.slot_loss_coef * slot_loss
        
        # 3. intent token softmax
        # if intent_token_ids is not None:
        #     if self.args.use_crf:
        #         intent_token_loss = self.crf(intent_token_logits, intent_token_ids, mask=attention_mask.byte, reduction='mean')
        #         intent_token_loss = -1 * intent_token_loss
        #     else:
        #         intent_token_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
        #         if attention_mask is not None:
        #             active_intent_loss = attention_mask.view(-1) == 1
        #             active_intent_logits = intent_token_logits.view(-1, self.num_intent_labels)[active_intent_loss]
        #             active_intent_tokens = intent_token_ids.view(-1)[active_intent_loss]
        #             intent_token_loss = intent_token_loss_fct(active_intent_logits, active_intent_tokens)
        #         else:
        #             intent_token_loss = intent_token_loss_fct(intent_token_logits.view(-1, self.num_intent_labels), intent_token_ids.view(-1))
                    
        # total_loss += self.args.slot_loss_coef * intent_token_loss
        
        # outputs = ((intent_logits, slot_logits, intent_token_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
