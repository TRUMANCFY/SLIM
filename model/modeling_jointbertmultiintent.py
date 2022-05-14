import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier, TagIntentClassifier
import logging

logger = logging.getLogger()

class JointBERTMultiIntent(BertPreTrainedModel):
    # multi_intent: 1,
    # intent_seq: 1,
    # tag_intent: 1,
    # bi_tag: 1,
    # cls_token_cat: 1,
    # intent_attn: 1,
    # num_mask: 4
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
        if args.intent_seq:
            self.intent_token_classifier = IntentTokenClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        
        if args.tag_intent:
            if args.cls_token_cat:
                self.tag_intent_classifier = TagIntentClassifier(2 * config.hidden_size, self.num_intent_labels, args.dropout_rate)
            else:
                self.tag_intent_classifier = TagIntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                intent_label_ids,
                slot_labels_ids,
                intent_token_ids,
                B_tag_mask,
                BI_tag_mask,
                tag_intent_label):
        """
            Args: B: batch_size; L: sequence length; I: the number of intents; M: number of mask; D: the output dim of Bert
            input_ids: B * L
            token_type_ids: B * L
            token_type_ids: B * L
            intent_label_ids: B * I
            slot_labels_ids: B * L
            intent_token_ids: B * L
            B_tag_mask: B * M * L
            BI_tag_mask: B * M * L
            tag_intent_label: B * M
        """
        # input_ids:  torch.Size([32, 50])
        # attention_mask:  torch.Size([32, 50])
        # token_type_ids:  torch.Size([32, 50])
        # intent_label_ids:  torch.Size([32, 10])
        # slot_labels_ids:  torch.Size([32, 50])
        # intent_token_ids:  torch.Size([32, 50])
        # B_tag_mask:  torch.Size([32, 4, 50])
        # BI_tag_mask:  torch.Size([32, 4, 50])
        # tag_intent_label:  torch.Size([32, 4])
        
        # (len_seq, batch_size, hidden_dim), (batch_size, hidden_dim)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # B * L * D
        sequence_output = outputs[0]
        # B * D
        pooled_output = outputs[1]  # [CLS]
        
        total_loss = 0
        
        # ==================================== 1. Intent Softmax ========================================
        # (batch_size, num_intents)
        intent_logits = self.multi_intent_classifier(pooled_output)
        intent_logits_cpu = intent_logits.data.cpu().numpy()
        
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1, self.num_intent_labels))
            else:
                # intent_loss_fct = nn.CrossEntropyLoss()
                # default reduction is mean
                intent_loss_fct = nn.BCELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels) + 1e-10, intent_label_ids.view(-1, self.num_intent_labels))
            # Question: do we need to add weight here
            total_loss += intent_loss
            
        if intent_label_ids.type() != torch.cuda.FloatTensor:
            intent_label_ids = intent_label_ids.type(torch.cuda.FloatTensor)
            
        # ==================================== 2. Slot Softmax ========================================
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output)
        
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    try:
                        active_loss = attention_mask.view(-1) == 1
                        attention_mask_cpu = attention_mask.data.cpu().numpy()
                        active_loss_cpu = active_loss.data.cpu().numpy()
                        active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                        active_labels = slot_labels_ids.view(-1)[active_loss]
                        slot_loss = slot_loss_fct(active_logits, active_labels)
                    except:
                        print('intent_logits: ', intent_logits_cpu)
                        print('attention_mask: ', attention_mask_cpu)
                        print('active_loss: ', active_loss_cpu)
                        logger.info('intent_logits: ', intent_logits_cpu)
                        logger.info('attention_mask: ', attention_mask_cpu)
                        logger.info('active_loss: ', active_loss_cpu)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += self.args.slot_loss_coef * slot_loss
        
        
        # ==================================== 3. Intent Token Softmax ========================================
        intent_token_loss = 0.0
        if self.args.intent_seq:
            # (batch_size, seq_len, num_intents)
            intent_token_logits = self.intent_token_classifier(sequence_output)

            if intent_token_ids is not None:
                if self.args.use_crf:
                    intent_token_loss = self.crf(intent_token_logits, intent_token_ids, mask=attention_mask.byte, reduction='mean')
                    intent_token_loss = -1 * intent_token_loss
                else:
                    intent_token_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                    if attention_mask is not None:
                        active_intent_loss = attention_mask.view(-1) == 1
                        active_intent_logits = intent_token_logits.view(-1, self.num_intent_labels)[active_intent_loss]
                        active_intent_tokens = intent_token_ids.view(-1)[active_intent_loss]
                        intent_token_loss = intent_token_loss_fct(active_intent_logits, active_intent_tokens)
                    else:
                        intent_token_loss = intent_token_loss_fct(intent_token_logits.view(-1, self.num_intent_labels), intent_token_ids.view(-1))
                
                total_loss += self.args.slot_loss_coef * intent_token_loss
        
        # convert the sequence_out to long
        if BI_tag_mask != None and  BI_tag_mask.type() != torch.cuda.FloatTensor:
            BI_tag_mask = BI_tag_mask.type(torch.cuda.FloatTensor)

        if B_tag_mask != None and B_tag_mask.type() != torch.cuda.FloatTensor:
            B_tag_mask = B_tag_mask.type(torch.cuda.FloatTensor)
        
        tag_intent_loss = 0.0
        if self.args.tag_intent:
            # B * M * D
            if self.args.BI_tag:
                tag_intent_vec = torch.einsum('bml,bld->bmd', BI_tag_mask, sequence_output)
            else:
                tag_intent_vec = torch.einsum('bml,bld->bmd', B_tag_mask, sequence_output)
            
            if self.args.cls_token_cat:
                cls_token = pooled_output.unsqueeze(1)
                # B * M * D
                cls_token = cls_token.repeat(1, self.args.num_mask, 1)
                # B * M * 2D
                tag_intent_vec = torch.cat((cls_token, tag_intent_vec), dim=2)
            
            tag_intent_vec = tag_intent_vec.view(tag_intent_vec.size(0) * tag_intent_vec.size(1), -1)
            
            # after softmax
            tag_intent_logits = self.tag_intent_classifier(tag_intent_vec)

            if self.args.intent_attn:
                # (batch_size, num_intent) => (batch_size * num_mask, num_intent) sigmoid [0, 1]
                intent_probs = intent_logits.unsqueeze(1)
                intent_probs = intent_probs.repeat(1, self.args.num_mask, 1)
                intent_probs = intent_probs.view(intent_probs.size(0) * intent_probs.size(1), -1)
                # (batch_size * num_mask, num_intent)
                tag_intent_logits = tag_intent_logits * intent_probs
                tag_intent_logits = tag_intent_logits.div(tag_intent_logits.sum(dim=1, keepdim=True))
            
            # tag_intent_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)

            # tag_intent_loss = tag_intent_loss_fct(tag_intent_logits, tag_intent_label.view(-1))

            nll_fct = nn.NLLLoss(ignore_index=self.args.ignore_index)
            
            tag_intent_loss = nll_fct(torch.log(tag_intent_logits + 1e-10), tag_intent_label.view(-1))
            
            total_loss += self.args.tag_intent_coef * tag_intent_loss
            
        if self.args.intent_seq and self.args.tag_intent:
            outputs = ((intent_logits, slot_logits, intent_token_logits, tag_intent_logits),) + outputs[2:]  # add hidden states and attention if they are here
        elif self.args.intent_seq:
            outputs = ((intent_logits, slot_logits, intent_token_logits),) + outputs[2:]
        elif self.args.tag_intent:
            outputs = ((intent_logits, slot_logits, tag_intent_logits),) + outputs[2:]
        else:
            outputs = ((intent_logits, slot_logits),) + outputs[2:]
        
        outputs = ([total_loss, intent_loss, slot_loss, intent_token_loss, tag_intent_loss],) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits