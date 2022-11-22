import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels, compute_metrics_multi_intent

from seqeval.metrics.sequence_labeling import get_entities

FLAG = False
logger = logging.getLogger(__name__)

class Trainer_multi(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        # set of intents
        self.intent_label_lst = get_intent_labels(args)
        # set of slots
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        logger.info(vars(self.args))
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        step_per_epoch = len(train_dataloader) // 2

        # record the evaluation loss
        eval_acc = 0.0
        MAX_RECORD = self.args.patience
        num_eval = -1
        eval_result_record = (num_eval, eval_acc)
        flag = False
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=FLAG)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'intent_token_ids': batch[5],
                          'B_tag_mask': batch[6],
                          'BI_tag_mask': batch[7],
                          'tag_intent_label': batch[8]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                losses = outputs[0]
                loss = losses[0]
                intent_loss = losses[1]
                slot_loss = losses[2]
                intent_token_loss = losses[3]
                tag_intent_loss = losses[4]
                # tag_intent_loss_softmax = losses[5]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    if self.args.logging_steps > 0 and global_step % step_per_epoch == 0:
                        logger.info("***** Training Step %d *****", step)
                        logger.info("  total_loss = %f", loss)
                        logger.info("  intent_loss = %f", intent_loss)
                        logger.info("  slot_loss = %f", slot_loss)
                        logger.info("  intent_token_loss = %f", intent_token_loss)
                        logger.info("  tag_intent_loss = %f", tag_intent_loss)
                        # logger.info("  tag_intent_loss_softmax = %f", tag_intent_loss_softmax)

                        dev_result = self.evaluate("dev")
                        test_result = self.evaluate("test")
                        num_eval += 1
                        if self.args.patience != 0:
                            if dev_result['sementic_frame_acc'] + dev_result['intent_acc'] + dev_result['slot_f1']   > eval_result_record[1]:
                                self.save_model()
                                eval_result_record = (num_eval, dev_result['sementic_frame_acc'] + dev_result['intent_acc'] + dev_result['slot_f1'] )
                            else:
                                cur_num_eval = eval_result_record[0]
                                if num_eval - cur_num_eval >= MAX_RECORD:
                                    # it has been ok
                                    logger.info(' EARLY STOP Evaluate: at {}, best eval {} intent_slot_acc: {} '.format(num_eval, cur_num_eval, eval_result_record[1]))
                                    flag = True
                                    break
                        else:
                            self.save_model()

                            

                        # we check whether there is an overfitting issue for mixsnips
                        

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()
                    
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if flag:
                train_iterator.close()
                break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        intent_token_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None
        out_intent_token_ids = None
        
        tag_intent_preds = None
        out_tag_intent_ids = None
        

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=FLAG):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'intent_token_ids': batch[5],
                          'B_tag_mask': batch[6],
                          'BI_tag_mask': batch[7],
                          'tag_intent_label': batch[8]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                if self.args.intent_seq and self.args.tag_intent:
                    tmp_eval_loss, (intent_logits, slot_logits, intent_token_logits, tag_intent_logits) = outputs[:2]
                elif self.args.intent_seq:
                    tmp_eval_loss, (intent_logits, slot_logits, intent_token_logits) = outputs[:2]
                elif self.args.tag_intent:
                    tmp_eval_loss, (intent_logits, slot_logits, tag_intent_logits) = outputs[:2]
                else:
                    tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

#                 eval_loss += tmp_eval_loss.mean().item()
#             nb_eval_steps += 1

            # ============================ Intent prediction =============================
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # ============================= Slot prediction ==============================
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)
            
            # ============================== Intent Token Seq =============================
            if self.args.intent_seq:
                if intent_token_preds is None:
                    if self.args.use_crf:
                        intent_token_preds = np.array(self.model.crf.decode(intent_token_logits))
                    else:
                        intent_token_preds = intent_token_logits.detach().cpu().numpy()

                    out_intent_token_ids = inputs["intent_token_ids"].detach().cpu().numpy()
                else:
                    if self.args.use_crf:
                        intent_token_preds = np.append(intent_token_preds, np.array(self.model.crf.decode(intent_token_logits)), axis=0)
                    else:
                        intent_token_preds = np.append(intent_token_preds, intent_token_logits.detach().cpu().numpy(), axis=0)

                    out_intent_token_ids = np.append(out_intent_token_ids, inputs["intent_token_ids"].detach().cpu().numpy(), axis=0)
        
            # slot_preds: (64 * n, 50, 74)    
        
#         eval_loss = eval_loss / nb_eval_steps
#         results = {
#             "loss": eval_loss
#         }

        # Intent result
        # (batch_size, )
        # intent_preds = np.argmax(intent_preds, axis=1)
        # (batch_size, num_intents)
        # we set the threshold to 0.5
        intent_preds = torch.as_tensor(intent_preds > 0.5, dtype=torch.int32)

        # Slot result
        # (batch_size, seq_len)
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        
        B_tag_mask_pred = []
        BI_tag_mask_pred = []
        
        # generate mask
        for i in range(out_slot_labels_ids.shape[0]):
            # record the padding position
            pos_offset = [0 for _ in range(out_slot_labels_ids.shape[1])]
            pos_cnt = 0
            padding_recording = [0 for _ in range(out_slot_labels_ids.shape[1])]
            
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                    pos_offset[pos_cnt+1] = pos_offset[pos_cnt]
                    pos_cnt += 1
                else:
                    pos_offset[pos_cnt] = pos_offset[pos_cnt] + 1
                    padding_recording[j] = 1
                    

            entities = get_entities(slot_preds_list[i])
            entities = [tag for entity_idx, tag in enumerate(entities) if slot_preds_list[i][tag[1]].startswith('B')]
            
            if len(entities) > self.args.num_mask:
                entities = entities[:self.args.num_mask]
            
            entity_masks = []
            
            for entity_idx, entity in enumerate(entities):
                entity_mask = [0 for _ in range(out_slot_labels_ids.shape[1])]
                start_idx = entity[1] + pos_offset[entity[1]]
                end_idx = entity[2] + pos_offset[entity[2]] + 1
                if self.args.BI_tag:
                    entity_mask[start_idx:end_idx] = [1] * (end_idx - start_idx)
                    for padding_idx in range(start_idx, end_idx):
                        if padding_recording[padding_idx]:
                            entity_mask[padding_idx] = 0
                else:
                    entity_mask[start_idx] = 1
                    
                entity_masks.append(entity_mask)
            
            for extra_idx in range(self.args.num_mask - len(entity_masks)):
                entity_masks.append([
                    0 for _ in range(out_slot_labels_ids.shape[1])
                ])

            
            if self.args.BI_tag:
                BI_tag_mask_pred.append(entity_masks)
            else:
                B_tag_mask_pred.append(entity_masks)
                
        if self.args.BI_tag:
            BI_tag_mask_pred_tensor = torch.FloatTensor(BI_tag_mask_pred)
        else:
            B_tag_mask_pred_tensor = torch.FloatTensor(B_tag_mask_pred)
        
        BI_tag_mask_pred_input = None
        B_tag_mask_pred_input = None
        
        for eval_idx, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", disable=FLAG):
            if self.args.BI_tag:
                BI_tag_mask_pred_input = BI_tag_mask_pred_tensor[eval_idx*self.args.eval_batch_size:(eval_idx+1)*self.args.eval_batch_size]
            else:
                B_tag_mask_pred_input = B_tag_mask_pred_tensor[eval_idx*self.args.eval_batch_size:(eval_idx+1)*self.args.eval_batch_size]
            
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'intent_token_ids': batch[5],
                          'B_tag_mask': B_tag_mask_pred_input,
                          'BI_tag_mask': BI_tag_mask_pred_input,
                          'tag_intent_label': batch[8]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                if self.args.intent_seq and self.args.tag_intent:
                    tmp_eval_loss, (intent_logits, slot_logits, intent_token_logits, tag_intent_logits) = outputs[:2]
                elif self.args.intent_seq:
                    tmp_eval_loss, (intent_logits, slot_logits, intent_token_logits) = outputs[:2]
                elif self.args.tag_intent:
                    tmp_eval_loss, (intent_logits, slot_logits, tag_intent_logits) = outputs[:2]
                else:
                    tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                # if mode == 'test':
                #     print(eval_idx, ' ', tmp_eval_loss)
                eval_loss += tmp_eval_loss[0].mean().item()
            nb_eval_steps += 1
            
            if self.args.tag_intent:
                size_1 = inputs['tag_intent_label'].size(0)
                size_2 = inputs['tag_intent_label'].size(1)
                
                if tag_intent_preds is None:
                    tag_intent_preds = tag_intent_logits.view(size_1, size_2, -1).detach().cpu().numpy()
                    out_tag_intent_ids = inputs['tag_intent_label'].detach().cpu().numpy()
                else:
                    tag_intent_preds = np.append(tag_intent_preds, tag_intent_logits.view(size_1, size_2, -1).detach().cpu().numpy(), axis=0)
#                     print('out_tag_intent_ids shape: ', out_tag_intent_ids.shape)
#                     print('tag_intent_label shape: ', inputs['tag_intent_label'].shape)
                    out_tag_intent_ids = np.append(
                        out_tag_intent_ids, inputs['tag_intent_label'].detach().cpu().numpy(), axis=0)
                
        
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        
        intent_token_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        out_intent_token_list = None
        intent_token_preds_list = None
        # ============================= Intent Seq Prediction ============================
        if self.args.intent_seq:
            if not self.args.use_crf:
                intent_token_preds = np.argmax(intent_token_preds, axis=2)
            out_intent_token_list = [[] for _ in range(out_intent_token_ids.shape[0])]
            intent_token_preds_list = [[] for _ in range(out_intent_token_ids.shape[0])]

            for i in range(out_intent_token_ids.shape[0]):
                for j in range(out_intent_token_ids.shape[1]):
                    if out_intent_token_ids[i, j] != self.pad_token_label_id:
                        out_intent_token_list[i].append(intent_token_map[out_intent_token_ids[i][j]])
                        intent_token_preds_list[i].append(intent_token_map[intent_token_preds[i][j]])
        
        out_tag_intent_list = None
        tag_intent_preds_list = None
        # ============================ Tag Intent Prediction ==============================
        if self.args.tag_intent:
            tag_intent_preds = np.argmax(tag_intent_preds, axis=2)
            out_tag_intent_list = [[] for _ in range(out_tag_intent_ids.shape[0])]
            tag_intent_preds_list = [[] for _ in range(out_tag_intent_ids.shape[0])]
            
            for i in range(out_tag_intent_ids.shape[0]):
                for j in range(out_tag_intent_ids.shape[1]):
                    if out_tag_intent_ids[i, j] != self.pad_token_label_id:
                        out_tag_intent_list[i].append(intent_token_map[out_tag_intent_ids[i][j]])
                        tag_intent_preds_list[i].append(intent_token_map[tag_intent_preds[i][j]])
                        
        total_result = compute_metrics_multi_intent(intent_preds,
                                       out_intent_label_ids,
                                       slot_preds_list,
                                       out_slot_label_list,
                                       intent_token_preds_list,
                                       out_intent_token_list,
                                       tag_intent_preds_list,
                                       out_tag_intent_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s_%s = %s", mode, key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")