import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert, JointBERTMultiIntent, JointBERTMultiIntentWoISeq

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer),
    'multibert': (BertConfig, JointBERTMultiIntent, BertTokenizer),
    'multibertWoISeq': (BertConfig, JointBERTMultiIntentWoISeq, BertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1',
    'multibert': 'bert-base-uncased',
    'multibertWoISeq': 'bert-base-uncased',
}

def get_intent_labels(args):
    # data/atis/intent_label.txt
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    # data/atis/slot_label.txt
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='./logs/{}_seed{}_TI{}_attn{}_cls{}_mask{}_ticoef{}.log'.format(args.task, args.seed, args.tag_intent, args.intent_attn, args.cls_token_cat, args.num_mask, args.tag_intent_coef))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results

def compute_metrics_multi_intent(intent_preds,
                                 intent_labels,
                                 slot_preds,
                                 slot_labels,
                                 intent_token_preds=None,
                                 intent_tokens=None,
                                 tag_intent_preds=None,
                                 tag_intent_ids=None):
    intent_seq_existence = (intent_token_preds != None and intent_tokens != None)
    
    tag_intent_existence = (tag_intent_preds != None and tag_intent_ids != None)
    
    if intent_seq_existence:
        assert len(intent_preds) == len(intent_labels) == len(slot_preds) \
        == len(slot_labels) == len(intent_token_preds) == len(intent_tokens), 'the length should be the same'
        
    if tag_intent_existence:
        assert len(intent_preds) == len(intent_labels) == len(slot_preds) \
        == len(slot_labels) == len(tag_intent_preds) == len(tag_intent_ids), 'the length should be the same'
    
    results = {}
    # intent_result = get_intent_acc(intent_preds, intent_labels)
    intent_result = get_multi_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    if intent_seq_existence:
        intent_token_result = get_intent_token_metrics(intent_token_preds, intent_tokens)
        
    if tag_intent_existence:
        tag_intent_result = get_tag_intent_metrics(tag_intent_preds, tag_intent_ids)
    # sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc_multi_intent(intent_preds,
                                                          intent_labels,
                                                          slot_preds,
                                                          slot_labels,
                                                          intent_token_preds,
                                                          intent_tokens,
                                                          tag_intent_preds,
                                                          tag_intent_ids)
    
    results.update(intent_result)
    results.update(slot_result)
    if intent_seq_existence:
        results.update(intent_token_result)
    if tag_intent_existence:
        results.update(tag_intent_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds, average='macro')
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def get_multi_intent_acc(intent_preds, intent_labels):
    intent_preds = intent_preds.tolist()
    intent_labels = intent_labels.tolist()
    records = []
    for ip, il in zip(intent_preds, intent_labels):
        one_sent = True
        for ipn, iln in zip(ip, il):
            if ipn != iln:
                one_sent = False
                break
        records.append(int(one_sent))
    
    return {
        "intent_acc": np.mean(records)
    }

def get_tag_intent_metrics(tag_intent_preds, tag_intent_ids):
    tag_intent_preds_flatten = [tag_intent for tag_intents in tag_intent_preds for tag_intent in tag_intents]
    tag_intent_ids_flatten = [tag_intent for tag_intents in tag_intent_ids for tag_intent in tag_intents]
    
    total_cnt = 0
    correct_cnt = 0
    
    for pred, gt in zip(tag_intent_preds_flatten, tag_intent_ids_flatten):       
        if pred == gt:
            correct_cnt += 1
            total_cnt += 1
        else:
            total_cnt += 1
    
    return {
        'tag_intent_acc': correct_cnt / total_cnt
    }


def getEntities(elems):
    """
    Compress the entities
    For example: ['O', 'Good', 'Good', 'O', 'O']
                 will be compressed as [('O', 0, 1), ('Good', 1, 3), ('O', 3, 5)], then we can remove O as we wish
    
    Args:
        elems: list of hashable elements (it will automatically flat the listoflist concatenated by O)
    
    Return:
        entities: list of tuples, (element, its starting index, its ending index)
    """
    
    if isinstance(elems[0], list):
        elems = [item for sub in elems for item in sub + ['O']]
    
    current_char = elems[0]

    elem_len = len(elems)
    
    current_idx = 0
    ptr = current_idx + 1
    
    entities = []
    
    while ptr < elem_len:
        if ptr == elem_len - 1:
            if elems[ptr] == current_char:
                entities.append((current_char, current_idx, ptr + 1))
            else:
                entities.append((current_char, current_idx, ptr))
                entities.append((elems[ptr], ptr, ptr+1))
            break
        
        if elems[ptr] != current_char:
            entities.append((current_char, current_idx, ptr))
            current_idx = ptr
            current_char = elems[ptr]
        
        ptr += 1
    
    return entities


def get_intent_token_metrics(intent_token_preds, intent_tokens):
    """
    Args:
        intent_token_preds: list(list(str)) [['O', 'PlayMusic', 'PlayMusic', 'O', 'O'], ...]=>(PlayMusic, 1, 3)
        intent_tokens: list(list(str))
        
    """
    pred_tokens = set([item for item in getEntities(intent_token_preds) if item[0] != 'O'])
    true_tokens = set([item for item in getEntities(intent_tokens) if item[0] != 'O'])
    
    # pred_tokens = [(PlayMusic, 1, 3), (ComposeMusic, 4, 5)]
    # true_tokens = [(PlayMusic, 1, 3), (ComposeMusic, 4, 6)]
    
    nb_correct = len(pred_tokens & true_tokens)
    nb_pred = len(pred_tokens)
    nb_true = len(true_tokens)
    
    pre = nb_correct / nb_pred if nb_pred > 0 else 0
    recall = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * pre * recall / (pre + recall) if pre + recall > 0 else 0
    
    return {
        'intent_token_precision': pre,
        'intent_token_recall': recall,
        'intent_token_f1': score,
    }

def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }


def get_sentence_frame_acc_multi_intent(intent_preds,
                                        intent_labels,
                                        slot_preds,
                                        slot_labels,
                                        intent_token_preds=None,
                                        intent_tokens=None,
                                        tag_intent_preds=None,
                                        tag_intent_labels=None):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    intent_token_existence = (intent_token_preds and intent_tokens)
    tag_intent_existence = (tag_intent_preds and tag_intent_labels)
    
    # Get the intent comparison result
    intent_result = []
    intent_preds = intent_preds.tolist()
    intent_labels = intent_labels.tolist()

    for ip, il in zip(intent_preds, intent_labels):
        one_sent = True
        for ipn, iln in zip(ip, il):
            if ipn != iln:
                one_sent = False
                break
        intent_result.append(int(one_sent))
    intent_result = np.array(intent_result)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    
    slot_result = np.array(slot_result)
    
    # Get the intent token comparison result
    if intent_token_existence:
        intent_token_result = []
        for preds, labels in zip(intent_token_preds, intent_tokens):
            assert len(preds) == len(labels)
            one_sent_result = True
            for p, l in zip(preds, labels):
                if p != l:
                    one_sent_result = False
                    break
            intent_token_result.append(one_sent_result)
        
        intent_token_result = np.array(intent_token_result)
    
    if tag_intent_existence:
        tag_intent_result = []
        for preds, labels in zip(tag_intent_preds, tag_intent_labels):
            one_sent_result = True
            for p, l in zip(preds, labels):
                if p != l:
                    one_sent_result = False
                    break
            tag_intent_result.append(one_sent_result)
        
        tag_intent_result = np.array(tag_intent_result)
    
    if tag_intent_existence and intent_token_existence:
        sementic_acc = np.multiply(np.multiply(np.multiply(intent_result, slot_result), intent_token_result), tag_intent_result).mean()
    elif tag_intent_existence:
        sementic_acc = np.multiply(np.multiply(intent_result, slot_result), tag_intent_result).mean()
    elif intent_token_existence:
        sementic_acc = np.multiply(np.multiply(intent_result, slot_result), intent_token_result).mean()
    else:
        sementic_acc = np.multiply(intent_result, slot_result).mean()
    
    intent_slot_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc,
        "intent_slot_acc": intent_slot_acc,
    }