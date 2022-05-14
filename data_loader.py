import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels

from seqeval.metrics.sequence_labeling import get_entities

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputExampleMultiIntent(InputExample):
    def __init__(self,
                 guid,
                 words,
                 intent_label=None,
                 slot_labels=None,
                 intent_tokens=None,
                 B_tag_mask=None,
                 BI_tag_mask=None,
                 tag_intent_label=None):
        super().__init__(guid, words, intent_label, slot_labels)
        self.intent_tokens=intent_tokens
        self.B_tag_mask=B_tag_mask
        self.BI_tag_mask=BI_tag_mask
        self.tag_intent_label=tag_intent_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeaturesMultiIntent(InputFeatures):
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 intent_label_id,
                 slot_labels_ids,
                 intent_tokens_ids,
                 B_tag_mask,
                 BI_tag_mask,
                 tag_intent_label):
        super().__init__(input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids)
        self.intent_tokens_ids = intent_tokens_ids
        self.B_tag_mask = B_tag_mask
        self.BI_tag_mask = BI_tag_mask
        self.tag_intent_label = tag_intent_label


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)

class JointProcessorMultiIntent(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        # data/atis/intent_label.txt
        self.intent_labels = get_intent_labels(args)
        # data/atis/slot_label.txt
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'
        self.intent_tokens_file = 'seq_intent.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """
        Read text file as lines
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, intent_tokens, set_type):
        """
        Creates examples for the training and dev sets.
        
        Args:
            texts: list of utterance (str; concat of tokens)
            intents: list of intents
            slots: bio tokens (str)
            intent_tokens (str)
            
        Return:
            examples: a list of examples (example will contains an id, list_of_words, intent, list_of_bio)
        """
        examples = []
        for i, (text, intent, slot, intent_token) in enumerate(zip(texts, intents, slots, intent_tokens)):
            # train-i
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent to list list(index)
            intent_label_token = [self.intent_labels.index(int_tok) if int_tok in self.intent_labels else self.intent_labels.index('UNK') for int_tok in intent.split('#')]
            # we have to convert it to an indicating list with the length of intents
            intent_label = [0 for _ in self.intent_labels]
            for i in intent_label_token:
                intent_label[i] = 1
            # 3. slot to list of index list(list(index))
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
            
            # 4. intent_token_str to index list(list(index))
            intent_token_list = []
            for s in intent_token.split():
                intent_token_list.append(self.intent_labels.index(s) if s in self.intent_labels else self.intent_labels.index('UNK'))
                
            # get entities in one utterance
            MAX_SLOT = self.args.num_mask
            # seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            # [('PER', 0, 1), ('LOC', 3, 3)]
            
            entities = get_entities(slot.split())
            
            if len(entities) > MAX_SLOT:
                entities = entities[:MAX_SLOT]
           
            # 5. B tag mask: B * M * L
            # BI tag mask: B * M * L
            # tag intent label: B * M
            B_tag_mask = [[0 for _ in slot.split()] for utter in range(MAX_SLOT)]
            BI_tag_mask = [[0 for _ in slot.split()] for utter in range(MAX_SLOT)]
            tag_intent_label = [self.intent_labels.index("PAD") for _ in range(MAX_SLOT)]
            
            try:
                for idx, tag in enumerate(entities):
                    B_tag_mask[idx][tag[1]] = 1
                    BI_tag_mask[idx][tag[1]:tag[2]+1] = [1./(tag[2]-tag[1]+1)] * (tag[2]-tag[1]+1)
                    # BI_tag_mask[idx][tag[1]:tag[2]+1] = [1] * (tag[2]-tag[1]+1)
                    tag_intent_label[idx] = intent_token_list[tag[1]]
                    assert tag_intent_label[idx] != self.intent_labels.index("UNK") and \
                    tag_intent_label[idx] != self.intent_labels.index("O"), 'The intent tagged is UNK or O!'
            except:
                logger.info('Error')
                logger.info(text)
                logger.info(slot.split())
                logger.info(entities)
                logger.info(intent_token_list)

                
            assert len(words) == len(slot_labels) == len(intent_token_list)
            examples.append(InputExampleMultiIntent(guid=guid,
                                                    words=words,
                                                    intent_label=intent_label,
                                                    slot_labels=slot_labels,
                                                    intent_tokens=intent_token_list,
                                                    B_tag_mask=B_tag_mask,
                                                    BI_tag_mask=BI_tag_mask,
                                                    tag_intent_label=tag_intent_label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        
        Returns:
            list of example
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     intent_tokens=self._read_file(os.path.join(data_path, self.intent_tokens_file)),
                                     set_type=mode)


processors = {
    "atis": JointProcessorMultiIntent,
    "snips": JointProcessorMultiIntent,
    'mixsnips': JointProcessorMultiIntent,
    'mixatis': JointProcessorMultiIntent,
    'mixsnips_large': JointProcessorMultiIntent,
    'atis_seq': JointProcessorMultiIntent,
    'snips_seq': JointProcessorMultiIntent,
    'mixsnips_single': JointProcessor,
    'dstc4': JointProcessorMultiIntent,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def convert_examples_to_features_multi(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    Convert the example (text, id, ...) into feature (different types of tensor)
    Args:
        examples: list of example
        max_seq_len: upper bound of token_length
        args: two functions:
        pad_token_label_id:
        cls_token_segment_id:
        sequence_a_segment_id:
    Returns:
        features:
        
    """
    
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        intent_tokens_ids = []
        B_tag_mask_list = []
        BI_tag_mask_list = []
        # for the B_tag_mask and BI_tag_mask in example, we need to zip them to make tokenization and padding simpler
        B_tag_mask = list(zip(*example.B_tag_mask))
        BI_tag_mask = list(zip(*example.BI_tag_mask))
        
        # the number of mask
        try:
            num_mask = len(B_tag_mask[0])
        except:
            print(example.words)
            print(example.slot_labels)
            print(example.intent_tokens)

        for word, slot_label, intent_token, B_pos_mask, BI_pos_mask in zip(
                example.words,
                example.slot_labels,
                example.intent_tokens,
                B_tag_mask,
                BI_tag_mask,
            ):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            #### IMPORTANT: This is the case mentioned in the paper ####
            # redbreast => red, ##bre, ##ast => we will only put the first one as the token
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            intent_tokens_ids.extend([int(intent_token)] + [pad_token_label_id] * (len(word_tokens) - 1))
            B_tag_mask_list.extend([B_pos_mask] + [tuple([0 for _ in range(num_mask)])] * (len(word_tokens) - 1))
            BI_tag_mask_list.extend([BI_pos_mask] + [tuple([0 for _ in range(num_mask)])] * (len(word_tokens) - 1))
            
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # limit the maximum length, please note no padding yet
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]
            intent_tokens_ids = intent_tokens_ids[:(max_seq_len - special_tokens_count)]
            B_tag_mask_list = B_tag_mask_list[:(max_seq_len - special_tokens_count)]
            BI_tag_mask_list = BI_tag_mask_list[:(max_seq_len - special_tokens_count)]
            
        # Add [SEP] token
        # sequence_a_segment_id: 0
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        intent_tokens_ids += [pad_token_label_id]
        B_tag_mask_list += [tuple([0 for _ in range(num_mask)])]
        BI_tag_mask_list += [tuple([0 for _ in range(num_mask)])]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        # cls_token_segment_id: 0
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        intent_tokens_ids = [pad_token_label_id] + intent_tokens_ids
        B_tag_mask_list = [tuple([0 for _ in range(num_mask)])] + B_tag_mask_list
        BI_tag_mask_list = [tuple([0 for _ in range(num_mask)])] + BI_tag_mask_list
        token_type_ids = [cls_token_segment_id] + token_type_ids
        
        # convert tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)
        intent_tokens_ids = intent_tokens_ids + ([pad_token_label_id] * padding_length)
        B_tag_mask_list = B_tag_mask_list + ([tuple([0 for _ in range(num_mask)])] * padding_length)
        BI_tag_mask_list = BI_tag_mask_list + ([tuple([0 for _ in range(num_mask)])] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)
        assert len(intent_tokens_ids) == max_seq_len, "Error with intent tokens length {} vs {}".format(len(intent_tokens_ids), max_seq_len)
        assert len(B_tag_mask_list) == max_seq_len, "Error with B_tag_mask_list length {} vs {}".format(len(B_tag_mask_list), max_seq_len)
        assert len(BI_tag_mask_list) == max_seq_len, "Error with BI_tag_mask_list length {} vs {}".format(len(BI_tag_mask_list), max_seq_len)
        
        # for multi-intent process, it is a list of int
        intent_label_id = [int(i) for i in example.intent_label]
        tag_intent_label = [int(i) for i in example.tag_intent_label]
        
        # convert the B_tag_mask and BI_tag_mask back
        B_tag_mask_list = list(zip(*B_tag_mask_list))
        BI_tag_mask_list = list(zip(*BI_tag_mask_list))
        B_tag_mask_list = [list(i) for i in B_tag_mask_list]
        BI_tag_mask_list = [list(i) for i in BI_tag_mask_list]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %s)" % (" ".join([str(i) for i in example.intent_label]),\
                                                        " ".join([str(i) for i in intent_label_id])))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))
            logger.info("intent_tokens: %s" % " ".join([str(x) for x in intent_tokens_ids]))

        features.append(
            InputFeaturesMultiIntent(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids,
                          intent_tokens_ids=intent_tokens_ids,
                          B_tag_mask=B_tag_mask_list,
                          BI_tag_mask=BI_tag_mask_list,
                          tag_intent_label=tag_intent_label,
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    """
    Generate the different types of dataloader
    
    Args:
        args:
        tokenizer:
        mode: train/dev/test
    
    Return:
        dataset: dataloader
    """
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            args.num_mask,
        )
    )
    
    # try to load from the cached data first
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        # Defaultly, pad id will be set to 0
        pad_token_label_id = args.ignore_index
        if args.multi_intent:
            features = convert_examples_to_features_multi(examples, args.max_seq_len, tokenizer,
                                                        pad_token_label_id=pad_token_label_id)
        else:
            features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                   pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    
    if args.multi_intent:
        # as the intent has been transfer to multiple intent
        # we have to transfer the intent to a list of binary
        all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.float)
        all_intent_tokens_ids = torch.tensor([f.intent_tokens_ids for f in features], dtype=torch.long)
        all_B_tag_mask = torch.tensor([f.B_tag_mask for f in features], dtype=torch.long)
        all_BI_tag_mask = torch.tensor([f.BI_tag_mask for f in features], dtype=torch.float)
        all_tag_intent_label = torch.tensor([f.tag_intent_label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids,
                                all_attention_mask,
                                all_token_type_ids,
                                all_intent_label_ids,
                                all_slot_labels_ids,
                                all_intent_tokens_ids,
                                all_B_tag_mask,
                                all_BI_tag_mask,
                                all_tag_intent_label)
    else:
        all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids,
                                all_attention_mask,
                                all_token_type_ids,
                                all_intent_label_ids,
                                all_slot_labels_ids)
    return dataset