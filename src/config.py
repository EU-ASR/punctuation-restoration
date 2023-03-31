from transformers import *

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# 'O' -> No punctuation
#punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}

punctuation_dict = {
    "O,O" : 0,
    "O,COMMA" : 1,
    "O,PERIOD" : 2,
    "O,QUESTION" : 3,
    "O,EXCLAMATION" : 4,

    "TitleCase,O" : 5,
    "TitleCase,COMMA" : 6,
    "TitleCase,PERIOD" : 7,
    "TitleCase,QUESTION" : 8,
    "TitleCase,EXCLAMATION" : 9,

    "ALL_CAPS,O" : 10,
    "ALL_CAPS,COMMA" : 11,
    "ALL_CAPS,PERIOD" : 12,
    "ALL_CAPS,QUESTION" : 13,
    "ALL_CAPS,EXCLAMATION" : 14,
}


# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'bert-base-uncased': (BertModel, BertTokenizer, 768, 'bert'),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024, 'bert'),
    'bert-base-multilingual-cased': (BertModel, BertTokenizer, 768, 'bert', "./pretrained_models/bert-base-multilingual-cased"), # multilingual BERT (Wikipedia), 710MB
    'bert-base-multilingual-uncased': (BertModel, BertTokenizer, 768, 'bert', "./pretrained_models/bert-base-multilingual-uncased"), # multilingual BERT (Wikipedia), 670MB
    'xlm-mlm-en-2048': (XLMModel, XLMTokenizer, 2048, 'xlm'),
    'xlm-mlm-100-1280': (XLMModel, XLMTokenizer, 1280, 'xlm', "./pretrained_models/xlm-mlm-100-1280"), # multilingual LM (Wikipedia), 1.1GB
    'roberta-base': (RobertaModel, RobertaTokenizer, 768, 'roberta', "./pretrained_models/roberta-base"), # English only...
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024, 'roberta'),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768, 'bert'),
    'distilbert-base-multilingual-cased': (DistilBertModel, DistilBertTokenizer, 768, 'bert'),
    'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer, 768, 'roberta', "./pretrained_models/xlm-roberta-base"), # multilingual BERT (2.5TB of CommonCrawl), 1.1GB
    'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer, 1024, 'roberta'),
    'albert-base-v1': (AlbertModel, AlbertTokenizer, 768, 'albert'),
    'albert-base-v2': (AlbertModel, AlbertTokenizer, 768, 'albert'),
    'albert-large-v2': (AlbertModel, AlbertTokenizer, 1024, 'albert'),
}
