import sys
import re
import torch

import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *

from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--pretrained-model', default='xlm-roberta-base', type=str, help='pretrained language model')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--in-file', default='data/test_en.txt', type=str, help='path to inference file')
    parser.add_argument('--weight-path', default='out_czech/weights.pt', type=str, help='model weight path')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--out-file', default='data/test_en_out.txt', type=str, help='output file location')

    args = parser.parse_args()
    return args


class CasePunctuator():
    def __init__(self, args):

        # Sequence length
        self.sequence_length = args.sequence_length

        # Tokenizer
        model_path = MODELS[args.pretrained_model][4]
        self.tokenizer = MODELS[args.pretrained_model][1].from_pretrained(model_path)
        self.token_style = MODELS[args.pretrained_model][3]

        # Model
        self.device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
        # - create model object
        self.deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
        self.deep_punctuation.to(self.device)
        # - load model parameters
        self.deep_punctuation.load_state_dict(torch.load(args.weight_path, map_location=self.device))
        self.deep_punctuation.eval()

        # PunctuationMap
        self.punctuation_map = { v : k for k,v in punctuation_dict.items() } # from config


    def apply(self, words: List[str]) -> List[str]:
        """ Apply the Casing & Punctuation model. """

        word_pos = 0
        sequence_len = self.sequence_length
        result = []
        decode_idx = 0

        tokenizer = self.tokenizer
        token_style = self.token_style
        device = self.device
        deep_punctuation = self.deep_punctuation
        punctuation_map = self.punctuation_map

        while word_pos < len(words):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y_mask = [0]

            while len(x) < sequence_len and word_pos < len(words):
                tokens = tokenizer.tokenize(words[word_pos])
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y_mask.append(0)
                    x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    y_mask.append(1)
                    word_pos += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y_mask.append(0)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

            x = torch.tensor(x).reshape(1,-1)
            y_mask = torch.tensor(y_mask)
            attn_mask = torch.tensor(attn_mask).reshape(1,-1)
            x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

            with torch.no_grad():
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            for i in range(y_mask.shape[0]):
                if y_mask[i] == 1:
                    # Process the predicted label:
                    label_predict = punctuation_map[y_predict[i].item()]
                    case_type, punct_type = label_predict.split(",")

                    word = words[decode_idx]
                    if case_type == "O":
                        pass
                    elif case_type == "TitleCase":
                        word = word.title()
                    elif case_type == "ALL_CAPS":
                        word = word.upper()
                    else:
                        raise Exception(f"Unknown 'case_type' {case_type}")

                    punctuation = ""
                    if punct_type == "O":
                        pass
                    elif punct_type == "COMMA":
                        punctuation = ","
                    elif punct_type == "PERIOD":
                        punctuation = "."
                    elif punct_type == "QUESTION":
                        punctuation = "?"
                    elif punct_type == "EXCLAMATION":
                        punctuation = "!"
                    else:
                        raise Exception(f"Unknown 'punct_type' {punct_type}")

                    # add word, punctuation to result
                    result.append(word)
                    if punctuation: result.append(punctuation)

                    decode_idx += 1

        return result


if __name__ == '__main__':
    args = parse_args()

    case_punctuator = CasePunctuator(args)

    with open(args.in_file, 'r', encoding="utf-8") as f_in:
        with open(args.out_file, 'w', encoding="utf-8") as f_out:
            while True:
                line = f_in.readline()
                if not line: break

                if len(line.strip().split()) > 1:
                    # parse input
                    key, text = line.strip().split(maxsplit=1)
                    text = re.sub(r"[,:.!;?]", '', text)
                    words = text.lower().split()

                    # apply the punctuator
                    words_out = case_punctuator.apply(words)

                    # output
                    out_str = f"{key} {' '.join(words_out)}"
                else:
                    out_str = line.strip()

                print(out_str)
                f_out.write(out_str + "\n")

