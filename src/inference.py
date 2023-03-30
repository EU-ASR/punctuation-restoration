import re
import torch

import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *

parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='xlm-roberta-base', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')
parser.add_argument('--in-file', default='data/test_en.txt', type=str, help='path to inference file')
parser.add_argument('--weight-path', default='out_czech/weights.pt', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=256, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--out-file', default='data/test_en_out.txt', type=str, help='output file location')

args = parser.parse_args()

# tokenizer
model_path = MODELS[args.pretrained_model][4]
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(model_path)
token_style = MODELS[args.pretrained_model][3]

# logs
model_save_path = args.weight_path

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)


def inference():
    deep_punctuation.load_state_dict(torch.load(model_save_path, map_location=device))
    deep_punctuation.eval()

    with open(args.in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        text = line.strip()

        text = re.sub(r"[,:.!;?]", '', text)
        #words_original_case = text.split()
        words = text.lower().split()

        word_pos = 0
        sequence_len = args.sequence_length
        result = ""
        decode_idx = 0

        #punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
        punctuation_map = { v : k for k,v in punctuation_dict.items() }

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
                if args.use_crf:
                    y = torch.zeros(x.shape[0])
                    y_predict = deep_punctuation(x, attn_mask, y)
                    y_predict = y_predict.view(-1)
                else:
                    y_predict = deep_punctuation(x, attn_mask)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
            for i in range(y_mask.shape[0]):
                if y_mask[i] == 1:
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
                        punctuation = " ,"
                    elif punct_type == "PERIOD":
                        punctuation = " ."
                    elif punct_type == "QUESTION":
                        punctuation = " ?"
                    elif punct_type == "EXCLAMATION":
                        punctuation = " !"
                    else:
                        raise Exception(f"Unknown 'punct_type' {punct_type}")

                    result += word + punctuation + ' '

                    decode_idx += 1

        print('Punctuated text')
        print(result)

        with open(args.out_file, 'a', encoding='utf-8') as f:
            f.write(result+'\n')


if __name__ == '__main__':
    inference()
