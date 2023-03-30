import os
import torch
from tqdm import tqdm
import numpy as np

import argparse
from dataset import Dataset
from model import DeepPunctuation, DeepPunctuationCRF
from config import *


parser = argparse.ArgumentParser(description='Punctuation restoration test')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='roberta-large', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')
parser.add_argument('--weight-path', default='out/weights.pt', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=256, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')

parser.add_argument('--test-data', default='data/cz/dev data/cz/pdtsc_test', type=str, nargs='+', help='files with test-set datasets')
parser.add_argument('--log-file', default='out_czech/punctuation-restore_test_logs.txt', type=str, help='log file for output')

args = parser.parse_args()


# tokenizer
model_path = MODELS[args.pretrained_model][4]
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(model_path)
token_style = MODELS[args.pretrained_model][3]

test_set = []
for test_file in args.test_data:
    test_set.append(Dataset(test_file, tokenizer=tokenizer,
                            sequence_len=args.sequence_length,
                            token_style=token_style, is_train=False))

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 0
}

test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
model_save_path = args.weight_path
log_file = args.log_file

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)


def test(data_loader):
    """
    :return: label_counts[numpy array], precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    label_counts = np.zeros(1+len(punctuation_dict), dtype='i8')
    tp = np.zeros(1+len(punctuation_dict), dtype='i8')
    fp = np.zeros(1+len(punctuation_dict), dtype='i8')
    fn = np.zeros(1+len(punctuation_dict), dtype='i8')
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype='i8')
    correct = 0
    total = 0

    dataset_name = data_loader.dataset.get_dataset_name()
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc=f"Test: {dataset_name}"):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                label_counts[cor] += 1
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return label_counts, precision, recall, f1, correct/total, cm


def run():
    # load model parameters
    deep_punctuation.load_state_dict(torch.load(model_save_path, map_location=device))

    # MAIN LOOP over test-set data loaders
    for loader in test_loaders:
        label_counts, precision, recall, f1, accuracy, cm = test(loader)

        dataset_name = loader.dataset.get_dataset_name()
        log = f"\n\n[[ TEST_SET: {dataset_name} ]]\n\n"

        # label counts in the test set:
        log += "[ Labels: ]\n"
        for label, count in zip(list(punctuation_dict), list(label_counts)):
            log += f"{count:7d} {label}\n"
        log += f"{label_counts.sum():7d} TOTAL\n\n"

        # disable word-wrapping for str(numpy)
        np.set_printoptions(linewidth=np.inf)

        # per-class statistics to file
        log += "[ PER-CLASS SCORES ]\n"
        log += 'Precis. Recall F1_score Label\n'
        for i in range(0, len(punctuation_dict) + 1):
            label = list(punctuation_dict)[i] if i < len(punctuation_dict) else "TOTAL, class O,O excluded"
            log += f"{(precision[i] * 100):6.2f} {(recall[i] * 100):6.2f} {(f1[i] * 100):6.2f} {label}\n"

        log += '\nAccuracy:' + str(accuracy) + '\n\n' + \
               '[ CONFUSION_MATRIX[corr,pred]: ]\n' + str(cm) + '\n'
        log += '\n-----\n'

        print(log)
        with open(log_file, 'a') as f:
            f.write(log)


if __name__ == "__main__":
    run()
