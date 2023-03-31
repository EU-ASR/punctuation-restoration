import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm

from dataset import Dataset
from model import DeepPunctuation, DeepPunctuationCRF
from config import *
import augmentation

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration')
    parser.add_argument('--name', default='punctuation-restore', type=str, help='name of run')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--pretrained-model', default='roberta-large', type=str, help='pretrained language model')
    parser.add_argument('--freeze-bert', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Freeze BERT layers or not')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')
    parser.add_argument('--data-path', default='data/', type=str, help='path to train/dev/test datasets')
    parser.add_argument('--language', default='english', type=str,
                        help='language, available options are english, bangla, english-bangla (for training with both)')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--augment-rate', default=0., type=float, help='token augmentation probability')
    parser.add_argument('--augment-type', default='all', type=str, help='which augmentation to use')
    parser.add_argument('--sub-style', default='unk', type=str, help='replacement strategy for substitution augment')
    parser.add_argument('--alpha-sub', default=0.4, type=float, help='augmentation rate for substitution')
    parser.add_argument('--alpha-del', default=0.4, type=float, help='augmentation rate for deletion')
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gradient-clip', default=-1, type=float, help='gradient clipping (default: -1 i.e., none)')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 10)')
    parser.add_argument('--save-path', default='out/', type=str, help='model and log save directory')

    args = parser.parse_args()
    return args

#torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
model_path = MODELS[args.pretrained_model][4]
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(model_path)
augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del
token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length
aug_type = args.augment_type

# Datasets
if args.language == 'english':
    train_set = Dataset(os.path.join(args.data_path, 'en/train2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_ref, test_set_asr]
elif args.language == 'czech':
    train_set = Dataset(os.path.join(args.data_path, 'cz/train'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    val_set = Dataset(os.path.join(args.data_path, 'cz/dev'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_pdtsc = Dataset(os.path.join(args.data_path, 'cz/pdtsc_test'), tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=False)
    test_set_LDC2000S89 = Dataset(os.path.join(args.data_path, 'cz/bnc_ldc_LDC2000S89_test'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_LDC2004S01 = Dataset(os.path.join(args.data_path, 'cz/bnc_ldc_LDC2004S01_test'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_LDC2009S02 = Dataset(os.path.join(args.data_path, 'cz/bnc_ldc_LDC2009S02_test'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_pdtsc, test_set_LDC2000S89, test_set_LDC2004S01, test_set_LDC2009S02]
else:
    raise ValueError('Incorrect language argument for Dataset')

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
os.makedirs(args.save_path, exist_ok=True)
model_save_path = os.path.join(args.save_path, 'weights.pt')
log_path = os.path.join(args.save_path, f"{args.name}_{args.pretrained_model}_{args.language}_logs.txt")


# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)


def validate(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                loss = deep_punctuation.log_likelihood(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration


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


def train():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0
    for epoch in range(args.epoch):
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        deep_punctuation.train()
        for x, y, att, y_mask in tqdm(train_loader, desc='train'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                loss = deep_punctuation.log_likelihood(x, att, y)
                # y_predict = deep_punctuation(x, att, y)
                # y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y = y.view(-1)
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

                correct += torch.sum(y_mask * (y_predict == y).long()).item()

            optimizer.zero_grad()
            train_loss += loss.item()
            train_iteration += 1
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
            optimizer.step()

            y_mask = y_mask.view(-1)

            total += torch.sum(y_mask).item()

        train_loss /= train_iteration
        log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)

        val_acc, val_loss = validate(val_loader)
        log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(deep_punctuation.state_dict(), model_save_path)

    print('Best validation Acc:', best_val_acc)
    deep_punctuation.load_state_dict(torch.load(model_save_path))

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
        with open(log_path, 'a') as f:
            f.write(log)


if __name__ == '__main__':
    train()
