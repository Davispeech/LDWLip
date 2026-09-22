import argparse
import os
import glob
from ptlip.utils.read_config import read_all_config


def parse_args():
    parser = argparse.ArgumentParser(description='test Model')
    # config
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args


args = parse_args()
cfg = read_all_config(args.config, 'common')

# 从根据数据集的标签自动获取一个字典
vocab_map_path = cfg.model.char_list
label_dirs = os.path.join(cfg.dataset.dataset_dir, 'label')
dic = dict()
samples_list = glob.glob(os.path.join(label_dirs, '*'))  # /data/Dataset/lipreading/dataset/SELF_DATASET/label/0001
samples_list = sorted(samples_list)
for sample_list in samples_list:
    labels_list = glob.glob(os.path.join(sample_list, '*.txt'))  # /data/Dataset/lipreading/dataset/SELF_DATASET/label/0001/0001.txt
    labels_list = sorted(labels_list)
    for i in range(len(labels_list)):
        label_path = labels_list[i]
        f = open(label_path, 'r', encoding='utf-8')
        line = f.readline().strip()
        for chara in line:
            if chara not in dic:
                dic[chara] = 1
            else:
                dic[chara] += 1
        f.close()

words = [a for a in dic]
words = sorted(words)

special_words = ['<PAD>', '<EOS>', '<BOS>', '<unknown>']
int_to_vocab = {idx: word for idx, word in enumerate(special_words + words)}
with open(vocab_map_path, 'w', encoding='utf-8') as f:
    for k, v in int_to_vocab.items():
        f.write(str(k) + ': ' + str(v) + '\n')
