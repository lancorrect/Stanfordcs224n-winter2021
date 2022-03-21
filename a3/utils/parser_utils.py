#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2020-2021: Homework 3
parser_utils.py: Utilities for training the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""

import time
import os
import logging
from collections import Counter
from utils.general_utils import get_minibatches
import sys
sys.path.append('..')
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'


class Config(object):
    language = 'english'
    with_punct = True
    unlabeled = True
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'  # 预训练好的词向量文件
    '''
    conll格式说明：CoNll是自然语言处理界的顶会，每年都是不同的共享任务，因此有很多CoNLL格式，每行代表包含一系列制表符分隔字段的单个单词
    具体的格式为：ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    ID: 单词索引，对于每个新句子，它是一个从1开始的整数;对于multiword token可能是一个范围;对于空节点是小数（小数可以低于1但必须大于0）
    FORM: 单词形式或标点符号符号
    LEMMA: 词形或词根
    UPOS: 通用词性标签 https://universaldependencies.org/u/pos/index.html 里面包含了日常所学的形容词副词一类的标签
    XPOS: 语言特定的词性标签，如果没有则用下划线代替
    FEATS: 来自universal feature inventory的形态特征列表或由已经定义的语言特定的扩展;如果不可用则使用下划线
    HEAD: 当前词的头部，它是ID或零（0）
    DEPREL: 当前词与头部的universal dependency relation（如果 head = 0，则是root），或这是一个定义的语言特定的子类型。
    DEPS: 以head-deprel对的列表形式的增强依赖关系图
    MISC: 其他注释
    
    在该格式中：
        每个单词表示在一行中
        每个句子由一个空行隔开
        每列代表一个特征(注解)
        在同一个句子中，每个词有相同数量的列数
        对于一个具体单词，每一个特征都是一个字符串
        如果在依赖句法中一个单词没有一个父节点(比如它是句法中的根结点ROOT，把它的HEAD设为0)
    '''


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)  # 统计键值对的数量
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]  # 找出出现次数最多的元素，得到的结果每个元素为一个元组(出现最多的元素，出现次数)，并按次数从高到低排列
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))  # 把ROOT的标签和单词的标签放入到列表中，且
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}  # 建立修饰关系的字典
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)  # L_PREFIX + NULL对应的值是tok2id字典的长度，但是长度里不包含这个键值对

        config = Config()
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep
        self.language = config.language

        if self.unlabeled:
            trans = ['L', 'R', 'S']  # transition based dependency parser中的是三个action，即SHIFT, LEFT-ARC, RIGHT-ARC
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}  # 建立action对id的字典
        self.id2tran = {i: t for (i, t) in enumerate(trans)}  # 建立id对action的字典

        # logging.info('Build dictionary for part-of-speech tags.')  建立词性字典
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # logging.info('Build dictionary for words.')  # 建立单词字典
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}  # tok2id的反向字典，即tok2id中的键是id2tok的值

        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0)  # 特征数量
        self.n_tokens = len(tok2id)

    def vectorize(self, examples):
        vec_examples = []
        # 循环每句话
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]  # 把单词都转换成索引值，如果不在tok2id字典中，则以UNK代替
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]  # 把词性都转换成索引值，如果不在tok2id字典中，则以P_UNK代替
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]  # 把标签都转换成索引值，如果不再tok2id中，则以-1代替
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})  # 把向量化好的结果放入到examples中
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        if stack[0] == "ROOT":
            stack[0] = 0

        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])  # 判断是不是LC，如果是LC必然是head的索引值大于箭头尾部的索引值

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []  # 初始化词性特征列表
        l_features = []  # 初始化依赖关系特征列表
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]  # 填入在stack中的单词索引值，如果没有就填入NULL的索引值
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf))  # 填入在buffer中的单词索引值，如果没有就填入NULL的索引值
        if self.use_pos:
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]  # 填入在stack中的词性索引值，如果没有就填入P_NULL的索引值
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf))  # 填入在buffer中的词性索引值，如果没有就填入P_NULL的索引值

        for i in range(2):  # 这个循环的目的是每次循环得到stack中的前两个元素，分别查看两个元素是否有LEFT-ARC, RIGHT-ARC
            if i < len(stack):
                k = stack[-i-1]
                lc = get_lc(k)  # 索引从小到大排序，其含义为离head越来越远的词
                rc = get_rc(k)  # 索引从大到小排序，其含义为离head越来越近的词
                llc = get_lc(lc[0]) if len(lc) > 0 else []  # llc是为了找到离head最远的词还有没有LEFT-ARC
                rrc = get_rc(rc[0]) if len(rc) > 0 else []  # rrc是为了找到离head最远的词还有没有RIGHT-ARC

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):
        if len(stack) < 2:
            return self.n_trans - 1

        i0 = stack[-1]  # stack的top元素
        i1 = stack[-2]  # stack的次顶点(second top)元素
        h0 = ex['head'][i0]  # top元素的头(head)
        h1 = ex['head'][i1]  # 次顶点元素的头
        l0 = ex['label'][i0]  # top元素的依赖关系
        l1 = ex['label'][i1]  # 次顶点元素的依赖关系

        if self.unlabeled:
            if (i1 > 0) and (h1 == i0):  # LEFT-ARC操作
                return 0
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return 1  # RIGHT-ARC操作，not any的含义是如果buffer中还有单词的head是stack中的top元素，则不能使用RIGHT-ARC操作
                # 否则会把top元素删掉，之后的buffer中的单词就找不到head了
            else:  # SHIFT操作
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1  # 每句话单词的个数，减1是因为最后有个UNK，这个不算在整个句子的长度中

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)  # action的序号，0为LEFT-ARC, 1为LEFT-ARC, 2为RIGHT-ARC
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))  # 提取的特征放到列表中
                if gold_t == self.n_trans - 1:  # SHIFT操作
                    stack.append(buf[0])
                    buf = buf[1:]
                elif gold_t < self.n_deprel:  # LEFT-ARC操作
                    arcs.append((stack[-1], stack[-2], gold_t))  # 更新依赖关系，(head, word, action)
                    stack = stack[:-2] + [stack[-1]]  # 在stack中删除次顶点元素
                else:
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))  # 更新依赖关系，(head, word, action)
                    stack = stack[:-1]  # 在stack中删除top元素
            else:
                succ += 1
                all_instances += instances

        return all_instances

    def legal_labels(self, stack, buf):  # 此函数的作用是为了之后预测action类型
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel  # 如果stack长度大于2，说明里面有至少三个元素，除了ROOT，还有两个单词，可能构成LEFT-ARC
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel  # 如果stack长度大于等于2，说明里面至少有两个元素，除了ROOT还有一个单词，可能构成RIGHT-ARC
        labels += [1] if len(buf) > 0 else [0]  # 如果buffer长度大于0，说明有几率构成SHIFT操作
        return labels

    def parse(self, dataset, eval_batch_size=5000):  # dataset中元素都是字典，每个字典中包括每句话的word向量，pos向量, head向量和label向量
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1  # 每句话词的个数
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i  # id()函数返回这句话的唯一标识符

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        UAS = all_tokens = 0.0
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = [-1] * len(ex['word'])
                for h, t, in dependencies[i]:
                    head[t] = h  # t代表单词索引，h是head索引，把单词索引和head索引对应起来
                for pred_h, gold_h, gold_l, pos in \
                        zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                        assert self.id2tok[pos].startswith(P_PREFIX)  # 确保开头是<p>字符
                        pos_str = self.id2tok[pos][len(P_PREFIX):]  # pos的向量，要提前去除掉<p>字符的长度
                        if (self.with_punct) or (not punct(self.language, pos_str)):  # 如果有标点符号，且该标点符号不在对应语言的标点符号列表内
                            UAS += 1 if pred_h == gold_h else 0  # 如果预测的和标准的一样
                            all_tokens += 1  # 单词数加一
                prog.update(i + 1)
        UAS /= all_tokens  # 计算UAS
        return UAS, dependencies


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]  # 提取出来特征
        mb_x = np.array(mb_x).astype('int32')  # numpy化
        mb_x = torch.from_numpy(mb_x).long()  # 转换到torch向量
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]  # 每一行都是一个行向量，元素为0或者1(例[0,1,1])

        pred = self.parser.model(mb_x)  # 模型预测过后的向量
        pred = pred.detach().numpy()  # 转变为列表numpy形式
        mb_l = 10000 * np.array(mb_l).astype('float32')
        pred = pred + mb_l
        pred = np.argmax(pred, 1)
        # 由于pred每一行是一个tensor，其中元素为小数，mb_l乘以10000后让每一行的1变成10000，这样在与pred相加时
        # 10000就会很大，其他数很小，便于之后选择每行最大数的索引，也就返回了action的索引，变相地知道了预测的操作
        # pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
        return pred


def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')  # strip移除空格，split以制表符分开，最后形成了列表，每个元素是一个字符
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:  # 这里的判断条件主要是因为当一个句子结束的时候，sp会是['']，只需要判断句子中有没有词，即word列表不为空，就可以形成一个句子
                # 把一个句子中的词，语言特定的词性标签，头和关系形成字典，并放入到所有句子存放的列表中
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []  # 初始化下一个句子的各个特征
                if (max_example is not None) and (len(examples) == max_example):  # 如果句子总数达到预定目标，则跳出循环
                    break
        if len(word) > 0:  # 如果词的列表中还有词，说明还有一个句子没有存放进去
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)  # 返回一个排列好的字典，其中顺序是值从大到小排列

    return {w[0]: index + offset for (index, w) in enumerate(ls)}  # 这里循环的是索引值和元组，最后组成单词：索引值加offset的字典


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)


def minibatches(data, batch_size):
    x = np.array([d[0] for d in data])  # 训练集中每个句子的transition based算法过程，全部是由向量表示的，向量中的元素是索引
    y = np.array([d[2] for d in data])  # 训练集中每句话的依赖关系
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1  # 制作标签的one-hot向量矩阵，y中的值代表的是第几列值为1
    return get_minibatches([x, one_hot], batch_size)


def load_and_preprocess_data(reduced=True):
    config = Config()

    print("Loading data...",)
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)  # 读取训练集
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)  # 读取验证集
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)  # 读取测试集
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...",)
    start = time.time()
    parser = Parser(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Loading pretrained embeddings...",)
    start = time.time()
    word_vectors = {}  # 初始化词向量字典
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]  # 形成词：向量的字典
    #  numpy.random.normal是从正态分布中抽取随机样本，第一个参数是分布的均值，第二个是分布的标准差，第三个是维度大小
    #  这里的维度表示parser的tok2id字典中，每个键都有一个对应的向量，该向量的长度为36(其实是特征的数量)，总共有n_tokens个键
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:  # 如果tok2id中的词有词向量
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:  # 否则查看该单词的小写有没有词向量
            embeddings_matrix[i] = word_vectors[token.lower()]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Vectorizing data...",)
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...",)
    start = time.time()
    train_examples = parser.create_instances(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    return parser, embeddings_matrix, train_examples, dev_set, test_set,


class AverageMeter(object):  # 管理变量的更新，最开始调用reset()函数来初始化，然后更新一些变量，例如val，sum(变量之和), count(变量数量), avg(平均变量)
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # load_and_preprocess_data()
    pass