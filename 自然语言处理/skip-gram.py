import io
import os
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F


# 下载语料用于训练word2vec
def download():
    # 已经存在文件则不下载
    if os.path.exists('./text8.txt'):
        return
    print('正在下载语料...')
    # 可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    # 使用python的requests包下载数据集到本地
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    # 把下载后的文件存储在当前目录的text8.txt文件内
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()


def load_text8():
    with open('./text8.txt', 'r') as f:
        corpus = f.read().strip('\n')
    f.close()

    return corpus


# 自然语言处理中，需要先对语料进行切词。对于英文来说，可以比较简单地直接使用空格进行切词
def data_preprocess(corpus):
    # 由于英文单词出现在句首的单词经常大写，所以我们把所有英文字母都转成小写，以便对语料进行归一化处理
    corpus = corpus.strip().lower()
    corpus = corpus.split(' ')
    return corpus


# 对语料进行统计，为每个词构造id。一般来说，可以根据每个词再语料中出现的频次构造id， 频次越高，id越小，便于对词典进行管理
def build_dict(corpus):
    # 首先统计不同词的出现频率，用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 1
        else:
            word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    # 一般来说，出现频率高的词往往是 I the you 这种代词，而出现频率低的往往是一些名词
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    # 构造三个不同的词典，分别存储

    # 每个词到id的映射关系
    word2id_dict = dict()
    # 每个id出现的频率
    word2id_freq = dict()
    # 每个id到词的映射关系
    id2word_dict = dict()

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[curr_id] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


# 把语料转为id序列
def convert_corpus_to_id(corpus, word2id_dict):

    corpus = [word2id_dict[word] for word in corpus]
    return corpus


# 使用二次采样算法（subsampling）处理语料，降低高频词在语料中出现的频次, 强化训练效果
def subsampling(corpus, word2id_freq):
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus


# 构造数据，准备模型训练
# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
# negative sample num代表了对于每个正样本，我们需要随机采样多少个负样本用于训练
# 一般来说，negative_sample_num的值越大，训练结果越稳定，但是训练速度越慢
def build_data(corpus, word2id_dict, word2id_freq, max_window_size=3, negative_sample_num=4):
    print("正在build_data...")
    # 使用一个list存储处理好的数据
    dataset = []

    # 从左至右，开始枚举每个中心点的位置
    for center_word_idx in range(len(corpus)):
        # 以max_window_size为上限， 随机采样一个window_size， 这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        # 当前中心词就是center_word_idx 所指向的词
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看成正样本
        positive_word_range = (max(0, center_word_idx-window_size), min(len(corpus), center_word_idx+window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1])]


        # 对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            # 首先把（中心词，正样本， label=1）的三元组数据放入dataset中
            # 这里label=1表示这个样本是正样本
            dataset.append((center_word, positive_word, 1))

            # 开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate not in positive_word_candidates:
                    # 把（中心词，负样本， label=0）的三元组放入dataset中
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((center_word, negative_word_candidate, 0))
                    i+=1

    return dataset


# 构造mini-batch，准备对模型进行训练
# 我们对不同类型的数据放到不同的tensor里，便于神经网络进行处理
# 并通过numpy的array函数，构造出不同的tensor， 并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):

    print("正在build batch...")
    # center_word_batch缓存batch_size个中心词
    center_word_batch = []
    # target_word_batch缓存batch_size个目标词
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []




    for epoch in range(epoch_num):
        # 每次开启一个新的epoch之前，都对数据进行一次随机打乱，提升训练效果
        random.shuffle(dataset)

        for center_word, target_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些样本送到不同的tensor里
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            # 当样本积攒到1个batch_size后，就把数据返回回来
            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype('int64'), \
                    np.array(target_word_batch).astype('int64'),\
                    np.array(label_batch).astype('float32')
                center_word_batch = []
                target_word_batch = []
                label_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype('int64'), \
              np.array(target_word_batch).astype('int64'), \
              np.array(label_batch).astype('float32')


# 定义skip-gram网络结构
class SkipGram(paddle.nn.Layer):
    # vocab_size定义了这个模型的词表大小
    # embedding_size定义了词向量的维度是多少
    # init_scale定义词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练

    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # 使用Embedding函数构造一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale))
        )

        # 使用Embedding函数构造另外一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding_out = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))

    # 定义网络的前向计算逻辑
    # center_words是一个tensor(mini_batch), 表示中心词
    # target_words是一个tensor(mini_batch), 表示目标词
    # label是一个tensor(mini_batch), 表示这个词是正样本还是负样本
    # 用于在训练过程中计算这个tensor对应的词的同义词， 用于观察模型训练效果
    def forward(self, center_words, target_words, label):
        # 首先，通过self.embedding参数，将mini-batch中的词转换为词向量
        # 这里center_words和eval_words_emb查询的是一个相同的参数,而target_words_emb查询的是另一个参数
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # 我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率
        word_sim = paddle.multiply(center_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])
        pred = F.sigmoid(word_sim)

        # 通过估计的输出概率定义损失函数，注意我们使用的是binary_cross_entropy_with_logits函数
        # 将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
        loss = F.binary_cross_entropy_with_logits(word_sim, label)
        loss = paddle.mean(loss)

        return pred, loss


# 定义一个使用word_embedding查询同义词的函数
# 这个函数query_token是要查询的词， k表示要返回多少个最相似的词， embed是我们学习到的word_embedding参数
# 我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
# x代表要查询词的embedding， Embedding参数矩阵W代表所有词的embedding
# 两者计算cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表
def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W*W, axis=1)*np.sum(x*x)+(1e-9))
    flat = cos.flatten()
    # 返回top_k的数
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))



def train():
    batch_size = 512
    epoch_num = 3
    embedding_size = 200
    step = 0
    learning_rate = 0.001

    print('开始训练...')

    skip_gram_model = SkipGram(vocab_size, embedding_size)


    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=skip_gram_model.parameters())

    batch_data = build_batch(dataset=dataset, batch_size=batch_size, epoch_num=epoch_num)

    # 使用build_batch函数，以mini-batch为单位，遍历训练数据，训练网络
    for center_words, target_words, label in batch_data:
        center_words_var = paddle.to_tensor(center_words)
        target_words_var = paddle.to_tensor(target_words)

        label_var = paddle.to_tensor(label)

        pred, loss = skip_gram_model(center_words_var, target_words_var, label_var)

        loss.backward()
        opt.step()
        opt.clear_grad()

        step += 1

        if step % 2000 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        # 每隔10000步，打印一次模型对以下查询词的相似词，这里我们使用词和词之间的向量点积作为衡量相似度的方法，只打印了5个最相似的词
        if step % 10000 == 0:
            get_similar_tokens('movie', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('chip', 5, skip_gram_model.embedding.weight)
        if step % 50000 == 0:
            paddle.save(skip_gram_model.state_dict(), './skip_gram_ckpt{}.pdparams'.format(step/50000))

    # 保存模型参数
    paddle.save(skip_gram_model.state_dict(), './skip_gram.pdparams')
    print('保存模型成功！')


def predict(embedding_size=200):
    skip_gram_model = SkipGram(vocab_size, embedding_size)
    skip_gram_model.set_state_dict(paddle.load('./skip_gram_ckpt3.0.pdparams'))
    skip_gram_model.eval()

    ipt_word = ''
    while ipt_word != ':q':
        ipt_word = input(">>请输入单词：")
        try:
            get_similar_tokens(ipt_word, 5, skip_gram_model.embedding.weight)
        except KeyError as e:
            print("词典中不存在这个词！")
    print("退出...")



if __name__ == '__main__':
    download()
    paddle.set_device('gpu')
    # print(load_text8()[:500])
    corpus = load_text8()
    corpus = data_preprocess(corpus)
    word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
    vocab_size = len(word2id_freq)
    # print("there are totoally %d different words in the corpus" % vocab_size)
    # for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    #    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))
    # 训练前要把语料库转为id
    corpus = convert_corpus_to_id(corpus, word2id_dict)
    corpus = subsampling(corpus, word2id_freq)
    print("%d tokens in the corpus" % len(corpus))
    # print(corpus[:50])

    mode = 'predict'
    if mode == 'train':
        # 缩减语料库，减少内存占用
        corpus_light = corpus[:int(len(corpus) * 0.5)]

        dataset = build_data(corpus_light, word2id_dict, word2id_freq)
        print("dataset_len:{}".format(len(dataset)))
        train()
    elif mode == 'predict':
        predict()

