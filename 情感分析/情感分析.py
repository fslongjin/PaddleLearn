import re, os
import random
import tarfile
import requests
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F
from paddle.nn import LSTM, Dropout, Linear


def download():
    if os.path.exists('./aclImdb_v1.tar.gz'):
        return
    print('正在下载数据集...')
    corpus_url = 'https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz'
    req = requests.get(corpus_url)
    corpus = req.content
    with open("./aclImdb_v1.tar.gz", "wb") as f:
        f.write(corpus)
    f.close()
    print('下载完成！')


def load_imdb(is_training):
    print('正在加载数据集...   is_training:{}'.format(is_training))
    data_set = []
    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的rarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感

    for label in ['pos', 'neg']:
        path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
            else "aclImdb/test/" + label + "/.*\.txt$"
        path_pattern = re.compile(path_pattern)
        base_dir = "aclImdb_v1/aclImdb/train/" + label
        files = os.listdir(base_dir)

        for file in files:


            if file[-4:] != '.txt':
                continue
            file = os.path.join(base_dir, file)
            if not os.path.isfile(file):
                continue
            # print(file)
            with open(file, 'rb') as f:
                sentence = f.read().decode()
                # print(sentence)
                sentence_label = 1 if label == 'pos' else 0
                data_set.append((sentence, sentence_label))


    print('加载数据集完成！')
    return data_set


# 一般来说，在自然语言处理中，需要先对语料进行切词，这里我们可以使用空格把每个句子切成若干词的序列
def data_preprocess(corpus):
    dataset = []
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.lower()
        dataset.append((sentence, sentence_label))

    return dataset


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    print('正在构造词典...')
    word_freq_dict = dict()

    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 1
            else:
                word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    # 我们使用了一个特殊的单词"[oov]"（out-of-vocabulary），用于表示词表中没有覆盖到的词。之所以使用"[oov]"这个符号，是为了处理某一些词，在测试数据中有，但训练数据没有的现象。
    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表

    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 0
    word2id_freq[1] = 1e10
    idx = len(word2id_dict)

    for word, freq in word_freq_dict:
        word2id_dict[word] = idx
        word2id_freq[idx] = freq
        idx += 1

    return word2id_dict, word2id_freq


# 把语料中的句子转为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    dataset = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict else word2id_dict['[oov]'] for word in sentence]
        dataset.append((sentence, sentence_label))

    return dataset


# 编写一个迭代器， 每次调用这个迭代器都会返回一个新的epoch， 用于训练或预测
def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True, drop_last=True):
    # 模型会接受两个输入：
    # 1、一个形状为[batchsize, max_seq_len]的张量， sentence_batch, 代表了一个mini-batch的句子
    # 2、一个形状为[batch_size,1]的张量，sentence_label_batch， 每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）

    sentence_batch = []
    sentence_label_batch = []
    for _ in range(epoch_num):

        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]

            # 填充
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype('int64'), np.array(sentence_label_batch).astype('int64')
                sentence_batch = []
                sentence_label_batch = []

        if not drop_last and len(sentence_batch) > 0:
            yield np.array(sentence_batch).astype('int64'), np.array(sentence_label_batch).astype('int64')

# 定义一个分类网络
class SentimentClassifier(paddle.nn.Layer):
    def __init__(self, hidden_size, vocab_size, embedding_size, class_num=2, num_steps=128, num_layers=1, init_scale=0.1, dropout_rate=None):
        # 参数含义如下：
        # 1.hidden_size，表示embedding-size，hidden和cell向量的维度
        # 2.vocab_size，模型可以考虑的词表大小
        # 3.embedding_size，表示词向量的维度
        # 4.class_num，情感类型个数，可以是2分类，也可以是多分类
        # 5.num_steps，表示这个情感分析模型最大可以考虑的句子长度
        # 6.num_layers，表示网络的层数
        # 7.dropout_rate，表示使用dropout过程中失活的神经元比例
        # 8.init_scale，表示网络内部的参数的初始化范围,长短时记忆网络内部用了很多Tanh，Sigmoid等激活函数，\
        # 这些函数对数值精度非常敏感，因此我们一般只使用比较小的初始化范围，以保证效果
        super(SentimentClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale

        # 声明一个lstm模型，用来把每个句子抽象成向量
        self.lstm_rnn = LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        # 声明一个embedding层，把句子中的每个词转换为向量
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, sparse=False,
                                   weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))


        # 使用上述语义向量映射到具体情感类别所需要使用的线性层
        self.cls_fc = Linear(in_features=self.hidden_size, out_features=self.class_num, weight_attr=None, bias_attr=None)

        # 一般在获取单词的embedding后，会使用dropout层，防止过拟合，提升模型泛化能力
        self.dropout_layer = Dropout(p=self.dropout_rate, mode='upscale_in_train')


    # input为输入的训练文本， 其shape为[batch_size, max_seq_len]
    # label为对应的标签， 其shape为[batch_size,1]
    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # 本实验默认使用1层的LSTM，首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        init_hidden_data = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_cell_data = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype='float32')

        # 设置stop_gradient=True，避免这些向量被更新，从而影响训练效果
        init_hidden = paddle.to_tensor(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = paddle.to_tensor(init_cell_data)
        init_cell.stop_gradient = True

        # 对应以上第2步，将输入的句子的mini-batch转换为词向量表示，转换后输入数据shape为[batch_size, max_seq_len, embedding_size]
        x_emb = self.embedding(inputs)
        x_emb = paddle.reshape(x_emb, shape=[-1, self.num_steps, self.embedding_size])
        # 在获取的词向量后添加dropout
        if self.dropout_rate is not None and self.dropout_rate >= 0.0:
            x_emb = self.dropout_layer(x_emb)

        # 对应以上第三步，使用LSTM网络，把每个句子转换为语义向量
        # 返回的last_hidden是最后一个时间步的输出， 其shape为[self.num_layers, batch_size, hidden_size]
        rnn_out, (last_hidden, last_cell) = self.lstm_rnn(x_emb, (init_hidden, init_cell))

        # 提取最后一层隐状态作为文本的语义向量，其shape为[batch_size, hidden_size]
        last_hidden = paddle.reshape(last_hidden[-1], shape=[-1, self.hidden_size])

        # 对应以上第四步，将每个句子的向量映射到具体的情感类别上，logits的维度为[batch_size, 2]
        logits = self.cls_fc(last_hidden)

        return logits


def train():
    epoch_num = 5
    batch_size = 128

    learning_rate = 0.01
    dropout_rate = 0.2
    num_layers = 3
    hidden_size = 256
    embedding_size = 256
    max_seq_len = 128
    vocab_size = len(word2id_freq)

    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    # 实例化模型
    sentiment_classifier = SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                               num_layers=num_layers, dropout_rate=dropout_rate)
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                parameters=sentiment_classifier.parameters())

    sentiment_classifier = paddle.DataParallel(sentiment_classifier)

    # 定义训练函数
    # 记录训练过程中的损失变化情况，可用于后续画图查看训练情况
    losses = []
    steps = []

    sentiment_classifier.train()

    train_loader = build_batch(word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len)

    for step, (sentences, labels) in enumerate(train_loader):
        sentences = paddle.to_tensor(sentences)
        labels = paddle.to_tensor(labels)

        logits = sentiment_classifier(sentences)
        loss = F.cross_entropy(input=logits, label=labels, soft_label=False)
        loss = paddle.mean(loss)

        loss.backward()
        opt.step()
        opt.clear_grad()

        if step % 100 == 0:
            losses.append(loss.numpy()[0])
            steps.append(step)

            print('step:{}, loss:{}'.format(step, loss.numpy()))


    model_name = 'sentiment_classifier'
    paddle.save(sentiment_classifier.state_dict(), "{}.pdparams".format(model_name))
    # 保存优化器参数，方便后续模型继续训练
    paddle.save(opt.state_dict(), "{}.pdopt".format(model_name))


def evaluate():
    print('正在评估...')
    batch_size = 128

    learning_rate = 0.01
    dropout_rate = 0.2
    num_layers = 1
    hidden_size = 256
    embedding_size = 256
    max_seq_len = 128
    vocab_size = len(word2id_freq)

    model = SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                               num_layers=num_layers, dropout_rate=dropout_rate)
    model_dict = paddle.load('sentiment_classifier.pdparams')
    model.set_state_dict(model_dict)
    model.eval()


    test_loader = build_batch(word2id_dict, test_corpus, batch_size, 1, max_seq_len=max_seq_len)

    # 定义统计指标
    tp, tn, fp, fn = 0, 0, 0, 0

    for sentences, label in test_loader:
        sentences = paddle.to_tensor(sentences)
        labels = paddle.to_tensor(label)

        logits = model(sentences)

        # 使用softmax进行归一化
        probs = F.softmax(logits)
        probs = probs.numpy()

        for i in range(len(probs)):
            if labels[i][0] == 1:
                # 模型预测是正例
                if probs[i][1] > probs[i][0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if probs[i][1] > probs[i][0]:
                    fp += 1
                else:
                    tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 输出最终评估的模型效果
    print("TP: {}\nFP: {}\nTN: {}\nFN: {}\n".format(tp, fp, tn, fn))
    print("Accuracy: %.4f" % accuracy)




if __name__ == '__main__':
    download()

    train_corpus = load_imdb(True)
    test_corpus = load_imdb(False)

    '''for i in range(5):
        print('sentence %d, %s' % (i, train_corpus[i][0]))
        print('sentence %d, label %d' % (i, train_corpus[i][1]))
    '''

    train_corpus = data_preprocess(train_corpus)
    test_corpus = data_preprocess(test_corpus)

    print(train_corpus[:5])
    print(test_corpus[:5])

    word2id_dict, word2id_freq = build_dict(train_corpus)
    vocab_size = len(word2id_freq)

    print("there are totoally %d different words in the corpus" % vocab_size)
    for _, (word, word_id) in zip(range(10), word2id_dict.items()):
        print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))

    train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
    test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)

    print("%d tokens in the corpus" % len(train_corpus))
    print(train_corpus[:5])
    print(test_corpus[:5])

    train()
    evaluate()

