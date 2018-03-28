# Task1: word2vec

标签（空格分隔）： python numpy

---
## 一、数据集 ##
使用PTB数据集，链接为 http://www.fit.vutbr.cz/~imikolov/rnnlm/
论文：Mikolov  T, SutskeverI, Chen K, et al. Distributed  representations  of words  and phrases  and their compositionality[C]//Adv ances  in neural  information  processing  systems.  2013: 3111-3119.

## 二、实现主要功能 ##
1. CBOW和skip-gram两种训练模式
2. hierarchical softmax和negative sampling两种输出结构
3. subsampling of frequent words
4. phrase2vec: 将训练数据变成短语形式的输入
5. 使用tensorboard进行降维可视化

## 三、实现细节 ##
1. 用python, numpy实现
2. sigmoid值提前算好，提升速度
3. 使用min_count去除词频较少的词
4. 训练多个epoches，每个epoches随机打乱训练集顺序

## 四、代码文件说明 ##
1. word2vec.py是训练word2vec的文件，其中实现了word2vec的主要功能。
2. word2phrase.py是针对论文中的提出的基于短语的训练的程序，主要实现的功能是把原始的训练数据文件变成基于短语形式的文件。
3. visualize.py是针对训练结果进行可视化的代码，用的tensorboard进行可视化。

## 五、实验结果 ##
1. CBOW，hierarchical softmax，subsample_rate=5e-3(根据训练集决定)
   ![cbow-hs-subsam.PNG-439.8kB][1]
2. skip-gram，hierarchical softmax，subsample_rate=5e-3
   ![sg-hs-subsam.PNG-527.1kB][2]
3. CBOW，negative sampling，subsample_rate=5e-3
   ![cbow-neg-subsam.PNG-147.8kB][3]
4. skip-gram，negative sampling，subsample_rate=5e-3
   ![sg-neg-subsam.PNG-250.8kB][4]
5. CBOW，hierarchical softmax，no subsampling
   ![cbow-hs-nosub.PNG-424.8kB][5]
6. (phrase2vec) CBOW，hierarchical softmax，subsample_rate=5e-3
   ![phrase-cbow-hs-subsam.PNG-358.9kB][6]


  [1]: http://static.zybuluo.com/njuzrs/8078o8zca0f2mk2z0n140t1s/cbow-hs-subsam.PNG
  [2]: http://static.zybuluo.com/njuzrs/12ybln6esumykxhsf1npkkmd/sg-hs-subsam.PNG
  [3]: http://static.zybuluo.com/njuzrs/ukehamcwvzarumm0v37d9as5/cbow-neg-subsam.PNG
  [4]: http://static.zybuluo.com/njuzrs/tpx9ik509wyh0keig6no1ikl/sg-neg-subsam.PNG
  [5]: http://static.zybuluo.com/njuzrs/6d8gfv9ah1p8q1od4hakcx3w/cbow-hs-nosub.PNG
  [6]: http://static.zybuluo.com/njuzrs/a2rstvb6v7fzqe9211xnoroa/phrase-cbow-hs-subsam.PNG