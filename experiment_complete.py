#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感分析实验完整代码
Kaggle比赛: Bag of Words Meets Bags of Popcorn

实验流程:
1. 数据加载 - 加载带标签数据、无标签数据和测试数据
2. 文本预处理 - HTML标签去除、小写转换、否定词处理
3. Word2Vec训练 - 使用Skip-gram算法训练词向量
4. 特征提取 - Word2Vec向量 + 手工特征
5. 模型训练 - 随机森林分类器
6. 模型评估 - AUC-ROC指标
7. 生成提交文件

改进点:
- 保留否定词（not, no, never等）
- 处理否定词组合（not_good）
- 多种池化策略（均值、最大值、最小值、标准差）
- 手工特征（文本长度、情感词计数、否定词计数）
- 集成Word2Vec和手工特征
"""

import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ==============================================
# 停用词列表（保留否定词）
# ==============================================
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
}

NEGATION_WORDS = {'not', 'no', 'never', 'nor', 'none', 'nothing', 'nowhere', 'neither', 'nobody'}
STOP_WORDS_FILTERED = STOPWORDS - NEGATION_WORDS

# 情感词列表
POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'beautiful', 
                  'perfect', 'awesome', 'fantastic', 'brilliant', 'outstanding', 'superb', 'terrific'}
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'poor',
                  'boring', 'dull', 'ugly', 'sad', 'angry', 'annoying', 'frustrating', 'waste'}

# ==============================================
# 文本预处理函数
# ==============================================
def preprocess_text(text):
    """
    文本预处理：
    1. 去除HTML标签
    2. 转小写
    3. 处理否定词组合（not + word -> not_word）
    4. 去除标点符号
    5. 分词
    6. 过滤停用词（保留否定词和否定词组合）
    """
    # 去除HTML标签
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 转小写
    text = text.lower()
    
    # 处理否定词组合（如 "not good" -> "not_good"）
    text = re.sub(r'\b(not|no)\s+(\w+)', r'\1_\2', text)
    
    # 去除标点符号
    text = re.sub(r'[^a-zA-Z\s_]', ' ', text)
    
    # 分词
    tokens = text.split()
    
    # 过滤停用词（保留否定词和否定词组合）
    tokens = [token for token in tokens if token not in STOP_WORDS_FILTERED or 
              token.startswith('not_') or token.startswith('no_')]
    
    # 去除单字符（除否定词外）
    tokens = [token for token in tokens if len(token) > 1 or token in NEGATION_WORDS]
    
    return tokens

# ==============================================
# 手工特征提取函数
# ==============================================
def extract_handcrafted_features(tokens):
    """
    提取手工特征：
    1. 文本长度
    2. 唯一词数量
    3. 积极情感词数量
    4. 消极情感词数量
    5. 否定词数量
    6. 否定+积极组合数量（如 not_good）
    7. 否定+消极组合数量（如 not_bad）
    """
    features = []
    
    # 基本特征
    text_length = len(tokens)
    unique_count = len(set(tokens))
    
    # 情感词计数
    pos_count = sum(1 for token in tokens if token in POSITIVE_WORDS or 
                    token.replace('not_', '') in POSITIVE_WORDS)
    neg_count = sum(1 for token in tokens if token in NEGATIVE_WORDS or 
                    token.replace('not_', '') in NEGATIVE_WORDS)
    
    # 否定词计数
    negation_count = sum(1 for token in tokens if token in NEGATION_WORDS or 
                         token.startswith('not_') or token.startswith('no_'))
    
    # 否定情感组合
    neg_positive_count = sum(1 for token in tokens if token.startswith('not_') and 
                             token.replace('not_', '') in POSITIVE_WORDS)
    neg_negative_count = sum(1 for token in tokens if token.startswith('not_') and 
                             token.replace('not_', '') in NEGATIVE_WORDS)
    
    features.extend([text_length, unique_count, pos_count, neg_count, 
                     negation_count, neg_positive_count, neg_negative_count])
    
    return np.array(features)

# ==============================================
# 句子向量转换函数
# ==============================================
def sentence_to_vector(tokens, model, vector_size=300):
    """
    将句子转换为向量：
    1. 否定词组合的向量取反（not_good -> -good向量）
    2. 使用多种池化策略：均值、最大值、最小值、标准差
    """
    vectors = []
    
    for token in tokens:
        if token.startswith('not_') or token.startswith('no_'):
            # 否定词组合：向量取反
            base_token = token.replace('not_', '').replace('no_', '')
            if base_token in model.wv:
                vectors.append(-model.wv[base_token])
        elif token in model.wv:
            vectors.append(model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(vector_size * 4)
    
    vectors = np.array(vectors)
    
    # 多种池化策略
    mean_vec = np.mean(vectors, axis=0)
    max_vec = np.max(vectors, axis=0)
    min_vec = np.min(vectors, axis=0)
    std_vec = np.std(vectors, axis=0)
    
    return np.concatenate([mean_vec, max_vec, min_vec, std_vec])

# ==============================================
# 主实验流程
# ==============================================
def run_experiment():
    print("="*70)
    print("情感分析实验 - Word2Vec + 随机森林")
    print("="*70)
    
    # --------------------------
    # 步骤1: 加载数据
    # --------------------------
    print("\n[步骤1/7] 加载数据...")
    labeled_path = 'labeledTrainData.tsv'
    unlabeled_path = '../unlabeledTrainData.tsv/unlabeledTrainData.tsv'
    test_path = '../testData.tsv/testData.tsv'
    
    labeled_df = pd.read_csv(labeled_path, sep='\t')
    unlabeled_df = pd.read_csv(unlabeled_path, sep='\t', on_bad_lines='skip')
    test_df = pd.read_csv(test_path, sep='\t')
    
    print(f"  带标签数据: {labeled_df.shape[0]} 条")
    print(f"  无标签数据: {unlabeled_df.shape[0]} 条")
    print(f"  测试数据: {test_df.shape[0]} 条")
    
    # --------------------------
    # 步骤2: 文本预处理
    # --------------------------
    print("\n[步骤2/7] 文本预处理...")
    labeled_df['tokens'] = labeled_df['review'].apply(preprocess_text)
    unlabeled_df['tokens'] = unlabeled_df['review'].apply(preprocess_text)
    test_df['tokens'] = test_df['review'].apply(preprocess_text)
    print("  预处理完成")
    
    # --------------------------
    # 步骤3: 提取手工特征
    # --------------------------
    print("\n[步骤3/7] 提取手工特征...")
    labeled_df['handcrafted'] = labeled_df['tokens'].apply(extract_handcrafted_features)
    test_df['handcrafted'] = test_df['tokens'].apply(extract_handcrafted_features)
    print("  手工特征提取完成")
    
    # --------------------------
    # 步骤4: 训练Word2Vec模型
    # --------------------------
    print("\n[步骤4/7] 训练Word2Vec模型...")
    all_sentences = list(labeled_df['tokens']) + list(unlabeled_df['tokens'])
    
    w2v_model = Word2Vec(
        sentences=all_sentences,
        vector_size=300,
        window=15,
        min_count=5,
        workers=4,
        epochs=30,
        sg=1,
        negative=20
    )
    
    print(f"  Word2Vec训练完成，词汇表大小: {len(w2v_model.wv)}")
    
    # --------------------------
    # 步骤5: 转换为向量特征
    # --------------------------
    print("\n[步骤5/7] 转换为向量特征...")
    X_w2v = np.array([sentence_to_vector(tokens, w2v_model) for tokens in labeled_df['tokens']])
    X_handcrafted = np.array(labeled_df['handcrafted'].tolist())
    X = np.hstack([X_w2v, X_handcrafted])
    
    X_test_w2v = np.array([sentence_to_vector(tokens, w2v_model) for tokens in test_df['tokens']])
    X_test_handcrafted = np.array(test_df['handcrafted'].tolist())
    X_test = np.hstack([X_test_w2v, X_test_handcrafted])
    
    y = labeled_df['sentiment'].values
    
    print(f"  训练特征维度: {X.shape}")
    print(f"  测试特征维度: {X_test.shape}")
    
    # --------------------------
    # 步骤6: 特征标准化和模型训练
    # --------------------------
    print("\n[步骤6/7] 训练模型...")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    # 训练随机森林
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=40,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 评估
    y_val_pred = clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    print(f"  验证集AUC: {auc_score:.4f}")
    
    # --------------------------
    # 步骤7: 生成提交文件
    # --------------------------
    print("\n[步骤7/7] 生成提交文件...")
    
    # 预测测试集
    test_df['sentiment'] = clf.predict_proba(X_test_scaled)[:, 1]
    
    # 生成提交文件
    submission = test_df[['id', 'sentiment']]
    submission.to_csv('submission.csv', index=False)
    
    # 保存实验结果
    with open('experiment_results.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("情感分析实验结果\n")
        f.write("="*50 + "\n")
        f.write("参数配置:\n")
        f.write("  Word2Vec维度: 300\n")
        f.write("  Word2Vec窗口: 15\n")
        f.write("  Word2Vec训练轮数: 30\n")
        f.write("  Word2Vec算法: Skip-gram\n")
        f.write("  随机森林树数量: 400\n")
        f.write("  随机森林最大深度: 40\n")
        f.write("-"*50 + "\n")
        f.write(f"验证集AUC: {auc_score:.4f}\n")
        f.write("="*50 + "\n")
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"验证集AUC: {auc_score:.4f}")
    print("生成的文件:")
    print("  - submission.csv (Kaggle提交文件)")
    print("  - experiment_results.txt (实验结果)")
    print("="*70)
    
    # 打印样本输出
    print("\n样本预测结果:")
    print(submission.head())
    
    return auc_score

# ==============================================
# 运行实验
# ==============================================
if __name__ == '__main__':
    run_experiment()
