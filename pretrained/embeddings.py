import os
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from mindspore import Parameter

## 获取词典对应的嵌入表
def get_embedding_table(vocab_dict, kv_model, embedding_table, save_path):
	"""
	param: Vocab: 词典
	param: kv_model: 读取预训练词向量后得到的向量表
	param: embedding_table: 嵌入层初始表引用
	"""

	for word, idx in vocab_dict.items():
		if kv_model.has_index_for(word):
			embedding_vec = kv_model[word]
		else:
			embedding_vec = 2 * np.random.random((1, 200)).astype(np.float32) - 1
		if embedding_vec is not None:
			embedding_table[idx] = embedding_vec
	
	np.savetxt(save_path, embedding_table)



# 获取词典嵌入表，没有则先创建
def create_embedding_table(vocab_dict, config):
	if os.path.exists(config.embedding_table_path):
		return np.loadtxt(config.embedding_table_path, delimiter=' ').astype(np.float32)
	else:
		## GloVe存储格式转化为Word2Vec存储格式
		glove_input_file = config.glove_vector_path
		word2vec_output_file = config.glove_word2vec_path
		if not os.path.exists(word2vec_output_file):
			glove2word2vec(glove_input_file, word2vec_output_file)
		## 加载嵌入向量
		kv_model = KeyedVectors.load_word2vec_format(word2vec_output_file)
		## 初始化嵌入表
		embedding_table = np.zeros([config.vocab_size, 200], dtype=np.float32)
		get_embedding_table(vocab_dict, kv_model, embedding_table, config.embedding_table_path)
		return embedding_table
	