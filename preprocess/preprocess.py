import mindspore
from mindspore.mindrecord import FileWriter
import mindspore.dataset as dataset
import numpy as np
import json
from easydict import EasyDict
import os
import matplotlib as mt
import matplotlib.pyplot as plt
from skimage import transform,data

def get_simplified_words(sentence):
	sentence = sentence.lower()
	sentence = sentence.replace('?', ' ?') \
		               .replace('.', ' .') \
					   .replace(',', ' ,') \
					   .replace('!', ' !') \
					   .replace(':', ':') \
					   .replace(';', ' ;') \
					   .replace('-', ' ')  \
					   .replace('\"', ' ') \
					   .replace('\'', ' ')
	words = sentence.split(' ')
	return words

# 只取那些answer长度为1的组合
def get_list(question_path, answer_path):
	# read question & answer from files
	answer_data = None
	question_data = None
	with open(answer_path, 'r',encoding='utf8')as fp1:
		answer_data = json.load(fp1)
	with open(question_path, 'r', encoding='utf-8') as fp2:
		question_data = json.load(fp2)
	# create question dictionary
	question_dict = dict()
	for question in question_data['questions']:
		question_dict[question['question_id']] = question
	# create image-question-answer lists
	answers = []
	questions = []
	images = []
	for answer in answer_data['annotations']:
		if len(get_simplified_words(answer['multiple_choice_answer'])) == 1:
			answers.append(get_simplified_words(answer['multiple_choice_answer']))
			questions.append(get_simplified_words(question_dict[answer['question_id']]['question']))
			images.append(answer['image_id'])

	return images, questions, answers

# read original data

def add_word_into_dict(sentences, dict):
	for sentence in sentences:
		for word in sentence:
			if not dict.__contains__(word):
				cur_idx = len(dict)
				dict[word] = cur_idx
	return dict

def get_vec_and_pad(sentences, dict, max_length):
	vectors = []
	for sentence in sentences:
		vector = []
		for word in sentence:
			vector.append(dict[word])
		if len(sentence) < max_length:
			vector += [0] * (max_length - len(sentence))
		else:
			vector = vector[:max_length]
		if(max_length == 1):
			vectors.append(vector[0])
		else:
			vectors.append(vector)
	return vectors

def read_image(images_list, path):
    img_list = []

    for i in images_list:
        sub_img = mt.image.imread(path + str(i).rjust(12, '0') + ".jpg")
        
        sub_img=transform.resize(sub_img, (224, 224, 3))
        img_list.append(sub_img)
    
    return np.array(img_list, dtype=np.int32)

# create MindRecord

def generate_mindrecord(mindrecord_path, num_splits, images, questions, answers):
	schema = {"image": {"type": "int32", "shape": [224, 224, 3]},
          "question": {"type": "int32", "shape": [-1]},
		  "answer": {"type": "int32", "shape": [-1]},
	}
	writer = FileWriter(mindrecord_path, num_splits, overwrite=True)
	print(mindrecord_path.split("/")[-1].split(".")[0])
	writer.add_schema(schema, "vqa "+mindrecord_path.split("/")[-1].split(".")[0])
	data_list = []
	for i, q, a in zip(images, questions, answers):
		i = np.array(i).astype(np.int32)
		q = np.array(q).astype(np.int32)
		a = np.array(a).astype(np.int32)
		data_json = {"image": i.reshape(224, 224, 3),
					 "question": q.reshape(-1),
					 "answer": a.reshape(-1)}
		data_list.append(data_json)
	writer.write_raw_data(data_list)
	writer.commit()


# load data from MindRecord

def generate_dataset(data_path, batch_size, epoch_size, device_num=1, rank=0):
	def read_operation(image, question, answer):
		return image, question, answer

	data_set = dataset.MindDataset(data_path, columns_list=["image", "question", "answer"],
								num_parallel_workers=4, num_shards=device_num, shard_id=rank)
	data_set = data_set.map(read_operation,
							input_columns=["image", "question", "answer"], 
							output_columns=["image", "question", "answer"], 
							column_order=["image", "question", "answer"])
	data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
	data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
	data_set = data_set.repeat(count=epoch_size)

	return data_set