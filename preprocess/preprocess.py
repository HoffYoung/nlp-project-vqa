import mindspore
from mindspore.mindrecord import FileWriter
import mindspore.dataset as dataset
import numpy as np
import json
from easydict import EasyDict
import os
import matplotlib as mt
import matplotlib.pyplot as plt
#import cv2
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision.c_transforms import Resize, Decode

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
					   .replace('\'', ' ') \

	words = sentence.split(' ')
	return words

# 只取那些answer长度为1的组合
def get_list(question_path, answer_path):
	# read question & answer from files
	answer_data = None
	question_data = None
	with open(answer_path, 'r',encoding='utf8') as fp1:
		answer_data = json.load(fp1)
	with open(question_path, 'r', encoding='utf-8') as fp2:
		question_data = json.load(fp2)
	# create question dictionary
	question_dict = dict()
	for question in question_data['questions']:
		question_dict[question['question_id']] = question
	# create image-question-answer lists
	answers = []
	options = list()
	questions = []
	images = []
	images_id = []
	for answer in answer_data['annotations']:
		# if len(get_simplified_words(answer['multiple_choice_answer'])) == 1:
		answers.append(answer['multiple_choice_answer'])
		options.append([item['answer'] for item in answer['answers']])
		questions.append(get_simplified_words(question_dict[answer['question_id']]['question']))
		images.append(answer['image_id'])
  
	return images, questions, answers, options

# read original data

def add_word_into_dict(sentences, dict):
	for sentence in sentences:
		for word in sentence:
			if not dict.__contains__(word):
				cur_idx = len(dict)
				dict[word] = cur_idx
	return dict

# read original answers

def add_answer_into_dict(answers, dict):
    for answer in answers:
        if not dict.__contains__(answer):
            cur_idx = len(dict)
            dict[answer] = cur_idx
    return dict

# word to index

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

# option word to index

def get_option_vec_and_pad(options, dict, max_length):
	vector_list = []
	for option in options:
		vectors = []
		for answer in option:
			vector = []
			for word in answer:
				vector.append(dict[answer])
			if len(word) < max_length:
				vector += [0] * (max_length - len(word))
			else:
				vector = vector[:max_length]
			if(max_length == 1):
				vectors.append(vector[0])
			else:
				vectors.append(vector)
		vector_list.append(vectors)

	return vector_list
    
# answer to index

def get_answer_to_idx(answers, dict):
    vectors = []
    bool_set = set()
    num_set = set()
    other_set = set()
    
    for answer in answers:
        vectors.append(dict[answer])
        if answer == "yes" or answer == "no":
            bool_set.add(dict[answer])
        elif answer.isdigit():
            num_set.add(dict[answer])
        else:
            other_set.add(dict[answer])
            
    return vectors, bool_set, num_set, other_set

# get filtered set according to frequency

def get_filtered_answer_set(total_answers_vec, min_frequency):
	unique, count = np.unique(total_answers_vec, return_counts=True)
	cnt = 0
	total_cnt = 0
	filtered_set = set()
	for item in zip(unique, count):
		if item[1] >= min_frequency:
			filtered_set.add(item[0])
			cnt += item[1]
		total_cnt += item[1]
	print(cnt / total_cnt)
 
	return filtered_set


def read_image(images_list, path):
    img_list = []

    for i in images_list:
        sub_img = mt.image.imread(path + str(i).rjust(12, '0') + ".jpg")
        img_list.append(sub_img)
    
    return img_list

# set to dict
def set_to_dict(least_set):
	answer_dict = dict()
	idx = 0
	while len(least_set) > 0:
		answer_dict[least_set.pop()] = idx
		idx += 1
	return answer_dict

# create MindRecord

def generate_mindrecord(mindrecord_path, image_path, num_splits, images, questions, answers, options):
	schema = {"image": {"type": "bytes"},
          "question": {"type": "int32", "shape": [-1]},
		  "answer": {"type": "int32"},
		  "image_id": {"type": "int32"},
		  "options": {"type": "int32", "shape": [-1]}
	}
	#writer = FileWriter(mindrecord_path, num_splits, overwrite=True)
	writer = FileWriter(mindrecord_path, num_splits)
	print(mindrecord_path.split("/")[-1].split(".")[0])
	writer.add_schema(schema, "vqa "+mindrecord_path.split("/")[-1].split(".")[0])
	data_list = []
	for i, q, a, o in zip(images, questions, answers, options):
		with open(image_path + str(i).rjust(12, '0') + ".jpg", "rb") as fp:
			q = np.array(q).astype(np.int32)
			o = np.array(o).astype(np.int32)
			data_json = {"image": fp.read(),
						"question": q.reshape(-1),
						"answer": a,
						"image_id": i,
      					"options":o.reshape(-1)
           				}
			data_list.append(data_json)
	writer.write_raw_data(data_list)
	writer.commit()


# load data from MindRecord

def generate_dataset(data_path, batch_size, epoch_size, seq_length, filter_dict, image_height=224, image_width=224, 
                     num_parallel_workers=4, device_num=1, rank=0):
	def read_answer(answer):
		if filter_dict.__contains__(answer.item()):
			return np.array(filter_dict[answer.item()])
		else:
			return np.array(len(filter_dict))

	def read_question(question):
		return question[0:seq_length] if seq_length < len(question) else question

	def read_options(options):
		return options

	def read_image_id(image_id):
		return image_id

	decode_op = Decode()
	resize_op = Resize([image_height, image_width], Inter.BICUBIC)
	transforms_list = [decode_op, resize_op]
	
	data_set = dataset.MindDataset(data_path, columns_list=["image", "question", "answer", "image_id", "options"],
								num_parallel_workers=num_parallel_workers, num_shards=device_num, shard_id=rank)
    
	data_set = data_set.map(operations=transforms_list, input_columns=["image"])
	data_set = data_set.map(operations=read_question, input_columns=["question"])
	data_set = data_set.map(operations=read_answer, input_columns=["answer"])
	data_set = data_set.map(operations=read_image_id, input_columns=["image_id"])
	data_set = data_set.map(operations=read_options, input_columns=["options"])
	data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
	data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
	data_set = data_set.repeat(count=epoch_size)
	
	return data_set