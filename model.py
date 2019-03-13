import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Behaviors_CNN(nn.Module):
	def __init__(self, in_channels, total_num_URL, total_num_TIME, total_num_LOC, out_channels, filter_widths):
		super(CharacterLevelCNN, self).__init__()
		url_embedding_size = int(math.ceil(math.log(total_num_URL, 2)))
		time_embedding_size = int(math.ceil(math.log(total_num_TIME, 2)))
		loc_embedding_size = int(math.ceil(math.log(total_num_LOC, 2)))

		self.URL_EmbeddingTable = nn.Embedding(total_num_URL, url_embedding_size)
		self.TIME_EmbeddingTable = nn.Embedding(total_num_TIME, time_embedding_size)
		self.LOC_EmbeddingTable = nn.Embedding(total_num_LOC, loc_embedding_size)

		self.behavior_embedding_size = url_embedding_size + time_embedding_size + loc_embedding_size

		self.filter_widths = filter_widths # like [3, 4, 5]
		for index, filter_width in enumerate(filter_widths):
			setattr(self, 'conv1d_'+str(index), nn.Sequential(
						nn.Conv1d(
							in_channels=in_channels,	  
							out_channels=out_channels,	
							kernel_size=(filter_width, self.behavior_embedding_size)
						),	  
						nn.ReLU()
					))

	def forward(self, index_batch_tuple):
		url_index_batch, time_index_batch, loc_index_batch = index_batch_tuple

		# batch for multiple users
		# [batch_size, max_sum_behaviors, embedding_size]
		url_embedding_batch = self.URL_EmbeddingTable(url_index_batch)
		time_embedding_batch = self.TIME_EmbeddingTable(time_index_batch)
		loc_embedding_batch = self.LOC_EmbeddingTable(loc_index_batch)

		behavior_embedding_batch = torch.cat([url_embedding_batch, time_embedding_batch, loc_embedding_batch])

		# '1' for 1 input_channel
		behavior_embedding_batch = behavior_embedding_batch.view(behavior_embedding_batch.size(0), 1, behavior_embedding_batch.size(1), behavior_embedding_batch.size(1))

		char_embs = []
		for index in range(len(self.filter_widths)):
			cnn_layer = getattr(self, 'conv1d_'+str(index))
			cnn_out = cnn_layer(behavior_embedding_batch)
			
			# rows_after_cov = max_sum_behaviors-filter_width+1
			# [batch_size, out_channels, rows_after_cov, in_channels]
			# --> [batch_size, out_channels, rows_after_cov] since in_channels =1
			cnn_out = cnn_out.view(cnn_out.size(0), cnn_out.size(1), cnn_out.size(2))

			cnn_out, _ = torch.max(cnn_out, 2)
			char_embs.append(cnn_out)

		# [batch_size, out_channels*len(filter_widths)]
		char_embs = torch.cat(char_embs, 1)

		return char_embs

class FFNN(nn.Module):
	"""docstring for FFNN"""
	def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
		super(FFNN, self).__init__()
		self.num_hidden_layers = num_hidden_layers
		for index in range(self.num_hidden_layers):
			if index == 0:
				h_input_size = input_size
			else:
				h_input_size = hidden_size
			setattr(self, 'hidden_layer_'+str(index), nn.Sequential(
						nn.Linear(h_input_size, hidden_size),	  
						nn.ReLU()
					))
		self.output_layer = nn.Linear(hidden_size, output_size)
	def forward(self, input_batch):
		outputs = input_batch
		for index in range(self.num_hidden_layers):
			hidden_layer = getattr(self, 'hidden_layer_'+str(index))
			outputs = hidden_layer(outputs)
		outputs = torch.softmax(self.output_layer(outputs), 1)
		return outputs

