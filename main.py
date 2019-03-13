import sys, os
import argparse
import torch

parser = argparse.ArgumentParser(description='Behavior Modeling')

parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':

	url_index_batch = []
	time_index_batch = []
	loc_index_batch = []
	for user in user_batch:
		user_urls = user['urls'] # urls of behaviors
		user_times = user['times']
		user_locs = user['locs']
		assert len(user_urls) == len(user_times) == len(user_locs)
		for i, url in enumerate(user_urls):
			user_urls.append(url2id[url])
			user_times.append(time2id[user_times[i]])
			user_locs.append(url2id[user_locs[i]])
		url_index_batch.append(user_urls)
		time_index_batch.append(user_times)
		loc_index_batch.append(user_locs)

	behavior_cnn = Behaviors_CNN(in_channels=1, total_num_URL=len(url2id), total_num_TIME=len(time2id), total_num_LOC=len(loc2id), out_channels=50, filter_widths=[3,4,5])
	behavior_cnn.to(device)
	fnn = FFNN(input_size=50*3, hidden_size=50, num_hidden_layers=1, output_size=2)

	# [batch_size, max_sum_behaviors, embedding_size]
	# TO DO: padding for url_index_batch, because different user may has diffrent number of behaviors
	url_index_batch = torch.LongTensor(url_index_batch).to(device)
	time_index_batch = torch.LongTensor(time_index_batch).to(device)
	loc_index_batch = torch.LongTensor(loc_index_batch).to(device)

	# [batch_size, out_channels*len(filter_widths)]
	output = behavior_cnn((url_index_batch, time_index_batch, loc_index_batch))
	# [batch_size, 2]
	male_female_scores = fnn(output)

