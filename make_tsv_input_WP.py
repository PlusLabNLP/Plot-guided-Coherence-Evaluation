import json
import math
import csv
import argparse
import os


if __name__=="__main__":
	#create train/valid/test file from AF output
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default='Data/WP/WP_Eval/', help='data path for WP_Eval data')
	parser.add_argument("--output_AF", type=str, default='AF_output.json', help='The result file from AF')
	parser.add_argument("--train_output", type=str, default='AF_ManPlts_train.tsv', help='The training file resulted from AF applied on WP manipulated plots to be used for training the evaluator')
	parser.add_argument("--valid_output", type=str, default='AF_ManPlts_valid.tsv', help='The validation file resulted from AF applied on WP manipulated plots to be used for validating the evaluator')
	parser.add_argument("--test_output", type=str, default='AF_ManPlts_test.tsv', help='The testing file resulted from AF applied on WP manipulated plots to be used for testing the evaluator')
	args = parser.parse_args()

	fr = open(os.path.join(args.data_path, args.output_AF), 'r')
	fw_train = open(os.path.join(args.data_path, 'adv_ManPlts/'+args.train_output), 'w')
	fw_valid = open(os.path.join(args.data_path, 'adv_ManPlts'+args.valid_output), 'w')
	fw_test = open(os.path.join(args.data_path, 'adv_ManPlts'+args.test_output), 'w')
	tsv_train = csv.writer(fw_train, delimiter='\t', lineterminator='\n')
	tsv_valid = csv.writer(fw_valid, delimiter='\t', lineterminator='\n')
	tsv_test = csv.writer(fw_test, delimiter='\t', lineterminator='\n')

	list_ctx = []
	list_gt = []
	list_gens = []
	num_stories= 0
	for line in fr:
		line = json.loads(line)
		list_ctx.append(line['ctx'])
		list_gt.append(line['gt_detok'])
		gens = []
		#line['assignment'][-1] shows the index of most challenging generated stories based on the applied AF
		sel_inds = line['assignment'][-1]
		for ind in sel_inds:
			gens.append(line['gens'][ind])
		list_gens.append(gens)

	num_stories=len(list_ctx)
	num_train_stories = math.ceil((60*num_stories)/100)
	num_valid_stories = math.ceil((20*num_stories)/100)

	start_ind={'train':0, 'valid':num_train_stories, 'test': num_train_stories + num_valid_stories}
	for mode in ['train', 'valid', 'test']:
		st_ind = start_ind[globals()['{}'.format(mode)]]
		globals()['{}_ctx'.format(mode)] = list_ctx[globals()[st_ind:st_ind+'num_{}_stories'.format(mode)]]
		globals()['{}_gt'.format(mode)] = list_gt[globals()[st_ind:st_ind+'num_{}_stories'.format(mode)]]
		globals()['{}_gens'.format(mode)] = list_gens[globals()[st_ind:st_ind+'num_{}_stories'.format(mode)]]

	
	for mode in ['train', 'valid', 'test']:
		line_ind = 0
		for ind, gt in enumerate(globals()['{}_gt'.format(mode)]):
			globals()['tsv_{}'.format(mode)].writerow([line_ind, 1, line_ind , gt])
			line_ind+=1
			for gen in globals()['{}_gens'.format(mode)][ind]:
				globals()['tsv_{}'.format(mode)].writerow([line_ind, 0, line_ind, gen])
				line_ind+=1
			



