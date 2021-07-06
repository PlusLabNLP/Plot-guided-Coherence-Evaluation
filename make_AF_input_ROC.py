import os
import torch
from fairseq.models.bart import BARTModel
import csv
import json
import argparse


class AF_Data_Generation():

	def __init__(self, model_path):
		
		self.bart = BARTModel.from_pretrained(
		    model_path,
		    checkpoint_file='checkpoint_best.pt',
		    data_name_or_path='rocstories_kw_story-bin'
		)
		self.bart.cuda()
		self.bart.eval()
		self.bart.half()


	def generate_implausible_stories(self, args, gt_stories, num_sents_gt_stories, file_type, ind_file):
		#This function takes the manipulated plots and leverages the conditional LM (finetuned BART) to generate implausible stories
		fr = open(args.man_plts, 'r')
		manipulated_plots = fr.readlines()
		tsv_file = open(os.path.join(args.data_path, 'ManPlts/{}_pos_neg_stories_{}.tsv'.format(file_type, ind_file)), 'w')
		tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')

		count = 1
		bsz = args.batch_size
		conv_lines = []
		gt_stories_lines = []
		num_sents_gt_lines = []
		indj = 0
		for ind_line, plots in  enumerate(manipulated_plots[:100]):
			plots = plots.strip()
			conv_lines.append(plots)
			gt_stories_lines.append(gt_stories[ind_line])
			num_sents_gt_lines.append(num_sents_gt_stories[ind_line])
			if count % bsz == 0:
				with torch.no_grad():
					hyps = self.bart.sample(conv_lines, sampling =True, lenpen=2.0, max_len_b=args.max_len_b, min_len=args.min_len, sampling_topk=args.sampling_topk, temperature=args.temperature, beam=args.beam)
				for ind, hypothesis in enumerate(hyps):
					hypothesis = hypothesis.strip().split('</s>')[:num_sents_gt_lines[ind]]
					hypothesis = '</s>'.join(hypothesis)
					hypothesis = hypothesis.replace('  ', ' ')
					tsv_writer.writerow([indj, '1', indj, gt_stories_lines[ind]])
					indj+=1
					tsv_writer.writerow([indj, '0', indj, hypothesis])
					indj+=1	
				num_sents_gt_lines=[]
				gt_stories_lines = []
				conv_lines = []
			count +=1
		if conv_lines != []:
			hyps = self.bart.sample(conv_lines, sampling=True, lenpen=2.0, max_len_b=args.max_len_b, min_len=args.min_len, sampling_topk=args.sampling_topk, temperature=args.temperature, beam=args.beam)
			for ind, hypothesis in enumerate(hyps):
				hypothesis = hypothesis.strip().split('</s>')[:num_sents_gt_lines[ind]]
				hypothesis = '</s>'.join(hypothesis)
				hypothesis = hypothesis.replace('  ', ' ')
				tsv_writer.writerow([indj, '1', indj, gt_stories_lines[ind]])
				indj+=1
				tsv_writer.writerow([indj, '0', indj, hypothesis])
				indj+=1


	def create_json_AF_input(self, args):
		#This function takes "num_negative_samples" tsv files that includes stories with different negative samples and creates a json file as an input for AF 
		for mode in ['train', 'valid', 'test']:
			for ind in range(args.num_negative_samples):
				globals()['fr_{}_{}'.format(mode,ind)] = open(os.path.join(args.data_path, 'ManPlts/{}_pos_neg_stories_{}.tsv'.format(mode, ind)), 'r')
				globals()['lines_{}_{}'.format(mode,ind)] = globals()['fr_{}_{}'.format(mode,ind)].readlines()
		fw = open(os.path.join(args.data_path, args.json_file), 'w')

		output = []
		for mode in ['train', 'valid', 'test']:
			ind=0
			for i in range(0, len(globals()['lines_{}_{}'.format(mode,ind)]), 2):
				output_text ={}
				gt_story = globals()['lines_{}_{}'.format(mode,ind)][i].split('\t')[3].split('\n')[0].strip()
				gens=[]
				for ind in range(args.num_negative_samples):
					globals()['gen_story{}'.format(ind)]=' '.join(globals()['lines_{}_{}'.format(mode,ind)][i+1].split('\t')[3].split('\n')[0].strip().split('</s>')[3:]).replace('  ', ' ').strip()
					gens.append(globals()['gen_story{}'.format(ind)])
				prmpt = ' . '.join(gt_story.split(' . ')[:3]).strip()
				if prmpt[-1] not in ['?', '.', '!']:
					prmpt = prmpt + ' .'
				gt_story = ' . '.join(gt_story.split(' . ')[3:])
				output_text["ctx"] = prmpt
				output_text["gt_detok"] = gt_story
				output_text["gens"] = gens 
				output.append(output_text)
				ind=0
							
		fw.write('\n'.join(json.dumps(i, ensure_ascii=False) for i in output))



if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--bart_model_path", type=str, default="Models/Ft_BART_Story_Generator/ROC/", help="model path including finetuned BART model as the conditional LM")
	parser.add_argument("--data_path", type=str, default='Data/ROC/ROC_Eval', help='data path for ROC_Eval data')
	parser.add_argument("--json_file", type=str, default='ROC_AF_input.json', help='json input files')
	parser.add_argument("--num_negative_samples", type=int, default=6, help="number of negative (implausible) samples to generate for each plausible story")
	parser.add_argument("--batch_size", type=int, default=32, help='batch size to generate samples')
	parser.add_argument("--max_len_b", type=int, default=200, help='max length of stories')
	parser.add_argument("--min_len", type=int, default=10, help='min length of stories')
	parser.add_argument("--sampling_topk", type=int, default=50, help='topk sampling')
	parser.add_argument("--temperature", type=float, default=0.8, help='temperature value')
	parser.add_argument("--beam", type=float, default=4, help='beam size')

	
	args = parser.parse_args()
	af = AF_Data_Generation(args.bart_model_path)
	
	for  file_type in ['train', 'valid', 'test']:
		#file including ground truth stories
		args.gt_stories=args.data_path+'/Rocstories_{}'.format(file_type)
		#file including manipulated plots
		args.man_plts=args.data_path+'/ManPlts/Rocstories_{}_manipulated_plts'.format(file_type)
		fr_gt = open(args.gt_stories, 'r')
		lines_gt_stories = fr_gt.readlines()
		gt_stories=[]
		num_sents_gt_stories=[]
		for ind_line, story in  enumerate(lines_gt_stories):
			gt_story = lines_gt_stories[ind_line].split('@')[0].split('<EOL>')[1].strip().split('</s>')[1:]
			gt_story = [g.strip() for g in gt_story]
			num_sents_gt_stories.append(len(gt_story))
			gt_story = ' '.join(gt_story).strip()
			gt_story = gt_story.replace('  ', ' ')
			gt_stories.append(gt_story)
	
		#Generate args.num_negative_samples different tsv files each including gt_stories as positive and generated negative stories as implausible ones
		for ind in range(args.num_negative_samples):
			af.generate_implausible_stories(args, gt_stories, num_sents_gt_stories, file_type, ind)

	af.create_json_AF_input(args)

