import csv
import numpy as np
from scipy.stats import spearmanr, kendalltau
import os
import argparse


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default="Data/AMT/")
	parser.add_argument("--data", type=str, default="WP or ROC")
	args = parser.parse_args()
	
	if args.data=='WP':
		fname_human_scores='AMT_WP.csv'
		fname_model_scores='Pred_advWP_longformer.txt'
		max_stoies_in_hit=2
	else:
		fname_human_scores='AMT_ROC.csv'
		fname_model_scores='Pred_advROC_roberta.txt'
		max_stoies_in_hit=5


	fr_preds = open(os.path.join(args.data_path, fname_model_scores), 'r')
	preds=fr_preds.readlines()
	preds_scores={}
	for line in preds:
		text, score = line.split('\t')
		text = text.strip()
		score = float(score.split('\n')[0])
		preds_scores[text] = score

	csv_ifile = open(os.path.join(args.data_path,fname_human_scores), 'r')
	csv_reader = csv.DictReader(csv_ifile)
	set_proc_texts = []
	proc_texts_scores={}
	for row in csv_reader:
		for i in range(1, max_stoies_in_hit+1):
			row['Input.text'+str(i)] = row['Input.text'+str(i)].split('\n')[0].strip()
			score =row['Answer.text'+str(i)]
			if row['Input.text'+str(i)] not in set_proc_texts:
				set_proc_texts.append(row['Input.text'+str(i)])
				proc_texts_scores[row['Input.text'+str(i)]] = str(score)
			else:
				proc_texts_scores[row['Input.text'+str(i)]] = proc_texts_scores[row['Input.text'+str(i)]] + ' ' +str(score)

	print(len(preds_scores))
	print(len(proc_texts_scores))	
	human_scores = []
	human_scores_mean = []
	model_scores = []
	for i, text_scores in enumerate(proc_texts_scores):
		human_scores.append([int(sc) for sc in proc_texts_scores[set_proc_texts[i]].split()])
		human_scores_mean.append(np.mean(human_scores[i]))
		model_scores.append(preds_scores[set_proc_texts[i]])
	texts_human_scores_mean_01 = (human_scores_mean-np.min(human_scores_mean))/(max(human_scores_mean)-min(human_scores_mean))


	spearman_sentavg_posneg, p_value_sentavg_posneg = spearmanr(texts_human_scores_mean_01, model_scores)
	kendall_sentavg_posneg, pk_value_sentavg_posneg = kendalltau(texts_human_scores_mean_01, model_scores)
	print('The Spearman correlation is {} ({})'.format(spearman_sentavg_posneg,p_value_sentavg_posneg ))
	print('The Kendall correlation is {} ({})'.format(kendall_sentavg_posneg,pk_value_sentavg_posneg ))
