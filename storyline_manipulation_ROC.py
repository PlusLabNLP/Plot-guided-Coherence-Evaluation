#This code is used for appyling different type of proposed manipulattions on the plots of the ROCstories
import torch
import os
import numpy as np
import nltk
import math
import spacy
import pyinflect
import argparse 
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
nlp = spacy.load('en_core_web_sm')
np.random.seed(100)



class Plt_manipulations():
	#class of plots manipulations
	def __init__(self, COMET_model_file, COMET_sampling_algorithm, device):
		opt, state_dict = interactive.load_model_file(COMET_model_file)
		self.data_loader, self.text_encoder = interactive.load_data("conceptnet", opt)
		n_ctx = self.data_loader.max_e1 + self.data_loader.max_e2 + self.data_loader.max_r
		n_vocab = len(self.text_encoder.encoder) + n_ctx
		self.comet_model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
		sampling_algorithm = COMET_sampling_algorithm	
		if args.device != "cpu":
			cfg.device = int(device)
			cfg.do_gpu = True
			torch.cuda.set_device(cfg.device)
			self.comet_model.cuda(cfg.device)
		else:
			cfg.device = "cpu"
		self.sampler = interactive.set_sampler(opt, sampling_algorithm, self.data_loader)
		fr = open("Data/conceptnet_antonym.txt", "r")	
		self.conceptnet_antonyms = fr.readlines()
		self.plt_antonyms = self.get_antonyms()
		

	def get_pos_plt(self, plt, pos_sent):
		#This function returns the part of speech tag of a plot in a sentence
		for token, pos in pos_sent:
			if token == plt:
				return pos
		return None
	
	def has_verb_tag(self, pos_ngram):
		#This function returns whether the ngram includes a verb  
		for token, pos in pos_ngram:
			if 'VB' in pos:
				return pos
		return None

	def get_verb_pos(self, pos_ngram):
		#This function returns all the verbs in an ngram 
		ind =-1
		results = []
		for  token, pos in pos_ngram:
			ind +=1
			if 'VB' in pos:
				results.append(ind)
		return results
	
	def get_antonyms(self):
		#This function returns a dictionary of conceptnet words with their antonyms  
		anotomy_word = {}
		for line in self.conceptnet_antonyms:
			tmp = line.strip().split("|||")
			if len(tmp) == 3:
            			h, t = tmp[0], tmp[2].split()
			if h in anotomy_word:
				anotomy_word[h] += t
			else:
				anotomy_word[h] = t[:]		
		return anotomy_word

	def repetition(self, plots):
		#This function applies the "Repetition Insertion" manipulation on the plots 
		#In ROCStory dataset all the sentences' plots of a story are combined with "#"
		sents_plots = plots.split('#') 
		num_sents = len(sents_plots)
		new_sents_plots = sents_plots
		num_plts = 0
		list_plts = []
		weighted_sents=[]
		plts_positions_sents = {} 
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			sent_plots = [i for i in sent_plots if i]
			num_plts+=len(sent_plots)
			weighted_sents.append(len(sent_plots))
			for plt in sent_plots:
				list_plts.append(plt.strip())
				plts_positions_sents[plt] = ind
			
		weighted_sents_prob = [(num_plts-w_sent)/num_plts for w_sent in weighted_sents]
		weighted_sents_prob = [w_sent/sum(weighted_sents_prob) for w_sent in weighted_sents_prob]		
		
	 	#we repeat 5% of plts in different positions
		num_plt_to_add = math.ceil((5*num_plts) / 100)

		#we repeat each selected plt in 33% of sentences of stories in the ROCStories dataset
		num_sent_to_insert_plt = num_sents // 3
		random_plt_inds = np.random.choice(num_plts, size=num_plt_to_add, replace=False)

		for random_plt_ind in random_plt_inds:
			#plt for repetition
			plt = list_plts[random_plt_ind]
			#we randomly select sentence positions that the selected plt would be repeated in those positions
			sent_inds_insert = np.random.choice(num_sents, size=num_sent_to_insert_plt , replace=False, p=weighted_sents_prob)
			for sent_ind_insert in sent_inds_insert:
				plts_sent = sents_plots[sent_ind_insert].split('\t')
				num_plts+=1
				weighted_sents[sent_ind_insert]+=1 
				#select a random position in the current selected sentence to repeat the selected plot
				pos_insertion = np.random.choice(len(plts_sent), size=1)[0]
				if pos_insertion ==0:
					pos_insertion+=1
				new_sents_plots[sent_ind_insert] = '\t'.join(sents_plots[sent_ind_insert].split('\t')[0:pos_insertion]) + '\t' + plt + '\t' + '\t'.join(sents_plots[sent_ind_insert].split('\t')[pos_insertion:]) 
			
		new_plots = '#'.join(new_sents_plots)
		if new_plots.startswith('\t'):
			new_plots = '\t'.join(new_plots.split('\t')[1:])
		return new_plots

	def notlogic_ordered(self, plots, text):
		#This function applies the "Non-logically Ordered" manipulation on the plots 
		sents_plots = plots.split('#')[:5]
		num_sents = len(sents_plots)
		new_sents_plots = sents_plots

		list_plts_verb = []
		list_plts_verb_tense = []      
	
		sents = text.split('</s>')[1:]
		sents_pos = {}
		for ind, sent in enumerate(sents):
			sent_tokens = nltk.word_tokenize(sent)
			pos_sent = nltk.pos_tag(sent_tokens)
			sents_pos[ind] = pos_sent

		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			for plt in sent_plots:
				plt = plt.strip()
				if ' ' not in plt:
					#what is the POS of the plot in its sentence
					pos_plt = self.get_pos_plt(plt, sents_pos[ind])
				else:
					plt_tokens = nltk.word_tokenize(plt)
					pos_plt_tokens = nltk.pos_tag(plt_tokens)
					#for the current ngram plot which tokens have verb type POS
					pos_plt = self.has_verb_tag(pos_plt_tokens)					
					
				if pos_plt != None and pos_plt in ['VBD', 'VB',  'VBZ', 'VBP'] and plt not in list_plts_verb:
					#We only select verb type plots to change their order 
					list_plts_verb.append(plt)
					#list_plts_verb_tense includes the tense of selected verb type plots
					list_plts_verb_tense.append(pos_plt)
				

		if list_plts_verb == []:
			#If there are no plots of type verb we can not apply Non-logically Ordered manipulation
			return plots
		num_plt_to_add = math.ceil((15*len(list_plts_verb)) / 100)

		#select a random list of verb type plots that we want to replace with other events
		random_plts = np.random.choice(list_plts_verb, size=num_plt_to_add, replace=False)
			
		for random_plt in random_plts:
			random_plt = random_plt.strip()
			ind_input_event_text = list_plts_verb.index(random_plt)
			tense_input_event_text = list_plts_verb_tense[ind_input_event_text]
			
			#get the verb from random_plt and change it to its simple tense 
			if ' ' in random_plt:
				#for an ngram plot it returns all the verbs that exist
				random_plt_tokens = nltk.word_tokenize(random_plt)
				random_plt_pos = nltk.pos_tag(random_plt_tokens)
				ind_verb_random_plt_pos = self.get_verb_pos(random_plt_pos)
				
				if len(ind_verb_random_plt_pos) != 1:
					#only select a plot from plot_list that has exactly one verb
					continue
				#get the simple tense of the verb in the plot
				random_plt_verb_simple_tense = nlp(random_plt_tokens[ind_verb_random_plt_pos[0]])[0]._.inflect('VB')
			
			else:
				random_plt_tokens = nltk.word_tokenize(random_plt)[0]
				random_plt_tokens = nlp(random_plt_tokens)	
				#get the simple tense of the verb in the plot
				random_plt_verb_simple_tense = random_plt_tokens[0]._.inflect('VB')
			
			#random select one relation type that has ordering sense
			relation = np.random.choice(['HasPrerequisite', 'Causes', 'HasFirstSubevent', 'HasLastSubevent'], size=1, replace=False)[0]
				
			#extract concepts that have specified relation with the random_plt (subject plot) in COMET
			outputs = interactive.get_conceptnet_sequence(random_plt, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation)
			for output in outputs:
				random_plot = nlp(random_plt)
				output_event = ''
				output = output.replace('your', '')
				output = output.replace('you', '')
				
				if  output != '' and random_plt_verb_simple_tense != None and random_plt_verb_simple_tense  not in output:#Object extracted from COMET should not include the verb which is in the subject plot
					if ' ' not in output and nlp(output)[0]._.inflect('VB') != None: #output is a verb type unigram
						#change output to the tense of subject plot
						output_event = nlp(output)[0]._.inflect(tense_input_event_text)
						output_event = output_event.strip()
					elif ' ' in output:
						output_tokens = nltk.word_tokenize(output)
						output_pos = nltk.pos_tag(output_tokens)
						ind_verb_output_pos = self.get_verb_pos(output_pos)
						if ind_verb_output_pos == [] or len(ind_verb_output_pos) != 1:
							#skip if none of the tokens in the object ngram is a verb or it has more than one verb
							continue
						#change the tense of object to be compatible with subject tense
						verb_chg_tense = nlp(output_tokens[ind_verb_output_pos[0]])[0]._.inflect(tense_input_event_text)
						if verb_chg_tense == None:
							continue
						output_tokens[ind_verb_output_pos[0]] = verb_chg_tense
						output_event = ' '.join(output_tokens).strip()
				
					else:
						continue								
					break
			if output_event == '':
				continue

			for ind in range(num_sents):
				if random_plt in sents_plots[ind].split('\t'):
					position_ind = sents_plots[ind].split('\t').index(random_plt)
					#concat_word = np.random.choice(['then', '', 'later', 'subsequently'], size=1)[0]
					concat_word = ''
					if relation == "HasPrerequisite" or relation== "HasFirstSubevent":
						if concat_word == '':
							changing_plots = random_plt  + '\t' + output_event		
						else:
							changing_plots = random_plt  + ' ' + concat_word + ' ' + output_event
						sents_plots[ind] = '\t'.join(sents_plots[ind].split('\t')[:position_ind]) + '\t' + changing_plots  + '\t' + '\t'.join(sents_plots[ind].split('\t')[position_ind+1:])
						
					elif relation == "Causes" or relation=="HasLastSubevent":
						if concat_word == '':  
							changing_plots = output_event  + '\t' + random_plt
						else:
							changing_plots = output_event  + ' '+ concat_word + ' ' + random_plt
						sents_plots[ind] = '\t'.join(sents_plots[ind].split('\t')[:position_ind]) + '\t' + changing_plots  + '\t' + '\t'.join(sents_plots[ind].split('\t')[position_ind+1:])
					new_sents_plots[ind] = sents_plots[ind]
					break
			
		new_sents_plots = '#'.join(new_sents_plots)
		if new_sents_plots.startswith('\t'):
			new_sents_plots = '\t'.join(new_sents_plots.split('\t')[1:])
		return new_sents_plots


	def plt_random_insertion(self,plots, set_plots):
		#This function applies the "Random Substitution" manipulation on the plots 
		sents_plots = plots.split('#')
		num_sents = len(sents_plots)
		new_sents_plots = sents_plots
		weighted_sents = []
		num_plots = 0
		short_sents_plots =[]
		for sent_plots in sents_plots:
			sent_plots = sent_plots.split('\t')
			sent_plots = [i for i in sent_plots if i]
			num_plots+=len(sent_plots)
			weighted_sents.append(len(sent_plots))
		
		weighted_sents = [(num_plots-w_sent)/num_plots for w_sent in weighted_sents]
		weighted_sents = [w_sent/sum(weighted_sents) for w_sent in weighted_sents]
		
		num_sents_insert_rand_plt = math.ceil((5 * num_sents) / 100)
		
		#select sentences to replace their plots with random plots
		ind_sents_insert = np.random.choice(num_sents, size=num_sents_insert_rand_plt , replace=False, p=weighted_sents)
			
		for ind_sent in ind_sents_insert:
			#select a random text to replace the current randomly selected plots with its plots
			rand_text_ind = np.random.choice(len(set_plots), size=1 , replace=False)[0]
			
			#select a sentence from the selected text
			list_sents = [i for i in range(num_sents-1)]
			rand_text_sent_ind = np.random.choice(list_sents, size=1 , replace=False)[0]
			
			rand_text_sent_plots = set_plots[rand_text_ind].split('#')[rand_text_sent_ind].split('\t')
			rand_text_sent_plots = [i for i in rand_text_sent_plots if i]
			
			while len(rand_text_sent_plots) <3:
				#selected sentence to be added to the plots should have at least 3 plots to have enough number of randomly substitued plots to show their effect
				list_sents= list(set(list_sents)-set([rand_text_sent_ind]))
				if list_sents == []:
					rand_text_ind = np.random.choice(len(set_plots), size=1 , replace=False)[0]
					list_sents = [i for i in range(num_sents-1)]
				rand_text_sent_ind = np.random.choice(list_sents, size=1 , replace=False)[0]
				rand_text_sent_plots = set_plots[rand_text_ind].split('#')[rand_text_sent_ind].split('\t')
				rand_text_sent_plots = [i for i in rand_text_sent_plots if i] 

			if len(rand_text_sent_plots) == 3:
				replace_plots = rand_text_sent_plots
			else:
				ind_replace_plots = np.random.choice(len(rand_text_sent_plots)-3, 1, replace=False)[0]
				replace_plots = rand_text_sent_plots[ind_replace_plots:ind_replace_plots+3]
			sents_plots[ind_sent] = ''	
			for plt in replace_plots:
				sents_plots[ind_sent] +='\t'
				sents_plots[ind_sent] = sents_plots[ind_sent].replace('\t\t', '\t')
				sents_plots[ind_sent] += plt+'\t'
			new_sents_plots[ind_sent] = sents_plots[ind_sent]

		new_sents_plots = '#'.join(new_sents_plots)
		if new_sents_plots.startswith('\t'):
			new_sents_plots = '\t'.join(new_sents_plots.split('\t')[1:])
		return new_sents_plots


	def insert_antonym(self, plots):
		#This function applies the "Contradiction Insertion" manipulation on the plots 
		sents_plots = plots.split('#')
		new_sents_plots = sents_plots
		plots_with_antonyms = []
		#the index of the sentences including plots with antonyms
		plts_positions_sents = {} 
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			sent_plots = [i for i in sent_plots if i]
			for plt in sent_plots:
				if plt in list(self.plt_antonyms.keys()) and plt not in plots_with_antonyms:
					plots_with_antonyms.append(plt) 
					plts_positions_sents[plt]=ind
		 
		num_plt_to_add = math.ceil((5*len(plots_with_antonyms)) / 100)
		random_plts = np.random.choice(plots_with_antonyms, size=num_plt_to_add, replace=False)
		
		for plt in random_plts:
			antonym_plt = np.random.choice(self.plt_antonyms[plt], size=1, replace=False)[0]
			position_ind = new_sents_plots[plts_positions_sents[plt]].split('\t').index(plt)
			new_sents_plots[plts_positions_sents[plt]] = '\t'.join(new_sents_plots[plts_positions_sents[plt]].split('\t')[:position_ind]) + '\t' + antonym_plt +'\t' + '\t'.join(new_sents_plots[plts_positions_sents[plt]].split('\t')[position_ind:])
		new_plots = '#'.join(new_sents_plots)	
		if new_plots.startswith('\t'):
			new_plots = '\t'.join(new_plots.split('\t')[1:])
		return new_plots	


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--COMET_model_file", type=str, default="pretrained_models/conceptnet_pretrained_model.pickle")
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--COMET_sampling_algorithm", type=str, default="beam-10")
	parser.add_argument("--data_dir", type=str, default="Data/ROC/ROC_Eval")
	parser.add_argument("--fname", type=str, default="Rocstories_train")

	args = parser.parse_args()

	fr_stories = open(os.path.join(args.data_dir, args.fname), 'r')
	lines = fr_stories.readlines()
	stories = []
	set_plots =[]

	for line in lines:
		storyplots =  line.split('<EOT>')[1].split('<EOL>')[0].strip()
		story_content =  line.split('<EOL>')[1].split('@')[0].strip()
		stories.append(story_content)
		set_plots.append(storyplots)
		
	plt_changes = Plt_manipulations(args.COMET_model_file, args.COMET_sampling_algorithm, args.device)
	
	num_gens= 1
	output_file = args.fname +'_manipulated_plts'
	if not os.path.isdir(args.data_dir+'ManPlts/'):
		os.mkdir(args.data_dir+'ManPlts/')
	fw_plts = open(os.path.join(args.data_dir+'ManPlts/', output_file), 'w')
	
	
	for ind, story_plots in enumerate(set_plots):
		print('******************{}****************'.format(ind))
		print('STORY PLOTS:\t'+story_plots)
		print('STORY:\t'+stories[ind])
		for i in range(num_gens):
			manipulated_story_plts = story_plots
			num_changes = np.random.choice([2,3,4], size=1, replace=False)[0]	
			ind_technique_apply = np.random.choice([1,2,3,4], size=num_changes, replace=False)
			print('number of changes {}'.format(num_changes))
			print('the techniques to apply is {}'.format(ind_technique_apply))
			for tech_ind in ind_technique_apply:
				if tech_ind ==0:
					manipulated_story_plts = plt_changes.insert_antonym(manipulated_story_plts)
					print('after antonym insertion {}'.format(manipulated_story_plts))
				elif tech_ind ==1:
					manipulated_story_plts = plt_changes.repetition(manipulated_story_plts)
					print('after repetitions {}'.format(manipulated_story_plts))
				elif tech_ind ==2:
					manipulated_story_plts = plt_changes.notlogic_ordered(manipulated_story_plts, stories[ind])
					print('after noot logically order {}'.format(manipulated_story_plts))
				elif tech_ind ==3:
					manipulated_story_plts = plt_changes.plt_random_insertion(manipulated_story_plts, set_plots)
					print('after random isertion {}'.format(manipulated_story_plts))
			print(manipulated_story_plts)
			fw_plts.write(manipulated_story_plts.strip() + '\n')
		print('_________________________________________')

