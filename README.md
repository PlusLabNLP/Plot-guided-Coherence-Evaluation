# Plot-guided-Coherence-Evaluation

This repository contains code for the [Plot-guided-Coherence-Evaluation](https://aclanthology.org/2021.naacl-main.343/) paper. For citation please use the following citation:
```
@inproceedings{ghazarian2021plot,
  title={Plot-guided Adversarial Example Construction for Evaluating Open-domain Story Generation},
  author={Sarik Ghazarian and Zixi Liu and Akash S M and Ralph Weischedel and Aram Galstyan and Nanyun Peng},
  booktitle={The 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  paper_url={https://www.aclweb.org/anthology/2021.naacl-main.343},
  pages={4334–-4344},
  slides_url={https://underline.io/events/122/sessions/4241/lecture/19650-plot-guided-adversarial-example-construction-for-evaluating-open-domain-story-generation},
  year={2021}
}
```

### Install Requirements
Please use requirements.txt file to get all the necessary packages to run the code. In our manipulations, we use COMET model to manipulate the logically ordered plots. You can download the model from https://github.com/atcbosselut/comet-commonsense. Download and save the model in a new directory called "pretrained_models".

## Data Creation, the evaluators training and testing steps
1. Our proposed four different approaches including Non-logically Ordered Plots, Contradiction Insertion, Repetition Insertion and Random Substitution manipulations can be applied by:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'python storyline_manipulation_WP.py --data_dir Data/WP/WP_Eval  --fname WP_train' 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python storyline_manipulation_ROC.py --data_dir Data/ROC/ROC_Eval/ --fname Rocstories_train

Note: To run these two codes, you need to first install https://github.com/atcbosselut/comet-commonsense and then put these codes in the home directory and run from there to get the manipulated plots.

2. In order to generate implausible stories conditioned on the manipulated plots, we use BART model as a conditional LM. We have finetuned BART on both ROC_LM and WP_LM data for three epochs using [Fairseq](https://github.com/pytorch/fairseq). You can download these models from [ft_BART_ROC](https://drive.google.com/file/d/1QaeZSGk9JFygUCOIacC_1Sm3tjXB_zQC/view?usp=sharing) and [ft_BART_WP](https://drive.google.com/file/d/1QFSlDUsK5nhIHVf_Wo8eifdVDKhODV0H/view?usp=sharing). The BART model finetuned on ROCstories dataset should be placed in Models/Ft_BART_Story_Generator/ROC/ while the finetuned BART model on the WP dataset should be placed in Models/Ft_BART_Story_Generator/WP/. 


3. We leverage the finetuned BART models to generate 6 different negative samples for each plausible story and then use the [Adversarial Filtering (AF) technique](https://arxiv.org/abs/1905.07830) proposed by Zellers et al. (2019) to select the three most challenging implausible ones for the evaluator. To generate negative samples and make the data ready for applying AF technique run:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python make_AF_input_WP.py --num_negative_samples 6

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python make_AF_input_ROC.py --num_negative_samples 6

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can set different generation parameters for generating various implausible stories.

4. We follow the code for [AF]( https://github.com/rowanz/hellaswag) on the [Data/WP/WP_Eval/WP_AF_input.json](https://drive.google.com/file/d/1z5_lP_yEM0iilyqtKqqfZg9TAXsNPIpP/view?usp=sharing) and Data/ROC/ROC_Eval/ROC_AF_input.json data to select the challenging implausible stories.

5. The output from AF technique is in json format. We convert it to tsv format which is a suitable input format for our evaluators. In this format, we have one plausible story with the label "1" and three implausible stories with the label "0".

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python make_tsv_input_WP.py 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python make_tsv_input_ROC.py

6. We use the run_glue.py code from [huggingface](https://github.com/huggingface/transformers) to finetune RoBERTa model for ROCstories and Longformer for WP dataset. You can download the evaluators from [ft_roberta](https://drive.google.com/file/d/1_byN9elrTO2J_nEGAnIjoBoYLZfCpsAp/view?usp=sharing) and [ft_longformer](https://drive.google.com/file/d/1Go25mHaKK4yYcfADK9qbtJUIZL0MPTch/view?usp=sharing). These models should be placed in Models/Ft_RoBERTa/ and Models/Ft_Longformer/ directories respectively. We also use run_glue.py code to predict the scores for the test data.


7. In order to examine the performance of our evaluators, we have collected human judgments through AMT. Data/AMT/AMT_ROC.csv and Data/AMT/AMT_WP.csv files consist of these human evaluations. To get the Spearman and Kendall correlations between predicted scores using our evaluators and human judgments you can run:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python Pearson_corr.py --data ROC

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python Pearson_corr.py --data WP



For any comments or issues feel free to contact [me](sarikgha@usc.edu).
