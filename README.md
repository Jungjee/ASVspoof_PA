ASVspoof2019 Physical Access
============================

#1. Overview
This repository contains modified pytorch codes for the paper 
Replay attack detection with complementary high-resolution information using end-to-end DNN for the ASVspoof 2019 Challenge.
(https://arxiv.org/abs/1904.10134, accepted for Interspeech 2019)

Our submission for the competition was conducted using Keras with Tensorflow backend. 
In this repository, we present easily reproducible codes with PyTorch. 
Although DNN architecture, and other detailed configurations might slightly differ (such as the negative slope value in Leaky ReLU),
we verified comaparable performance through our internal comparison experiments. 
(EER of 1.7% was shown in Keras, EER of 1.6 was shown in PyTorch, using only Magnitude spectrogram, on a partial subset of validation set comprising 5,400 bona-fide utt + 5,400 spoofed utts)

Also, for the experiment showing EER of 1.6%, we share a comet ml experiment, which enables all of you to look the details of conducted experiment! 

	* Experiment settings, libraries, and other configurations including the code, yaml configurations are accessible via

	  https://www.comet.ml/jungjee/public-comet-for-fithub/d463da20d0b84c70beeab387c94e8fee


To be added continuously..
***

#2. Script Usage
* g0 	:  Extracts spectrograms and align for later usage. 
* g1	:  Conducts model training and validation (CNN model, achieves ~ 4.2% EER on validation set)
	+ Set parameters via yaml file(.yaml)
* g2	:  Conducts model training and validation (CNN-GRU model, achieves ~ 1.7% EER on validation set)
	+ Set parameters via yaml file(.yaml)

#3. Acknowledgement.
* For the Keras-alike model summary, we used the implementation presented in https://github.com/Jungjee/pytorch-summary

 
