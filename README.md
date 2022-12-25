# Part-of-Speech-Tagging-Multiple-Models
This repository holds the code for my thesis on "Active Learning and Part of Speech Tagging". 

This is the first of two parts of code, and explores different models used as Part of Speech Taggers. The second part can be found on https://github.com/Xenonas/Active-Learning-for-Part-of-Speech-Tagging/

After downloading the files, you need to also download word2vec pretrained model for english from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g if you wish toy use word embeddings.

The models used are: 
  - RNN
  - LSTM
  - Bi-LSTM
  - Bi-LSTM with CRF layer
  - CRF
  - HMM
  - n-grams
  
  By executing the files main.py, hmm.py and crf.py you can find the accuracy on english and greek on the Universal Dependencies EWT and GDT datasets.
