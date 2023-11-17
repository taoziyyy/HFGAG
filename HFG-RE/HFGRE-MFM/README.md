# Construction of a Hair Follicle Growth Association Gene Dataset Using Text-Mining
***

## Introduction
MOTIVATION: Accurate knowledge of key genes that promote hair follicle growth and development is of great value in
the field of hair research and dermatology. Compared with the traditional time-consuming and laborious experimental
methods for obtaining key genes, literature mining methods can extract proven key genes for hair follicle growth from
the vast amount of literature more rapidly and comprehensively. In detail, such methods extract key genes by performing
two major tasks: named entity recognition (NER) of related entities and relationship extraction (RE) between entities.
However, existing literature mining techniques for automatic information extraction of hair follicle growth-associated
genes are fewer in number, and suffer from the lack of standardized annotated datasets and the lack of models effectively
adapted to the above two major tasks. Therefore, we created a labeled corpus containing 500 literature abstracts and
proposed a model for extracting key genes related to hair follicle growth based on literature mining to address the above
problems.
***

## Prerequisition
### Basics
HFGRE-MFM requires Python 3.7.  

```shell
pip install pytorch
pip install pandas
pip install numpy  
```

To check the version of Tensorflow you have installed:  

```shell
python -c 'import tensorflow as tf; print(tf.__version__)'
```
To do testing using trained models, CPU will suffice. To train a new model, a high-end GPU and the GPU version of Tensorflow is needed. To install the GPU version of tensorflow:
```shell
pip install pytorch==1.7.1
pip install pytorch-transformers==1.2.0
conda install pytorch-transformers==1.2.0
conda install torchvision==0.8.2
//安装cudnn
conda install cudnn=7.6.3
```

## Quick Start

## Build a HFGRE-MFM(PubMedBERT-Attention) Model
```shell
	python train_pubmedbert_attention.py --ab_fn ../train/abstracts.txt --stc_fn ../train/sentences.txt --ner_fn ../train/anns.txt --label_fn ../train/labels.csv --output_dir ../model/renet/ --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --batch_size 5 --epochs 20
```
## Build a GPT2 Model
```shell
	#python -m torch.distributed.launch --nproc_per_node=4 train_GPT2.py --ab_fn ../train/abstracts.txt --stc_fn ../train/sentences.txt --ner_fn ../train/anns.txt --label_fn ../train/labels.csv --output_dir ./Model/renet/ --world_size 4
    python train_GPT2_noDDP.py --ab_fn ../train/abstracts.txt --stc_fn ../train/sentences.txt --ner_fn ../train/anns.txt --label_fn ../train/labels.csv --output_dir ../model/renet/ --model_name "gpt2" --batch_size 5 --epochs 20
    python -m torch.distributed.launch --nproc_per_node=4 train_GPT3.py --ab_fn ../train/abstracts.txt --stc_fn ../train/sentences.txt --ner_fn ../train/anns.txt --label_fn ../train/labels.csv --output_dir ./model/renet/ --world_size 4
```

### Validate test results
```shell
  python test_pubmedbert_attention.py --ab_fn ../predication_data2/abstracts.txt --stc_fn ../predication_data2/sentences.txt --ner_fn ../predication_data2/anns.txt --label_fn ./predication_data2/labels.csv --output_dir ../model/renet/ --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" --batch_size 5 --epochs 1
```
***
## Understand Output File
There are four columns in the outputfile classification_result.txt:  

1 | 2 | 3                  | 4 | 
--- | --- |--------------------| --- | 
Article PubMed Id | Gene ID (Entrez) | Follicle Id (MESH) | Predict 

The column Predict shows the predict result of HFGAG, 1 means HFGAG believes it is a true associaton, 0 otherwise.

