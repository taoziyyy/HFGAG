***

## Prerequisition
### Basics
HFGAG requires Python 2.7.  
Make sure you have Tensorflow ≥ 1.10.0 installed, the following commands install the lastest CPU version of Tensorflow

```shell
pip install tensorflow
pip install pandas
pip install numpy  
```

To check the version of Tensorflow you have installed:  

```shell
python -c 'import tensorflow as tf; print(tf.__version__)'
```
To do testing using trained models, CPU will suffice. To train a new model, a high-end GPU and the GPU version of Tensorflow is needed. To install the GPU version of tensorflow:
```shell
pip install tensorflow-gpu           
//注意必须到当前虚拟环境
pip install tensorflow-gpu==1.10
conda install tensorflow-gpu=1.10
//安装cuda
conda install cudatoolkit=9.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
 
//安装cudnn
conda install cudnn=7.1.2
```

## Quick Start


### train
```shell
python ./renet/train.py --ab_fn train/abstracts.txt --stc_fn train/sentences.txt --ner_fn train/anns.txt --label_fn train/labels.csv --output_dir ./model/renet/
```
***
## test
```shell
cd ../testing_data
python ../renet/evaluate.py --model_dir ../model/renet/ --ab_fn abstracts.txt --stc_fn sentences.txt --ner_fn anns.txt --label_fn labels.csv --output_fn classification_result.txt
```
***
## Understand Output File
There are four columns in the outputfile classification_result.txt:  

1 | 2 | 3                  | 4 | 
--- | --- |--------------------| --- | 
Article PubMed Id | Gene ID (Entrez) | Follicle Id (MESH) | Predict 

The column Predict shows the predict result of HFGAG, 1 means HFGAG believes it is a true associaton, 0 otherwise.

***
Literature references and code URLs for this part of the code
Wu Y, Luo R, Leung H C M, et al. Renet: A deep learning approach for extracting gene-disease associations from literature[C]//Research in Computational Molecular Biology: 23rd Annual International Conference, RECOMB 2019, Washington, DC, USA, May 5-8, 2019, Proceedings 23. Springer International Publishing, 2019: 272-284.

https://github.com/alexwuhkucs/gda-extraction
***
