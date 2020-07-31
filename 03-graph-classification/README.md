# Transfer Learning for Molecular Property Prediction 

This tutorial consists of i) Pretraining with multi regression task on topological indices; ii) Finetuning for property prediction task. We use transfer learning because data for the primary, property prediction, task is usually highly imbalanced. In other words, there is an abundance of data with label 0 and lack of data with label 1. In order to overcome this class imbalance, we pretrain with multi regression task on prediction three set of topological indices, {wiener index, hyper wiener index, zagreb index}. Studies have found that there is a strong correlation between molecular topology and chemical properties such as boiling point, toxicity. For example, wiener index is highly correlated with boiling point. To transfer the model to property prediction task, we simply remove the last  output layer of the pretrained neural network and replace with  a new output layer. Because pretraining on multi regression task on topological indices is self supervised learning, it is also useful for large, unlabeled molecule data.  

## Table of Contents

* [Data](#data)
* [Pretraining](#pretraining)
* [Finetuning](#finetuning)
* [Model variants](#model_variants) 

## Data <a name="data"></a>
For pretraining, we use Aurora Fine Chemicals LLC database [Aurora Fine Chemicals LLC database](https://aurorafinechemicals.com) available on PubChem which has 34,304,433 molecules. Because this is a tutorial purpose, we used a small subset, 5000, of the original data. However, in real practice user should use a much larger data size than 5000 for pretraining purpose. It should be at least larger than the finetuning task data size. 
For finetuning, we directly use E.Coli dataset from <https://github.com/yangkevin2/coronavirus_data/blob/master/data/ecoli.csv>  Note that train, validation, test splits have already been specified in the linked repository. 


## Pretraining <a name="pretraining"></a>
Figure 1 shows that there is a clear difference between true positive and negative distributions, although the difference may be insignificant. Even though the difference is small and may be insignificant (we did not run t-test t confirm this), some signal, although a weak one, may exist. 

To pretrain and save the model, run:
~~~~
python pretrain.py -Ts -f <data_path> <wiener_index_path> <hyper_wiener_index_path> <zagreb_index_path>
~~~~
Where '<data_path>' is path to pickle file of pretraining data, '<wiener_index_path>', '<hyper_wiener_index_path>',  '<zagreb_index_path>' is path to .npy file of computed wiener index, hyper wiener index, zagreb index respectively. 

For example:
~~~~
python model.py -Ts -f data/5000_mols.pickle data/wiener_idx.npy data/hyper_wiener_idx.npy data/zagreb_idx.npy
~~~~

## Finetuning <a name="finetuning"></a>
During finetuning, the hope is that low level features learned from large, unlabeled data via self supervision on topological indices helps learning the primary task. 

To load pretrained model and finetune, run:
~~~
python finetuning.py -lF -f <data path> 
~~~
Where '<data_path>' is path to CSV file of the dataset. 

For example:
~~~
python finetuning.py -F -f data/ecoli.csv
~~~

## Model Variants <a name="model_variants"></a>
 In this tutorial, during finetuning phase we retrain the entire neural network (i.e., shared layers’ parameters + output layer’s parameters). However, if the primary, property prediction, task data is very small, user can also only re train on the last output layer. 

