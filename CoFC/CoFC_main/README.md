# Conjugation-aware Fully Connected Graph Neural Networks for Molecular Property Prediction  
  
## CoFC is  a graph neural network model, combining spatial and conjugation information.  
  
## Dependencies  
To successfully run this project, the following required libraries and dependencies must be installed on your system:  
```bash  
pip install paddle  
pip install pahelix  
pip install rdkit
```  
## Pre-training data  
We leveraged the data provided by [PubChemQC](https://chibakoudai.sharepoint.com/:u:/s/stair02/Ed9Z16k0ctJKk9nQLMYFHYUBp_E9zerPApRaWTrOIYN-Eg) for pre-training. Download dataset and using following command to tranform pretrainning data:  
```bash  
mkdir pretrain_data

cd pretrain_data

tar xzf pubchemqc_jcim2017_jsons.10150017a15274edd1e5ed06ad5831de.tar.gz 

mkdir -p json1 && find json/ -maxdepth 1 -name "*.tar.gz" | xargs -I {} tar -xzf {} -C json1

python -m pretrain_data.py
```  
## Spatial pre-training
Use the following command to run spitial pretrain.(note that the "init_model" should past the address of best 
spatial pretrain parameters)
```bash  
sh spitial_pretrain.sh
```  
## Conj pre-training
Use the following command to run conj pretrain .
```bash  
sh conj_pretrain.sh
``` 
We also supplement pretrained model in:  
```bash  
CoFC/pretrain_models/best.pdparams
``` 
  
## Fine-tuning the Model  
For the fine-tune the model on the downstream tasks, you can use the command. This command already load pretrain model :  
```bash  
bash finetune_class.sh 
bash finetune_rege.sh 
```  
The results are show in:
```bash  
 CoFC/log/pre-{data name}/final_result
```
 
