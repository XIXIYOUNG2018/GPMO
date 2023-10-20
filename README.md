![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
# Molecular Optimization by Capturing Chemist's Intuition Using Deep Neural Networks
## Description
Implementation of the Seq2Seq with attention and the Transformer used in [Molecular Optimization by Capturing Chemist's Intuition Using Deep Neural Networks](https://chemrxiv.org/articles/preprint/Molecular_Optimization_by_Capturing_Chemist_s_Intuition_Using_Deep_Neural_Networks/12941744).
Given a molecule and desirable property changes, the goal is to generate molecules with desirable property changes. This problem can be viewed as a machine translation problem in natural language processing. Property changes are incorporated into input together with SMILES. 


## Usage
Create environment 

```
conda env create -f environment.yml
source activate gpmo
```
**1. Preparation before training the model**

create folder, download pre-training checkpoint, and pre-process customered data before training.
***1.1 create fold for checkpoint of pre-training and finetune  and  evaluation***
```
cd experiments
mkdir evaluation_transformer #create folder for evaluation
mkdir train_transformer/checkpointpretrain  #create folder for pretraining checkpoint
then download the pre-training checkpoint from here() and put it into the pretraining checkpoint
mkdir  train_transformer/checkpoint #create folder for finetune checkpoint.
```
***1.2 If you want to train the model with customered data***
If you use the data provide in data/chembl_02, skip this step.
 Encode property change, build vocabulary, and split data into train, validation, and test. Outputs are saved in the same directory with input data path.
```
python preprocess_prop.py --input-data-path data/data_name/data.csv
```
**2. Train model**

 Train the model and save results and logs to `experiments/save_directory/`; The model from each epoch is saved in 
`experiments/save_directory/checkpoint/`; The training loss, validation loss and validation accuracy are saved in `experiments/save_directory/tensorboard/`.
```
python trainper.py --data-path data/chembl_02 --save-directory train_transformer --model-choice transformer transformer
``` 
A pre-trained Transformer model can be found [here](https://zenodo.org/record/5571965#.YWmMoZpBybi).

**3. Generate molecules**

Use the model saved at a given epoch (e.g. 60) to generate molecules for the given test filename, and save the results to `experiments/save_directory/test_file_name/evaluation_epoch/generated_molecules.csv`. The three test sets used in our paper can be found in `data/chembl_02/` as below,

- Test-Original ->` data/chembl_02/test.csv`

```
python generate.py --model-choice transformer --data-path baseline_dataset_modof/ --test-file-name test --model-path experiments/train_transformer/checkpoint --save-directory evaluation_transformer --epoch 60
```   
**4. Compute properties for generated molecules**

Since we build the property prediction model based on the in-house experimental data, we can't make it public. But the computed properties can be found in `experiments/evaluation_transformer/test_file_name/evaluation_60/generated_molecules_prop.csv`

**5.Evaluate the generated molecules in term of satisfying the desirable properties and draw molecules**
```

python evaluate.py --data-path experiments/evaluation_transformer/test/evaluation_60/generated_molecules_prop.csv
```
**6. Matched molecular pair analysis between starting molecules and generated molecules**

- Download [mmpdb](https://github.com/rdkit/mmpdb) for matched molecular pair generation
- Parse the downloaded mmpdb path (i.e. path/mmpdb/) to --mmpdb-path of mmp_analysis.py

Between starting molecules and all the generated molecules
```
python mmp_analysis.py --data-path experiments/evaluation_transformer/test/evaluation_130/generated_molecules.csv  --train-path data/chembl_02/train_dmo_prop_encoded.csv --mmpdb-path path/mmpdb/
```

Between starting molecules and all the generated molecules with desirable properties
```
python mmp_analysis.py --data-path experiments/evaluation_transformer/test/evaluation_130/generated_molecules_statistics.csv --train-path data/chembl_02/train.csv --mmpdb-path path/mmpdb/ --only-desirable
```
### License
The code is copyright 2020 by Jiazhen He and distributed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.
