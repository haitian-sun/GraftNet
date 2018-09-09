# GraftNet

This is the implementation of GraftNet described in EMNLP 2018 paper [Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text](https://arxiv.org/abs/1809.00782).

### Prerequisites
The recommended way to install the required packages is using Conda and the provided `environment.yml` file. See [this](https://conda.io/docs/user-guide/install/index.html) page on how to install conda. Create the environment by running the following command:
```
conda env create --name=graftnet --file=environment.yml
```

Then activate the environment using:
```
source activate graftnet
```

### Downloads
Pre-processed datasets:
1. [WikiMovies](http://curtis.ml.cmu.edu/datasets/graftnet/data_wikimovie.zip)
2. [WebQuestionsSP](http://curtis.ml.cmu.edu/datasets/graftnet/data_webqsp.zip)

Pre-trained models:
1. [WikiMovies](http://curtis.ml.cmu.edu/datasets/graftnet/model_wikimovie.zip)
2. [WebQuestionsSP](http://curtis.ml.cmu.edu/datasets/graftnet/model_webqsp.zip)

### Demo
This repo contains pretrained models with the full text corpus and knowledge base, and generated datasets with incomplete knowledge base (10%, 30%, 50%). The default folder structure is as follows:

```
GraftNet
├── *.py
├── config
├── datasets
    ├── webqsp
    └── wikimovie
└── model
    ├── webqsp
    └── wikimovie

```

 - ./config: configuration files for Wikimovies and WebQuestionsSP
 - ./model: pretrained models on full text corpus and knowledge base 
 - ./datasets: retrieved text from full text corpus and sampled knowledge base 


 
To reproduce the result, (1) download data and pre-trained model, and save them under ./model and ./datasets folders respectively, and (2) change the following values in the *.yml file:

 - data_folder: Folder in ./datasets you would like to run
 - to_save_model: True if you would like to save the best model
 - save_model_file: Path to save model
 - pred_file: Path to generate and save predictions
 - load_model_file: Path to load pretrained model. At training time, it will initialize your model with the model pointed by this value. Set it to "null" if you would like to train from scratch.
 - fact_dropout: [0, 1) for training. We won't use it at test time.
 - use_kb: True if use kb
 - use_doc: True if use doc

Then run the following commands:

Train:
```
python main.py --train config/wikimovie.yml
python main.py --train config/webqsp.yml
```
Test:
```
python main.py --test config/wikimovie.yml
python main.py --test config/webqsp.yml
```
Evaluate:
```
python script.py wikimovie KB_PRED DOC_PRED HYBRID_PRED
python script.py webqsp KB_PRED DOC_PRED HYBRID_PRED
```
where *_PRED are the "pred_file" in the .yml file under different settings (with different combinations of "use_doc" and "use_kb").

### Contributors
If you use this code please cite the following:

Sun, H., Dhingra, B., Zaheer, M., Mazaitis, K., Salakhutdinov, R., & Cohen, W. W. (2018). Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text. EMNLP.
```
@article{sun2018open,
  title={Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text},
  author={Sun, Haitian and Dhingra, Bhuwan and Zaheer, Manzil and Mazaitis, Kathryn and Salakhutdinov, Ruslan and Cohen, William W},
  journal={EMNLP},
  year={2018}
}
```
