# BERTFinanceNeg

BERT-based Finance Negation Detection
* Baseline for [金融信息负面及主体判定, CCF Big Data & Computing Intelligence Contest, CCF BDCI](https://www.datafountain.cn/competitions/353)
* [Chen Zhang](https://genezc.github.io).

## Requirements

* Python 3.6
* PyTorch 1.0.0
* pytorch-pretrained-bert

## Usage

* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python train.py --model_name bert --batch_size 16 --save True 
```
* Infer with [infer.py](/infer.py)

## Model

An overview of the BERT-based baseline is given below

![model](/assets/bert_spc.png)

## Credits

* For any issues or suggestions about this work, don't hesitate to create an issue or directly contact me via [gene_zhangchen@163.com](mailto:gene_zhangchen@163.com) !