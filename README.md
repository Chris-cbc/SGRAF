# SGRAF
PyTorch implementation for AAAI2021 paper of [**“Similarity Reasoning and Filtration for Image-Text Matching”**](https://drive.google.com/file/d/1tAE_qkAxiw1CajjHix9EXoI7xu2t66iQ/view?usp=sharing).  
It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) and [Cross-modal_Retrieval_Tutorial](https://github.com/Paranioar/Cross-modal_Retrieval_Tutorial).

## Introduction

**The framework of SGRAF:**

<img src="./fig/model.png" width = "100%" height="50%">

**The updated results (Better than the original paper)**
<table>
   <tr> <td rowspan="2">Dataset</td> <td rowspan="2", align="center">Module</td> 
        <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td rowspan="3">Flick30k</td>
        <td>SAF</td> <td>75.6</td><td>92.7</td><td>96.9</td> <td>56.5</td><td>82.0</td><td>88.4</td> </tr>
   <tr> <td>SGR</td> <td>76.6</td><td>93.7</td><td>96.6</td> <td>56.1</td><td>80.9</td><td>87.0</td> </tr>
   <tr> <td>SGRAF</td> <td>78.4</td><td>94.6</td><td>97.5</td> <td>58.2</td><td>83.0</td><td>89.1</td> </tr>
   <tr> <td rowspan="3">MSCOCO1k</td>
        <td>SAF</td> <td>--</td><td>--</td><td>--</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>SGR</td> <td>--</td><td>--</td><td>--</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>SGRAF</td> <td>--</td><td>--</td><td>--</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td rowspan="3">MSCOCO5k</td>
        <td>SAF</td> <td>--</td><td>--</td><td>--</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>SGR</td> <td>--</td><td>--</td><td>--</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>SGRAF</td> <td>--</td><td>--</td><td>--</td> <td>--</td><td>--</td><td>--</td> </tr>
  

   
</table> 

## Requirements 
We recommended the following dependencies.

*  Python **(2.7 not 3.\*)**  
*  [PyTorch](http://pytorch.org/) **(0.4.1)**  
*  [NumPy](http://www.numpy.org/) **(>1.12.1)** 
*  [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)  
*  Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data and vocab
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, which can be downloaded by using:

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
```

## Pre-trained models and evaluation
Modify the **model_path**, **data_path**, **vocab_path** in the `evaluation.py` file. Then run `evaluation.py`:

```bash
python evaluation.py
```

Note that `fold5=True` is only for evaluation on MSCOCO1K test set (5 folders average) while `fold5=False` for MSCOCO5K and Flickr30K. Pretrained models and Log files can be downloaded from [Flickr30K_SGRAF](https://drive.google.com/file/d/1OBRIn1-Et49TDu8rk0wgP0wKXlYRk4Uj/view?usp=sharing) and MSCOCO_SGRAF (soon).

## Training new models from scratch
Modify the **data_path**, **vocab_path**, **model_name**, **logger_name** in the `opts.py` file. Then run `train.py`:

For MSCOCO:

```bash
(For SGR) python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SGR
(For SAF) python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SAF
```

For Flickr30K:

```bash
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --lr_update 30 --module_name SGR
(For SAF) python train.py --data_name f30k_precomp --num_epochs 30 --lr_update 20 --module_name SAF
```

## Reference

If SGRAF is useful for your research, please cite the following paper:

    @inproceedings{Diao2021SGRAF,
      title={Similarity Reasoning and Filtration for Image-Text Matching},
      author={Diao, Haiwen and Zhang, Ying and Ma, Lin and Lu, Huchuan},
      booktitle={AAAI},
      year={2021}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


