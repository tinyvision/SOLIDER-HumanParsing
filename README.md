# SOLIDER on [Human Parsing]

This repo provides details about how to use [SOLIDER](https://github.com/tinyvision/SOLIDER) pretrained representation on human parsing task.
We modify the code from [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing), and you can refer to the original repo for more details.

## Installation and Datasets

Details of installation and dataset preparation can be found in [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing).

## Prepare Pre-trained Models
Step 1. Download models from [SOLIDER](https://github.com/tinyvision/SOLIDER), or use [SOLIDER](https://github.com/tinyvision/SOLIDER) to train your own models.

Steo 2. Put the pretrained models under the `pretrained` file, and rename their names as `./pretrained/solider_swin_tiny(small/base).pth`

## Training
Train with single GPU or multiple GPUs:

```shell
sh train_swin.sh
```

## Performance

| Method | Model | LIP(MIoU) |
| ------ | :---: | :---: | 
| SOLIDER | Swin Tiny | 57.41 | 
| SOLIDER | Swin Small | 60.21 | 
| SOLIDER | Swin Base | 60.50 | 

- We use the pretrained models from [SOLIDER](https://github.com/tinyvision/SOLIDER).
- The semantic weight we used in these experiments is 0.8.

## Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{chen2023beyond,
  title={Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks},
  author={Weihua Chen and Xianzhe Xu and Jian Jia and Hao Luo and Yaohua Wang and Fan Wang and Rong Jin and Xiuyu Sun},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
