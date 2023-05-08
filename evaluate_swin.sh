name='aline_swin-s_sota'
python evaluate.py --arch swin_small --data-dir /home/xianzhe.xxz/datasets/HumanParsing/LIP --model-restore ./logs/${name}/schp_4_checkpoint.pth.tar --input-size 572,384 --multi-scales 0.5,0.75,1.0,1.25,1.5 --flip
