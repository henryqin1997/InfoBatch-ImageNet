python3 finetune_notprune.py -a resnet50 --dist-url 'tcp://127.0.0.1:12346' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /tmp/data/IN1K -b 1024 --lr 0.16 ;
python3 /mnt/bn/dc-in-nas/zhanka.py