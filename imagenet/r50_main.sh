python3 main.py -a resnet50 --dist-url 'tcp://127.0.0.1:12347' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /tmp/data/IN1K -b 1024 --lr 6.4 --epochs 200 > r50_main_log_200epoch.txt ;
python3 /mnt/bn/dc-in-nas/zhanka.py