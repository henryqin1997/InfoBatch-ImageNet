#python3 prune_experiment_unsup.py -a resnet18 --dist-url 'tcp://127.0.0.1:12346' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /tmp/data/IN1K -b 1024 --lr 3.8 ;
python3 prune_experiment_unsup.py -a resnet50 --dist-url 'tcp://127.0.0.1:12346' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /tmp/data/IN1K -b 1024 --lr 6.4 --epochs 200 > r50_unsup_log_200epoch.txt ;
python3 /mnt/bn/dc-in-nas/zhanka.py