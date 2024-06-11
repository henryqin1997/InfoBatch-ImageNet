python3 lrfinder.py -a resnet101 -b 1024 --dist-url 'tcp://127.0.0.1:12346' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /tmp/data/IN1K > res101lrfinderlog.txt ;
echo "code end, output at res101lrfinderlog.txt" ;
python3 /mnt/bn/dc-in-nas/zhanka.py