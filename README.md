# cv-projects

Some commands to run training:

```
python pretrain_yolo.py --num_epochs 10 --batch_size 4 --lr_scheduler fixed --lr 0.001 --dropout 0 --run_name cpu_run_on_image_net
python pretrain_yolo.py --num_epochs 100 --batch_size 1024 --lr_scheduler 0.001 --dropout 0 --run_name image_net_train_deeper_network_and_dropout --device cuda
python pretrain_yolo.py --num_epochs 10 --batch_size 32 --lr_scheduler 0.001 --dropout 0 --run_name cpu_run_on_image_net
```

## To continue training on cpu test

```
python pretrain_yolo.py --num_epochs 10 --batch_size 4 --lr_scheduler fixed --lr 0.001 --dropout 0 --run_name run_cont_from_IM_76 --continue_from_checkpoint_signature IM-76:checkpoints/epoch_5 --log_interval 1
```

## To continue training on gpu test

```
python pretrain_yolo.py --num_epochs 100 --batch_size 256 --lr_scheduler custom --lr 0.001 --dropout 0 --run_name run_cont_from_IM-50-with-slower_lr_decay_from_epoch_4 --continue_from_checkpoint_signature IM-50:checkpoints/epoch_4 --device cuda
python pretrain_yolo.py --num_epochs 10 --batch_size 32 --lr_scheduler fixed --lr 0.001 --dropout 0 --run_Name run_cont_from_IM-50 --continue_from_checkpoint_signature IM-50:checkpoints/epoch_5 --device cuda
```
