# cv-projects

Some commands to run training:

```
python train.py --num_epochs 10 --batch_size 4 --lr_scheduler fixed --lr 0.001 --run_name cpu_run_on_image_net
python train.py --num_epochs 100 --batch_size 1024 --lr_scheduler 0.001 --run_name image_net_train_deeper_network_and_dropout --device cuda
python train.py --num_epochs 10 --batch_size 32 --lr_scheduler 0.001 --run_name cpu_run_on_image_net
```

## To continue training on cpu test

```
python train.py --num_epochs 10 --batch_size 4 --lr_scheduler fixed --lr 0.01 --continue_from_checkpoint_run_id IM-74 --continue_from_checkpoint_path checkpoints/epoch_5 --log_interval 1
```

## To continue training on gpu test

```
python train.py --num_epochs 100 --batch_size 512 --lr_scheduler 0.001 --continue_from_checkpoint_run_id IM-50 --continue_from_checkpoint_path checkpoints/epoch_5
python train.py --num_epochs 10 --batch_size 32 --lr_scheduler 0.001 --continue_from_checkpoint_run_id IM-50 --continue_from_checkpoint_path checkpoints/epoch_5
```
