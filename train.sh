# ViT-S+ICS
#CUDA_VISIBLE_DEVICES=0 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 --conv-stem -pp ../../model/vit_small_ics_cfs_lup.pth --logs-dir ../../log/cluster_contrast_reid/msmt17/vit_small_ics_cfs_lup --eval-step 50

# VIT-S
CUDA_VISIBLE_DEVICES=0 python cluster_contrast_train_usl.py -b 256 -a vit_small -d market1501 --iters 200 --eps 0.6 --self-norm \
--use-hard --hw-ratio 2 --num-instances 8 --temperature 1.0 --layers 1 --k 16 --w 1.0 \
-pp /home/ps/D/data-1/hqs/2023_v2/vit_small_cfs_lup.pth --logs-dir ./log_market1


CUDA_VISIBLE_DEVICES=0 python cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm \
--use-hard --hw-ratio 2 --num-instances 8 --temperature 1.0 --layers 1 --k 16 --w 1.0 \
-pp /home/ps/D/data-1/hqs/2023_v2/vit_small_cfs_lup.pth --logs-dir ./log_msmt1


# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp /mnt1/michuan.lh/log/dino/lup_filter/deit_small_251w_forget/checkpoint.pth --logs-dir ../../log/cluster_contrast_reid/msmt17/vit_small_cfs_lup/checkpoint --eval-step 50
