torchrun --standalone --nnodes=1 --nproc-per-node=1 DDP_train.py --use_amp --exp_name football \
--batch_size_per_gpu 256 \
--verbose info \
--total_iters 100000 \
--coeff 0.05 \
--alpha 1 \
--val_batch_size 128 \
--tb_port 7000