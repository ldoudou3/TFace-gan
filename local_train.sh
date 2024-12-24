
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
python3 -u -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 train.py