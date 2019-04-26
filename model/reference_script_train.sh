
###########################################
# train - CUDA_VISIBLE_DEVICES=0
###########################################

CUDA_VISIBLE_DEVICES=0 python train_clip.py --batch_size 512 --mem_dim 300 --lr 0.001 --dr 0.7 --num_train_steps 100000 --graph_prefix 'compAggr_clip_Ploss' --is_save 0 --clip 0 --corpus 'RQE'


###########################################
# train - LTC
###########################################

CUDA_VISIBLE_DEVICES=0 python train_clip.py --batch_size 512 --mem_dim 300 --lr 0.001 --dr 0.7 --num_train_steps 100000 --graph_prefix 'compAggr_clip_Ploss' --is_save 0 --clip 0 --corpus 'RQE' --ltc 1 --ltc_t 2 --ltc_mem 100 --ltc_dr 0.8

###########################################
# train - SquadT transfer - just example
###########################################

CUDA_VISIBLE_DEVICES=0 python train_clip.py --batch_size 30 --mem_dim 300 --lr=0.001 --dr 0.7 --num_train_steps 100000 --graph_prefix 'SquadT_compAggr_clip_Ploss' --is_save 0 --clip 1 --corpus 'SquadT_WikiQA' --ltc 1 --ltc_t 2 --ltc_mem 100 --ltc_dr 0.8 --pre_model './save/SquadT_premodel/'