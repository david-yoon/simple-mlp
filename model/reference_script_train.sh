
###########################################
# Emobase
###########################################

CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 1024 --lr 0.001 --hidden 500 --num_layer 3 --lr 0.001 --dr 1.0 --num_train_steps 100000 --graph_prefix 'bigcomp' --is_save 0 --corpus 'bigcomp' --data_path '../data/target/fold_1/'