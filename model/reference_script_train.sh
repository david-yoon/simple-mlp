
###########################################
# MLP
###########################################

CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 1024 --lr 0.001 --hidden 256 --num_layer 2 --lr 0.001 --dr 1.0 --num_train_steps 100000 --graph_prefix 'bigcomp_emobase' --is_save 0 --corpus 'bigcomp' --data_path '../data/target/emobase/'


###########################################
# CNN
###########################################
CUDA_VISIBLE_DEVICES=0 python train.py --model 'cnn' --cnn_filters 16 --cnn_stride 10 --batch_size 256 --lr 0.001 --lr 0.001 --dr 1.0 --num_train_steps 100000 --graph_prefix 'bigcomp_emobase' --is_save 0 --corpus 'bigcomp' --data_path '../data/target/emobase/'