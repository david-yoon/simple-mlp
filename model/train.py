#-*- coding: utf-8 -*-

"""
what    : MLP, CNN
data    : ..
"""
import tensorflow as tf
import os
import time
import argparse
import datetime
import random

from model import *
from model_cnn import *
from process_data import *
from evaluation import *

from params import *

# for training         
def train_step(sess, model, batch_gen):
    
    list_d, list_l = batch_gen.get_batch(
                                    data=batch_gen.train,
                                    batch_size=model.params.BATCH_SIZE,
                                    is_test=False
                                    )

    # prepare data which will be push from pc to placeholder
    input_feed = {}
    input_feed[model.batch_d] = list_d
    input_feed[model.batch_l] = list_l
    input_feed[model.phase]   = True
    input_feed[model.dr_prob] = 1.0
    
    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
    
    return summary

def train_model(params, model, batch_gen, graph_dir_name='default', pre_model=''):
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    summary = None
    val_summary = None
    
    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter('./graph/'+graph_dir_name, sess.graph)
        sess.run(tf.global_variables_initializer())
                
        early_stop_count = params.MAX_EARLY_STOP_COUNT
        
        # if exists check point, starts from the check point
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pre_model))
        if ckpt and ckpt.model_checkpoint_path:
            print ('[load] pre_model check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)

        initial_time = time.time()
        
        min_ce = 1000000
        best_dev_f1 = 0
        test_f1_at_best_dev = 0
        test_zip_at_best_dev = None
        
        for index in range(params.NUM_TRAIN_STEPS):

            try:
                # run train 
                summary = train_step(sess, model, batch_gen)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except Exception as e:
                print ("excepetion occurs in train step", e)
                pass
                
            
            # run validation
            if (index + 1) % params.VALID_FREQ == 0:
                
                dev_ce, dev_accr, dev_f1, dev_zip, dev_summary = run_test(sess=sess,
                                                                   model=model, 
                                                                   batch_gen=batch_gen,
                                                                   data=batch_gen.dev
                                                                   )
                
                writer.add_summary( dev_summary, global_step=model.global_step.eval() )
                

                end_time = time.time()

                if index > params.CAL_ACCURACY_FROM:

                    if ( dev_ce < min_ce ):
                        min_ce = dev_ce

                        # save best result
                        if params.IS_SAVE is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )
                            
                        early_stop_count = params.MAX_EARLY_STOP_COUNT

                        if dev_accr > float(params.QUICK_SAVE_BEST):
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() ) 
                        
                        test_ce, test_accr, test_f1, test_zip, _ = run_test(sess=sess,
                                                                    model=model,
                                                                    batch_gen=batch_gen,
                                                                    data=batch_gen.test
                                                                    )
                        
                        best_dev_f1 = dev_f1
                        test_f1_at_best_dev = test_f1
                        test_zip_at_best_dev = test_zip
                        
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print ("early stopped")
                            break
                             
                        test_f1 = 0
                        early_stop_count = early_stop_count -1
                        
                        
                    train_ce, train_accr, train_f1, train_zip, _ = run_test(sess=sess,
                                           model=model,
                                           batch_gen=batch_gen,
                                           data=batch_gen.train
                                           )
                        
                        
                    print (str( int((end_time - initial_time)/60) ) + " mins" + \
                          " step/seen/itr: " + str( model.global_step.eval() ) + "/ " + \
                           str( model.global_step.eval() * model.params.BATCH_SIZE ) + "/" + \
                           str( round( model.global_step.eval() * model.params.BATCH_SIZE / float(len(batch_gen.train)), 2)  ) + \
                           "\tdev f1: " + '{:.3f}'.format(dev_f1)  + \
                           "  test f1: " + '{:.3f}'.format(test_f1) + \
                           "  dev loss: " + '{:.2f}'.format(dev_ce) + \
                           "  train f1: " + '{:.3f}'.format(train_f1))
                
        writer.close()

        print('Total steps : {}'.format(model.global_step.eval()) )
        
        
        # unzip final result
        accr_class, recall_class, f1_class, accr_avg, recall_avg, f1_avg = test_zip_at_best_dev
        
        print(accr_class, recall_class, f1_class)
        
        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    batch_gen.data_path.split('/')[-2] + '\t' + \
                    graph_dir_name + '\t' + \
                    str(accr_class[0]) + '\t' + \
                    str(recall_class[0]) + '\t' + \
                    str(f1_class[0]) + '\t\t' + \
                    str(accr_class[1]) + '\t' + \
                    str(recall_class[1]) + '\t' + \
                    str(f1_class[1]) + '\t\t' + \
                    str(accr_class[2]) + '\t' + \
                    str(recall_class[2]) + '\t' + \
                    str(f1_class[2]) + '\t\t' + \
                    str(accr_class[3]) + '\t' + \
                    str(recall_class[3]) + '\t' + \
                    str(f1_class[3]) + '\t\t' + \
                    str(accr_class[4]) + '\t' + \
                    str(recall_class[4]) + '\t' + \
                    str(f1_class[4]) + '\t\t' + \
                    str(accr_class[5]) + '\t' + \
                    str(recall_class[5]) + '\t' + \
                    str(f1_class[5]) + '\t\t' + \
                    str(accr_avg) + '\t' + \
                    str(recall_avg) + '\t' + \
                    str(f1_avg) + \
                    '\n')

    

def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(params, graph_dir_name):
    
    print(params.DATA_PATH)
    create_dir('save/')
    
    if params.IS_SAVE is 1:
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/'+graph_dir_name)

    batch_gen = ProcessData(params)
    
    if (params.MODEL=='mlp'):
        print("model: ", params.MODEL)
        model = SimpleMLP(params=params)
    elif (params.MODEL=='cnn'):
        print("model: ", params.MODEL)
        model = SimpleCNN(params=params)
        
    model.build_graph()
    
    params.VALID_FREQ = int( len(batch_gen.train) * params.EPOCH_PER_VALID_FREQ / float(params.BATCH_SIZE)  ) + 1
    
    print ("[Info] valid freq: ",        str(params.VALID_FREQ))
    
    train_model(params, model, batch_gen, graph_dir_name)
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=str, default='bigcomp')
    p.add_argument('--data_path', type=str, default='../data/target/')
    
    p.add_argument('--model', type=str, default='mlp')
    
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--num_layer', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dr', type=float, default=1.0)
    
    # cnn
    p.add_argument('--cnn_filters', type=int, default=16)
    p.add_argument('--cnn_stride', type=int, default=10)
    
    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    
    args = p.parse_args()
    
    if args.corpus == ('bigcomp'):
        params = Params()
        graph_name = 'bigcomp'
    
    params.MODEL     = args.model
    params.DATA_PATH = args.data_path
    
    # CNN
    params.NUM_FILTERS = args.cnn_filters
    params.STRIDE      = args.cnn_stride
    
    params.BATCH_SIZE = args.batch_size
    params.NUM_HIDDEN = args.hidden
    params.NUM_LAYERS = args.num_layer
    params.LR = args.lr
    params.DR = args.dr
    
    params.NUM_TRAIN_STEPS = args.num_train_steps
    params.IS_SAVE = args.is_save
    
    graph_name = args.graph_prefix + \
                    '_b' + str(params.BATCH_SIZE) + \
                    '_H' + str(params.NUM_HIDDEN) + \
                    '_L' + str(params.NUM_LAYERS) + \
                    '_dr' + str(params.DR)

    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")
    
    print('[INFO] model: ', params.MODEL)
    if ( params.MODEL == 'cnn'):
        print('[INFO] # filters : ', params.NUM_FILTERS)
        print('[INFO] list_kernels : ', params.LIST_KERNELS)
        print('[INFO] stride : ', params.STRIDE)
        print('[INFO] pool : ', params.POOL)
    print('[INFO] data_path: ', params.DATA_PATH)
    print('[INFO] batch_size: ', params.BATCH_SIZE)
    print('[INFO] hidden: ', params.NUM_HIDDEN)
    print('[INFO] layers: ', params.NUM_LAYERS)
    print('[INFO] learning rate: ', params.LR)
    print('[INFO] drop out: ', params.DR)
    print('[INFO] is save: ', params.IS_SAVE)

    main(
        params  = params,
        graph_dir_name = graph_name
        )