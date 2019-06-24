#-*- coding: utf-8 -*-
"""
what    : compareAggregate
data    : 
"""
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical

from tensorflow.core.framework import summary_pb2
import numpy as np

class SimpleCNN:
    
    def __init__(self,
                 params=None
                ):
        
        self.params = params
        
        self.encoder_inputs = []
        self.encoder_seq_length =[]
        self.y_labels =[]

        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    

    def _create_placeholders(self):
        print ('[launch] placeholders')
        with tf.name_scope('text_placeholder'):
            
            # [batch,time_step]
            self.batch_d     = tf.placeholder(tf.float32, shape=[self.params.BATCH_SIZE, self.params.FEATURE_DIM], name="batch_data")
            
            self.batch_l     = tf.placeholder(tf.float32, shape=[self.params.BATCH_SIZE, self.params.N_CLASS], name="batch_label")
            self.dr_prob     = tf.placeholder(tf.float32, name="dropout")
            self.phase       = tf.placeholder(tf.bool, name='phase')
   
           

    def _apply_CNN(self):
        print('[INFO] CNN')
        
        self._pooled_outputs = []
        _num_filters  = self.params.NUM_FILTERS
        _kernel_sizes = self.params.LIST_KERNELS
        _stride       = self.params.STRIDE
        
        data_reshape = tf.reshape( self.batch_d, [self.params.BATCH_SIZE, -1, 1])
        
        for i, kernel_size in enumerate(_kernel_sizes):
            with tf.name_scope("conv-maxpool-%s" % kernel_size):
        
                cnn_1d = tf.layers.conv1d(
                                            inputs      = data_reshape,
                                            filters     = _num_filters,
                                            kernel_size = kernel_size,
                                            strides     = _stride,
                                            padding     = "SAME",
                                            activation  = tf.nn.relu,
                                            name        = "cnn"+str(i)
                                            )

                if (self.params.POOL == 'max'):
                    pooled = tf.layers.max_pooling1d(
                                                inputs      = cnn_1d,
                                                pool_size   = self.params.FEATURE_DIM,
                                                strides     = self.params.FEATURE_DIM,
                                                padding     = 'SAME',
                                                name        = "pool"+str(i)
                                            )
                elif (self.params.POOL == 'average'):
                    pooled = tf.layers.average_pooling1d(
                                                inputs      = cnn_1d,
                                                pool_size   = self.params.FEATURE_DIM,
                                                strides     = self.params.FEATURE_DIM,
                                                padding     = 'SAME',
                                                name        = "pool"+str(i)
                                            )
                else:
                    print("error in pooling")
                
                self._pooled_outputs.append( tf.squeeze(pooled) )
        
        num_filters_total = _num_filters * len(_kernel_sizes)
        self.pool_final   = tf.concat(self._pooled_outputs, -1)
        
        self.dense = tf.layers.dense(inputs=tf.squeeze(self.pool_final), units=32, activation=tf.nn.tanh)
            
        self.final_encoding = self.dense
        
        
    def _create_output_layers(self):
        print('[launch] create output projection layer')
        
        with tf.name_scope('text_output_layer') as scope:

            self.output = tf.layers.dense(inputs=tf.squeeze(self.final_encoding), units=self.params.N_CLASS, activation=None)
            
            self.batch_preds = tf.squeeze( tf.nn.sigmoid(self.output) )
            self.y_labels  = self.batch_l
    
            
        with tf.name_scope('loss') as scope:

            vars   = tf.trainable_variables()
            self.lossL2      = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * self.params.L2_LOSS_RATIO
            self.batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.batch_preds, labels=self.y_labels)
            self.loss       = tf.reduce_sum( self.batch_loss ) + self.lossL2
            
            
    def _create_optimizer(self):
        print('[launch] create optimizer')
        
        with tf.name_scope('optimizer') as scope:
        
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
        
                opt_func = tf.train.AdamOptimizer(learning_rate=self.params.LR)
                gradients = opt_func.compute_gradients(self.loss)
                capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gradients]
                self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)
            
    
    def _create_summary(self):
        print('[launch] create summary')
        
        with tf.name_scope('summary'):
            tf.summary.scalar('batch_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    
    def build_graph(self):
        
        self._create_placeholders()
        
        self._apply_CNN()
        self._create_output_layers()
        
        self._create_optimizer()
        self._create_summary()
        
        