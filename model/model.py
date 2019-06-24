#-*- coding: utf-8 -*-
"""
what    : MLP
data    : ..
"""
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical

from tensorflow.core.framework import summary_pb2
import numpy as np

class SimpleMLP:
    
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
   
           
    def _dense(self, x, size, scope):
        return tf.contrib.layers.fully_connected(x, size, 
                                                 activation_fn=None,
                                                 scope=scope)

    def _dense_batch_relu(self, x, size, phase, scope):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(x, size, 
                                                   activation_fn=None,
                                                   scope='dense')
            h2 = tf.contrib.layers.batch_norm(h1, 
                                              center=True, scale=True, 
                                              is_training=phase,
                                              scope='bn')
            return tf.nn.relu(h2, 'relu')


    def _apply_MLP(self):
        print('[INFO] MLP')
        
        self.encoding = self.batch_d
        
        for i in range(self.params.NUM_LAYERS):
            
            self.encoding = self._dense_batch_relu(
                                                    x = self.encoding, 
                                                    size = self.params.NUM_HIDDEN, 
                                                    phase = True,
                                                    #phase = self.phase,
                                                    scope='L'+str(i+1)
                                                    )
            
        self.final_encoding = self.encoding
        
        
    def _create_output_layers(self):
        print('[launch] create output projection layer')
        
        with tf.name_scope('text_output_layer') as scope:

            self.output      = self._dense(self.final_encoding, self.params.N_CLASS, 'logits')
            
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
        
        self._apply_MLP()
        self._create_output_layers()
        
        self._create_optimizer()
        self._create_summary()
        
        