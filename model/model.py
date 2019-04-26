#-*- coding: utf-8 -*-
"""
what    : compareAggregate
data    : 
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
   
           
    def _MLP(self, in_tensor, out_dim, reuse=None, scope="mlp_scope", v_name="default-mlp"):
        print ('[launch] MLP (reuse, scope): ', reuse, scope)
        with tf.name_scope(scope):
            with tf.variable_scope(name_or_scope=scope, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):

                initializer = tf.contrib.layers.xavier_initializer(
                                                                uniform=True,
                                                                seed=None,
                                                                dtype=tf.float32
                                                                )

                layer_output = tf.contrib.layers.fully_connected( 
                                                            inputs = in_tensor,
                                                            num_outputs =  out_dim,
                                                            activation_fn = tf.nn.tanh,
                                                            normalizer_fn=None,
                                                            normalizer_params=None,
                                                            weights_initializer=initializer,
                                                            weights_regularizer=None,
                                                            biases_initializer=tf.zeros_initializer(),
                                                            biases_regularizer=None,
                                                            trainable=True,
                                                            reuse=reuse,
                                                            scope=scope+'mlp'
                                                        )
                #node_mlp_L1 = tf.nn.dropout( node_mlp_L1, keep_prob=self.dr_prob )

                return layer_output
            
            
    def _apply_MLP(self):
        print('[INFO] MLP')
        
        self.encoding = self.batch_d
        
        for i in range(self.params.NUM_LAYERS):
            
            self.encoding = self._MLP(
                                    self.encoding, 
                                    self.params.NUM_HIDDEN, 
                                    reuse=False, 
                                    scope='L'+str(i+1),
                                    v_name='MLP'
                                 )
            
        self.final_encoding = self.encoding
        
    def _create_output_layers(self):
        print('[launch] create output projection layer')
        
        with tf.name_scope('text_output_layer') as scope:
    
            self.W_output = tf.Variable(tf.random_normal([self.params.NUM_HIDDEN, self.params.N_CLASS],
                                                    mean=0.0,
                                                    stddev=0.01,
                                                    dtype=tf.float32
                                                   ),
                                               trainable = True,
                                               name='W_output')
            self.bias_output = tf.Variable(tf.zeros([self.params.N_CLASS]), name='bias_output')
            
            self.output      = tf.matmul(self.final_encoding, self.W_output, name="output") + self.bias_output
            
            self.batch_preds = tf.squeeze( tf.nn.sigmoid(self.output) )
            
            self.y_labels  = self.batch_l
    
            
        with tf.name_scope('loss') as scope:

            vars   = tf.trainable_variables()
            self.lossL2      = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * self.params.L2_LOSS_RATIO
            self.batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.batch_preds, labels=self.y_labels)
            self.loss       = tf.reduce_sum( self.batch_loss ) + self.lossL2
            
            
    def _create_optimizer(self):
        print('[launch] create optimizer')
        
        with tf.name_scope('optimizer') as scope:
            
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
        
        