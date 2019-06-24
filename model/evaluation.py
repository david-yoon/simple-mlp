#-*- coding: utf-8 -*-
"""
what    : evaluation
data    : ..
"""
from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

"""
    desc  : 
    
    inputs: 
        sess  : tf session
        model : model for test
        data  : such as the dev_set, test_set...
            
    return:
        sum_batch_ce : sum cross_entropy
        accr         : accuracy
        
"""
def run_test(sess, model, batch_gen, data):
    
    MAP_rst = 0
    MRR_rst = 0
    
    list_pred     = []
    list_label    = []
    list_loss      = []
    
    preds, labels  = [], []
    loss           = 0.0
    mean_loss      = 0.0
    
    max_loop  = int( len(data) / model.params.BATCH_SIZE )

    # run 1 more loop for the remaining
    for test_itr in range( int(max_loop + 1) ):
        
        list_d, list_l = batch_gen.get_batch(
                                            data=data,
                                            batch_size=model.params.BATCH_SIZE,
                                            is_test=True,
                                            start_index= (test_itr* model.params.BATCH_SIZE)
                                            )
        
        # prepare data which will be push from pc to placeholder
        input_feed = {}
        input_feed[model.batch_d] = list_d
        input_feed[model.batch_l] = list_l
        input_feed[model.phase]   = False
        input_feed[model.dr_prob] = 1.0
        
        try:
            preds, labels, loss = sess.run([model.batch_preds, model.y_labels, model.batch_loss], input_feed)
    
        except Exception as e:
            print ("excepetion occurs in valid step : ", e)
            pass
        
        
        # batch loss
        list_loss.append( loss )
    
        # batch accuracy    
        list_pred.extend( np.argmax(preds, axis=1) )
        list_label.extend( np.argmax(labels, axis=1) )
        
    list_pred     = list_pred[:len(data)]
    list_label    = list_label[:len(data)]
    list_loss     = list_loss[:len(data)]

    
    
    accr_class = precision_score(y_true=list_label,
                           y_pred=list_pred,
                           average=None)
        
    recall_class = recall_score(y_true=list_label,
                           y_pred=list_pred,
                           average=None)
        
    f1_class = f1_score(y_true=list_label,
                  y_pred=list_pred,
                  average=None)
    
    
    # weighted : ignore class unbalance (avg of each accr)
    # macro    : unweighted mean (take label imbalance into account)
    accr_avg = precision_score(y_true=list_label,
                           y_pred=list_pred,
                           average=model.params.ACCURACY_AVG)
        
    recall_avg = recall_score(y_true=list_label,
                           y_pred=list_pred,
                           average=model.params.RECALL_AVG)
        
    f1_avg = f1_score(y_true=list_label,
                  y_pred=list_pred,
                  average=model.params.F1_AVG)
    
#     accr_WA = precision_score(y_true=list_label,
#                            y_pred=list_pred,
#                            average='weighted')
    
#     accr_UA = precision_score(y_true=list_label,
#                            y_pred=list_pred,
#                            average='macro')

    result_zip = [accr_class, recall_class, f1_class, accr_avg, recall_avg, f1_avg]
    
    sum_loss = np.sum( list_loss )
        
    
    value1 = summary_pb2.Summary.Value(tag="dev_loss", simple_value=sum_loss)
    value2 = summary_pb2.Summary.Value(tag="dev_accr", simple_value=accr_avg )
    summary = summary_pb2.Summary(value=[value1, value2])

    
    return sum_loss, accr_avg, f1_avg, result_zip, summary