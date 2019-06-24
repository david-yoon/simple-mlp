class Params():

    def __init__(self):
        print('bigcomp')
        self.name = "bigcomp"
        
    ################################
    #     dataset
    ################################     
    DATA_PATH    = ''
    
        
    ################################
    #     dataset
    ################################     
    DATA_TRAIN   = 'train.pkl'
    DATA_DEV     = 'dev.pkl'
    DATA_TEST    = 'test.pkl'
    
    ################################
    #     training
    ################################
    BATCH_SIZE = 10
    LR = 1e-3
    DR = 1.0
    
    CAL_ACCURACY_FROM      = 0         # run iteration without excuting validation
    MAX_EARLY_STOP_COUNT   = 8
    EPOCH_PER_VALID_FREQ   = 0.3
    VALID_FREQ             = 0
    QUICK_SAVE_BEST        = 0.755
    L2_LOSS_RATIO          = 2e-4     #2e-4, (3e-4)/2
    
    NUM_TRAIN_STEPS = 100000
    IS_SAVE = 0
    
    ################################
    #     model
    ################################
    MODEL        = 'mlp'
    
    N_CLASS      = 6
    FEATURE_DIM  = 100     # emobase 1582, compar 6373

    # MLP
    PAD_INDEX = 0
    NUM_HIDDEN = 100
    NUM_LAYERS = 1
    
    # CNN
    NUM_FILTERS  = 16
    LIST_KERNELS = [10, 50, 100, 200 ,300, 400]
    STRIDE       = 10
    
    
    ###############################
    # ETC
    ###############################
    IS_RESULT_LOGGING = False
    