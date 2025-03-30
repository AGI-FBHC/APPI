
__all__ = ['Config']


class Config():
    seed = 2020
    MAP_CUTOFF = 14
    DIST_NORM = 15
    hidden = 256
    hops = 8
    num_layer = 1
    dropout = 0.1
    dropEdge = 0
    alpha = 0.7
    LAMBDA = 1.5
    excitation_rate = 1.0
    activate = 'softmax'  # 'softmax', 'tanh'
    use_se = False

    learning_rate = 1E-3
    weight_decay = 0
    batch_size = 1
    num_workers = 4
    num_classes = 2  # [not bind, bind]
    epochs = 50  # 20 for bert

    base_input_dim = 62
    bert_input_dim = 1024
    esm650_input_dim = 1280
    graph_ppis_input_dim = 54
    output_dim = 256
    feature_path = "./Feature/"
    graph_path = './Graph/'
    center = 'SC/'
    psepos = 'SC.pkl'
    Test60_psepos_path = './Feature/psepos/Test60_psepos_'
    Test315_28_psepos_path = './Feature/psepos/Test315-28_psepos_'
    Btest31_psepos_path = './Feature/psepos/Test60_psepos_'
    UBtest31_28_psepos_path = './Feature/psepos/UBtest31-6_psepos_'
    Train335_psepos_path = './Feature/psepos/Train335_psepos_'
    dataset_path = "./Dataset/"
    virtual_node = 3
    AlphaFold3_pred = False
    graph_type = 'weight/' # 'normalize/' , 'weight/', 'weight_no_norm/', 'ffk_norm/'
    test_type = 4  #