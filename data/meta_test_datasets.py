META_TEST_DATASET_DICT = {
    # LCBench
    'lcbench': [
        'higgs',
        'jannis',
        'jasmine',
        'jungle_chess_2pcs_raw_endgame_complete',
        'kc1',
        'kr-vs-kp',
        'mfeat-factors',
        'nomao',
        'numerai28.6',
        'phoneme',
        'segment',
        'shuttle',
        'sylvine',
        'vehicle',
        'volkert'],

    # ODBench
    'odbench': [
        'd8_a1',
        'd8_a2',
        'd8_a3',
        'd9_a1',
        'd9_a2',
        'd9_a3',
        'd10_a1',
        'd10_a2',
        'd10_a3'],

    # TaskSet
    'taskset': [
        'rnn_text_classification_family_seed8', 'rnn_text_classification_family_seed82', 'rnn_text_classification_family_seed89',
        'word_rnn_language_model_family_seed84', 'word_rnn_language_model_family_seed98', 'word_rnn_language_model_family_seed99',
        'char_rnn_language_model_family_seed84', 'char_rnn_language_model_family_seed94', 'char_rnn_language_model_family_seed96'],

    # PD1
    'pd1': [
    'imagenet_resnet_batch_size_512',
    'translate_wmt_xformer_translate_batch_size_64',
    'mnist_max_pooling_cnn_tanh_batch_size_256',
    'fashion_mnist_max_pooling_cnn_tanh_batch_size_256',
    'svhn_no_extra_wide_resnet_batch_size_256',
    'cifar100_wide_resnet_batch_size_256',
    'cifar10_wide_resnet_batch_size_256'
    ]
}