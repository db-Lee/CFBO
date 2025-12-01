import torch

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
        'volkert'
    ],

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
        'd10_a3'
    ],

    # TaskSet
    'taskset': [
        'rnn_text_classification_family_seed8', 'rnn_text_classification_family_seed82', 'rnn_text_classification_family_seed89',
        'word_rnn_language_model_family_seed84', 'word_rnn_language_model_family_seed98', 'word_rnn_language_model_family_seed99',
        'char_rnn_language_model_family_seed84', 'char_rnn_language_model_family_seed94', 'char_rnn_language_model_family_seed96'
    ],

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

VALID_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 17, 
    18, 24, 25, 26, 28, 29, 30, 31, 33, 35, 37, 
    42, 43, 44, 46, 53, 56, 58, 61, 63, 64, 68, 
    72, 74, 75, 76, 78, 79, 82, 83, 85, 87, 89, 
    91, 93, 95, 96, 97, 98, 99, 100, 101, 102, 
    103, 106, 107, 108, 109, 110, 111, 112, 113, 
    114, 117, 120, 122, 123, 124, 125, 127, 128, 
    129, 132, 133, 134, 136, 137, 139, 140, 141, 
    143, 144, 145, 147, 148, 149, 151, 152, 154, 
    155, 157, 159, 160, 161, 162, 167, 169, 171, 
    172, 173, 174, 176, 177, 178, 180, 181, 185, 
    186, 187, 188, 189, 190, 192, 193, 194, 196, 
    198, 199, 206, 207, 208, 214, 215, 216, 217, 
    218, 220, 221, 222, 223, 224, 226, 227, 230, 
    231, 232, 235, 237, 238, 243, 244, 245, 247, 
    249, 252, 253, 254, 256, 257, 258, 261, 262, 
    264, 266, 268, 271, 272, 273, 274, 276, 277, 
    278, 279, 280, 281, 282, 284, 285, 286, 288, 
    291, 294, 295, 299, 300, 301, 303, 304, 309, 
    310, 312, 313, 314, 316, 317, 321, 322, 323, 
    324, 325, 326, 327, 328, 329, 331, 333, 334, 
    335, 337, 338, 340, 341, 343, 344, 345, 346, 
    348, 350, 352, 353, 354, 355, 356, 358, 361, 
    366, 370, 371, 372, 373, 376, 377, 378, 380, 
    382, 383, 385, 386, 388, 390, 391, 393, 394, 
    395, 396, 397, 399
]

# https://github.com/automl/lcpfn/blob/ba892f6f451027f69c50edf00c765ded98c75d30/lcpfn/utils.py#L324
def pfn_normalize(
    lb: torch.Tensor = torch.tensor(float('-inf')),
    ub: torch.Tensor = torch.tensor(float('inf')),
    soft_lb: float = 0.0,
    soft_ub: float = 1.0,
    minimize: bool = False):
    """
    LC-PFN curve prior assumes curves to be normalized within the range [0,1] and to be maximized.
    This function allows to normalize and denormalize data to fit this assumption.
    
    Parameters:
        lb (torch.Tensor): Lower bound of the data.
        ub (torch.Tensor): Upper bound of the data.
        soft_lb (float): Soft lower bound for normalization. Default is 0.0.
        soft_ub (float): Soft upper bound for normalization. Default is 1.0.
        minimize (bool): If True, the original curve is a minization. Default is False.
    
    Returns: Two functions for normalizing and denormalizing the data.
    """
    assert(lb <= soft_lb and soft_lb < soft_ub and soft_ub <= ub)
    # step 1: linearly transform [soft_lb,soft_ub] [-1,1] (where the sigmoid behaves approx linearly)
    #    2.0/(soft_ub - soft_lb)*(x - soft_lb) - 1.0
    # step 2: apply a vertically scaled/shifted the sigmoid such that [lb,ub] --> [0,1]

    def cinv(x):
        return 1-x if minimize else x

    def lin_soft(x):
        return 2/(soft_ub - soft_lb)*(x - soft_lb)-1

    def lin_soft_inv(y):
        return (y+1)/2*(soft_ub - soft_lb)+soft_lb
    
    try:
        if torch.exp(-lin_soft(lb)) > 1e300: raise RuntimeError
        # otherwise overflow causes issues, treat these cases as if the lower bound was -infinite
        # print(f"WARNING: {lb} --> NINF to avoid overflows ({np.exp(-lin_soft(lb))})")
    except RuntimeError:
        lb = torch.tensor(float('-inf'))
    if torch.isinf(lb) and torch.isinf(ub):
        return lambda x: cinv(1/(1 + torch.exp(-lin_soft(x)))), lambda y: lin_soft_inv(torch.log(cinv(y)/(1-cinv(y))))
    elif torch.isinf(lb):
        a = 1 + torch.exp(-lin_soft(ub))
        return lambda x: cinv(a/(1 + torch.exp(-lin_soft(x)))), lambda y: lin_soft_inv(torch.log((cinv(y)/a)/(1-(cinv(y)/a))))
    elif torch.isinf(ub):
        a = 1/(1-1/(1+torch.exp(-lin_soft(lb))))
        b = 1-a
        return lambda x: cinv(a/(1 + torch.exp(-lin_soft(x))) + b), lambda y: lin_soft_inv(torch.log(((cinv(y)-b)/a)/(1-((cinv(y)-b)/a))))
    else:
        a = (1 + torch.exp(-lin_soft(ub)) + torch.exp(-lin_soft(lb)) + torch.exp(-lin_soft(ub)-lin_soft(lb))) / (torch.exp(-lin_soft(lb)) - torch.exp(-lin_soft(ub)))
        b = - a / (1 + torch.exp(-lin_soft(lb)))
        return lambda x: cinv(a/(1 + torch.exp(-lin_soft(x))) + b), lambda y: lin_soft_inv(torch.log(((cinv(y)-b)/a)/(1-((cinv(y)-b)/a))))