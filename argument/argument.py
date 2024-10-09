import argparse

parser = argparse.ArgumentParser(description='template of demosaick')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--b_cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_gpu', type=int, default=1,
                    help='number of GPU')
parser.add_argument('--b_cudnn', type=bool, default=True,
                    help='use cudnn')
parser.add_argument('--n_seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0',
                    help='CUDA_VISIBLE_DEVICES')

# Model specifications
parser.add_argument('--s_model', '-m', default='PRNet.Restoration',
                    help='model name')
parser.add_argument('--b_save_all_models', default=False,
                    help='save all intermediate models')
parser.add_argument('--b_load_best', type=bool, default=True,
                    help='use best model for testing')

# Data specifications
parser.add_argument('--dir_dataset', type=str, default='../Dataset/Blind_x4',
                    help='dataset directory')
parser.add_argument('--n_patch_size', type=int, default=128,
                    help='output patch size')
parser.add_argument('--scale', type=int, default=4,
                    help='output scale size')
parser.add_argument('--n_rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--data_pack', type=str, default='packet/packet',  # train/test
                    choices=('packet', 'bin', 'ori'),
                    help='make binary data')

# Evaluation specifications
parser.add_argument('--s_eval_dataset', default='Adobe_test.Adobe_test',
                    help='evaluation dataset')
parser.add_argument('--b_test_only', type=bool, default=True,
                    help='set this option to test the model')
parser.add_argument('--pre_train', type=str, default='PRNet_x4.pth',
                    help='pre-trained model directory')

# Log specifications
parser.add_argument('--s_experiment_name', type=str, default='test_x4',
                    help='file name to save')
parser.add_argument('--b_save_results', type=bool, default=False,
                    help='save output results')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
