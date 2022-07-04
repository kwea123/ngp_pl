import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nsvf'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    parser.add_argument('--scale', type=float, default=1.0,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')

    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')

    parser.add_argument('--hard_sampling', action='store_true', default=False,
                        help='whether to hard sample rays with high loss')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    parser.add_argument('--lr', type=float, default=3e-3,
                        help='learning rate')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_freq', type=int, default=10,
                        help='how often to save checkpoint')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')

    return parser.parse_args()
