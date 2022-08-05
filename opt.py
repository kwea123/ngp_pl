import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # common args for all datasets
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='mgtv',
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test_a'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=0.5,
                        help='downsample factor (<=1.0) for the images')

    parser.add_argument('--scene', type=str, default='F1_06')
    parser.add_argument('--take', type=str, default='000000')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')

    # training options
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='number of rays in a batch')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-2,
                        help='learning rate')

    # misc
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    return parser.parse_args()
