import argparse
import cv2
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--split', type=str, default='val',
                        help='use which split to train')
    parser.add_argument('--scene', type=str, default='F1_06')
    parser.add_argument('--take', type=str, default='000000')

    args = parser.parse_args()

    for cam in range(92): # read all cameras
        xml_path = os.path.join(args.root_dir,
            "camera_parameters", args.scene, str(cam+1), "intrinsic.xml")
        fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
        K = fs.getNode('M').mat()
        if K[0, 0] < 4000: # hack to get image height and width
            H = 2048
            W = 2592
        else:
            H = 3072
            W = 4096
        # self.Ds += [fs.getNode('D').mat()]

        xml_path = os.path.join(args.root_dir, 
            "camera_parameters", args.scene, str(cam+1), "extrinsics.xml")
        fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
        R = fs.getNode('R').mat()
        T = fs.getNode('T').mat() # in meters

        w2c = np.eye(4)
        w2c[:3] = np.concatenate([R, T], 1) # (3, 4)
        c2w = np.linalg.inv(w2c)
        if args.scene=='M3_02':
            c2w[:3, 3] /= 3.3
        else:
            c2w[:3, 3] /= 2
        c2w[2, 3] += 0.45
        w2c = np.linalg.inv(c2w)
        w2c[:, :2] *= -1

        with open(os.path.join(args.root_dir, args.split, args.scene, args.take,
                    f'image.cam{cam+1:02d}_{args.take}.cam'), 'w') as f:
            s1 = f'{w2c[0, 3]} {w2c[1, 3]} {w2c[2, 3]} '
            s1+= f'{w2c[0, 0]} {w2c[0, 1]} {w2c[0, 2]} '
            s1+= f'{w2c[1, 0]} {w2c[1, 1]} {w2c[1, 2]} '
            s1+= f'{w2c[2, 0]} {w2c[2, 1]} {w2c[2, 2]}\n'

            s2 = f'{K[0, 0]/W} 0 0 {K[1, 1]/K[0, 0]} {K[0, 2]/W} {K[1, 2]/H}\n'
            f.write(s1+s2)