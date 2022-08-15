'''
error_code:  
-1 error: video number unmatch
-2 error: image not found
-3 error: image size unmatch

'''
import os
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim

import numpy as np
import argparse
import glob
from PIL import Image
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import cv2

def MSE(x,y):
	return compare_mse(x,y)

def PSNR(ximg,yimg):
    return compare_psnr(ximg,yimg,data_range=255)

def SSIM(y,t,value_range=255):   
    try:
        ssim_value = ssim(y, t, gaussian_weights=True, data_range=value_range, multichannel=True)
    except ValueError:
        #WinSize too small
        ssim_value = ssim(y, t, gaussian_weights=True, data_range=value_range, multichannel=True, win_size=3)
    return ssim_value


def Evaluate_meta(pred, gt, meth):
    if meth == "SSIM":
        res = SSIM(pred,  gt)
    else:
        res = PSNR(pred,  gt)
    return res


def Evaluate(files_gt, files_pred, methods = [PSNR,MSE,SSIM], num_processing=32):
    score = {}

    for meth in methods:

        pool = ThreadPool(processes=num_processing)
        results = []
        frame_num=len(files_gt)
        for frame in range(frame_num):
            async_result = pool.apply_async(Evaluate_meta, (files_pred[frame],files_gt[frame], meth.__name__,) )
            return_val = async_result.get()
            results.append(return_val)
        pool.close()
        pool.join()
        mres = np.mean(results)
        stdres=np.std(results)

        score['mean']=mres
        score['std']=stdres
    return score


def evaluate(gt_folder, pred_folder):   
    error_code=0
    error_flag='successful.'
    final_result=[]
    print("starting evaluate | images")
    
    # load folder
    grountruth_folder_list = sorted(glob.glob(os.path.join(gt_folder, '*/00*')))
    prediction_folder_list = sorted(glob.glob(os.path.join(pred_folder,'*/00*')))    
    
    if len(grountruth_folder_list) != len(prediction_folder_list): 
        error_code=-1
        error_flag='folder number unmatch.'
        print(error_flag)
        return error_code, error_flag, 0    

    for i in tqdm(range(len(grountruth_folder_list))):
        # load images
        image_gt_list=[]
        image_mask_list=[]
        image_pred_list=[]
        image_list = sorted(glob.glob(os.path.join(grountruth_folder_list[i],'image.cam*.jpg')))

        image_reading_cost = 0

        for j in range(len(image_list)):
            image_gt_path=image_list[j]
            tic = time.time()
            image_gt_list.append(np.array(Image.open(image_gt_path)).astype(np.uint8))
            image_mask_list.append(np.array(Image.open(image_gt_path.replace('image','mask/image'))).astype(np.uint8))
            image_predict_path=os.path.join(pred_folder,image_gt_path.split('crop_gt_/')[1])

            image_reading_cost += (time.time() - tic)
            try: 
                # tic = time.time()
                image_pred=np.array(Image.open(image_predict_path)).astype(np.uint8)
                # TODO: do this in testing not here
                image_pred = cv2.resize(image_pred, 
                                (image_mask_list[j].shape[1], image_mask_list[j].shape[0]))
                #apply mask
                image_pred=cv2.bitwise_and(image_pred, image_pred, mask=image_mask_list[j])
                image_pred_list.append(image_pred)
                image_reading_cost += (time.time() - tic)
            except Exception as e:
                error_code=-2
                error_flag= 'read ' + image_predict_path +' failed.'
                print("error!! {}".format(e))
                return error_code, error_flag, 0
        
        # check image size
        for j in range(len(image_list)):
            if image_gt_list[j].shape!=image_pred_list[j].shape:
                error_code=-3
                error_flag= 'image size unmatch.' + image_pred_list[j]
                return error_code, error_flag, 0

        print("starting evaluate | evaluate")
        tic = time.time()
        psnr_res = Evaluate(image_gt_list, image_pred_list, methods=[PSNR])
        ssim_res = Evaluate(image_gt_list, image_pred_list, methods=[SSIM])
        dist_calc_cost = (time.time() - tic)

        print("image_reading cost:", image_reading_cost)
        print("dist_calculate cost:", dist_calc_cost)
        
        psnr_res_norm=min(100,psnr_res['mean'])         
        ssim_res_norm=(ssim_res['mean']*100)*0.5

        result=psnr_res_norm+ssim_res_norm
        print(grountruth_folder_list[i],psnr_res_norm,ssim_res_norm,result)

        final_result.append(result)
    return error_code, error_flag, np.mean(final_result) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth_folder',
                        default='/home/ubuntu/hdd/data/mgtv/val_crop_gt_')
    parser.add_argument('--prediction_folder',
                        default='results/mgtv')
    # usage: python evaluate.py --groundtruth_folder ./val_crop_gt --prediction_folder ./val_crop_predict
    args = parser.parse_args()
    gt_folder = args.groundtruth_folder
    pred_folder = args.prediction_folder

    print(time.time())
    tic = time.time()
    error_code, error_flag, final_result = evaluate(gt_folder, pred_folder)
    print(time.time())
    toc = time.time()
    print("time cost : ", toc - tic)
    print(final_result)


