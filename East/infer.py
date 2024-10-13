import os
import config as cfg
from model import East
import torch
import utils
import preprossing
import cv2
import numpy as np
import time

def predict(image_path,device,model_checkpoint):

    print('Loading Image')
    im = cv2.imread(image_path)[:,:,::-1]

    print('EAST <==> Prepare <==> Network <==> Begin')
    model = East()
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu_ids)

    bbox_result_dict = dict()
    bbox_result_dict['detections'] = []
    bbox_result = "bbox_detection.txt"
    image_result = "image_detection.jpg"
    
    # Load Model
    print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(model_checkpoint))
    checkpoint = torch.load(model_checkpoint,map_location=torch.device(device))
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(model_checkpoint))

    model.eval()

    # image resize
    im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
    im_resized = im_resized.astype(np.float32)
    
    # Convert image to tensor format
    im_resized = im_resized.transpose(2, 0, 1)
    im_tensor = torch.from_numpy(im_resized)
    im_tensor = im_tensor.cpu()
    
    # Image data adds one dimension
    im_tensor = im_tensor.unsqueeze(0)

    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()

    # Input network for inference
    score, geometry = model(im_tensor)

    timer['net'] = time.time() - start
    score = score.permute(0, 2, 3, 1)
    geometry = geometry.permute(0, 2, 3, 1)
    score = score.data.cpu().numpy()
    geometry = geometry.data.cpu().numpy()
    
    # Text box detection
    boxes, timer = utils.detect(score_map=score, geo_map=geometry, timer=timer,
                                score_map_thresh=cfg.score_map_thresh, box_thresh=cfg.box_thresh,
                                nms_thres=cfg.box_thresh)
    print('EAST <==> TEST <==> model:{:.2f}ms, restore:{:.2f}ms, nms:{:.2f}ms'
          .format(timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    # save to txt file
    if boxes is not None:
        res_file = os.path.join(
            cfg.res_img_path,
            'res_{}.txt'.format(
                os.path.basename(bbox_result).split('.')[0]))

        with open(res_file, 'w') as f:
            for box in boxes:
                # to avoid submitting errors
                box = utils.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                ))
                cv2.polylines(im[:,:,::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                              color=(255, 255, 0), thickness=1)
                bbox_result_dict['detections'].append([box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]])
            print('EAST <==> TEST <==> Save txt   at:{} <==> Done'.format(res_file))

    # 图片输出
    if cfg.write_images:
        img_path = os.path.join(cfg.res_img_path, os.path.basename(image_result))
        print(img_path)
        cv2.imwrite(img_path, im[:,:,::-1])
        print('EAST <==> TEST <==> Save image at:{} <==> Done'.format(img_path))

    print('EAST <==> TEST <==> Record and Save <==> Done')

    return bbox_result_dict
