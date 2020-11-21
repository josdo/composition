"""
Predicts on image collection of stories and saves bounding box overlays 
and people maps.
"""
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

def format_for_prediction(img_path, compound_coef, use_cuda = True, use_float16 = False):
    force_input_size = None  # set None to use default size
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    return ori_imgs, framed_metas, x

def load_model(weights_path, obj_list, compound_coef, use_cuda = True, use_float16 = False):
    cudnn.fastest = True 
    cudnn.benchmark = True

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 # TODO: replace this part with your project's anchor config
                                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load(weights_path)) # TODO: just weight path?
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
        
    return model

def main():
    # Choose batch and model settings
    dirs = {
        "imgs":"datasets/stories/imgs",
        "maps":"datasets/stories/pplmaps",
        "preds":"datasets/stories/predictions",
    }

    weights_path = './logs/peopleart_coco/efficientdet-d0_149_600.pth' # 150 epochs, last layer

    # for model
    threshold = 0.5
    iou_threshold = 0.5
    obj_list = ['person']

    compound_coef = 0
    use_cuda = True
    use_float16 = False

    # for pplmap
    map_dim = 200 # along width and height
    map_shade = 64 # allows up to 4 boxes to overlap

    # Load model
    model = load_model(weights_path, obj_list, compound_coef, use_cuda, use_float16)

    # Process images 
    for story in ["abraham_isaac"]: # "adam_eve","last_supper", 
        print("Predicting on images in %s" % story)
        path = Path(os.path.join(dirs['imgs'], story))
        batch_paths = [p.as_posix() for p in path.glob('*.jpg')]

        for img_path in tqdm(batch_paths):
            # Pre-process
            ori_imgs, framed_metas, x = format_for_prediction(img_path, compound_coef, use_cuda, use_float16)

            # Predict
            with torch.no_grad():
                features, regression, classification, anchors = model(x)
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)

            # Post-process
            out = invert_affine(framed_metas, out)

            for i in range(len(ori_imgs)):
                img = ori_imgs[i].copy()
                bbox_coords = out[i]['rois'].astype(np.int)

                if not len(bbox_coords):
                    print("No predicted boxes for img %s" % img_path)
                    continue

                # Single channel pplmap
#                 mapw, maph = np.asarray(img.shape[:2]) // downscale
                pplmap = np.zeros((map_dim, map_dim))
                x_scale, y_scale = map_dim / np.array(img.shape[:2])
#                 print("scaling by %.4f %.4f" % (x_scale, y_scale))
                
                for j in range(len(bbox_coords)):
                    (x1, y1, x2, y2) = bbox_coords[j]
                    # Update pplmap
                    # np.array((x1, y1, x2, y2)) // downscale
#                     print("BOX COORDS")
#                     print(bbox_coords[j])
                    
                    mapx1, mapx2 = np.floor([x1 * x_scale, x2 * x_scale]).astype(np.int32)
                    mapy1, mapy2 = np.floor([y1 * y_scale, y2 * y_scale]).astype(np.int32)
#                     print("SCALED COORDS")
#                     print(mapy1,mapy2, mapx1,mapx2)
                    pplmap[mapy1:mapy2+1, mapx1:mapx2+1] += map_shade

                    # Plot bbox predictions
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    obj = obj_list[out[i]['class_ids'][j]]
                    score = float(out[i]['scores'][j])

                    cv2.putText(img, '{}, {:.3f}'.format(obj, score),
                                (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
                    plt.imshow(img)

                # Save as files
                map_path = os.path.join(dirs['maps'], story, os.path.basename(img_path))
                cv2.imwrite(map_path, cv2.cvtColor(pplmap.astype('uint8'), cv2.COLOR_GRAY2BGR))

                viz_path = os.path.join(dirs['preds'], story, os.path.basename(img_path))
                plt.savefig(viz_path)
                plt.close()

if __name__ == "__main__":
    main()