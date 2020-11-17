import argparse
import os
from misc import pyutils
import train_cam, make_cam, cam_to_pseudo_labels
import torch
torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    #parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", default='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/', type=str, 
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    #parser.add_argument("--train_list", default="voc12/train.txt", type=str)
    parser.add_argument("--train_list", default="voc12/sequences", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int) # WAS 16
    parser.add_argument("--cam_num_epoches", default=200, type=int)
    parser.add_argument("--cam_learning_rate", default=0.001, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0), help="Multi-scale inferences")
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Output Path
    parser.add_argument("--cam_weights_name", default="saved/decom_cam.pth", type=str)
    '''
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--pseudo_labels_out_dir", default="result/pseudo_labels", type=str)
    '''
    parser.add_argument("--cam_out_dir", default="result_seq/cam_seq", type=str)
    parser.add_argument("--pseudo_labels_out_dir", default="result_seq/pseudo_labels_seq", type=str)
    args = parser.parse_args()
    os.makedirs("saved", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.pseudo_labels_out_dir, exist_ok=True)

    print(vars(args))

    # Train resnet on pascal voc for classification
    timer = pyutils.Timer('step.train_cam:')
    print(args.num_workers)

    ##train_cam.run(args)
    # Generate class activation maps from pretrained resnet
    timer = pyutils.Timer('step.make_cam:')
    make_cam.run(args)
    # Generate pseudo labels from CAMs
    timer = pyutils.Timer('step.cam_to_ir_label:')
    cam_to_pseudo_labels.run(args)
