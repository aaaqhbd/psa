
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist
def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_subcls4", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default="VOC2012", type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_la_crf", default=None, type=str)
    parser.add_argument("--out_ha_crf", default=None, type=str)
    parser.add_argument("--out_cam_pred", default=None, type=str)
    parser.add_argument("--session_name", default="vgg_cls", type=str)
    args = parser.parse_args()
    # if args.infer_list=='val':
    #     p='val'
    #     args.infer_list="voc12/val.txt"
    # elif args.infer_list=='train_aug':
    #     p = 'train_aug'
    #     args.infer_list = "voc12/train_aug.txt"
    # print(args.infer_list)
    # args.out_cam=os.path.join(args.session_name,p,'outcam')
    # args.out_la_crf = os.path.join(args.session_name,p, 'out_la_crf')
    # args.out_ha_crf = os.path.join(args.session_name,p, 'out_ha_crf')
    # args.out_cam_pred = os.path.join(args.session_name,p, 'out_cam_pred')
    # makedir(args.session_name)
    # makedir(os.path.join(args.session_name,p))
    # makedir(args.out_cam_pred)
    # makedir(args.out_ha_crf)
    # makedir(args.out_la_crf)
    # makedir(args.out_cam)
    # if args.weights[0:3]=='vgg':
    #     if 'k' in args.weights:
    #         args.network="network.vgg16_subcls"
    #     else:
    #         args.network = "network.vgg16_cls"
    # elif args.weights[0:3]=='res':
    #     if 'k' in args.weights:
    #         if 'res1' in args.weights:
    #             args.network = "network.resnet38_subcls1"
    #         else:
    #             args.network="network.resnet38_subcls"
    #     elif 'cls1' in args.weights:
    #         args.network = "network.resnet38_cls1"
    #     else:
    #         args.network = "network.resnet38_cls"
    def get_iou(weights):
        model = getattr(importlib.import_module(args.network), 'Net')()
        model.load_state_dict(torch.load(weights))

        model.eval()
        model.cuda()

        infer_dataset = voc12.data.VOC12ClsDatasetMSFseg(args.infer_list, voc12_root=args.voc12_root,
                                                       scales=(1, 0.5, 1.5, 2.0),
                                                       inter_transform=torchvision.transforms.Compose(
                                                           [np.asarray,
                                                            model.normalize,
                                                            imutils.HWC_to_CHW]))

        infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        n_gpus = torch.cuda.device_count()
        model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
        preds, gts = [], []
        for iter, (img_name, img_list, label,labelseg) in enumerate(infer_data_loader):
            img_name = img_name[0]; label = label[0]

            img_path = voc12.data.get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]
            # print(len(img_list))
            # for im in img_list:
            #     print (im.shape)
            def _work(i, img):
                with torch.no_grad():
                    with torch.cuda.device(i%n_gpus):
                        cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                        cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                        if i % 2 == 1:
                            cam = np.flip(cam, axis=-1)
                        return cam

            thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                                batch_size=12, prefetch_size=0, processes=args.num_workers)

            cam_list = thread_pool.pop_results()

            sum_cam = np.sum(cam_list, axis=0)
            print(sum_cam.shape)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

            bg_score = [np.ones_like(norm_cam[0])*0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

            if iter%50==0:
                print(iter,weights)

            preds += list(pred)
            gts += list(labelseg[0].cpu().numpy())
        score = scores(gts, preds, n_class=21)
        print(weights)
        print(score)

    # for i in range(3):
    #     weights = args.weights +'/'+args.weights + str(i) + '0.pth'
    #     get_iou(weights)
    # weights = args.weights + '.pth'
    # get_iou(weights)
    # for i in range(0,3):
    #     weights = args.weights +'/'+args.weights + str(i) + '14.pth'
    #     get_iou(weights)
    # for i in range(0,1):
    #     # weights = args.weights +'/'+args.weights + str(i) + '0.pth'
    #     # get_iou(weights)
    #     weights = args.weights + str(i) + '7.pth'
    #     get_iou(weights)

    for i in range(0,1):
        # weights = args.weights +'/'+args.weights + str(i) + '0.pth'
        # get_iou(weights)
        weights = args.weights + str(i) + '14.pth'
        get_iou(weights)
    # for i in range(6,7):
    #     weights = args.weights +'/'+args.weights + '0'+str(i) + '.pth'
    #     get_iou(weights)
    # for i in range(1,4):
    #     weights = args.weights +'/'+args.weights + str(i) + '1.pth'
    #     get_iou(weights)
    # for i in range(0,4):
    #     weights = args.weights +'/'+args.weights + str(i) + '0.pth'
    #     get_iou(weights)