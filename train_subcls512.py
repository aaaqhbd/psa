
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils,kmeans
import argparse
import importlib
import torch.nn.functional as F
import os
import sys
path=os.path.basename(sys.argv[0])
f=open(path,'r')
lines=f.read()
f.close()
print(lines)
def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')
    val_loss_metersub = pyutils.AverageMeter('losssub')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]
            label = pack[2].cuda(non_blocking=True)

            xf, x, xsub = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})
            # losssub = F.multilabel_soft_margin_loss(xsub, labelssubi)

            # val_loss_metersub.add({'losssub': losssub.item()})
    model.train()

    print('loss:', val_loss_meter.pop('loss'))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--network", default="network.resnet38_subcls4", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", default='resnet38_cls1/resnet38_cls112.pth', type=str) # trained weight using train_cls.py at 12 epoch which is better than 14 epoch
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="res138k", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default="VOC2012", type=str)
    args = parser.parse_args()




    model = getattr(importlib.import_module(args.network), 'Net')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(256, 512),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_data_loader_no_shuffle = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    val_dataset = voc12.data.VOC12ClsDataset(args.val_list, voc12_root=args.voc12_root,
                                             transform=transforms.Compose([
                        np.asarray,
                        model.normalize,
                        imutils.CenterCrop(500),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.get_parameter_groups()


    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls" or args.network == "network.resnet38_subcls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    elif args.weights[-11:] == '.caffemodel':
        assert args.network == "network.vgg16_cls" or "network.vgg16_subcls"
        import network.vgg16d
        weights_dict = network.vgg16d.convert_caffe_to_torch(args.weights)
        torch.save(weights_dict,'vgg16_20M.pth')
    else:
        weights_dict = torch.load(args.weights)
    # del weights_dict['fc8.weight']
    # print(type(weights_dict), weights_dict.keys())

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')
    avg_metersub = pyutils.AverageMeter('losssub')
    timer = pyutils.Timer("Session started: ")

    channel=model.module.dim[0]
    n_cluster=10
    centers = torch.zeros((20, n_cluster, channel))
    cneterready = torch.zeros((20,))
    for j in range(3):
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
            {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
            {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
        ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

        xfs = []
        idxs = []
        labels = []
        model.eval()
        with torch.no_grad():
            for iter, pack in enumerate(train_data_loader_no_shuffle):
                # if iter>10:
                #     break
                img = pack[1]
                label = pack[2].cuda(non_blocking=True)
                idx = pack[3]
                # print (idx,label.shape)
                xf, x, xsub = model(img)
                xfs.append(xf)
                idxs.append(idx)
                labels.append(label)
        xfs = torch.cat(xfs, dim=0).squeeze()
        idxs = torch.cat(idxs, dim=0)
        labels = torch.cat(labels, dim=0)
        model.train()
        labelssub1 = []
        xfs = xfs / torch.norm(xfs, p=2, dim=1, keepdim=True)
        for i in range(labels.size(1)):
            mask = labels[:, i] == 1
            labeli = torch.zeros((labels.shape[0], n_cluster))
            xfsi = xfs[mask]
            if not cneterready[i]:
                print ('not')
                cluster_ids_x, cluster_centers = kmeans.kmeans(
                    X=xfsi, num_clusters=n_cluster, iter_limit=1000, cluster_centers=[], tol=1e-4,
                    tqdm_flag=True, distance='euclidean', device=torch.device('cuda:0')
                )
                cneterready[i] = 1
            else:
                print ('ok')
                cluster_ids_x, cluster_centers = kmeans.kmeans(
                    X=xfsi, num_clusters=n_cluster, iter_limit=1000, cluster_centers=centers[i], tol=1e-4,
                    tqdm_flag=True, distance='euclidean', device=torch.device('cuda:0')
                )
            ones = torch.sparse.torch.eye(n_cluster)
            ids = ones.index_select(0, cluster_ids_x)
            labeli[mask] = ids
            centers[i] = cluster_centers
            labelssub1.append(labeli)
        labelssub = torch.cat(labelssub1, dim=1)
        for ep in range(args.max_epoches):
            # if ep >1:
            #     break


            for iter, pack in enumerate(train_data_loader):

                img = pack[1]
                label = pack[2].cuda(non_blocking=True)
                idx = pack[3]
                # print (idx)
                # idx=torch.clamp(idx,max=100)
                labelssubi=labelssub[idx].cuda(non_blocking=True)
                xf, x, xsub = model(img)
                lossp = F.multilabel_soft_margin_loss(x, label)
                losssub = F.multilabel_soft_margin_loss(xsub, labelssubi)
                avg_meter.add({'loss': lossp.item()})
                avg_metersub.add({'losssub': losssub.item()})
                loss=(losssub*5+lossp)/6
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (optimizer.global_step-1)%1 == 0:
                    timer.update_progress(optimizer.global_step / max_step)

                    print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                          'Loss:%.4f' % (avg_meter.pop('loss')),
                          'Losssub:%.4f' % (avg_metersub.pop('losssub')),
                          'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                          'Fin:%s' % (timer.str_est_finish()),
                          'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            else:
                validate(model, val_data_loader)
                timer.reset_stage()
            torch.save(model.module.state_dict(), args.session_name +str(j)+ str(ep)+'.pth')
