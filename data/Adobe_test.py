import os
import torch
import numpy as np
import cv2

from data import data_base
from argument.argument import args


class Adobe_test(data_base.DataBase):
    def __init__(self, args, name='TESTING', b_train=False, b_benchmark=False):
        self.ext = ('npy',)
        self.dir_root = os.path.join(args.dir_dataset, name)
        self.dir_label = os.path.join(self.dir_root, 'GT')
        self.dir_data = os.path.join(self.dir_root, 'TrainingSet')
        self.dir_isp = os.path.join(self.dir_root, 'ISP')
        self.dir_RAWImage = os.path.join(self.dir_root, 'RAWImage')
        self.data_range = '0-0/1-150'
        super(Adobe_test, self).__init__(args, name=name, b_train=b_train, b_benchmark=b_benchmark)

    def __getitem__(self, index):
        isp = cv2.cvtColor(cv2.imread(self.name_isp[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        isp = np.expand_dims(isp, axis=0)[0]
        data_res = np.expand_dims(np.load(self.name_data[index]), axis=0)[0]
        label = np.expand_dims(np.load(self.name_label[index]), axis=0)[0]
        xlin = np.expand_dims(np.load(self.name_RAWImage[index]), axis=0)[0]

        w = data_res.shape[0] // 2 * 2
        h = data_res.shape[1] // 2 * 2
        data_res = data_res[:w, :h]
        isp = isp[:w, :h]
        label = label[:args.scale * w, :args.scale * h, :]
        xlin = xlin[:args.scale * w, :args.scale * h, :]
        data = np.zeros((w, h, 3), dtype=float)
        
        data[0::2, 0::2, 0] = data_res[0::2, 0::2]
        data[1::2, 0::2, 1] = data_res[1::2, 0::2]
        data[0::2, 1::2, 1] = data_res[0::2, 1::2]
        data[1::2, 1::2, 2] = data_res[1::2, 1::2]

        im_data = np.ascontiguousarray(data.transpose((2, 0, 1)))
        im_isp = np.ascontiguousarray(isp.transpose((2, 0, 1)))
        im_label = np.ascontiguousarray(label.transpose((2, 0, 1)))
        im_xlin = np.ascontiguousarray(xlin.transpose((2, 0, 1)))

        return torch.from_numpy(im_data).float(), torch.from_numpy(im_isp).float()/255, \
               torch.from_numpy(im_xlin).float(), torch.from_numpy(im_label).float()


