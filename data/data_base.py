import os
import glob
import torch.utils.data as data


class DataBase(data.Dataset):
    def __init__(self, args, name='', b_train=True, b_benchmark=False):
        super(DataBase, self).__init__()

        self.args = args
        self.name = name
        self.b_benchmark = b_benchmark
        self.b_train = b_train

        self.name_data = []
        self.name_isp = []
        self.name_label = []
        self.name_RAWImage = []
        self.data_begin = 0
        self.data_end = 0
        self.data_paris = 0

        for e in self.ext:
            self.name_data += sorted(glob.glob(os.path.join(self.dir_data, '*' + e)))

        for f in self.name_data:
            filename, _ = os.path.splitext(os.path.basename(f))
            self.name_label.append(os.path.join(self.dir_label, '{}{}'.format(filename[:-4], _)))
            self.name_RAWImage.append(os.path.join(self.dir_RAWImage, '{}{}'.format(filename[:-4], _)))
            self.name_isp.append(os.path.join(self.dir_isp, '{}{}'.format(filename, _)).replace('.npy', '.jpeg'))

        range = [r.split('-') for r in self.data_range.split('/')]
        data_range = range[0] if self.b_train else range[1]
        self.data_begin = data_range[0]
        self.data_end = data_range[1]
        self.data_paris = int(self.data_end) - int(self.data_begin) + 1

        self.name_data = self.name_data[int(self.data_begin) - 1:int(self.data_end)]
        self.name_isp = self.name_isp[int(self.data_begin) - 1:int(self.data_end)]
        self.name_label = self.name_label[int(self.data_begin) - 1:int(self.data_end)]
        self.name_RAWImage = self.name_RAWImage[int(self.data_begin) - 1:int(self.data_end)]

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.data_paris
