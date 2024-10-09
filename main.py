import os
import time
import random
import numpy as np

import torch
import torch.nn.parallel.data_parallel
import torch.backends.cudnn
import torch.optim.lr_scheduler
import torch.distributed
import torch.multiprocessing
from torch.utils.data import DataLoader
from argument import argument
import common
from argument.argument import args
from utils import log, check_point, utils

os.environ['CUDA_VISIBLE_DEVICES'] = argument.args.CUDA_VISIBLE_DEVICES


def main():
    torch.cuda.empty_cache()
    random.seed(args.n_seed)
    np.random.seed(args.n_seed)
    torch.manual_seed(args.n_seed)
    torch.backends.cudnn.enabled = args.b_cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device('cpu' if args.b_cpu else 'cuda')

    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    info_log.write('Experiment: {} ({})'.format(
        args.s_experiment_name,
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ))

    model = utils.import_fun('models', args.s_model.strip())(args)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))) if args.n_gpu > 1 and not args.b_cpu else model
    model = model.to(device)

    ckp = check_point.CheckPoint('./experiments', args.s_experiment_name)

    data_loader_test = [(DataLoader(
                            dataset=utils.import_fun('data', dataset)(args, b_train=False),
                            num_workers=0,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=not args.b_cpu
                        )) for dataset in args.s_eval_dataset.strip().split('+')]

    ckp.save_config(args, model)

    if args.b_test_only:
        pth = ckp.load(device)
        model.load_state_dict(pth.get('model'))
        info_log.write('Resume model for testing')
        for ds in data_loader_test:
            info_log.write('Testing database: {}({})'.format(ds.dataset.name, len(ds)))
        psnr, ssim, test_time = common.test(model, data_loader_test)

        info_log.write('[Testing: {:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}]'.format(
            psnr[:, -1].item(),
            psnr[:, 0].item(), psnr[:, 1].item(), psnr[:, 2].item(),
            ssim[-1].item(),
        ))
        info_log.write('Testing elapsed: {:.3f}s'.format(test_time))

    info_log.write('Completed !!!')


if __name__ == '__main__':
    main()
