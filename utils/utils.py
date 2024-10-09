import sys
import torch

from skimage.metrics import structural_similarity
from importlib import import_module

from argument.argument import args


def crop_test(train_img, isp_img):
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    train_data = []
    isp_data = []
    scale = args.scale
    if args.scale == 2:
        crop_size = 256
    elif args.scale == 4:
        crop_size = 128
    test_step = crop_size // 2
    h = train_img.shape[2] // 2 * 2
    w = train_img.shape[3] // 2 * 2
    isp_img = isp_img[:, :, :h, :w]
    train_img = train_img[:, :, :h, :w]
    mask = torch.zeros(1, 3, scale * h, scale * w, device=device, dtype=torch.float)
    for i in range(0, h - crop_size + test_step, test_step):
        for j in range(0, w - crop_size + test_step, test_step):
            i = min(h - crop_size, i)
            j = min(w - crop_size, j)
            ie = i + crop_size
            je = j + crop_size
            mask[:, :, scale * i:scale * ie, scale * j:scale * je] += 1
            t_data = train_img[:, :, i:ie, j:je]
            i_data = isp_img[:, :, i:ie, j:je]
            train_data.append(t_data)
            isp_data.append(i_data)
    return train_data, isp_data, h, w, mask


def cat_test(crop_img_list, h, w, mask):
    device = torch.device('cpu' if args.b_cpu else 'cuda')
    scale = args.scale
    if args.scale == 2:
        crop_size = 256
    elif args.scale == 4:
        crop_size = 128
    test_step = crop_size // 2
    res = torch.zeros([1, 3, scale * h, scale * w], device=device, dtype=torch.float)
    index = 0
    for i in range(0, scale * (h - crop_size + test_step), scale * test_step):
        for j in range(0, scale * (w - crop_size + test_step), scale * test_step):
            i = min(scale * (h - crop_size), i)
            j = min(scale * (w - crop_size), j)
            ie = i + scale * crop_size
            je = j + scale * crop_size
            res[:, :, i:ie, j:je] += crop_img_list[index][:, :, :ie - i, :je - j]
            index += 1
    res = res / mask
    return res


def psnr(input, target, rgb_range):
    r_input, g_input, b_input = input.split(1, 1)
    if target.shape[1] == 3:
        r_target, g_target, b_target = target.split(1, 1)
    if target.shape[1] == 4:
        r_target, g_target1, g_target2, b_target = target.split(1, 1)
        g_target = (g_target1 + g_target2)/2

    mse_r = (r_input - r_target).pow(2).mean()
    mse_g = (g_input - g_target).pow(2).mean()
    mse_b = (b_input - b_target).pow(2).mean()

    cpsnr = 10 * (rgb_range * rgb_range / ((mse_r + mse_g + mse_b) / 3)).log10()

    psnr = torch.tensor([[10 * (rgb_range * rgb_range / mse_r).log10(),
                         10 * (rgb_range * rgb_range / mse_g).log10(),
                         10 * (rgb_range * rgb_range / mse_b).log10(),
                         cpsnr]]).float()

    return psnr


def ssim(input, target, rgb_range):
    c_s1 = structural_similarity(input[:, :, 0], target[:, :, 0], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                 use_sample_covariance=False)
    c_s2 = structural_similarity(input[:, :, 1], target[:, :, 1], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                 use_sample_covariance=False)
    c_s3 = structural_similarity(input[:, :, 2], target[:, :, 2], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                 use_sample_covariance=False)

    return torch.tensor([[(c_s1+c_s2+c_s3)/3]]).float()


def import_fun(fun_dir, module):
    fun = module.split('.')
    m = import_module(fun_dir + '.' + fun[0])
    return getattr(m, fun[1])


def catch_exception(exception):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print('{}: {}.'.format(exc_type, exception), exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno)


