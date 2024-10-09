import os
import torch
import numpy as np
import cv2
from argument.argument import args
from utils import log, timer, utils


def test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            for batch_index, (data, isp, lin, sRGB) in enumerate(d):
                data, isp, lin, sRGB = data.to(device, non_blocking=True), \
                                       isp.to(device, non_blocking=True), \
                                       lin.to(device, non_blocking=True), \
                                       sRGB.to(device, non_blocking=True)
                try:
                    target = sRGB
                    crop_output = []
                    crop_data, crop_isp, h, w, mask = utils.crop_test(data, isp)
                    timer_test.restart()
                    for i in range(len(crop_data)):
                        crop_model_out = model(crop_data[i], crop_isp[i])
                        crop_model_rgb = crop_model_out if len(crop_model_out) == 1 else crop_model_out[0]
                        crop_model_rgb = crop_model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                        crop_output.append(crop_model_rgb)
                    model_out = utils.cat_test(crop_output, h, w, mask)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()

                    all_ssim = utils.ssim(out_data, out_label, args.n_rgb_range).to(device)
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    info_log.write('{}_{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                        d.dataset.name,
                        batch_index,
                        all_psnr[:, -1].item(),
                        all_psnr[:, 0].item(),
                        all_psnr[:, 1].item(),
                        all_psnr[:, 2].item(),
                        all_ssim.item(),
                    ))

                    if args.b_save_results:
                        path = os.path.join('./experiments', args.s_experiment_name, d.dataset.name,
                                            'result_' + str(batch_index) + '.bmp')
                        cv2.imwrite(path, cv2.cvtColor(np.uint8(out_data * 255), cv2.COLOR_RGB2BGR))
                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)

            info_log.write('{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
            ))

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim = im_ssim.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim, timer_test_elapsed_ticks
