import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils.utils import *
from LFTramba import Net
from tqdm import tqdm
import h5py
import numpy as np
from torchvision.transforms import ToTensor
import torch
from matplotlib import pyplot as plt
import einops

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='LFTramba')   
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")

    parser.add_argument('--testset_dir', type=str, default='/home/lhs/LF/LFTramba/data_for_inference/') 
    parser.add_argument('--testdata', type=str, default='SR_5x5_4x/')
    parser.add_argument("--patchsize", type=int, default=32, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=16,
                        help="The stride between two test patches is set to patchsize/2")
    parser.add_argument('--channels', type=int, default=128, help='channels') 
    parser.add_argument('--model_path', type=str, default='/home/lhs/LF/LFTramba/pth/LFTramba_4xSR_5x5_epoch_63.pth')

    parser.add_argument('--save_path', type=str, default='/SRdata/lhsDATA/bs/') # SR result saving path (.mat format)
    parser.add_argument('--tta', type=bool, default=True) 

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):
    net = Net(cfg.angRes, cfg.upscale_factor, cfg.channels)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:1': cfg.device})
        try:
            net.load_state_dict(model['state_dict'])
        except:
            net.load_state_dict({k.replace('module.netG.', ''): v for k, v in model['state_dict'].items()})
    else:
        print("=> no model found at '{}'".format(cfg.model_path))

    with torch.no_grad():
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF = inference(cfg, test_loader, test_name, net)
            pass
        pass

def inference(cfg, test_loader, test_name, net):
  
    for idx_iter, (data, label) in tqdm((enumerate(test_loader)), total=len(test_loader), ncols=70):
        
        data = data.to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label 
        outLF = inference_no_pad(cfg,data,net)

        save_path = cfg.save_path + '/' + cfg.model_name + '/' + cfg.testdata

        isExists = os.path.exists(save_path + test_name)
        if not (isExists):
            os.makedirs(save_path + test_name)

        from scipy import io
        scipy.io.savemat(save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                         {'LF': outLF.cpu().numpy()})
        pass

    return outLF

def tta(lr_images,net):
    """TTA based on _restoration_augment_tensor
    """
    hr_preds = []
    hr_preds.append(inference1(lr_images,net))
    hr_preds.append(inference1(lr_images.rot90(1, [2, 3]).flip([2]),net).flip([2]).rot90(3, [2, 3]))
    hr_preds.append(inference1(lr_images.flip([2]),net).flip([2]))
    hr_preds.append(inference1(lr_images.rot90(3, [2, 3]),net).rot90(1, [2, 3]))
    hr_preds.append(inference1(lr_images.rot90(2, [2, 3]).flip([2]),net).flip([2]).rot90(2, [2, 3]))
    hr_preds.append(inference1(lr_images.rot90(1, [2, 3]),net).rot90(3, [2, 3]))
    hr_preds.append(inference1(lr_images.rot90(2, [2, 3]),net).rot90(2, [2, 3]))
    hr_preds.append(inference1(lr_images.rot90(3, [2, 3]).flip([2]),net).flip([2]).rot90(1, [2, 3]))
    return torch.stack(hr_preds, dim=0).mean(dim=0)
  
def inference1(lr_images,net):
    """general inference pipeline
    """
    model = net
    hr_preds = model(lr_images.to(cfg.device))

    return hr_preds

#psw++
class LF_divide_integrate_pswpp(object):
  def __init__(self, scale, patch_size, stride):
    self.scale = scale
    self.patch_size = patch_size
    self.stride = stride
    self.bdr = (patch_size - stride) // 2
    self.pad = torch.nn.ReflectionPad2d(padding=(self.bdr, self.bdr + stride - 1, self.bdr, self.bdr + stride - 1))

  def LFdivide(self, LF):
    assert LF.size(0) == 1, 'The batch_size of LF for test requires to be one!'
    LF = LF.squeeze(0)
    [c, u0, v0, h0, w0] = LF.size()
    stride = self.stride
    patch_size = self.patch_size

    self.sai_h = h0
    self.sai_w = w0

    sub_lf = []
    numU = 0
    for y in range(0, h0, stride):
      numV = 0
      for x in range(0, w0, stride):
        if y + patch_size > h0 and x + patch_size <= w0:
          sub_lf.append(LF[..., h0 - patch_size:, x: x + patch_size])
        elif y + patch_size <= h0 and x + patch_size > w0:
          sub_lf.append(LF[..., y: y + patch_size, w0 - patch_size:])
        elif y + patch_size > h0 and x + patch_size > w0:
          sub_lf.append(LF[..., h0 - patch_size:, w0 - patch_size:])
        else:
          sub_lf.append(LF[..., y: y + patch_size, x: x + patch_size])
        numV += 1
      numU += 1

    LF_divided = torch.stack(sub_lf, dim=0)
    return LF_divided

  def LFintegrate(self, LF_divided):
    # each SAI size
    stride = self.stride * self.scale
    patch_size = self.patch_size * self.scale
    bdr = self.stride // 2

    # rearrange to SAI views
    _, c, u, v, h, w = LF_divided.size()
    h1 = self.sai_h * self.scale
    w1 = self.sai_w * self.scale

    # allocate space
    out = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    mask = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)

    # colllect outter for patch_size
    idx = 0
    for y in range(0, h1, stride):
      for x in range(0, w1, stride):
        if y + patch_size > h1 and x + patch_size <= w1:
          out[..., h1 - patch_size:, x: x + patch_size] += LF_divided[idx]
          mask[..., h1 - patch_size:, x: x + patch_size] += 1
        elif y + patch_size <= h1 and x + patch_size > w1:
          out[..., y: y + patch_size, w1 - patch_size:] += LF_divided[idx]
          mask[..., y: y + patch_size, w1 - patch_size:] += 1
        elif y + patch_size > h1 and x + patch_size > w1:
          out[..., h1 - patch_size:, w1 - patch_size:] += LF_divided[idx]
          mask[..., h1 - patch_size:, w1 - patch_size:] += 1
        else:
          out[..., y: y + patch_size, x: x + patch_size] += LF_divided[idx]
          mask[..., y: y + patch_size, x: x + patch_size] += 1
        idx += 1
    # final = out / mask

    # collect inner for patch_size
    idx = 0
    out_in = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    mask_in = torch.zeros(c, u, v, h1, w1).to(LF_divided.device)
    for y in range(0, h1, stride):
      for x in range(0, w1, stride):
        if y + patch_size > h1 and x + patch_size <= w1:
          pass
        elif y + patch_size <= h1 and x + patch_size > w1:
          pass
        elif y + patch_size > h1 and x + patch_size > w1:
          pass
        else:
          out_in[..., y + bdr: y + bdr + stride, x + bdr: x + bdr + stride] += LF_divided[idx][..., bdr: bdr + stride, bdr: bdr + stride]  # nopep8
          mask_in[..., y + bdr: y + bdr + stride, x + bdr: x + bdr + stride] += 1
        idx += 1

    # inner to zero
    mask[mask_in != 0] = 0
    out[mask_in != 0] = 0
    final = (out + out_in) / (mask + mask_in)

    return final

def inference_no_pad(cfg,lr_images,net):
    assert lr_images.size(0) == 1, f"require input batchsize should be 1."
    device = cfg.device
    scale = cfg.upscale_factor
    angular = cfg.angRes
    stride = cfg.stride
    patch_size = cfg.patchsize
    Processor = LF_divide_integrate_pswpp(cfg.upscale_factor, cfg.patchsize, cfg.stride)
    img_h, img_w = lr_images.shape[-2:]
 
    # expand to aperture mode
    sub_lf = einops.rearrange(lr_images, 'b c (u h) (v w) -> b c u v h w', u=angular, v=angular)

    # crop to patches: [70, 1, 5, 5, 32, 32] -> [70, 1, 5 * 32, 5 * 32]
    sub_lf = Processor.LFdivide(sub_lf)
    # print(sub_lf.shape)
    sub_lf = einops.rearrange(sub_lf, 'n c u v h w -> n c (u h) (v w)', u=angular, v=angular)
    sub_lf_out = torch.zeros_like(sub_lf).repeat(1, 1, scale, scale)

    # # loop for every pathces
    for i in range(sub_lf.size(0)):
      if cfg.tta:
        sub_lf_out[i: i + 1] = tta(sub_lf[i: i + 1],net)
      else:
        sub_lf_out[i: i + 1] = inference1(sub_lf[i: i + 1],net)
    
    # # intergrate into one image
    sub_lf_out = einops.rearrange(sub_lf_out, 'n c (u h) (v w) -> n c u v h w', u=angular, v=angular)
    sub_lf_out = Processor.LFintegrate(sub_lf_out)
    #sub_lf_out = einops.rearrange(sub_lf_out, 'c u v h w -> 1 c (u h) (v w)', u=angular, v=angular)
    sub_lf_out = einops.rearrange(sub_lf_out, 'c u v h w -> u v h w c', u=angular, v=angular)
    
    return sub_lf_out


def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
