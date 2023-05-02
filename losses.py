import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import torchvision
from math import ceil as ceil
# from skimage.metrics import structural_similarity as ssim
# from sobel import CannyLoss
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# import numpy

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'MSELoss','VGGWPL']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
    
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        loss_type = nn.MSELoss(reduction="sum")
        loss = loss_type(input, target)
        
        return loss

# class MSELossLPIPS(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss_lpips = lpips.LPIPS(net='vgg')
#         self.loss_lpips = self.loss_lpips.cuda()
#     def forward(self, input, target, weight_MSE=1, weight_LPIPS=255):
#         loss_mse = nn.MSELoss(reduction="mean")
 
#         loss1 = loss_mse(input, target)
        
#         input = torch.squeeze(input)
#         target = torch.squeeze(target)
#         target = target.clamp(-1., 1.)
#         input = input.clamp(-1., 1.)
#         # input = input.cpu()
#         # target = target.to("cuda")
#         loss2 = self.loss_lpips(input, target)
        
#         return weight_MSE*loss1, weight_LPIPS*loss2
    
class Loss3(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_lpips = lpips.LPIPS(net='vgg')
        self.loss_lpips = self.loss_lpips.cuda()
        self.loss_L1 = nn.L1Loss(reduction = "mean")

        
    def forward(self, input, target, weight_l1=0.7, weight_LPIPS=0.3): 
        loss1 = self.loss_L1(input, target)
        loss4 = self.loss_lpips(input, target)
        loss4 = loss4.mean()
        return weight_l1*loss1, weight_LPIPS*loss4
    
class Loss5(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_lpips = lpips.LPIPS(net='vgg')
        self.loss_lpips = self.loss_lpips.cuda()

    def forward(self, input, target):
        input = torch.squeeze(input)
        target = torch.squeeze(target)
        input = torch.unsqueeze(input,0)
        input = torch.unsqueeze(input,0)
        target = torch.unsqueeze(target,0)          
        target = torch.unsqueeze(target,0)  
        loss = self.loss_lpips(input, target)
        
        return loss

def CenterCrop(tensor,ccsize=224):
    """
    

    Parameters
    ----------
    tensor : A 4-Dimensional pytorch tensor, shaped [BS, C, H, W]
    ccsize : Type: positive integer
        DESCRIPTION. The default is 224. Size of the square shaped window. 

    Returns
    -------
    tensor : 4-Dimensional pytorch tensor, shaper [BS, C, ccsize, ccsize], 
    extracted from the input parameter 'tensor'. It will be the center-cropped 
    tensor.

    """
    bs,c,h,w = tensor.shape
    h0 = ceil(h/2)-1 - ceil(ccsize/2)
    h1 = ceil(h/2)-1 + ceil(ccsize/2)
    w0 = ceil(w/2)-1 - ceil(ccsize/2)
    w1 = ceil(w/2)-1 + ceil(ccsize/2)
    tensor = tensor[:,:,h0:h1,w0:w1]
    return tensor

class VGGWPL(torch.nn.Module):
    #source:https://gist.github.com/brucemuller/37906a86526f53ec7f50af4e77d025c9
    def __init__(self, resize=True):
        super(VGGWPL, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0], style_layers=[]):
        #input and target shapes are [BS, C, H, W], with C being 1 for DCCR-Net
        
        bs, c, h, w =  input.shape
        p_size = 224
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize or (h,w)<(p_size,p_size):
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            nstacked = 1.0
        else:
            if p_size/h < 0.7 or p_size/w < 0.7:
                npat_h = ceil(h/p_size)
                npat_w = ceil(w/p_size)
                stride_h = h-p_size
                stride_w = w-p_size
                input_patches = input.unfold(2,p_size,stride_h).unfold(3,p_size,stride_w)
                target_patches = target.unfold(2,p_size,stride_h).unfold(3,p_size,stride_w)
                nstacked = npat_h*npat_w
                input = input_patches.reshape((bs*nstacked,3,p_size,p_size))
                target = target_patches.reshape((bs*nstacked,3,p_size,p_size))
            else:
                input = CenterCrop(input)
                target = CenterCrop(target)
                nstacked = 1.0
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                # if i == 0:
                #     lambda_n = torch.abs(lambda_1)
                #     loss_VGG_1 = torch.nn.functional.l1_loss(x, y)
                # elif i == 1:
                #     lambda_n = torch.abs(lambda_2)
                #     loss_VGG_2 = torch.nn.functional.l1_loss(x, y)
                # elif i == 2:
                #     lambda_n = torch.abs(lambda_3)
                #     loss_VGG_3 = torch.nn.functional.l1_loss(x, y)
                # elif i == 3:
                #     lambda_n = torch.abs(lambda_4)
                #     loss_VGG_4 = torch.nn.functional.l1_loss(x, y)
                # elif i == 4:
                #     lambda_n = torch.abs(lambda_5)
                #     loss_VGG_5 = torch.nn.functional.l1_loss(x, y)
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
# class MCLF(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, input, target):
        

# class Loss3(nn.Module):
#     def __init__(self):
#         super().__init__()   
#         self.loss_L1 = nn.L1Loss(reduction = "mean")
#         self.msssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        
        
#     def forward(self, input, target, weight_L1=100, weight_MSSSIM = 100, alpha = 0.84):
#         #Input was size 448x448 and target was 1x1x448x448 before this operation during training, so we set the size manually to prevent further errors
#         template1 = torch.BoolTensor(1,448,448)
#         input = input.reshape(template1.shape)
#         target = target.reshape(template1.shape)
        
#         loss1 = self.loss_L1(input, target)
#         loss4 = 1-self.msssim_loss(torch.unsqueeze(input,0),torch.unsqueeze(target,0))
#         if numpy.isnan(loss4.item()):
#             print("WARNING: NAN VALUE OF MSSSIM LOSS GENERATED")
#         loss4 = (1-alpha)*weight_MSSSIM*loss4
#         loss1 = alpha*weight_L1*loss1

#         return loss1, loss4
    
# class Loss4(nn.Module):
#     def __init__(self):
#         super().__init__()   
#         self.loss_L1 = nn.L1Loss(reduction = "mean")
#         self.msssim_loss = 1-SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        
        
#     def forward(self, input, target, weight_L1=100, weight_MSSSIM = 100, alpha = 0.84):
#         #Input was size 448x448 and target was 1x1x448x448 before this operation during training, so we set the size manually to prevent further errors
#         template1 = torch.BoolTensor(1,448,448)
#         input = input.reshape(template1.shape)
#         target = target.reshape(template1.shape)
        
#         loss1 = self.loss_L1(input, target)
#         loss4 = self.msssim_loss(torch.unsqueeze(input,0),torch.unsqueeze(target,0))
#         if numpy.isnan(loss4.item()):
#             print("WARNING: NAN VALUE OF MSSSIM LOSS GENERATED")
#         loss4 = (1-alpha)*weight_MSSSIM*loss4
#         loss1 = alpha*weight_L1*loss1

#         return loss1, loss4