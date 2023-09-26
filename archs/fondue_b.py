import torch
from torch import nn
import random

__all__ = ['FONDUE']


class CBU(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # self.relu = nn.PReLU(num_parameters=middle_channels,init=0.1)
        self.relu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #BLOCK_1
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        x0_bn = torch.unsqueeze(x, 4)
        x1_bn = torch.unsqueeze(out, 4)  # Add Singleton Dimension along 5th
        x1 = torch.cat((x1_bn, x0_bn), dim=4)  # Concatenating along the 5th dimension
        x1_max, _ = torch.max(x1, 4)
        
        x1=self.relu1(x1_max)
        x1 = self.conv2(x1)
        x2_bn = self.bn2(x1)
        
        x2_bn = torch.unsqueeze(x2_bn, 4)
        x1_max = torch.unsqueeze(x1_max, 4)
        x2 = torch.cat((x2_bn, x1_max), dim=4)  # Concatenating along the 5th dimension
        x2_max, _ = torch.max(x2, 4)
        
        x2 = self.relu1(x2_max)
        out = self.conv3(x2)
        out = self.bn3(out)

        return out
    
class CBUInput(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        # self.relu = nn.PReLU()
        self.relu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #Batch normalization for input:
        x0_bn = self.bn0(x)
        x0 = self.conv1(x0_bn)
        x1_bn = self.bn1(x0)
        
        x1 = self.relu1(x1_bn)
        x1 = self.conv2(x1)
        x2_bn = self.bn2(x1)
        
        #First MaxOut
        x1_bn = torch.unsqueeze(x1_bn, 4) # [BS, C, H, W, 1]
        x2_bn = torch.unsqueeze(x2_bn, 4)  # Add Singleton Dimension along 5th [BS, C, H, W, 1]
        x2 = torch.cat((x2_bn, x1_bn), dim=4)  # Concatenating along the 5th dimension [BS, C, H, W, 2]
        x2_max, _ = torch.max(x2, 4) #[BS, C, H, W, 1]
        
        out=self.relu1(x2_max)
        return out

class FONDUE(nn.Module):
    def __init__(self, num_classes, input_channels=7, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = 64

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2), padding = (0,0), return_indices = True, ceil_mode = True)
        self.up_even = nn.MaxUnpool2d(kernel_size = (2,2), stride = (2,2), padding = (0,0))
        
        self.up_odd_both = nn.MaxUnpool2d(kernel_size = (1,1), stride = (2,2), padding = (0,0))
        self.up_odd_h = nn.MaxUnpool2d(kernel_size = (1,2), stride = (2,2), padding = (0,0))
        self.up_odd_w = nn.MaxUnpool2d(kernel_size = (2,1), stride = (2,2), padding = (0,0))
        
        self.lambda_1 = nn.Parameter(torch.ones(1)*(1/4))
        self.lambda_2 = nn.Parameter(torch.ones(1)*(1/4))
        self.lambda_3 = nn.Parameter(torch.ones(1)*(1/4))
        self.lambda_4 = nn.Parameter(torch.ones(1)*(1/4))
        
        self.alpha_1 = nn.Parameter(torch.ones(1)*(1/6))
        self.alpha_2 = nn.Parameter(torch.ones(1)*(1/6))
        self.alpha_3 = nn.Parameter(torch.ones(1)*(1/6))
        self.alpha_4 = nn.Parameter(torch.ones(1)*(1/6))
        self.alpha_5 = nn.Parameter(torch.ones(1)*(1/6))
        self.alpha_6 = nn.Parameter(torch.ones(1)*(1/6))
        
        
        self.conv0_0 = CBUInput(input_channels, nb_filter, nb_filter)
        self.conv1_0 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv2_0 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv3_0 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv4_0 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv5_0 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv6_0 = CBU(nb_filter, nb_filter, nb_filter)

        self.conv0_1 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv1_1 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv2_1 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv3_1 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv4_1 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv5_1 = CBU(nb_filter, nb_filter, nb_filter)

        self.conv0_2 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv1_2 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv2_2 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv3_2 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv4_2 = CBU(nb_filter, nb_filter, nb_filter)

        self.conv0_3 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv1_3 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv2_3 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv3_3 = CBU(nb_filter, nb_filter, nb_filter)

        self.conv0_4 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv1_4 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv2_4 = CBU(nb_filter, nb_filter, nb_filter)
        
        self.conv0_5 = CBU(nb_filter, nb_filter, nb_filter)
        self.conv1_5 = CBU(nb_filter, nb_filter, nb_filter)
        
        self.conv0_6 = CBU(nb_filter, nb_filter, nb_filter)
        self.norm_intensity = nn.Hardtanh(min_val = 0., max_val = 1., inplace = True)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
            self.final6 = nn.Conv2d(nb_filter, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter*4, num_classes, kernel_size=1)
        #self.final = nn.ConvTranspose2d(nb_filter, num_classes, kernel_size=1)
        # self.final = self.final.cuda()
        

    

    

    def forward(self, input, orig_zoom):
        
        # def ceil_next_even(f): #rounds to closest number divisible by 32 with integer remainder
        #     return 32*round((f / 32.))
        
        def ceil_next_even(f): #rounds to closest number divisible by 32 with integer remainder
            return 32*round((f / 32.))
        
        # def dim_changed(before, after):
        #     h_before = before.size()[2]
        #     w_before = before.size()[3]
            
        #     h_after = after.size()[2]
        #     w_after = after.size()[3]
            
        #     if h_before/2.0 != h_after:
        #         h_ch = True
        #     else:
        #         h_ch = False
        #     if w_before/2.0 != w_after:
        #         w_ch = True
        #     else:
        #         w_ch = False
            
        #     if h_ch == True and w_ch == False:
        #         return "only_h"
        #     elif h_ch == False and w_ch == True:
        #         return "only_w"
        #     elif h_ch == True and w_ch == True:
        #         return "both"
        #     else:
        #         return "none"
        
        # def unpool(dim_changed, filters, indices):
        #     if dim_changed == "only_h":
        #         return self.up_odd_h(filters, indices)
        #     elif dim_changed == "only_w":
        #         return self.up_odd_w(filters, indices)
        #     elif dim_changed == "both":
        #         return self.up_odd_both(filters, indices)
        #     else:
        #         return self.up_even(filters, indices)
        
        def dim_changed(before, after):
            h_before = before.size()[2]
            w_before = before.size()[3]
            
            h_after = after.size()[2]
            w_after = after.size()[3]
            
            if h_before/2.0 != h_after:
                h_ch = True
            else:
                h_ch = False
            if w_before/2.0 != w_after:
                w_ch = True
            else:
                w_ch = False
            
            if h_ch == True and w_ch == False:
                return "only_h"
            elif h_ch == False and w_ch == True:
                return "only_w"
            elif h_ch == True and w_ch == True:
                return "both"
            else:
                return "none"
        
        def unpool(dim_changed, filters, indices):
                return self.up_even(filters, indices)
        
        input = torch.squeeze(input)
        shapes = input.size()
        # if len(input.size()) == 3 and shapes[0] == 7:
        #     print("size of input is: " + str(input.size()) + ". Attempting unsqueeze at dim=1")
        #     input = torch.unsqueeze(input,1)
        #     print("New input size is: " + str(input.size()))
        # elif len(input.size()) == 3 and shapes[0] != 7:
        #     print("Error: first dimension is not thick slice shape")
        if len(input.size()) == 3 and shapes[0] == 7:
            # print("size of input is: " + str(input.size()) + ". Attempting unsqueeze at dim=0")
            input = torch.unsqueeze(input,0)
            shapes = input.size()
            # print("New input size is: " + str(input.size()))
        elif len(input.size()) == 3 and shapes[0] != 7:
            print("Error: first dimension is not thick slice shape")
        # bs = shapes[0]
        # nfilt = shapes[1]
        
        h = shapes[2]
        w = shapes[3]
        # orig_shape2 = (bs, nfilt, h, w)
        orig_shape = (h,w)
        
        
        inner_zoom = 0.5
        orig_zoom = orig_zoom[0]
        alpha = random.gauss(0, 0.1)
        factor = (inner_zoom / orig_zoom) + alpha
        # factor = factor.item()
        # inner_shape2 = (bs, nfilt, ceil_next_even(h*factor), ceil_next_even(w*factor))
        inner_shape = (ceil_next_even(h/factor), ceil_next_even(w/factor))
        #x0_0:
        x0_0 = self.conv0_0(input) #input is [BS, 7, max_size, max_size]. x0_0 is [BS, Filtnum, max_size, max_size] and max_size is 320
        
        x0_02 = torch.nn.functional.interpolate(x0_0, size=inner_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        x1_0 = self.conv1_0(x0_02) #"x1_0" is [BS, Filtnum, max_size/2, max_size/2]
        
        #x0_1:
        #Same_level_left_side:
        c1 = x0_0 #[BS, Filtnum, max_size, 448]
        c1 = torch.unsqueeze(c1,4)#[BS, Filtnum, max_size, max_size, 1]
        #Diagonal_inferior
        c2 = torch.nn.functional.interpolate(x1_0, size=orig_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        c2 = torch.unsqueeze(c2,4) #[BS, Filtnum, max_size, max_size]
        c_all = torch.cat((c1,c2),dim=4) #[BS, Filtnum, max_size, max_size, 1]
        c_max, _ = torch.max(c_all,4) #[BS, Filtnum, max_size, max_size, 2]
        x0_1 = self.conv0_1(c_max) #[BS, Filtnum, max_size, max_size, 1]
        
        pool_x1_0, indices_x1 = self.pool(x1_0) #"pool_x1_0" is [BS, Filtnum, max_size/4, max_size/4]       
        dim_change_1 = dim_changed(x1_0, pool_x1_0)
        x2_0 = self.conv2_0(pool_x1_0)#"x2_0 is [BS, Filtnum, max_size/4, max_size/4]
        
        c1 = x1_0
        c1 = torch.unsqueeze(c1,4)
        
        
        c2 = unpool(dim_change_1, x2_0, indices_x1)
        c2 = torch.unsqueeze(c2,4)
        c_all = torch.cat((c1,c2),dim=4)
        c_max, _ = torch.max(c_all,4)
        x1_1 = self.conv1_1(c_max)
        
        #x0_2:
        #Same_level_left_side:
        c1 = x0_0
        c1 = torch.unsqueeze(c1,4)
        c2 = x0_1
        c2 = torch.unsqueeze(c2,4)
        #Diagonal inferior:
        c3 = torch.nn.functional.interpolate(x1_1, size=orig_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        c3 = torch.unsqueeze(c3,4)
        c_all = torch.cat((c1,c2,c3),dim=4)
        c_max, _ = torch.max(c_all,4)
        x0_2 = self.conv0_2(c_max) #2GB added to VRAM
        
        pool_x2_0, indices_x2 = self.pool(x2_0)
        dim_change_2 = dim_changed(x2_0, pool_x2_0)
        x3_0 = self.conv3_0(pool_x2_0)
        
        #Computing x2_1:
        #Same_level_left_side:
        c1 = x2_0
        c1 = torch.unsqueeze(c1,4)
        #Diagonal_inferior:
        c2 = unpool(dim_change_2, x3_0, indices_x2)
        c2 = torch.unsqueeze(c2,4)
        c_all = torch.cat((c1,c2),dim=4)
        c_max, _ = torch.max(c_all,4)
        x2_1 = self.conv2_1(c_max)
        
        #Computing x1_2:
        #Same_level_left_side:
        c1 = x1_0
        c1 = torch.unsqueeze(c1,4)
        c2 = x1_1
        c2 = torch.unsqueeze(c2,4)
        #Diagonal_inferior:
        c3 = unpool(dim_change_1,x2_1,indices_x1)
        c3 = torch.unsqueeze(c3,4)
        c_all = torch.cat((c1,c2,c3),dim=4)
        c_max, _ = torch.max(c_all,4)
        x1_2 = self.conv1_2(c_max)
        
        #Computing x0_3:
        #Same_level_left_side
        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = torch.nn.functional.interpolate(x1_2, size=orig_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c_all = torch.cat((c1,c2,c3,c4),dim=4)
        c_max, _ = torch.max(c_all,4)
        x0_3 = self.conv0_3(c_max) #2GB added to the model
        pool_x3_0, indices_x3 = self.pool(x3_0)
        dim_change_3 = dim_changed(x3_0, pool_x3_0)
        x4_0 = self.conv4_0(pool_x3_0)
        
        #Computing x3_1:
        #Same_level_left_side
        c1 = x3_0
        c2 = unpool(dim_change_3, x4_0, indices_x3)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c_all = torch.cat((c1,c2),dim=4)
        c_max, _ = torch.max(c_all,4)
        x3_1 = self.conv3_1(c_max)
        
        
        c1 = x2_0
        c2 = x2_1
        c3 = unpool(dim_change_2, x3_1, indices_x2)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c_all = torch.cat((c1,c2,c3),dim=4)
        c_max, _ = torch.max(c_all,4)
        x2_2 = self.conv2_2(c_max) #Until here theres 20.2GB of VRAM used
        
        
        c1 = x1_0
        c2 = x1_1
        c3 = x1_2
        c4 = unpool(dim_change_1, x2_2, indices_x1)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c_all = torch.cat((c1,c2,c3,c4),dim=4)
        c_max, _ = torch.max(c_all,4)
        x1_3 = self.conv1_3(c_max)
        
        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = x0_3
        c5 = torch.nn.functional.interpolate(x1_3, size=orig_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c5 = torch.unsqueeze(c5,4)
        c_all = torch.cat((c1,c2,c3,c4,c5),dim=4)
        c_max, _ = torch.max(c_all,4) #Up until here theres 22.2GB of VRAM used
        x0_4 = self.conv0_4(c_max)
        
        pool_x4_0, indices_x4 = self.pool(x4_0)
        dim_change_4 = dim_changed(x4_0, pool_x4_0)
        x5_0 = self.conv5_0(pool_x4_0)
        
        c1 = x4_0
        
        c2 = unpool(dim_change_4, x5_0, indices_x4)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c_all = torch.cat((c1,c2),dim=4)
        c_max, _ = torch.max(c_all,4)
        x4_1 = self.conv4_1(c_max)
        
        c1 = x3_0
        c2 = x3_1
        c3 = unpool(dim_change_3, x4_1, indices_x3)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c_all = torch.cat((c1,c2,c3),dim=4)
        c_max, _ = torch.max(c_all,4)
        x3_2 = self.conv3_2(c_max) 
        
        c1 = x2_0
        c2 = x2_1
        c3 = x2_2
        c4 = unpool(dim_change_2, x3_2, indices_x2)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c_all = torch.cat((c1,c2,c3,c4),dim=4)
        c_max, _ = torch.max(c_all,4)
        x2_3 = self.conv2_3(c_max)
        
        c1 = x1_0
        c2 = x1_1
        c3 = x1_2
        c4 = x1_3
        c5 = unpool(dim_change_1, x2_3, indices_x1)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c5 = torch.unsqueeze(c5,4)
        c_all = torch.cat((c1,c2,c3,c4,c5),dim=4)
        c_max, _ = torch.max(c_all,4) #Up until here theres 22.2GB of VRAM used
        x1_4 = self.conv1_4(c_max)
        
        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = x0_3
        c5 = x0_4
        # c6 = self.up(x1_4, indices_x0)
        c6 = torch.nn.functional.interpolate(x1_4, size=orig_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c5 = torch.unsqueeze(c5,4)
        c6 = torch.unsqueeze(c6,4)
        c_all = torch.cat((c1,c2,c3,c4,c5,c6),dim=4)
        c_max, _ = torch.max(c_all,4) #Up until here theres 22.2GB of VRAM used
        x0_5 = self.conv0_5(c_max)
        
        pool_x5_0, indices_x5 = self.pool(x5_0)
        dim_change_5 = dim_changed(x5_0, pool_x5_0)
        x6_0 = self.conv6_0(pool_x5_0)
        
        c1 = x5_0
        c2 = unpool(dim_change_5, x6_0, indices_x5)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c_all = torch.cat((c1,c2),dim=4)
        c_max, _ = torch.max(c_all,4)
        x5_1 = self.conv5_1(c_max)
        
        c1 = x4_0
        c2 = x4_1
        c3 = unpool(dim_change_4, x5_1, indices_x4)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c_all = torch.cat((c1,c2,c3),dim=4)
        c_max, _ = torch.max(c_all,4)
        x4_2 = self.conv4_2(c_max) 
        
        c1 = x3_0
        c2 = x3_1
        c3 = x3_2
        c4 = unpool(dim_change_3, x4_2, indices_x3)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c_all = torch.cat((c1,c2,c3,c4),dim=4)
        c_max, _ = torch.max(c_all,4)
        x3_3 = self.conv3_3(c_max)
        
        c1 = x2_0
        c2 = x2_1
        c3 = x2_2
        c4 = x2_3
        c5 = unpool(dim_change_2, x3_3, indices_x2)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c5 = torch.unsqueeze(c5,4)
        c_all = torch.cat((c1,c2,c3,c4,c5),dim=4)
        c_max, _ = torch.max(c_all,4) #Up until here theres 22.2GB of VRAM used
        x2_4 = self.conv2_4(c_max)
        
        c1 = x1_0
        c2 = x1_1
        c3 = x1_2
        c4 = x1_3
        c5 = x1_4
        c6 = unpool(dim_change_1, x2_4, indices_x1)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c5 = torch.unsqueeze(c5,4)
        c6 = torch.unsqueeze(c6,4)
        c_all = torch.cat((c1,c2,c3,c4,c5,c6),dim=4)
        c_max, _ = torch.max(c_all,4) #Up until here theres 22.2GB of VRAM used
        x1_5 = self.conv1_5(c_max)
        
        c1 = x0_0
        c2 = x0_1
        c3 = x0_2
        c4 = x0_3
        c5 = x0_4
        c6 = x0_5
        # c7 = self.up(x1_5, indices_x0)
        c7 = torch.nn.functional.interpolate(x1_5, size=orig_shape, mode='bilinear', align_corners=None, recompute_scale_factor=None)
        c1 = torch.unsqueeze(c1,4)
        c2 = torch.unsqueeze(c2,4)
        c3 = torch.unsqueeze(c3,4)
        c4 = torch.unsqueeze(c4,4)
        c5 = torch.unsqueeze(c5,4)
        c6 = torch.unsqueeze(c6,4)
        c7 = torch.unsqueeze(c7,4)
        c_all = torch.cat((c1,c2,c3,c4,c5,c6,c7),dim=4)
        c_max, _ = torch.max(c_all,4) #Up until here theres 22.2GB of VRAM used
        x0_6 = self.conv0_6(c_max)
        
        

        if self.deep_supervision:
            output1 = torch.mul(self.final1(x0_1),self.alpha_1)
            output2 = torch.mul(self.final2(x0_2),self.alpha_2)
            output3 = torch.mul(self.final3(x0_3),self.alpha_3)
            output4 = torch.mul(self.final4(x0_4),self.alpha_4)
            output5 = torch.mul(self.final4(x0_5),self.alpha_5)
            output6 = torch.mul(self.final4(x0_6),self.alpha_6)
            alpha_sum = self.alpha_1 + self.alpha_2 + self.alpha_3 + self.alpha_4 + self.alpha_5 + self.alpha_6
            output = (output1 + output2 + output3 + output4 + output5 + output6) / alpha_sum
            input = input[:,3]
            input = torch.unsqueeze(input,1)
            denoised = input - output
            denoised = self.norm_intensity(denoised)
            return denoised, self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, self.alpha_5, self.alpha_6

        # else:
        #     output = self.final(x0_4)
        #     return input-output
        
        else:
            supercat = torch.cat((x0_1, x0_2, x0_3, x0_4),dim=1)
            #supercat = torch.max(supercat,dim=1)
            output = self.final(supercat) #should be size [BS, 1, 448, 448]
            #output = self.bbproj(output)
            
            input = input[:,3,:,:] #To convert output from [BS, 7, 448, 448] to [BS, 448, 448].
            input = torch.unsqueeze(input,1) #Output has shape [BS, 448, 448] as of now
            return input-output
