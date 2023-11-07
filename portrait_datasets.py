import os, cv2, copy
import torch.utils.data as data
import numpy as np
from augmentation import *

class PortraitDataset(data.Dataset):
    def __init__(self, args, split):
        self.args = args 
        self.split = split
        self.istrain = split == 'train'
        self.img_ids = []
        if 'supervisely_face' in args.dataset_list:
            img_ids = open(os.path.join(args.root_path, getattr(args.supervisely_face, split))).readlines()
            img_ids = [(path, 'supervisely_face') for path in img_ids]
            self.img_ids += img_ids
        
        if 'eg1800' in args.dataset_list:
            img_ids = open(os.path.join(args.root_path, getattr(args.eg1800, split))).readlines()
            img_ids = [(path, 'eg1800') for path in img_ids]
            self.img_ids += img_ids
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_path, source = self.img_ids[index]

        if source == 'supervisely_face':
            img_path = os.path.join(self.args.root_path, self.args.supervisely_face.root, img_path.strip())
            annopath = img_path.replace('/img/', '/ann/')
            img_name = img_path[img_path.rfind('/')+1:]
            img = cv2.imread(img_path)
            mask = cv2.imread(annopath, 0)
            mask[mask>0] = 1
        elif source == 'eg1800':
            img_path = os.path.join(self.args.root_path, self.args.eg1800.root, img_path.strip())
            annopath = img_path.replace('Images', 'Labels').replace('.jpg', '.png')
            img_name = img_path[img_path.rfind('/')+1:]
            img = cv2.imread(img_path)
            mask = cv2.imread(annopath, 0)
            mask[mask>1] = 0
            # print(annopath)
            # print(mask[mask>0])
        else:
            raise NotImplementedError

        height, width, channel = img.shape
        bbox = [0, 0, width-1, height-1]
        H = aug_matrix(width, height, bbox, self.args.input_width, self.args.input_height,
                    angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.args.input_height/4)
        
        if not self.istrain:
            img_aug_ori, mask_aug_ori = padding(img, mask, size=self.args.input_width, padding_color=self.args.portrait.padding_color)
            
            # ===========add new channel for video stability============
            input_norm = Normalize_Img(img_aug_ori, scale=self.args.img_scale, mean=self.args.img_mean, val=self.args.img_val)
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = copy.deepcopy(input)
        else:
            img_aug = cv2.warpAffine(np.uint8(img), H, (self.args.input_width, self.args.input_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(self.args.padding_color, self.args.padding_color, self.args.padding_color)) 
            mask_aug = cv2.warpAffine(np.uint8(mask), H, (self.args.input_width, self.args.input_height), 
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
            prior = np.zeros((self.args.input_height, self.args.input_width, 1))
            
            # add augmentation
            img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))  
            img_aug = data_aug_color(img_aug)
            img_aug = np.asarray(img_aug)
            img_aug = data_aug_blur(img_aug)
            img_aug = data_aug_noise(img_aug)
            img_aug = np.float32(img_aug[:,:,::-1]) # BGR, like cv2.imread
            
            input_norm = Normalize_Img(img_aug, scale=self.args.img_scale, mean=self.args.img_mean, val=self.args.img_val)
            input_ori_norm = Normalize_Img(img_aug_ori, scale=self.args.img_scale, mean=self.args.img_mean, val=self.args.img_val)
            
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = np.transpose(input_ori_norm, (2, 0, 1))
            
        use_float_mask = False
        if use_float_mask == True:
            output_mask = cv2.resize(mask_aug_ori, (self.args.input_width, self.args.input_height), interpolation=cv2.INTER_NEAREST)
            cv2.normalize(output_mask, output_mask, 0, 1, cv2.NORM_MINMAX)
            output_mask[output_mask>=0.5] = 1
            output_mask[output_mask<0.5] = 0
        else:
            output_mask = cv2.resize(np.uint8(mask_aug_ori), (self.args.input_width, self.args.input_height), interpolation=cv2.INTER_NEAREST)
            
            # add mask blur
            output_mask = np.uint8(cv2.blur(output_mask, (5,5)))
            output_mask[output_mask>=0.5] = 1
            output_mask[output_mask<0.5] = 0

        edge = show_edge(output_mask)
        # edge_blur = np.uint8(cv2.blur(edge, (5,5)))/255.0
        # return {'input_ori': input_ori, 'input': input, 'edge': edge, 'output_mask': output_mask}
        return input_ori, input, edge, output_mask
    
    # def collate_fn(feature)


# import hydra
# @hydra.main(config_path=".", config_name="config.yaml", version_base='1.1')
# def main(args):
#     dataset = PortraitDataset(args)
#     print(dataset[0])
    
# if __name__ == '__main__':
#     main()