### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# We downsample the source image  saliency map and fixation map in this code for MySALICON model
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        # dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        # self.A_paths = sorted(make_dataset(self.dir_A))
        
        # dir_A = 'maps/train/' if self.opt.label_nc == 0 else '_label'
        dir_A = 'images/train/' if self.opt.label_nc == 0 else '_label' # in our task, sourece domain A is the Image
        # dir_A = 'images/val/' if self.opt.label_nc == 0 else '_label' # in our task, sourece domain A is the Image
        # dir_A = 'image/img/Reference/' if self.opt.label_nc == 0 else '_label' # in our task, sourece domain A is the Image£¬ for MyDistorted Dataset
        # dir_A = 'images5/' if self.opt.label_nc == 0 else '_label' # CAT2000 subset, in our task, sourece domain A is the Image
        # dir_A = 'image/' if self.opt.label_nc == 0 else '_label' # MIT1003 in our task, sourece domain A is the Image
        # dir_A = 'rescaleMIT300/' if self.opt.label_nc == 0 else '_label' # MIT300 in our task, sourece domain A is the Image
        # dir_A = 'image/img/zz_shan_difficult_examples/' if self.opt.label_nc == 0 else '_label' # in our task, sourece domain A is the Image£¬ for MyDistorted Dataset
        # dir_A = '' if self.opt.label_nc == 0 else '_label' # Saliency4ASD test subset 
        self.dir_A = os.path.join(opt.dataroot, dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        # if opt.isTrain or opt.use_encoded_image:
          #  dir_B = '_B' if self.opt.label_nc == 0 else '_img'
          #  self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
          #  self.B_paths = sorted(make_dataset(self.dir_B))
        
        if opt.isTrain or opt.use_encoded_image:
            # dir_B = 'images/train/' if self.opt.label_nc == 0 else '_img'
            dir_B = 'maps/train/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # dir_B = 'maps/val/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # dir_B = 'map/Reference/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # dir_B = 'maps5/' if self.opt.label_nc == 0 else '_img' # CAT2000 subset, in our task, sourece domain B is the Saliency Map
            # dir_B = 'map/' if self.opt.label_nc == 0 else '_img' # MIT1003, in our task, sourece domain B is the Saliency Map
            # dir_B = 'rescaleMIT300/' if self.opt.label_nc == 0 else '_img' # MIT300, in our task, sourece domain B is the Saliency Map
            # dir_B = 'image/img/zz_shan_difficult_examples/' if self.opt.label_nc == 0 else '_img' # difficult examples
            # dir_B = '' if self.opt.label_nc == 0 else '_img' # Saliency4ASD test subset 
            self.dir_B = os.path.join(opt.dataroot, dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

            dir_C = 'fixations_img/train/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # dir_C = 'fixations_img/val/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # dir_C = 'fixation_img/Reference/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # dir_C = 'fixations5/' if self.opt.label_nc == 0 else '_img' # CAT2000 subset, in our task, sourece domain B is the Saliency Map
            # dir_C = 'fixations_img/' if self.opt.label_nc == 0 else '_img' # MIT1003, in our task, sourece domain B is the Saliency Map
            # dir_C = 'rescaleMIT300/' if self.opt.label_nc == 0 else '_img' # MIT300, in our task, sourece domain B is the Saliency Map
            # dir_C = 'image/img/zz_shan_difficult_examples/' if self.opt.label_nc == 0 else '_img' # difficult examples
            # dir_C = '' if self.opt.label_nc == 0 else '_img' # Saliency4ASD test subset 
            self.dir_C = os.path.join(opt.dataroot, dir_C)  
            self.C_paths = sorted(make_dataset(self.dir_C))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)   
        A = A.resize((640, 512), Image.BILINEAR) # for SALICON only     
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
        

        # B_tensor = inst_tensor = feat_tensor = 0
        B_tensor = C_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            # B = B.resize((80, 60), Image.BILINEAR) # for SALICON only  
            B = B.resize((160, 128), Image.BILINEAR) # for SALICON only , the small size should be 64 so that the smallest dimention of feature is even number, for better feature alignment
            # B = B.resize((320, 256), Image.BILINEAR) # for SALICON only , the small size should be 64 so that the smallest dimention of feature is even number, for better feature alignment
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

            C_path = self.C_paths[index]   
            C = Image.open(C_path).convert('RGB')
            # C = C.resize((80, 60), Image.BILINEAR) # for SALICON only  
            C = C.resize((160, 128), Image.BILINEAR) # for SALICON only  
            # C = C.resize((320, 256), Image.BILINEAR) # for SALICON only  
            transform_C = get_transform(self.opt, params)      
            C_tensor = transform_C(C)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            
        
        
        # print("A_tensor :", A_tensor.size())
        # print("B_tensor :", B_tensor.size())
        # print("C_tensor :", C_tensor.size())
        

        # input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,  
          #            'feat': feat_tensor, 'path': A_path}
        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,  'fixpts': C_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'