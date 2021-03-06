# GazeGAN: A New Saliency Prediction Model 

# This work has been Accepted by IEEE Transactions on Image Processing: https://ieeexplore.ieee.org/abstract/document/8866748/algorithms#algorithms


# Citation:

```
@article{ZhaohuiTIP19,
  title={How is Gaze Influenced by Image Transformations? Dataset and Model},
  author={Z. Che and A. Borji and G. Zhai and X. Min and G. Guo and P. L. Callet},  
  booktitle={IEEE Transactions on Image Processing},
  year={2019}
}
```

**Leaderboard of MIT300 Benchmark：**
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/mit300_2.jpg)

# The scores of our model on the MIT300 benchmark (See https://saliency.tuebingen.ai for more details):

AUC metric: 0.86070

shuffled AUC metric: 0.73163

Normalized Scanpath Saliency metric: 2.21181

Cross-correlation metric: 0.75789

KL divergence metric: 1.33904

Similarity metric: 0.64914

# The scores of our model on the CAT2000 benchmark (See https://saliency.tuebingen.ai for more details):

AUC metric: 0.87298

shuffled AUC metric: 0.59597

Normalized Scanpath Saliency metric: 2.31690

Cross-correlation metric: 0.82201

KL divergence metric: 0.71665

Similarity metric: 0.68483




We update the  **"Pytorch version"** !!! Please refer to **"GazeGAN_using_CSC"** and **"GazeGAN_LocalGlobal_Pytorch"**.
1. **"GazeGAN_using_CSC"** utilizes local and global modified U-Nets equipped with cross-scale Center-Surround-Connections, 
2. while **"GazeGAN_LocalGlobal_Pytorch"** utilizes local and global modified U-Nets without using Center-Surround-Connections, which serves as a **baseline** for comparison.

We also provide the **Tensorflow version**. Notice that the tensorflow version only utilizes Local modified U-Net without using Global U-Net.
It serves as a baseline for comparison.
The tensorflow codes of this repository are listed in **"versionGitHub"**.

**Requirements:**
1. display memory: at least 3000MiB.
2. Pytorch version: 3.5.2
3. Tensorflow version: 1.2.0

The **main contributions** of the proposed model (called as **GazeGAN**):
1. The proposed saliency detection model is based on Generative Adversarial Model. The generator is based on an U-Net (with Center-Surround Connection, optional) and a ResNet Block, and the discriminator contains 4 convolutional layers. 
2. The loss function is a weighted sum of the following losses. Pixel-wise losses: L1 loss, KLD loss, CC loss, NSS loss, and we also propose a new  Histogram loss (Alternative Chi-Square) which can improve the smoothness of the generated saliency maps (tending to the ground truth human gaze maps), and has the higher correlation with the popular sAUC metric.
3. We establish a new eye-movement database which contains several kinds of common distortions and transformations. And we proposed a valid data augmentation strategy for saliency prediction field based on sufficient experiments. And we use the proposed data augmentation strategy to boost the deep saliency model performance.


**Model Architecture of GazeGAN：**
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/ModelArchi11.jpg)

 
**How to use our model (Pytorch) from scratch?**
1. Set the training/testing path in "./GazeGAN_LocalGlobal_Pytorch/options/base_options.py" ('--dataroot') and "./GazeGAN_LocalGlobal_Pytorch/data/aligned_dataset.py" (dir_A, dir_B and dir_C represent the subpaths of source image, dense saliency map, and discrete fixation map)
2. Training Command: "python3 train.py --name label2city_512p --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none"
3. Fine-tuning Command: "python3 train.py --name label2city_512p --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none --continue_train"
4. Inference Command: "python3 test.py --name label2city_512p --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance"

**How to use our model (tensorflow) from scratch?**

**Warnings: The tensorflow code is the initial version of this repository. After that, we do some slight changes and improvements to the initial version. Therefore, the performance of the tensorflow code is not as good as our final version (Pytorch code). Currently, we recommend the users to adopt the Pytorch version, which is consistent with our final paper. We will update the final version of the tensorflow code as soon as possible.**

1. Data Pre-processing Step : Run "build_dataset_New_3.py" to transfer the training and testing images as .npy format (Notice that you should change the image path in our code as your own file path)
2. Training Step : Run "example_New_4.py" to train the model from scratch. Besides, you can also run "example_New_5.py" to fine-tune the pre-trained model on your own task-specific datasets.
3. Testing Step : Run "val_New_3.py" to generate the predicted saliency maps and save the generated results to the appointed file path.

Notice : You can download the **pre-trained model** at : 

1. https://mega.nz/#!Jr4GDaaA!e_0VC360LBLIVE5lYfrJGDlvYHK6wtzys9q30Aihzjs (tensorflow, trained on SALICON dataset)
2. https://mega.nz/#F!snwADCLa!FsfL7WoGcpd8LHecz8J0vg (Pytorch, without using Center-Surround-Connection,  trained on SALICON dataset)
3. https://mega.nz/#F!Qj4DzIJQ!EdnZ77GPeSp0X3Mhxa0lpQ (Pytorch, using Center-Surround-Connection,  trained on SALICON dataset)
4. https://mega.nz/#F!kyhjTIIS!kztBfqPJrNucXC3O0IaOCA (Pytorch, using Center-Surround-Connection,  pretrained on SALICON dataset, then fine-tuned on MIT1003 dataset)
5. https://drive.google.com/drive/folders/1cSrgqXdPQbt2_eNz_sdOpGj6YmJzsqqn (If you can NOT download the pre-trained models from MEGA cloud, try Google-Drive. This link contains the same pre-trained models as "3." and "4.")

For visualization our performance, we list the saliency maps generated by our model on **MIT300 Benchmark Dataset** as **"ResultsOnMIT300.rar"**

The **proposed dataset** with different transformations are available at : 
1. https://drive.google.com/drive/folders/1qXVU6deYqdM2ZTyJQTxJyVWId9bTUceJ?usp=sharing    (Google Drive, 3.7GB in total, including the high-resolution source images)
or
2. https://mega.nz/#!kn5B1CaJ!fIISTsQp4ox8CTHfyZwP-yE1Enml55UyrrtDlObkRlE    (MEGA cloud, 2.68GB in total. High-resolution images are not included here)



The supplementary material (containing more finer-grained visualizations and comparisons) of this project is available at: 
https://drive.google.com/file/d/151CbvR1dGDXnMzt50gjR-BJYeQT5U5Ml/view?usp=sharing

The code is heavily inspired by the following projects, and thanks for their great contributions:

1. **"https://github.com/NVIDIA/pix2pixHD"**
2. **"https://github.com/cvzoya/saliency"** 
3. **"https://github.com/Eyyub/tensorflow-pix2pix"** 
4. **"https://github.com/marcellacornia/sam"**   
5. **"https://github.com/TadasBaltrusaitis/OpenFace"**

**Figures represent Visualizations on MIT300 benchmark: (The first column is the predicted saliency map, the second column is the original image, the ground-truth are not published)**
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_1.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_2.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_3.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_4.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_6.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_7.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_8.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_9.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_10.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_11.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_12.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_13.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_14.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_15.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_16.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_17.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_18.jpg)
![image](https://github.com/CZHQuality/Sal-CFS-GAN/blob/master/Screenshot_19.jpg)

 
