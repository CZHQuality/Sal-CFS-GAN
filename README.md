Please download the "versionGitHub"

The main contributions of the proposed model :
1. The proposed saliency detection model is based on Generative Adversarial Model. The generator is based on an U-Net (with Center-Surround Connection, optional) and a ResNet Block, and the discriminator contains 4 convolutional layers. 
2. The loss function is a weighted sum of the following losses. Pixel-wise losses: L1 loss, KLD loss, CC loss, NSS loss, and we also propose a new  Histogram loss (Alternative Chi-Square) which can improve the smoothness of the generated saliency maps (tending to the ground truth human gaze maps), and has the higher correlation with the popular sAUC metric.
3. We establish a new eye-movement database which contains several kinds of common distortions and transformations. And we proposed a valid data augmentation strategy for saliency prediction field based on sufficient experiments. And we use the proposed data augmentation strategy to boost the deep saliency model performance.

How to use our model from scratch?
1. Data Pre-processing Step : Run "build_dataset_New_3.py" to transfer the training and testing images as .npy format (Notice that you should change the image path in our code as your own file path)
2. Training Step : Run "example_New_4.py" to train the model from scratch. Besides, you can also run "example_New_5.py" to fine-tune the pre-trained model on your own task-specific datasets.
3. Testing Step : Run "val_New_3.py" to generate the predicted saliency maps and save the generated results to the appointed file path. 

Notice : You can download the pre-trained model at : https://mega.nz/#!Jr4GDaaA!e_0VC360LBLIVE5lYfrJGDlvYHK6wtzys9q30Aihzjs

The code is highly inspired by the following projects, and thanks for their contributions : 
https://github.com/cvzoya/saliency
https://github.com/Eyyub/tensorflow-pix2pix
https://github.com/marcellacornia/sam
https://github.com/TadasBaltrusaitis/OpenFace
