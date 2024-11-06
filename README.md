# Welcome to the CNN side of the stain app!

- ### This model is trained using the [FABRIC STAIN DATASET](https://www.kaggle.com/datasets/priemshpathirana/fabric-stain-dataset).
- ### And then using a [Mask-RCNN Trained on the coco dataset](https://github.com/matterport/Mask_RCNN), we retrain the model on our new stain data.
- ### I used followed [this tutorial](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) for the most part.

## To run this model:
### 1. Run stains_con_cotton_cnn.ipynb
### 2. Run the command `python3 stain.py train --dataset=/path/to/dataset --model=coco`