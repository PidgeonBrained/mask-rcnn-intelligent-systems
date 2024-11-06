"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
from PIL import Image
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, "prebuilt_mask_rcnn")) # To find local version of the library
from prebuilt_mask_rcnn.mrcnn.config import Config
from prebuilt_mask_rcnn.mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "prebuilt_mask_rcnn", "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "prebuilt_mask_rcnn", "logs")

############################################################
#  Configurations
############################################################


class StainConfig(Config):
    """Configuration for training on the stain dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "stain"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2 #I'm just gonna leave this on 2 :)

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + stain

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class StainDataset(utils.Dataset):

    def load_stain(self, dataset_dir, subset):
        """Load a subset of the Stain dataset.
        dataset_dir: Root directory of the dataset. In our case, ./fabric_stain_dataset
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("stain", 1, "stain")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        image_dir = os.path.join(dataset_dir, "split_images", subset)
        annotation_dir = os.path.join(dataset_dir, "annotations", "stain")

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image_num = image_name.split(".")[0]
            annotations_path = os.path.join(annotation_dir, image_num + ".txt")
            annotations = []
            with open(annotations_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    annotations.append([float(part) for part in parts])
            
            # Get the image dimensions- balloon.py makes it sounds like loading the image in this manner is time consuming. If model takes too long to trian, we could normalize the image size.
            image = Image.open(image_path)
            image_height, image_width = image.shape

            boxes = []
            for annotation in annotations:
                class_id, x_center, y_center, width, height = annotation
                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height
                
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                boxes.append((x_min, y_min, x_max, y_max))
            
            self.add_image(
                "stain",
                image_id=image_name,  # use file number as a unique image id
                path=image_path,
                width=width, height=height, #of image itself
                boxes=boxes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a stain dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "stain":
            return super(self.__class__, self).load_mask(image_id)

        # Convert boxes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["boxes"])],
                        dtype=np.uint8)
        for i, b in enumerate(info["boxes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            x_min, y_min, x_max, y_max = b
            rr, cc = skimage.draw.rectangle(start=(y_min, x_min), end=(y_max, x_max))
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "stain":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = StainDataset()
    dataset_train.load_stain(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = StainDataset()
    dataset_val.load_stain(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're starting from COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30, #TODO: maybe increase epochs? Balloon dataset was much smaller than ours
                layers='heads')


# def color_splash(image, mask):
#     """Apply color splash effect.
#     image: RGB image [height, width, 3]
#     mask: instance segmentation mask [height, width, instance count]

#     Returns result image.
#     """
#     # Make a grayscale copy of the image. The grayscale copy still
#     # has 3 RGB channels, though.
#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#     # Copy color pixels from the original color image where mask is set
#     if mask.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         mask = (np.sum(mask, -1, keepdims=True) >= 1)
#         splash = np.where(mask, image, gray).astype(np.uint8)
#     else:
#         splash = gray.astype(np.uint8)
#     return splash


# def detect_and_color_splash(model, image_path=None, video_path=None):
#     assert image_path or video_path

#     # Image or video?
#     if image_path:
#         # Run model detection and generate the color splash effect
#         print("Running on {}".format(args.image))
#         # Read image
#         image = skimage.io.imread(args.image)
#         # Detect objects
#         r = model.detect([image], verbose=1)[0]
#         # Color splash
#         splash = color_splash(image, r['masks'])
#         # Save output
#         file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#         skimage.io.imsave(file_name, splash)
#     elif video_path:
#         import cv2
#         # Video capture
#         vcapture = cv2.VideoCapture(video_path)
#         width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = vcapture.get(cv2.CAP_PROP_FPS)

#         # Define codec and create video writer
#         file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
#         vwriter = cv2.VideoWriter(file_name,
#                                   cv2.VideoWriter_fourcc(*'MJPG'),
#                                   fps, (width, height))

#         count = 0
#         success = True
#         while success:
#             print("frame: ", count)
#             # Read next image
#             success, image = vcapture.read()
#             if success:
#                 # OpenCV returns images as BGR, convert to RGB
#                 image = image[..., ::-1]
#                 # Detect objects
#                 r = model.detect([image], verbose=0)[0]
#                 # Color splash
#                 splash = color_splash(image, r['masks'])
#                 # RGB -> BGR to save image to video
#                 splash = splash[..., ::-1]
#                 # Add image to video writer
#                 vwriter.write(splash)
#                 count += 1
#         vwriter.release()
#     print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect stains.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/stain/dataset/",
                        help='Directory of the Stain dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect stains on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to detect stains on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = StainConfig()
    else:
        class InferenceConfig(StainConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
