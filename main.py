import argparse
import os
import datetime
import numpy as np
import skimage.draw
from mrcnn.dataloader import ButterflyDataset
from mrcnn.config import ButterflyConfig
from mrcnn.model import MaskRCNN
from mrcnn.utils import Utils
from imgaug import augmenters as iaa


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the
        # mask into one layer
        mask = np.sum(mask, -1, keepdims=True) >= 1
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r["masks"])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now()
        )
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(
            datetime.datetime.now()
        )
        vwriter = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r["masks"])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def train(model, path, config):
    """Train the model."""
    # Training dataset.
    dataset_train = ButterflyDataset()
    dataset_train.load_data(path, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_test = ButterflyDataset()
    dataset_test.load_data(path, "val")
    dataset_test.prepare()

    # image argumentations
    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    augs = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(percent=(-0.05, 0.05))),
            sometimes(
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={
                        "x": (-0.2, 0.2),
                        "y": (-0.2, 0.2),
                    },  # translate by -20 to +20 percent (per axis)
                    rotate=(-15, 15),  # rotate by -45 to +45 degrees
                    shear=(-10, 10),  # shear by -16 to +16 degrees
                )
            ),
            iaa.SomeOf(
                (0, 4),
                [
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                        ]
                    ),
                    iaa.Sharpen(
                        alpha=(0, 1.0), lightness=(0.75, 1.5)
                    ),  # sharpen images
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),  # add gaussian noise to images
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.AddToHueAndSaturation(
                        (-20, 20)
                    ),  # change hue and saturation
                    iaa.ContrastNormalization(
                        (0.5, 2.0), per_channel=0.5
                    ),  # improve or worsen the contrast
                ],
                random_order=True,
            ),
        ],
        random_order=True,
    )

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(
        dataset_train,
        dataset_test,
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers="heads",
        augmentation=augs,
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN to detect balloons."
    )
    parser.add_argument(
        "command", metavar="<command>", help="'train' or 'splash'"
    )
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/balloon/dataset/",
        help="Directory of the Balloon dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default="./logs",
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--image",
        required=False,
        metavar="path or URL to image",
        help="Image to apply the color splash effect on",
    )
    parser.add_argument(
        "--video",
        required=False,
        metavar="path or URL to video",
        help="Video to apply the color splash effect on",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert (
            args.image or args.video
        ), "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ButterflyConfig()
    else:

        class InferenceConfig(ButterflyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    COCO_WEIGHTS_PATH = "./mask_rcnn_coco.h5"
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            Utils.download_trained_weights(weights_path)
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
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=[
                "mrcnn_class_logits",
                "mrcnn_bbox_fc",
                "mrcnn_bbox",
                "mrcnn_mask",
            ],
        )
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, config)
    elif args.command == "splash":
        detect_and_color_splash(
            model, image_path=args.image, video_path=args.video
        )
    else:
        print(
            "'{}' is not recognized. "
            "Use 'train' or 'splash'".format(args.command)
        )
