import os
import re
import datetime
import multiprocessing
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from fpn.resnet import resnet_graph
from fpn.utils import Utils
from fpn.dataloader import data_generator
from fpn.layers import (
    ProposalLayer,
    DetectionTargetLayer,
    DetectionLayer,
    build_rpn_model,
    fpn_classifier_graph,
    rpn_class_loss_graph,
    rpn_bbox_loss_graph,
    fpn_class_loss_graph,
    fpn_bbox_loss_graph,
)


class FPN:
    """Encapsulates the FPN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ["training", "inference"]
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build FPN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ["training", "inference"]

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )

        # Inputs
        input_image = KL.Input(shape=[None, None, 3], name="input_image")
        input_image_meta = KL.Input(
            shape=[config.IMAGE_META_SIZE], name="input_image_meta"
        )
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32
            )
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32
            )

            # Detection GT (class IDs, and bounding boxes)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32
            )
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32
            )
            # Normalize coordinates
            gt_boxes = KL.Lambda(
                lambda x: Utils.norm_boxes_graph(x, K.shape(input_image)[1:3])
            )(input_gt_boxes)

        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th
        # item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, stage5=True, train_bn=config.TRAIN_BN
            )
        else:
            _, C2, C3, C4, C5 = resnet_graph(
                input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN
            )
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c5p5")(C5)
        P4 = KL.Add(name="fpn_p4add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c4p4")(C4),
            ]
        )
        P3 = KL.Add(name="fpn_p3add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c3p3")(C3),
            ]
        )
        P2 = KL.Add(name="fpn_p2add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c2p2")(C2),
            ]
        )
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2"
        )(P2)
        P3 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3"
        )(P3)
        P4 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4"
        )(P4)
        P5 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5"
        )(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        fpn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(
                input_image
            )
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(
            config.RPN_ANCHOR_STRIDE,
            len(config.RPN_ANCHOR_RATIOS),
            config.TOP_DOWN_PYRAMID_SIZE,
        )
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs, output_names)
        ]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = (
            config.POST_NMS_ROIS_TRAINING
            if mode == "training"
            else config.POST_NMS_ROIS_INFERENCE
        )
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config,
        )([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the
            # dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: Utils.parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(
                    shape=[config.POST_NMS_ROIS_TRAINING, 4],
                    name="input_roi",
                    dtype=np.int32,
                )
                # Normalize coordinates
                target_rois = KL.Lambda(
                    lambda x: Utils.norm_boxes_graph(x, K.shape(input_image)[1:3])
                )(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, and gt_boxes, are zero
            # padded. Equally, returned rois and targets are zero padded.
            res = DetectionTargetLayer(config, name="proposal_targets")(
                [target_rois, input_gt_class_ids, gt_boxes]
            )
            rois, target_class_ids, target_bbox = res

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            fpn_class_logits, fpn_class, fpn_bbox = fpn_classifier_graph(
                rois,
                fpn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(
                lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
            )([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(
                lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
            )([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(
                lambda x: fpn_class_loss_graph(*x), name="fpn_class_loss"
            )([target_class_ids, fpn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(
                lambda x: fpn_bbox_loss_graph(*x), name="fpn_bbox_loss"
            )([target_bbox, target_class_ids, fpn_bbox])

            # Model
            inputs = [
                input_image,
                input_image_meta,
                input_rpn_match,
                input_rpn_bbox,
                input_gt_class_ids,
                input_gt_boxes,
            ]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [
                rpn_class_logits,
                rpn_class,
                rpn_bbox,
                fpn_class_logits,
                fpn_class,
                fpn_bbox,
                rpn_rois,
                output_rois,
                rpn_class_loss,
                rpn_bbox_loss,
                class_loss,
                bbox_loss,
            ]
            model = KM.Model(inputs, outputs, name="fpn")
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            fpn_class_logits, fpn_class, fpn_bbox = fpn_classifier_graph(
                rpn_rois,
                fpn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            # Detections
            # output is [batch, num_detections,
            # (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="fpn_detection")(
                [rpn_rois, fpn_class, fpn_bbox, input_image_meta]
            )

            model = KM.Model(
                [input_image, input_image_meta, input_anchors],
                [detections, fpn_class, fpn_bbox, rpn_rois, rpn_class, rpn_bbox],
                name="fpn",
            )

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno

            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir),
            )
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("fpn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno

            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name)
            )
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError("`load_weights` requires h5py.")
        f = h5py.File(filepath, mode="r")
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = (
            keras_model.inner_model.layers
            if hasattr(keras_model, "inner_model")
            else keras_model.layers
        )

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, "close"):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file

        TF_WEIGHTS_PATH_NO_TOP = "https://github.com/fchollet/deep-learning-models/" "releases/download/v0.2/" "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        weights_path = get_file(
            "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
            md5_hash="a268eb855778b3df3c7506639542a6af",
        )
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum, clipnorm=self.config.GRADIENT_CLIP_NORM
        )
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",
            "rpn_bbox_loss",
            "fpn_class_loss",
            "fpn_bbox_loss",
        ]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = tf.reduce_mean(
                layer.output, keepdims=True
            ) * self.config.LOSS_WEIGHTS.get(name, 1.)
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
            / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if "gamma" not in w.name and "beta" not in w.name
        ]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer, loss=[None] * len(self.keras_model.outputs)
        )

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = tf.reduce_mean(
                layer.output, keepdims=True
            ) * self.config.LOSS_WEIGHTS.get(name, 1.)
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            Utils.log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = (
            keras_model.inner_model.layers
            if hasattr(keras_model, "inner_model")
            else keras_model.layers
        )

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == "Model":
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == "TimeDistributed":
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                Utils.log(
                    "{}{:20}   ({})".format(
                        " " * indent, layer.name, layer.__class__.__name__
                    )
                )

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/fpn_coco_0001.h5
            regex = (
                r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/fpn\_[\w-]+(\d{4})\.h5"
            )
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                )
                # Epoch number in file is 1-based, and in Keras
                # code it's 0-based.
                # So, adjust for that then increment by one to
                # start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print("Re-starting from epoch %d" % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now)
        )

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that
        # get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir, "fpn_{}_*epoch*.h5".format(self.config.NAME.lower())
        )
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(
        self,
        train_dataset,
        val_dataset,
        learning_rate,
        epochs,
        layers,
        augmentation=None,
    ):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, and classifier of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gausssian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fpn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fpn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fpn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(fpn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(
            train_dataset,
            self.config,
            shuffle=True,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
        )
        val_generator = data_generator(
            val_dataset, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE
        )

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=False,
            ),
            keras.callbacks.ModelCheckpoint(
                self.checkpoint_path, verbose=0, save_weights_only=True
            ),
        ]

        # Train
        Utils.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        Utils.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is "nt":
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = Utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE,
            )
            molded_image = Utils.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = Utils.compose_image_meta(
                0,
                image.shape,
                molded_image.shape,
                window,
                scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32),
            )
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, original_image_shape, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in
                    normalized coordinates
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the
                image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, and scores
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = Utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = Utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0
        )[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            N = class_ids.shape[0]

        return boxes, class_ids, scores

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert (
            len(images) == self.config.BATCH_SIZE
        ), "len(images) must be equal to BATCH_SIZE"

        if verbose:
            Utils.log("Processing {} images".format(len(images)))
            for image in images:
                Utils.log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert (
                g.shape == image_shape
            ), "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            Utils.log("molded_images", molded_images)
            Utils.log("image_metas", image_metas)
            Utils.log("anchors", anchors)
        # Run object detection
        detections, _, _, fpn_mask, _, _, _ = self.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )
        # Process detections
        results = []
        for i, image in enumerate(images):
            res = self.unmold_detections(
                detections[i], image.shape, molded_images[i].shape, windows[i]
            )
            final_rois, final_class_ids, final_scores = res
            results.append(
                {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                }
            )
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also retruned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert (
            len(molded_images) == self.config.BATCH_SIZE
        ), "Number of images must be equal to BATCH_SIZE"

        if verbose:
            Utils.log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                Utils.log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            Utils.log("molded_images", molded_images)
            Utils.log("image_metas", image_metas)
            Utils.log("anchors", anchors)
        # Run object detection
        detections, _, _, fpn_mask, _, _, _ = self.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=0
        )
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            res = self.unmold_detections(
                detections[i], image.shape, molded_images[i].shape, window
            )
            final_rois, final_class_ids, final_scores, final_masks = res
            results.append(
                {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                }
            )
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = Utils.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = Utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE,
            )
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = Utils.norm_boxes(
                a, image_shape[:2]
            )
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == "TimeDistributed":
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for layer in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            layer = self.find_trainable_layer(layer)
            # Include layer if it has weights
            if layer.get_weights():
                layers.append(layer)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and noramlized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            Utils.log(k, v)
        return outputs_np
