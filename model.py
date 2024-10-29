from PIL import Image  # For handling image loading and processing
import torch
from torch import topk
from torch.nn import functional as F, SmoothL1Loss
from pycocotools.coco import COCO  # To manage COCO dataset annotations
import numpy as np
import torch.nn as nn
from torchvision.ops import nms  # For non-max suppression to remove overlapping boxes
from torchvision.transforms import transforms

class COCOObjectDetectionDataset():
    """
    Custom Dataset class to load and handle COCO-formatted object detection data.
    """

    def __init__(self, coco_annotation_file, image_directory, transform=None):
        """
        Initializes the dataset with COCO annotations and images.

        Args:
            coco_annotation_file (str): Path to COCO annotations JSON file.
            image_directory (str): Directory containing images referenced by COCO annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.coco = COCO(coco_annotation_file)
        self.image_directory = image_directory
        self.transform = transform
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """Fetches an image and its annotations for a given index."""
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_directory}/{image_info['file_name']}"

        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)

        # Retrieve annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and labels
        boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
        labels = torch.tensor([ann['category_id'] for ann in annotations], dtype=torch.int64)

        # Apply transformations if provided
        if self.transform:
            image = self.transform()(image)
            image = torch.Tensor(np.array(image))

        return image, {'boxes': boxes, 'labels': labels}


class MobileNetInspiredDetector(nn.Module):
    """
    Object detection model inspired by MobileNet architecture with depthwise separable convolutions.
    """

    def __init__(self, num_classes, boxes, input_ch, s=2):
        """
        Initializes the model layers.

        Args:
            num_classes (int): Number of classes for object detection.
            boxes (int): Number of bounding boxes per cell.
            input_ch (int): Number of input channels (e.g., 3 for RGB images).
            s (int): Stride value for the initial convolution.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = boxes

        # Function to create depthwise separable convolution layers
        def depthwise_separable_conv(in_ch, out_ch, stride=1):
            """Performs a depthwise separable convolution with BatchNorm and ReLU activation."""
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Initial convolution to reduce the number of input channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_ch, 32, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Series of depthwise separable convolutions
        self.depthwise_layers = nn.Sequential(
            depthwise_separable_conv(32, 64, stride=1),
            depthwise_separable_conv(64, 128, stride=2),
            depthwise_separable_conv(128, 128, stride=1),
            depthwise_separable_conv(128, 256, stride=2),
            depthwise_separable_conv(256, 256, stride=1),
            depthwise_separable_conv(256, 512, stride=2),
        )

        # Additional depthwise separable layers for deeper feature extraction
        self.depthwise_deep = nn.Sequential(
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 512, stride=1),
            depthwise_separable_conv(512, 1024, stride=2)
        )

        # Additional layers for feature extraction at different levels
        self.extras = nn.Sequential(
            depthwise_separable_conv(1024, 512, stride=2),
            depthwise_separable_conv(512, 256, stride=2),
            depthwise_separable_conv(256, 128, stride=1),
        )

        # Layers for predicting bounding box locations
        self.loc_layers = nn.Sequential(
            nn.Conv2d(512, self.num_boxes * 2, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_boxes * 2, kernel_size=3, padding=1),
            nn.Conv2d(128, self.num_boxes * 2, kernel_size=3, padding=1)
        )

        # Layers for predicting class confidence scores
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, self.num_boxes * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.num_boxes * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(128, self.num_boxes * num_classes, kernel_size=3, padding=1)
        ])

        # Additional location coordinate layer for output
        self.loc_coor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 6 * self.num_boxes * 4, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """Processes input image to return bounding box locations, confidence scores, and feature maps."""
        x = x.permute(2, 0, 1).contiguous()  # Adjust channel ordering for PyTorch
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])

        x = self.init_conv(x)
        features = []
        x = self.depthwise_layers(x)
        feats_depth_layers = x  # Save feature map for analysis
        x = self.depthwise_deep(x)
        feats_depth_deep = x  # Save deeper feature map for analysis

        # Extract features from additional layers
        for layer in self.extras:
            x = layer(x)
            features.append(x)

        locs = []
        confs = []
        loc_coor = self.loc_coor(x)  # Location coordinates output

        # Collect location and confidence predictions across feature maps
        for i, feature in enumerate(features):
            locs.append(self.loc_layers[i](feature).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf_layers[i](feature).permute(0, 2, 3, 1).contiguous())

        # Concatenate predictions
        loc = torch.cat([loc.view(loc.size(0), -1, 2) for loc in locs], 1)
        conf = torch.cat([conf.view(conf.size(0), -1, self.num_classes) for conf in confs], 1)
        conf = F.sigmoid(conf)
        loc_coor = loc_coor.view(loc_coor.shape[0], -1, 4)

        # Adjust location coordinates using score-based scaling
        locn_coor = loc_coor.clone()
        locn_coor[:, :, 0] = loc[:, :, 0] * loc_coor[:, :, 0]
        locn_coor[:, :, 1] = loc[:, :, 1] * loc_coor[:, :, 1]
        locn_coor[:, :, 2] = loc[:, :, 0] * loc_coor[:, :, 2]
        locn_coor[:, :, 3] = loc[:, :, 1] * loc_coor[:, :, 3]

        return loc_coor, conf, features, feats_depth_deep, feats_depth_layers


def data_loader(ann_file, img_file, init_batch, end_batch):
    """
    Loads a batch of images and corresponding targets for object detection.

    Args:
        ann_file (str): COCO annotations file.
        img_file (str): Path to directory containing images.
        init_batch (int): Starting batch index.
        end_batch (int): Ending batch index.
    """
    target = []
    images = []
    aobj = COCOObjectDetectionDataset(coco_annotation_file=ann_file, image_directory=img_file, transform=transforms.ToPILImage)

    # Loop through specified batch range
    for ix in range(init_batch, end_batch):
        img, target_obj = aobj[ix]
        target.append(target_obj)
        images.append(img)

    return images, target

class BoundingBox_LossProcessor(nn.Module):
    """
    Computes the loss for object detection by filtering boxes, performing Non-Max Suppression (NMS),
    and calculating both location and confidence losses.
    """
    def __init__(self, num_classes, alpha=0.25, gamma=2.0):
        """
        Initializes the loss processor.

        Args:
            num_classes (int): Number of classes.
            alpha (float): Weighting factor for the focal loss to address class imbalance.
            gamma (float): Focusing parameter for the focal loss, reducing the loss for well-classified examples.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fxn = SmoothL1Loss(reduction='sum')  # Smooth L1 Loss for bounding box regression

    def loss_forward(self, predictions, targets, conf_threshold=0.6, iou_threshold=0.5):
        """
        Calculates the batch loss by applying the loss function across all predictions and targets.

        Args:
            predictions (tuple): Contains predicted locations and confidence scores.
            targets (dict): Contains ground-truth boxes and labels.
            conf_threshold (float): Confidence threshold to filter predictions.
            iou_threshold (float): IOU threshold for Non-Max Suppression (NMS).

        Returns:
            float: Total loss for the batch.
        """
        locations, confidences = predictions
        total_loss = 0

        # Calculate single-batch loss and accumulate
        batch_loss = self.process_single_batch(
            locations, confidences,
            targets['boxes'], targets['labels'],
            conf_threshold, iou_threshold
        )
        print("Batch loss: ", batch_loss)
        total_loss += batch_loss
        return total_loss

    def process_single_batch(self, loc, conf, target_boxes, target_labels, conf_threshold, iou_threshold):
        """
        Processes a single batch of images to calculate localization and confidence losses.

        Args:
            loc (tensor): Predicted locations.
            conf (tensor): Predicted confidence scores.
            target_boxes (tensor): Ground-truth boxes for the batch.
            target_labels (tensor): Ground-truth labels for the batch.
            conf_threshold (float): Confidence threshold to filter predictions.
            iou_threshold (float): IOU threshold for NMS.

        Returns:
            float: Total loss for the batch, combining localization and confidence losses.
        """
        device = loc.device
        target_boxes = target_boxes.to(device)
        target_labels = target_labels.to(device)

        # If conf is 3D, reduce its dimensionality for easier handling
        if conf.dim() == 3:
            conf = conf.squeeze(0)

        # Calculate max confidence scores and class predictions
        confidence_scores, predicted_classes = conf.max(dim=1)

        # Filter out low-confidence predictions
        mask = confidence_scores > conf_threshold
        if not mask.any():
            print("Not found any mask.")
            return torch.tensor(0.001, device=device, requires_grad=True)

        # Apply mask to filter locations and confidences
        filtered_loc = loc[:, mask, :]
        filtered_conf = conf[mask]
        filtered_scores = confidence_scores[mask]

        # Check for any valid locations or target boxes
        if filtered_loc.numel() == 0 or target_boxes.numel() == 0:
            print("No locations found")
            return torch.tensor(0.001, device=device, requires_grad=True)

        # Remove extra dimension if necessary
        filtered_loc = filtered_loc.squeeze(0)
        score = filtered_conf[:, 0]

        # Perform Non-Max Suppression
        nms_boxes = nms(filtered_loc, score, iou_threshold)
        if not nms_boxes.any():
            print("No mask.")
            return torch.tensor(0.001, device=device, requires_grad=True)

        # Get boxes and confidences that passed NMS
        matched_loc = filtered_loc[nms_boxes, :]
        matched_conf = filtered_conf[nms_boxes, :]

        # Select top-k matched confidences and indices to compare against ground-truth labels
        k = target_labels.shape[0]
        matched_conf, indices = topk(matched_conf, k, dim=0)
        matched_conf = (matched_conf[:, 0] > 0.5).to(torch.float64)

        matched_loc = matched_loc[indices, :]
        matched_target_boxes = target_boxes
        matched_target_labels = target_labels.to(torch.float64)

        # Calculate localization loss between predicted and ground-truth boxes
        pred_box = matched_loc
        loc_loss = self.loss_fxn(pred_box, matched_target_boxes)

        # Calculate confidence loss using focal loss
        conf_loss = self.focal_loss(matched_conf, matched_target_labels)

        # Calculate number of positives for normalization
        num_positives = nms_boxes.sum().float()

        # Average total loss over positive predictions
        total_loss = (loc_loss + conf_loss) / num_positives
        print("Reached here.")

        # Clean up unnecessary variables to free memory
        del score, matched_loc, target_boxes, target_labels
        del num_positives, loc_loss, conf_loss, matched_conf
        del matched_target_boxes, matched_target_labels, filtered_conf
        del filtered_scores, filtered_loc, conf_threshold, iou_threshold

        return total_loss

    def focal_loss(self, pred, target):
        """
        Calculates focal loss to address class imbalance.

        Args:
            pred (tensor): Predicted class probabilities.
            target (tensor): Ground-truth labels.

        Returns:
            float: Focal loss.
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct classification
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # Focal loss computation
        return loss.sum()
