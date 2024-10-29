import random
import torch
import os
import gc
from torch import topk
from PIL import Image
import json
from torch.nn.functional import interpolate
from torchvision.utils import draw_bounding_boxes
import numpy as np

def save_checkpoint(model, optimizer, epoch, batch_index, loss, filename):
    """Saves the model state to a specified file path.

    Args:
        model (torch.nn.Module): The model instance to save.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        epoch (int): The current epoch number.
        batch_index (int): Index of the current batch.
        loss (float): The loss value for the current batch.
        filename (str): Path to save the checkpoint file.
    """
    torch.save({
        'epoch': epoch,
        'batch_index': batch_index,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)


def load_checkpoint(model, optimizer, filename):
    """Loads the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model instance to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        filename (str): Path of the checkpoint file.

    Returns:
        tuple: Loaded model, optimizer, epoch number, batch index, and loss value.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_index = checkpoint['batch_index']
        loss = checkpoint['loss']
        return model, optimizer, epoch, batch_index, loss
    return model, optimizer, 0, 0, None


def train_incrementally(
        model, scaler, loss_file_path, checkpoint_path, bbox_processor, optimizer, images,
        targets, num_epochs, checkpoint_interval, max_batches_per_run, batch_range, load_prev=False
):
    """Trains the model incrementally, saving losses and checkpoints at intervals.

    Args:
        model (torch.nn.Module): The model to train.
        scaler (torch.cuda.amp.GradScaler): For mixed-precision training.
        loss_file_path (str): Path to save the loss values.
        checkpoint_path (str): Path to save checkpoints.
        bbox_processor (object): Object to calculate bounding box loss.
        optimizer (torch.optim.Optimizer): Optimizer for model.
        images (torch.Tensor): Input images.
        targets (torch.Tensor): Target bounding box data.
        num_epochs (int): Number of epochs to train.
        checkpoint_interval (int): Interval to save checkpoints.
        max_batches_per_run (int): Max batches per training run.
        batch_range (tuple[int]): Start and end batch indices for resuming training.
        load_prev (bool): Whether to load from a previous checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_prev and batch_range[1] != 0:
        print("Loading previous checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])

    model.to(device)
    loss_dict = {}
    start_epoch, start_batch, end_batch = batch_range

    with open(loss_file_path, 'a') as loss_file:
        loss_file.write("Batch,Loss\n")

        for epoch in range(start_epoch, num_epochs):
            for batch_index, imgs_batch, tgt_batch in zip(range(start_batch, end_batch), images, targets):
                try:
                    imgs_batch = imgs_batch.to(device)
                    optimizer.zero_grad()
                    predictions = model(imgs_batch)
                    loss = bbox_processor.loss_forward(predictions=predictions, targets=tgt_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if epoch == num_epochs - 1:
                        loss_file.write(f"{batch_index},{loss.item()}\n")
                        loss_file.flush()

                    if (batch_index + 1) % checkpoint_interval == 0:
                        save_checkpoint(model, optimizer, epoch, batch_index + 1, loss.item(), checkpoint_path)
                        print(f"Checkpoint saved at Epoch {epoch} - Batch {batch_index + 1}, Loss: {loss.item()}")

                    if batch_index + 1 >= max_batches_per_run + end_batch:
                        save_checkpoint(model, optimizer, epoch, batch_index + 1, loss.item(), checkpoint_path)
                        print(f"Reached max batches per run, saving and exiting at Epoch {epoch} - Batch {batch_index + 1}")
                        return False
                except torch.cuda.OutOfMemoryError as error:
                    print(f"GPU memory error encountered: {error}. Exiting training.")
                    return True
    return True


def infer_image(annotation_data, image_dir):
    """Fetches a random image and returns its tensor, for inference and bounding box visualization.

    Args:
        annotation_data (dict): JSON dictionary with annotation data.
        image_dir (str): Directory where images are stored.

    Returns:
        tuple: Original image array, inference tensor, boxed image in PIL format, and permuted tensor.
    """
    annotation_list = annotation_data['annotations']
    selected_annotation = random.choice(annotation_list)
    image_id = selected_annotation['image_id']
    image_name = annotation_data['images'][image_id]['file_name']
    image_path = os.path.join(image_dir, image_name)
    bbox_coordinates = selected_annotation['bbox']

    bbox_tensor = torch.tensor(bbox_coordinates)
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array)

    permuted_image_tensor = image_tensor.permute(2, 0, 1)
    boxed_image = draw_bounding_boxes(permuted_image_tensor, bbox_tensor, width=3)
    boxed_image_array = np.array(boxed_image.permute(1, 2, 0), dtype=np.uint8)
    boxed_image_pil = Image.fromarray(boxed_image_array)

    image_for_inference = torch.tensor(image_array, dtype=torch.float32)
    return image_array, image_for_inference, boxed_image_pil, permuted_image_tensor


def topk_sort(k, predicted_boxes, predicted_scores):
    """Sorts and returns top-k predictions based on confidence scores.

    Args:
        k (int): Number of top predictions to retrieve.
        predicted_boxes (torch.Tensor): Predicted bounding boxes.
        predicted_scores (torch.Tensor): Confidence scores of the predictions.

    Returns:
        tuple: Sorted bounding boxes, sorted confidence scores, and labels as strings.
    """
    top_k_values = topk(predicted_scores, k)
    sorted_boxes = predicted_boxes[top_k_values[1], :]
    sorted_scores = predicted_scores[top_k_values[1]]

    score_labels = [str(score) for score in sorted_scores.tolist()]
    return sorted_boxes, sorted_scores, score_labels


def tensor_to_image_with_boxes(permuted_tensor, labels, bounding_boxes):
    """Converts a tensor with bounding boxes and labels into a PIL image.

    Args:
        permuted_tensor (torch.Tensor): Tensor of the image.
        labels (list): Labels for bounding boxes.
        bounding_boxes (torch.Tensor): Tensor with bounding box coordinates.

    Returns:
        PIL.Image: Image with bounding boxes and labels.
    """
    boxed_tensor = draw_bounding_boxes(permuted_tensor, bounding_boxes, labels=labels, width=3)
    boxed_image_array = np.array(boxed_tensor.permute(1, 2, 0), dtype=np.uint8)
    boxed_image_pil = Image.fromarray(boxed_image_array)
    return boxed_image_pil


def get_image_files(directory):
    """Gets a list of all image files in a given directory.

    Args:
        directory (str): The path to the directory containing images.

    Returns:
        list[str]: A list of paths to image files.
    """
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return image_files


def img_to_tensor(img):
    """Converts an image to tensor and interpolates it to required model size.

    Args:
        img (PIL.Image): Image to convert.

    Returns:
        tuple: Original array, interpolated tensor, and tensor ready for plotting.
    """
    array = np.array(img)
    img_tensor = torch.tensor(array, dtype=torch.float32)
    ip_feed = img_tensor.permute(2, 0, 1).unsqueeze(0)
    ip_tensor = interpolate(ip_feed, size=[640, 640], mode='bilinear')
    img_tensor = ip_tensor.squeeze(0)
    tensor_fplot = img_tensor.to(torch.uint8)
    return array, img_tensor, tensor_fplot


def pick_random_image(image_files):
    """Picks a random image from a list of image files.

    Args:
        image_files (list[str]): A list of paths to image files.

    Returns:
        str: The path to the randomly selected image.
    """
    if not image_files:
        raise ValueError("No image files were provided.")

    random_index = random.randint(0, len(image_files) - 1)
    random_image = image_files[random_index]
    return random_image
