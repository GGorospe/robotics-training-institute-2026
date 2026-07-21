"""
train_utils.py

Shared utilities for training CNN image classifiers during the RTI workshop.
Used by B3a (and any later notebook that trains a classifier, whether it has
two classes or three -- nothing here assumes a fixed number of classes).

Models and training records are written to /home/explorer/Models/ by default:
  - best_model_<model_name>.pth   -- the best checkpoint from this run
  - best_model_<model_name>.png   -- a plot of that run's training history
  - training_log.txt              -- one JSON line appended per training run,
                                      a running history across the whole week
"""

import os
import json
import random
from datetime import datetime

from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from capture_utils import ensure_directory

DEFAULT_MODELS_DIR = "/home/explorer/Models/"


IGNORED_DIRECTORY_NAMES = {'.ipynb_checkpoints'}


def list_class_names(data_dir):
    """Lists valid class subdirectory names in `data_dir`, skipping
    .ipynb_checkpoints (and other Jupyter/OS housekeeping directories),
    which are not classes even though they live alongside the real ones.

    Args:
        data_dir (str): dataset root, containing one subdirectory per class

    Returns:
        list[str]: sorted class directory names
    """
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in IGNORED_DIRECTORY_NAMES
    ])


def list_image_files(class_dir):
    """Lists files inside a class directory, skipping .ipynb_checkpoints
    and any other subdirectories that might live inside it.

    Args:
        class_dir (str): path to a single class's image directory

    Returns:
        list[str]: filenames (not full paths) of files in class_dir
    """
    return [
        f for f in os.listdir(class_dir)
        if f not in IGNORED_DIRECTORY_NAMES
        and os.path.isfile(os.path.join(class_dir, f))
    ]


def get_class_image_counts(data_dir):
    """Counts how many image files are in each class subdirectory of a
    dataset folder. Ignores .ipynb_checkpoints directories, both as a
    "class" itself and as stray contents inside a real class folder.

    Args:
        data_dir (str): dataset root, containing one subdirectory per class

    Returns:
        dict[str, int]: maps each class name to how many files are in its
            folder, e.g. {'blue': 52, 'red': 48}
    """
    class_names = list_class_names(data_dir)

    return {
        class_name: len(list_image_files(os.path.join(data_dir, class_name)))
        for class_name in class_names
    }


def preview_dataset(data_dir, num_samples_per_class=3):
    """Prints per-class image counts and displays a few sample images from
    each class, so students can sanity-check their dataset -- confirming
    it's labeled correctly and roughly balanced -- before spending time
    training on it.

    Args:
        data_dir (str): dataset root, containing one subdirectory per class
        num_samples_per_class (int): how many sample images to show per class

    Returns:
        None
    """
    class_counts = get_class_image_counts(data_dir)
    class_names = sorted(class_counts.keys())

    print("Dataset preview:")
    for class_name in class_names:
        print(f"  {class_name}: {class_counts[class_name]} images")

    fig, axes = plt.subplots(
        len(class_names), num_samples_per_class,
        figsize=(3 * num_samples_per_class, 3 * len(class_names))
    )
    if len(class_names) == 1:
        axes = [axes]

    for row, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = list_image_files(class_dir)
        sample_files = random.sample(image_files, min(num_samples_per_class, len(image_files)))

        for col in range(num_samples_per_class):
            ax = axes[row][col] if len(class_names) > 1 else axes[col]
            ax.axis('off')
            if col < len(sample_files):
                image = Image.open(os.path.join(class_dir, sample_files[col]))
                ax.imshow(image)
                if col == 0:
                    ax.set_title(class_name, fontsize=14, loc='left')

    plt.tight_layout()
    plt.show()


class _FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder that ignores .ipynb_checkpoints (and other Jupyter/OS
    housekeeping directories) when discovering classes, instead of
    treating an empty checkpoints folder as a real, empty class -- which
    would otherwise crash dataset loading.
    """

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory)
            if entry.is_dir() and entry.name not in IGNORED_DIRECTORY_NAMES
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")

        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx


def prepare_dataloaders(data_dir, batch_size, test_fraction=0.2):
    """Builds train/test DataLoaders from a folder of labeled image
    subdirectories (one subdirectory per class, e.g. "red/", "blue/").

    Works for any number of classes -- the number of classes is however
    many labeled subdirectories are found in `data_dir`.

    Args:
        data_dir (str): path to the dataset root; must contain one
            subdirectory per class, each containing that class's images
        batch_size (int): number of images per training batch
        test_fraction (float): fraction of images held out for testing,
            e.g. 0.2 reserves 20% of images to evaluate the model on
            images it never trained on

    Returns:
        (DataLoader, DataLoader, list[str]): train_loader, test_loader,
            and class_names (in the label-index order PyTorch assigned them)

    Raises:
        ValueError: if a class folder contains no images, or if there
            isn't enough data to build both a training and a test set
    """
    class_counts = get_class_image_counts(data_dir)

    for class_name, count in class_counts.items():
        if count == 0:
            raise ValueError(f"No data was found in the '{class_name}' class folder.")

    dataset = _FilteredImageFolder(
        data_dir,
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes:")
    for class_name in dataset.classes:
        print(f"  {class_name}: {class_counts[class_name]} images")

    test_size = max(1, int(len(dataset) * test_fraction))
    train_size = len(dataset) - test_size

    if train_size < 1:
        raise ValueError(
            "Not enough images to create a training set. Collect more images and try again."
        )

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader, dataset.classes


def build_model(num_classes):
    """Loads a pretrained resnet18 and replaces its final layer so it
    outputs `num_classes` predictions instead of the original 1000.

    Args:
        num_classes (int): number of classes this model should predict
            (2 for most models this week, 3 for at least one)

    Returns:
        (torch.nn.Module, torch.device): the model (already moved to the
            best available device) and that device
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cpu':
        print("No GPU found -- training will run on the CPU and will be slower than usual.")

    model = model.to(device)

    return model, device


def validate_model_name(model_name, models_dir=DEFAULT_MODELS_DIR):
    """Checks that `model_name` isn't already in use, so a student can't
    accidentally overwrite a model (their own, or a teammate's) they meant
    to keep.

    Args:
        model_name (str): the name the student chose for this model
        models_dir (str): directory where models are saved

    Raises:
        ValueError: if a model with this name already exists
    """
    model_path = os.path.join(models_dir, f"best_model_{model_name}.pth")
    if os.path.exists(model_path):
        raise ValueError(
            f"A model named '{model_name}' already exists at {model_path}. "
            "Please choose a different model_name so you don't overwrite it."
        )


def plot_training_history(history, model_name, save_path=None):
    """Plots training loss and test accuracy across epochs on a shared
    x-axis, and optionally saves the plot as an image.

    Args:
        history (list[dict]): one entry per epoch, each with keys
            'epoch', 'train_loss', 'test_accuracy'
        model_name (str): used in the plot title
        save_path (str, optional): if given, the plot is saved here

    Returns:
        None
    """
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    test_accuracy = [h['test_accuracy'] for h in history]

    fig, loss_axis = plt.subplots(figsize=(8, 5))

    loss_axis.set_xlabel('Epoch')
    loss_axis.set_ylabel('Training Loss', color='tab:red')
    loss_axis.plot(epochs, train_loss, color='tab:red', label='Training Loss')
    loss_axis.tick_params(axis='y', labelcolor='tab:red')

    accuracy_axis = loss_axis.twinx()
    accuracy_axis.set_ylabel('Test Accuracy', color='tab:blue')
    accuracy_axis.plot(epochs, test_accuracy, color='tab:blue', label='Test Accuracy')
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Training History: {model_name}')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")

    plt.show()


def print_training_summary(record):
    """Prints a compact, plain-language summary of a completed training
    run -- meant to be short enough for a student to copy into their
    physical training log by hand.

    Args:
        record (dict): a training record, as produced by train_model()

    Returns:
        None
    """
    hyperparameters = record['hyperparameters']

    print("=" * 50)
    print("TRAINING SUMMARY -- copy this into your notebook!")
    print("=" * 50)
    print(f"Model name:          {record['model_name']}")
    print(f"Classes:             {', '.join(record['class_names'])}")
    print(f"Training images:     {record['num_train_images']}")
    print(f"Test images:         {record['num_test_images']}")
    print(f"Epochs:              {hyperparameters['epochs']}")
    print(f"Learning rate:       {hyperparameters['learning_rate']}")
    print(f"Momentum:            {hyperparameters['momentum']}")
    print(f"Batch size:          {hyperparameters['batch_size']}")
    print(f"Best epoch:          {record['best_epoch']}")
    print(f"Best test accuracy:  {record['best_test_accuracy']:.1%}")
    print(f"Model saved to:      {record['saved_model_path']}")
    print("=" * 50)


def log_training_run(record, models_dir=DEFAULT_MODELS_DIR):
    """Appends a training record as one JSON line to training_log.txt,
    creating the models directory and/or the log file if they don't
    already exist. Never overwrites -- this file accumulates a full
    history of every training run all week.

    Args:
        record (dict): a JSON-serializable training record
        models_dir (str): directory the log file lives in

    Returns:
        str: the path to training_log.txt
    """
    ensure_directory(models_dir)
    log_path = os.path.join(models_dir, "training_log.txt")

    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    return log_path


def train_model(model, train_loader, test_loader, device, class_names, data_dir,
                 model_name, epochs, learning_rate, momentum,
                 models_dir=DEFAULT_MODELS_DIR):
    """Trains `model` for `epochs` epochs, saving the best-performing
    checkpoint, a training plot, and a training log entry.

    Args:
        model (torch.nn.Module): a model from build_model()
        train_loader (DataLoader): training data
        test_loader (DataLoader): test/validation data
        device (torch.device): from build_model()
        class_names (list[str]): from prepare_dataloaders()
        data_dir (str): dataset directory used, recorded for reference
        model_name (str): a unique name for this model/run -- used to
            name the saved checkpoint, plot, and log entry
        epochs (int): number of passes through the full training set
        learning_rate (float): how large a step the optimizer takes
            after each batch
        momentum (float): how much of the previous update carries into
            the next one, helping smooth out noisy gradients
        models_dir (str): where to save the checkpoint, plot, and log

    Returns:
        (list[dict], dict): the per-epoch history, and the training
            record that was logged and printed
    """
    validate_model_name(model_name, models_dir)
    ensure_directory(models_dir)

    model_path = os.path.join(models_dir, f"best_model_{model_name}.pth")
    plot_path = os.path.join(models_dir, f"best_model_{model_name}.png")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    num_train_images = len(train_loader.dataset)
    num_test_images = len(test_loader.dataset)

    history = []
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / num_train_images

        model.eval()
        test_error_count = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

        test_accuracy = 1.0 - test_error_count / num_test_images

        print(f"Epoch {epoch + 1}/{epochs} -- train loss: {train_loss:.4f}, test accuracy: {test_accuracy:.1%}")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'test_accuracy': test_accuracy})

        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), model_path)
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

    print("Training complete!")

    plot_training_history(history, model_name, save_path=plot_path)

    training_record = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "dataset_dir": data_dir,
        "class_names": list(class_names),
        "num_train_images": num_train_images,
        "num_test_images": num_test_images,
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "batch_size": train_loader.batch_size,
        },
        "best_epoch": best_epoch,
        "best_test_accuracy": best_accuracy,
        "saved_model_path": model_path,
        "training_plot_path": plot_path,
    }

    log_training_run(training_record, models_dir)
    print_training_summary(training_record)

    return history, training_record
