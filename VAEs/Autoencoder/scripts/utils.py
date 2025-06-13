import matplotlib.pyplot as plt
import numpy as np

def display(
    images,
    labels=None,
    class_names=None,  # map label IDs to class names
    n=10,
    size=(20, 3),
    cmap=None,
    as_type="float32",
    save_to=None
):
    """
    Displays n images with optional labels.
    
    Args:
        images: Array of shape (N, H, W, C)
        labels: Optional array of label IDs
        class_names: Optional list of class names for decoding label IDs
        n: Number of images to show
        size: Figure size
        cmap: Colormap (use None for RGB)
        as_type: Casting image dtype for display
        save_to: File path to save the figure
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        img = images[i].astype(as_type)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")

        if labels is not None:
            label = labels[i]
            if class_names:
                label = class_names[label]
            ax.set_title(label, fontsize=9)

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()



def sample_batch(dataset):
    """Apply to a tf.data.Dataset"""

    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def preprocess(imgs):
    """
    Normalize and pad color images (CIFAR-100).
    Input shape: (N, 32, 32, 3)
    Output shape: (N, 36, 36, 3)
    """
    imgs = imgs.astype("float32") / 255.0

    # imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=0.0)

    return imgs