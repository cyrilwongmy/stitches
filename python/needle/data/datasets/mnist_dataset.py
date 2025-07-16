from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        image_file_handle = gzip.open(image_filename, "rb")
        label_file_handle = gzip.open(label_filename, "rb")
        image_file_handle.read(16)
        label_file_handle.read(8)
        image_data = image_file_handle.read()
        label_data = label_file_handle.read()
        image_file_handle.close()
        label_file_handle.close()
        X = np.frombuffer(image_data, dtype=np.uint8).reshape(-1, 28*28).astype(np.float32)

        self.X = X / 255.0
        self.y = np.frombuffer(label_data, dtype=np.uint8)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Return the image and label of the index-th sample.
        Args:
            index: the index of the sample
        Returns:
            tuple: (image, label)
            image: 1 x H x W x C NDArray of an image
            label: int
        """
        ### BEGIN YOUR SOLUTION
        # apply transforms
        x = self.apply_transforms(self.X[index].reshape(28, 28, -1))
        return x.reshape(-1, 28*28), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        Returns:
            int: the number of samples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION