import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  img_path = []
  for purpose in glob.glob(dir_name + "/*"):
      for category in glob.glob(purpose + "/*"):
          for img in glob.glob(category + "/*"):
              img_path.append(img)
  scaler = StandardScaler()
  for img in img_path:
      image = Image.open(img).convert("L")
      arr = np.asarray(image)
      arr = arr/255
      arr = arr.reshape(-1,1)
      scaler.partial_fit(arr)
    
  mean = scaler.mean_
  std = scaler.scale_

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std

