import re
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import os

class RetinaDataset(Dataset):
  def __init__(self, image_path, csv_path, transform=None, drop_skip=False):
    self.df = pd.read_csv(csv_path)
    self.image_path = image_path
    self.transform = transform

    self.df['sex_binary'] = self.df['sex'].map({'F': 0, 'M': 1})
    bins = [0,45,55,65,75,82,150] #po analizie wizualnej histogramu
    labels = ["skip", "45-54", "55-64", "65-74", "75-82", ">82"]
    self.df["age_bin"] = pd.cut(
            self.df["age"], bins=bins, labels=labels, right=False
        )

    if drop_skip:
      self.df = self.df[self.df["age_bin"] != "skip"].reset_index(
                drop=True
            )
    self.df["age_bin_encoded"] = (
            self.df["age_bin"].cat.remove_unused_categories().cat.codes
        )

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    check_label = self.df.iloc[idx]['image_id']

    match = re.search(r'_(\d+)$', check_label)

    if match:
      num = int(match.group(1))
      prefix = check_label[:match.start()]

      if num == 1:
        img_name = prefix
      else:
        img_name = f"{prefix}_{num - 1}_"
    else:
      img_name = check_label

    img_name = os.path.join(self.image_path, f"{img_name}.png")

    img = io.imread(img_name)
    gender = self.df.iloc[idx]['sex_binary']
    age_bin = self.df.iloc[idx]['age_bin_encoded']

    if self.transform:
      img = self.transform(img)

    sample = {'image': img, 'gender': gender, 'age_bin': age_bin}

    return sample


  @property
  def classes(self):
    return self.df.classes
