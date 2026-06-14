from retinaDataset import RetinaDataset
from config import IMAGE_PATH,LABEL_PATH
from visualize import plot_image


data = RetinaDataset(IMAGE_PATH,LABEL_PATH)
plot_image(data[0]['image'])


