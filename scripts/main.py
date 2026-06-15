from retinaDataset import RetinaDataset
from config import IMAGE_PATH,LABEL_PATH,BATCH_SIZE,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS
from visualize import plot_image, plot_hist
from models import GenderClassifierRS, GenderClassifierEN, GenderClassifierRETFound
from experiments import gender_prediction_noXAI
import numpy as np

data = RetinaDataset(IMAGE_PATH,LABEL_PATH)
#plot_hist(data.df['age'],"Histogram wieku w badanej populacji")
u,c = (np.unique(data.df['sex'], return_counts=True))
print(u,c)

#gender_prediction_noXAI(data,GenderClassifierRS,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS,BATCH_SIZE)
