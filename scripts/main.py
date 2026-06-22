from retinaDataset import RetinaDataset
from config import IMAGE_PATH,LABEL_PATH,BATCH_SIZE,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS, TRANSFORM, OUTPUT_DIR, MODEL_DIR
from visualize import plot_image, plot_hist, plot_age_hist_bins
from models import GenderClassifierRS, GenderClassifierRETFound, AgeClassifierRETFound, AgeClassifierRS
from experiments import gender_prediction_noXAI, age_prediction_noXAI
import os


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

data = RetinaDataset(IMAGE_PATH,LABEL_PATH, transform=TRANSFORM)
data_age = RetinaDataset(IMAGE_PATH,LABEL_PATH,transform=TRANSFORM, drop_skip=True)


gender_prediction_noXAI(data,GenderClassifierRS,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS,BATCH_SIZE)
gender_prediction_noXAI(data,GenderClassifierRETFound,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS,BATCH_SIZE)

# age_prediction_noXAI(data,AgeClassifierRS,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS,BATCH_SIZE)
# age_prediction_noXAI(data,AgeClassifierRETFound,LEARNING_RATE,RANDOM_STATE,NUMBER_OF_EPOCHS,BATCH_SIZE)
