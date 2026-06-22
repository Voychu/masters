import torchvision.transforms as transforms 

IMAGE_PATH = "/home/wojtek/Dokumenty/PWR/MGR/SEM_3/Praca/projekt/Dane/Images"
LABEL_PATH = "/home/wojtek/Dokumenty/PWR/MGR/SEM_3/Praca/projekt/Dane/labels/labels.csv"

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
RETFOUND_REPO = "iszt/RETFound_mae_meh"
RANDOM_STATE = 71830
BATCH_SIZE=32
NUMBER_OF_EPOCHS = 10
LEARNING_RATE = 0.0001
OUTPUT_DIR = "results"
MODEL_DIR = "models"
