import torchvision.transforms as transforms 

IMAGE_PATH = "/home/wojtek/Dokumenty/PWR/MGR/SEM_3/Praca/projekt/Dane/Images"
LABEL_PATH = "/home/wojtek/Dokumenty/PWR/MGR/SEM_3/Praca/projekt/Dane/labels/labels.csv"

TRANSFORM_RS50 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

RETFOUND_REPO = "iszt/RETFound_mae_meh"