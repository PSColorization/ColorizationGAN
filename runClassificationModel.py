# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = "resnet18_places365.pth.tar"
# model_file = "/Users/enesguler/PycharmProjects/SceneClassifier/ModelTrain/model_best.pth.tar"
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
# model = models.__dict__[arch](num_classes=1000)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)


def getSuggestions(img_path):
    # load the test image
    img = Image.open(img_path)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.tolist()
    # print('{} prediction on {}'.format(arch, img_path))
    # output the prediction
    preds = list()
    for i in range(0, 5):
        preds.append(classes[idx[i]])
    return {'probs': probs[:5],
            "classes": preds}


if __name__ == '__main__':
    from glob import glob

    path = '/content/drive/MyDrive/ColorizationGAN/TEST/DemoTestImages'

    images = glob(f'{path}/**/*.png', recursive=True)
    images.sort()

    for image in images:
        results = getSuggestions(image)
        for p, c in zip(results['probs'], results['classes']):
            print(round(p, 2), '-> ', c)
        print("-"*50)
