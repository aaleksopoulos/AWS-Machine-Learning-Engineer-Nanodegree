import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import smdebug
except:
    pass

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features = model.fc.in_features
    #model.fc = nn.Sequential(
    #    nn.Linear(num_features, 512),
    #    nn.ReLU(),
    #    nn.Linear(512, 254),
    #    nn.ReLU(),
    #    nn.Linear(254, 133))
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    return model

def model_fn(model_dir):
    model = net()
    model_path = os.path.join(model_dir, 'model.pth')
    logger.info('model path : {}'.format(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (device == torch.device("cpu")) or (device=="cpu"):
        logger.info('got in here')
        model.load_state_dict(torch.load(model_path, map_location=device))
                              
    else:
        model.load_state_dict(torch.load(model_path))

    # with open(os.path.join(model_dir, "model.bin"), "rb") as f:
    #     model.load_state_dict(torch.load(f))
    logger.info('Successfully loaded the model')
    return model.to(device)

#def model_fn(model_dir):
#    logger.info("In model_fn. Model directory is {}".format(model_dir))
#
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model = net().to(device)
#    
#    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
#        print("Loading the dog-classifier model, path is {}".format(f))
#        checkpoint = torch.load(f, map_location=device)
#        model.load_state_dict(checkpoint)
#        print('MODEL-LOADED')
#        logger.info('model loaded successfully')
#    model.eval()
#    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    print('in input_fn')
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_object = input_object.to(device)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction
