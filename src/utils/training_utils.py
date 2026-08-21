import numpy as np
import random
import torch
import os
import onnx2pytorch
import onnx
import re

from models import resnet, vgg, mobilenet, mobilefacenet, efficientnet

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def set_seed(seed):
    """Seed Python, NumPy, PyTorch, and cuDNN for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def load_pretrained_weights(model, model_name):
    """Load pretrained weights into the model, ignoring mismatched layers."""
    weights_folder = os.path.join(project_root, "models", "weights")
    weights_list = os.listdir(weights_folder)
    weight_file = None
    for file in weights_list:
        if model_name in file:
            weight_file = file
            break
    if weight_file is None:
        print(f"No pretrained weights found for model: {model_name}")
        return model, None  # Return the model without loading weights
    
    file_name = os.path.basename(weight_file)
    root, extension = os.path.splitext(file_name)
    dataset_name = root.split("_")[1]  # Assuming the format is model_dataset.pth or model_dataset.pt

    if extension == ".pth":
        pretrained_weights = torch.load(os.path.join(weights_folder, weight_file), map_location=torch.device('cpu'))
    elif extension == ".pt":
        pretrained_weights = torch.load(os.path.join(weights_folder, weight_file), map_location=torch.device('cpu'))
    elif extension == ".onnx":
        onnx_model = onnx.load(os.path.join(weights_folder, weight_file))
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        pretrained_weights = pytorch_model.state_dict()
    else:
        raise ValueError(f"Unsupported weight file format: {extension}")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model, dataset_name


def load_model(model_name, num_channels=3, num_outputs=1, dropout_rate=0.3, freezed=False):
    MODEL_CLASSES = {
        "resnet": resnet.ResNetRegression,
        "vgg": vgg.VGGRegression,
        "mobilenet": mobilenet.MobileNet,
        "mobilefacenet": mobilefacenet.MobileFaceNet,
        "efficientnet": efficientnet.EfficientNetB0
    }

    if "resnet" in model_name or "vgg" in model_name:
        model_name_letters = re.findall(r'[a-zA-Z]+', model_name)
        model_class = MODEL_CLASSES.get(model_name_letters[0], None)
        if model_class is None:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        model = model_class(model_name=model_name, num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)
    elif model_name in MODEL_CLASSES:
        model_class = MODEL_CLASSES[model_name]
        model = model_class(num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} / {total_params}")
    if trainable_params == 0:
        print("ERROR: No trainable parameters! Check freeze_backbone logic.")

    return model
