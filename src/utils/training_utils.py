import numpy as np
import random
import torch
import os
import onnx2pytorch
import onnx
import re
from torch.nn import functional as F
import numpy as np

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

"""
def conv_orth_dist(kernel, stride = 1):

    new_s = stride*(w-1) + w#np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s*new_s*i_c).reshape((new_s*new_s*i_c, i_c, new_s,new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s*new_s*i_c, -1))
    Vmat = out[np.floor(new_s**2/2).astype(int)::new_s**2, :]
    temp= np.zeros((i_c, i_c*new_s**2))
    for i in range(temp.shape[0]):temp[i,np.floor(new_s**2/2).astype(int)+new_s**2*i]=1
    return torch.norm( Vmat@torch.t(out) - torch.from_numpy(temp).float().cuda() )

"""
    
def deconv_orth_dist(kernel, stride = 2, padding = 1):
    """
    Enforces spatial orthogonality via self-convolution: || Conv(K, K) - Dirac ||_F.
    Used for standard convolutional layers (3x3, etc.).
    """
    o_c, i_c, h, w = kernel.shape

    if h!= w:
        return orth_dist(kernel)

    try:
        output = F.conv2d(kernel, kernel, stride=stride, padding=padding)

        h_out, w_out = output.shape[-2], output.shape[-1]
        target = torch.zeros((o_c, o_c, h_out, w_out), device=kernel.device)

        ct_y = int(np.floor(h_out/2))
        ct_x = int(np.floor(w_out/2))

        ct_y = min(ct_y, h_out-1)
        ct_x = min(ct_x, w_out-1)

        target[:,:,ct_y,ct_x] = torch.eye(o_c, device=kernel.device)
        return torch.norm( output - target )
    
    except RuntimeError:
        return orth_dist(kernel)

    
def orth_dist(mat, stride=None):
    """
    Enforces matrix orthogonality: || K^T K - I ||_F.
    Used for Linear layers and 1x1 Conv shortcuts.
    """
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)

    eye = torch.eye(mat.shape[1], device=mat.device)
    return torch.norm( torch.t(mat)@mat - eye)


def compute_orth_loss_model(model):
    loss = 0

    named_params = list(model.named_parameters())

    bottleneck_parents = set()
    for name, param in named_params:
        if ".conv3.weight" in name:
            parts = name.split(".")
            if len(parts) >= 2:
                parent_name = ".".join(parts[:-2])
                bottleneck_parents.add(parent_name)

    for name, param in named_params:
        if "weight" not in name:
            continue

        is_conv = len(param.shape) == 4
        is_linear = len(param.shape) == 2

        if is_conv:
            if ".conv2" in name:
                parts = name.split(".")
                if len(parts) >= 2:
                    parent_name = ".".join(parts[:-2])
                    if parent_name not in bottleneck_parents:
                        continue

            try:
                layer_name = name.rsplit(".", 1)[0]
                layer = model.get_submodule(layer_name)

                stride = layer.stride[0] if hasattr(layer, 'stride') else 1
                padding = layer.padding[0] if hasattr(layer, 'padding') else 0

                if ".shortcut" in name:
                    loss += orth_dist(param)
                else:
                    loss += deconv_orth_dist(param, stride=stride, padding=padding)
            except Exception:
                loss += orth_dist(param)

        elif is_linear:
            loss += orth_dist(param)

    return loss


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__(reduction=reduction)

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))