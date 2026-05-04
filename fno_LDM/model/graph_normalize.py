import os
from copy import deepcopy
import torch
import sys; sys.path.append(os.getcwd())


def convert2dict(file_path):
    model_data = torch.load(file_path, weights_only=False, map_location='cpu')
    
    weights_dict = {}
    for name, tensor in model_data.items():
        if "weight" in name or "bias" in name:
            weights_dict[name] = tensor.cpu()
    
    return weights_dict

def merge_weights_dicts(weight_dicts_list):
    merged_dict = {}
    
    for weights_dict in weight_dicts_list:
        for key, value in weights_dict.items():
            if key not in merged_dict:
                merged_dict[key] = []
            merged_dict[key].append(value)
    
    return merged_dict

def build_normlize_weights(merged_dict):
    weights_norm_dict = {}
    
    for layer_name, weights_list in merged_dict.items():
        weights_tensor = torch.stack(weights_list)
        # Check if the weights tensor is complex
        if torch.is_complex(weights_tensor):
            print('complex', layer_name, weights_tensor.shape)
            
            real_weights = torch.real(weights_tensor)
            imag_weights = torch.imag(weights_tensor)
            
            # Compute min/max for both real and imaginary parts
            real_xmin = real_weights.amin(dim=(0,3,4), keepdim=True)[0]
            real_xmax = real_weights.amax(dim=(0,3,4), keepdim=True)[0]
            imag_xmin = imag_weights.amin(dim=(0,3,4), keepdim=True)[0]
            imag_xmax = imag_weights.amax(dim=(0,3,4), keepdim=True)[0]
            
            # Store the real and imaginary min/max separately
            weights_norm_dict[layer_name] = ((real_xmin, imag_xmin), (real_xmax, imag_xmax))
        
        else:
            print('real', layer_name, weights_tensor.shape)
            
            if len(weights_tensor.shape) == 4:
                dimensions = (0,3)
            elif len(weights_tensor.shape) == 2:
                dimensions = (0,)
            
            xmin = weights_tensor.amin(dim=dimensions, keepdim=True)[0]
            xmax = weights_tensor.amax(dim=dimensions, keepdim=True)[0]
            weights_norm_dict[layer_name] = (xmin, xmax)
    
    return weights_norm_dict

def normalize_weights(model, state_dict, norm_dict):
    model_ = deepcopy(model)
    state_dict_new = {}
    
    for layer_name, (xmin, xmax) in norm_dict.items():
        if layer_name not in state_dict:
            raise KeyError(f"Layer '{layer_name}' not found in model's state_dict.")

        # Check if the weights are complex
        if torch.is_complex(state_dict[layer_name]):
            real_xmin, real_xmax = xmin[0], xmax[0]
            imag_xmin, imag_xmax = xmin[1], xmax[1]
            
            # Normalize real and imaginary parts separately
            state_dict_new[layer_name] = torch.complex(
                (state_dict[layer_name].real.cpu() - real_xmin) / (real_xmax - real_xmin),
                (state_dict[layer_name].imag.cpu() - imag_xmin) / (imag_xmax - imag_xmin)
            )
        else:
            state_dict_new[layer_name] = state_dict[layer_name].cpu().sub(xmin).div(xmax - xmin)

    model_.load_state_dict(state_dict_new)
    return model_


def renormalize_weights(model, state_dict, norm_dict):
    model_ = deepcopy(model)
    state_dict_new = {}

    for layer_name, (xmin, xmax) in norm_dict.items():
        if layer_name not in state_dict:
            raise KeyError(f"Layer '{layer_name}' not found in model's state_dict.")

        # Check if the weights are complex
        if torch.is_complex(state_dict[layer_name]):
            real_xmin, real_xmax = xmin[0], xmax[0]
            imag_xmin, imag_xmax = xmin[1], xmax[1]
            
            # Renormalize real and imaginary parts separately
            state_dict_new[layer_name] = torch.complex(
                state_dict[layer_name].real.cpu() * (real_xmax - real_xmin) + real_xmin,
                state_dict[layer_name].imag.cpu() * (imag_xmax - imag_xmin) + imag_xmin
            )
        else:
            state_dict_new[layer_name] = state_dict[layer_name].cpu().mul(xmax - xmin).add(xmin)

    model_.load_state_dict(state_dict_new)
    return model_
