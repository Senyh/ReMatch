import torch
import random


class ActivationExtraction(torch.nn.Module):
    """ extracting activations from targetted intermediate layers """

    def __init__(self, model, texture_layer, shape_layer):
        super(ActivationExtraction, self).__init__()
        self.model = model
        self.texture_activation = []
        self.shape_activation = []
        self.texture_grad = []
        self.shape_grad = []
        self.texture_handle = texture_layer.register_forward_hook(self.forward_texture_hook_fn)
        self.shape_handle = shape_layer.register_forward_hook(self.forward_shape_hook_fn)

    def forward_texture_hook_fn(self, module, input, output):
        activation = output
        self.texture_activation.append(activation.detach())

    def forward_shape_hook_fn(self, module, input, output):
        activation = output
        self.shape_activation.append(activation.detach())

    @staticmethod
    def instance_max_min_normalize(input_tensor, is_3d=True):
        b = input_tensor.shape[0]
        if is_3d:
            min_vals = torch.min(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1, 1)
            max_vals = torch.max(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1, 1)
        else:
            min_vals = torch.min(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
            max_vals = torch.max(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
        normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
        return normalized_tensor
    @staticmethod
    def instance_mean_std_normalize(input_tensor, is_3d=True):
        b = input_tensor.shape[0]
        if is_3d:
            mean_vals = torch.mean(input_tensor.view(b, -1), dim=1).view(b, 1, 1, 1, 1)
            std_vals = torch.std(input_tensor.view(b, -1), dim=1).view(b, 1, 1, 1, 1)
        else:
            mean_vals = torch.mean(input_tensor.view(b, -1), dim=1).view(b, 1, 1, 1)
            std_vals = torch.std(input_tensor.view(b, -1), dim=1).view(b, 1, 1, 1)
        normalized_tensor = (input_tensor - mean_vals + 1e-5) / (std_vals + 1e-5)
        return normalized_tensor

    def forward(self, x):
        self.texture_activation = []
        self.shape_activation = []
        return self.model(x)

    def __del__(self):
        self.texture_handle.remove()
        self.shape_handle.remove()