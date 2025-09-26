import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from collections import OrderedDict


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, cid, model_name, feature_dim, num_classes):
        super().__init__()

        if model_name == "resnet18":
            self.base = torchvision.models.resnet18(pretrained=False)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        
        head = None # you may need more code for pre-existing heterogeneous heads
        if hasattr(self.base, 'heads'):
            head = self.base.heads
            self.base.heads = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'head'):
            head = self.base.head
            self.base.head = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'fc'):
            head = self.base.fc
            self.base.fc = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'classifier'):
            head = self.base.classifier
            self.base.classifier = nn.AdaptiveAvgPool1d(feature_dim)
        else:
            raise('The base model does not have a classification head.')

        
        self.head = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


class FedEXTModel(BaseHeadSplit):
    """
    Improved FedEXT model with fine-grained layer-based splitting.
    Supports extraction of intermediate features at any split point.
    Specifically designed to handle ResNet18 and FedAvgCNN architectures.
    """

    def __init__(self, cid, model_name, feature_dim, num_classes, encoder_ratio):
        super().__init__(cid, model_name, feature_dim, num_classes)

        # Store encoder ratio for automatic split calculation
        self.encoder_ratio = encoder_ratio
        self.cid = cid

        # Layer management
        self.layers_list = None
        self.split_point = None
        self.global_layers = None
        self.local_layers = None
        self.layer_split_index = 0
        
        # Parameter tracking for aggregation
        self.global_param_names = set()
        self.local_param_names = set()

        # Identify model type and build sequential representation
        self.model_type = self._identify_model_type()
        self._build_sequential_model()
        
        # Automatically set split based on ratio
        self._set_split_by_ratio(self.encoder_ratio)

    def _identify_model_type(self):
        """Identify the type of base model for proper layer extraction"""
        base_class_name = self.base.__class__.__name__
        
        if 'ResNet' in base_class_name:
            return 'resnet'
        elif 'FedAvgCNN' in base_class_name or (hasattr(self.base, 'conv1') and hasattr(self.base, 'conv2')):
            return 'fedavgcnn'
        elif 'CNN' in base_class_name:
            return 'cnn'
        else:
            return 'generic'

    def _build_sequential_model(self):
        """
        Build a fine-grained layer representation for splitting.
        Handles different model architectures with better granularity.
        """
        self.layers_list = []

        if self.model_type == 'resnet':
            self._extract_resnet_layers_fine_grained()
        elif self.model_type == 'fedavgcnn':
            self._extract_fedavgcnn_layers()
        else:
            self._extract_generic_layers()

        # Always add head at the end
        if isinstance(self.head, nn.Sequential):
            for i, module in enumerate(self.head):
                self.layers_list.append((f"head.{i}", module))
        else:
            self.layers_list.append(("head", self.head))

    def _extract_resnet_layers_fine_grained(self):
        """
        Extract layers from ResNet architecture with fine granularity.
        Breaks down residual layers into individual blocks.
        """
        # Initial layers
        if hasattr(self.base, 'conv1'):
            self.layers_list.append(("base.conv1", self.base.conv1))
        if hasattr(self.base, 'bn1'):
            self.layers_list.append(("base.bn1", self.base.bn1))
        if hasattr(self.base, 'relu'):
            self.layers_list.append(("base.relu", self.base.relu))
        if hasattr(self.base, 'maxpool'):
            self.layers_list.append(("base.maxpool", self.base.maxpool))

        # Residual layers - extract individual blocks
        for layer_idx in range(1, 5):
            layer_name = f'layer{layer_idx}'
            if hasattr(self.base, layer_name):
                layer = getattr(self.base, layer_name)
                # Extract individual blocks from each layer
                if isinstance(layer, nn.Sequential):
                    for block_idx, block in enumerate(layer):
                        self.layers_list.append((f"base.{layer_name}.{block_idx}", block))
                else:
                    # Fallback: treat entire layer as one unit
                    self.layers_list.append((f"base.{layer_name}", layer))

        # Final pooling
        if hasattr(self.base, 'avgpool'):
            self.layers_list.append(("base.avgpool", self.base.avgpool))

    def _extract_fedavgcnn_layers(self):
        """Extract layers from FedAvgCNN architecture"""
        if hasattr(self.base, 'conv1'):
            self.layers_list.append(("base.conv1", self.base.conv1))
        if hasattr(self.base, 'conv2'):
            self.layers_list.append(("base.conv2", self.base.conv2))
        if hasattr(self.base, 'fc1'):
            self.layers_list.append(("base.fc1", self.base.fc1))

    def _extract_generic_layers(self):
        """Generic layer extraction for unknown architectures"""
        children = list(self.base.named_children())
        if children:
            for name, module in children:
                if name not in ['fc', 'classifier', 'head', 'heads']:
                    self.layers_list.append((f"base.{name}", module))
        else:
            self.layers_list.append(("base", self.base))

    def _set_split_by_ratio(self, ratio):
        """
        Automatically set the split point based on the encoder ratio.
        
        Args:
            ratio: Float between 0 and 1, proportion of layers to be global
        """
        if self.layers_list is None:
            self._build_sequential_model()
        
        total_layers = len(self.layers_list)
        # Calculate split index based on ratio
        split_index = int(math.ceil(total_layers * ratio))
        
        # Ensure at least one layer is local (the head)
        split_index = min(split_index, total_layers - 1)
        
        self.set_layer_split(split_index)
        
        print(f"[Client {self.cid}] Auto-split at layer {split_index}/{total_layers} (ratio={ratio:.2f})")

    def set_layer_split(self, layer_split_index):
        """
        Set the split point based on layer index.
        Layers before split_index are global (FedAvg), after are local (group).
        """
        if self.layers_list is None:
            self._build_sequential_model()

        self.layer_split_index = layer_split_index

        if layer_split_index == 0:
            # All layers are local
            self.global_layers = []
            self.local_layers = self.layers_list
        elif layer_split_index >= len(self.layers_list):
            # All layers are global
            self.global_layers = self.layers_list
            self.local_layers = []
        else:
            # Split at layer_split_index
            self.global_layers = self.layers_list[:layer_split_index]
            self.local_layers = self.layers_list[layer_split_index:]

        # Update parameter tracking
        self._update_parameter_tracking()
        
        # Debug output
        self._print_split_info()

    def _update_parameter_tracking(self):
        """
        Track which parameters belong to global vs local parts.
        This is crucial for correct aggregation in federated learning.
        """
        self.global_param_names.clear()
        self.local_param_names.clear()
        
        # Track global parameters
        for layer_name, layer in self.global_layers:
            for param_name, _ in layer.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                self.global_param_names.add(full_name)
        
        # Track local parameters
        for layer_name, layer in self.local_layers:
            for param_name, _ in layer.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                self.local_param_names.add(full_name)

    def get_global_params(self):
        """
        Get parameters for global (FedAvg) aggregation.
        Returns only the parameters from layers before the split point.
        """
        global_dict = OrderedDict()
        
        # Extract parameters from global layers
        for layer_name, layer in self.global_layers:
            for param_name, param in layer.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                # Remove prefix for compatibility with server aggregation
                clean_name = full_name.replace("base.", "").replace("head.", "")
                global_dict[clean_name] = param.data.clone()
        
        return global_dict

    def get_local_params(self):
        """
        Get parameters for local (group) aggregation.
        Returns only the parameters from layers after the split point.
        """
        local_dict = OrderedDict()
        
        # Extract parameters from local layers
        for layer_name, layer in self.local_layers:
            for param_name, param in layer.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                # Remove prefix for compatibility with server aggregation
                clean_name = full_name.replace("base.", "").replace("head.", "")
                local_dict[clean_name] = param.data.clone()
        
        return local_dict

    def set_global_params(self, state_dict):
        """Update only global parameters"""
        for layer_name, layer in self.global_layers:
            layer_state = {}
            for param_name in layer.state_dict():
                # Try different key formats
                for prefix in ["", "base.", "head."]:
                    key = f"{layer_name.replace('base.', '').replace('head.', '')}.{param_name}"
                    if prefix + key in state_dict:
                        layer_state[param_name] = state_dict[prefix + key]
                        break
                    if key in state_dict:
                        layer_state[param_name] = state_dict[key]
                        break
            
            if layer_state:
                layer.load_state_dict(layer_state, strict=False)

    def set_local_params(self, state_dict):
        """Update only local parameters"""
        for layer_name, layer in self.local_layers:
            layer_state = {}
            for param_name in layer.state_dict():
                # Try different key formats
                for prefix in ["", "base.", "head."]:
                    key = f"{layer_name.replace('base.', '').replace('head.', '')}.{param_name}"
                    if prefix + key in state_dict:
                        layer_state[param_name] = state_dict[prefix + key]
                        break
                    if key in state_dict:
                        layer_state[param_name] = state_dict[key]
                        break
            
            if layer_state:
                layer.load_state_dict(layer_state, strict=False)

    def extract_global_features(self, x):
        """
        Extract features from the globally aggregated part of the model.
        Execute layers up to the split point and return intermediate features.
        """
        if self.global_layers is None or len(self.global_layers) == 0:
            # No global layers (pure group mode), return original input
            # Don't flatten here as local layers may expect 4D input
            return x

        # Execute global layers sequentially
        out = x
        for layer_name, layer in self.global_layers:
            out = self._forward_layer(out, layer_name, layer)

        # Prepare output for next layers
        out = self._prepare_for_next_layers(out)
        
        return out

    def _forward_layer(self, x, layer_name, layer):
        """
        Forward pass for a single layer with proper handling of special cases.
        """
        if 'avgpool' in layer_name and self.model_type == 'resnet':
            out = layer(x)
            return torch.flatten(out, 1)
        elif 'fc' in layer_name and len(x.shape) > 2:
            return layer(torch.flatten(x, 1))
        else:
            return layer(x)

    def _prepare_for_next_layers(self, x):
        """
        Prepare features for the next set of layers.
        Handles flattening when necessary.
        """
        if self.local_layers and len(self.local_layers) > 0:
            next_layer_name = self.local_layers[0][0]
            # Check if next layer expects flattened input
            if ('fc' in next_layer_name or 'head' in next_layer_name) and len(x.shape) > 2:
                return torch.flatten(x, 1)
        return x

    def forward_split(self, x):
        """
        Forward pass that returns intermediate features at split point.
        
        Returns:
            features: Features at the split point
            output: Final output after all layers
        """
        if self.global_layers is None or len(self.global_layers) == 0:
            # Pure group mode: all layers are local
            # Execute all local layers to get both features and output
            out = x
            for layer_name, layer in self.local_layers:
                out = self._forward_layer(out, layer_name, layer)
            
            # For feature extraction in pure group mode, 
            # return the penultimate layer output as features
            if len(self.local_layers) > 1:
                # Re-run to get features from second-to-last layer
                features = x
                for layer_name, layer in self.local_layers[:-1]:
                    features = self._forward_layer(features, layer_name, layer)
                # Ensure features are flattened for compatibility
                if len(features.shape) > 2:
                    features = torch.flatten(features, 1)
            else:
                # Only one layer, return flattened input as features
                features = torch.flatten(x, 1) if len(x.shape) > 2 else x
            
            return features, out
        else:
            # Normal case: execute global layers to get features
            features = self.extract_global_features(x)
            
            # Execute local layers if they exist
            if self.local_layers and len(self.local_layers) > 0:
                out = features
                for layer_name, layer in self.local_layers:
                    out = self._forward_layer(out, layer_name, layer)
                return features, out
            else:
                return features, features

    def forward(self, x):
        """
        Override forward to use the split-aware forward pass.
        """
        if self.layers_list is not None and (self.global_layers is not None or self.local_layers is not None):
            # Use split-aware forward if configured
            _, output = self.forward_split(x)
            return output
        else:
            # Fall back to original forward
            return super().forward(x)

    def _print_split_info(self):
        """Print information about the current split configuration"""
        total_params = sum(p.numel() for p in self.parameters())
        
        global_params = 0
        for _, layer in self.global_layers:
            global_params += sum(p.numel() for p in layer.parameters())
        
        local_params = total_params - global_params
        
        print(f"\n[Client {self.cid}] Model Split Configuration:")
        print(f"  Total layers: {len(self.layers_list)}")
        print(f"  Split at layer: {self.layer_split_index}")
        print(f"  Global layers: {len(self.global_layers)} ({global_params:,} params, {global_params/total_params*100:.1f}%)")
        print(f"  Local layers: {len(self.local_layers)} ({local_params:,} params, {local_params/total_params*100:.1f}%)")
        
        if len(self.global_layers) > 0:
            global_names = [name for name, _ in self.global_layers]
            print(f"  Global: {global_names[:3]}..." if len(global_names) > 3 else f"  Global: {global_names}")
        
        if len(self.local_layers) > 0:
            local_names = [name for name, _ in self.local_layers]
            print(f"  Local: {local_names[:3]}..." if len(local_names) > 3 else f"  Local: {local_names}")

    def update_split_ratio(self, new_ratio):
        """
        Dynamically update the split ratio during training.
        Useful for adaptive splitting strategies.
        """
        self.encoder_ratio = new_ratio
        self._set_split_by_ratio(new_ratio)

    def get_split_info(self):
        """
        Get information about current split configuration.
        Useful for server-side tracking and analysis.
        """
        return {
            'client_id': self.cid,
            'split_index': self.layer_split_index,
            'total_layers': len(self.layers_list),
            'encoder_ratio': self.encoder_ratio,
            'global_layers_count': len(self.global_layers),
            'local_layers_count': len(self.local_layers),
            'global_param_count': len(self.global_param_names),
            'local_param_count': len(self.local_param_names)
        }