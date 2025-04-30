# backbone_module.py (Corrigé et Reformaté pour 480x720 In/Out)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
import warnings
import os

# =============================================================================
# === SECTION 1 : BACKBONE MULTITASK (RegNet + BiFPN + Têtes Seg/Depth) =====
# =============================================================================

# --- Fonctions utilitaires de RegNet ---
def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)

def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    min_len = min(len(gs), len(ws_bot))
    gs = [min(gs[i], ws_bot[i]) for i in range(min_len)]
    ws_bot = ws_bot[:min_len]
    ws_bot = [quantize_float(ws_bot[i], gs[i]) for i in range(min_len)]
    if len(bms) > len(ws_bot):
        bms = bms[:len(ws_bot)]
    elif len(bms) < len(ws_bot):
        if bms:
            bms.extend([bms[-1]] * (len(ws_bot) - len(bms)))
        else:
            raise ValueError("bms cannot be empty if ws_bot is not")
    # Avoid division by zero if bms[i] happens to be 0
    ws = [int(ws_bot[i] / bms[i]) if bms[i] != 0 else 0 for i in range(len(ws_bot))]
    return ws, gs

def get_stages_from_blocks(ws_in, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws_in + [0], [0] + ws_in, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws_in, ts[:-1]) if t]
    diff_indices = [d for d, t in zip(range(len(ts)), ts) if t]
    s_ds = [] if len(diff_indices) < 2 else np.diff(diff_indices).tolist()
    return s_ws, s_ds

def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0, "Invalid RegNet params"
    ws_cont = np.arange(d) * w_a + w_0
    eps = 1e-8
    # Handle potential division by zero or log of non-positive number
    valid_indices = ws_cont > 0
    log_input = np.maximum(ws_cont / w_0, eps) # Use maximum to ensure positive input for log
    ks = np.zeros_like(ws_cont, dtype=float) # Initialize ks as float
    if np.any(valid_indices):
        ks[valid_indices] = np.round(np.log(log_input[valid_indices]) / np.log(w_m))

    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    unique_ws = np.unique(ws)
    num_stages = len(unique_ws) if len(unique_ws) > 0 else 0
    max_stage = 0 if ks.size == 0 else ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont

# --- Blocs de construction RegNet ---
class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""
    def __init__(self, in_w=3, out_w=32):
        super().__init__()
        self.conv = nn.Conv2d(in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_w)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block."""
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1."""
    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super().__init__()
        w_b = int(round(w_out * bm))
        # Ensure group width is positive and sensible
        gw = max(1, gw or 1)
        num_gs = max(1, w_b // gw)
        # Adjust w_b to be divisible by num_gs and non-zero
        w_b = max(num_gs, int(round(w_b / num_gs) * num_gs))
        w_b = max(w_b, 1) # Ensure w_b is at least 1

        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b)
        self.b_relu = nn.ReLU(inplace=True)
        # Add SE block only if se_r is provided and positive
        if se_r and se_r > 0:
            w_se = max(1, int(round(w_in * se_r)))
            self.se = SE(w_b, w_se)
        else:
            self.se = nn.Identity()
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out)
        # Add marker for potential special BN initialization
        self.c_bn.final_bn = True

    def forward(self, x):
        x = self.a(x); x = self.a_bn(x); x = self.a_relu(x)
        x = self.b(x); x = self.b_bn(x); x = self.b_relu(x)
        x = self.se(x)
        x = self.c(x); x = self.c_bn(x)
        return x

class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""
    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super().__init__()
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out)
        else:
            # Explicitly set to None if not used
            self.proj, self.bn = None, None
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.bn(self.proj(x)) if self.proj_block else x
        f_out = self.f(x)
        x = shortcut + f_out
        x = self.relu(x)
        return x

class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""
    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super().__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Ensure block_fun is callable
            if not callable(block_fun):
                 raise TypeError(f"block_fun must be callable, got {type(block_fun)}")
            self.add_module(f"b{i + 1}", block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

# --- Classe AnyNet modifiée pour extraction ---
class AnyNetFeatureExtractor(nn.Module):
    """AnyNet model modified to extract features from stages."""
    def __init__(self, **kwargs):
        super().__init__()
        # Ensure required arguments are present
        required_args = ["stem_w", "ds", "ws", "ss"]
        for arg in required_args:
            if arg not in kwargs or not kwargs[arg]:
                # Allow ds to be empty if ws is also empty (num_stages=0 case)
                if arg == "ds" and not kwargs.get("ws"):
                    continue
                raise ValueError(f"Missing or empty required argument '{arg}' for AnyNetFeatureExtractor")

        self.stem_w = kwargs["stem_w"]
        self.ds = kwargs["ds"]
        self.ws = kwargs["ws"]
        self.ss = kwargs["ss"]
        self.bms = kwargs.get("bms", [1.0] * len(self.ds))
        self.gws = kwargs.get("gws", [1] * len(self.ds))
        self.se_r = kwargs.get("se_r") # Can be None

        # Ensure lists have consistent lengths if ds is not empty
        if self.ds:
             expected_len = len(self.ds)
             for name, lst in [("ws", self.ws), ("ss", self.ss), ("bms", self.bms), ("gws", self.gws)]:
                 if len(lst) != expected_len:
                      raise ValueError(f"Length mismatch: len({name})={len(lst)} != len(ds)={expected_len}")

        self.stage_params = list(zip(self.ds, self.ws, self.ss, self.bms, self.gws))

        self.stem = SimpleStemIN(3, self.stem_w)
        block_fun = ResBottleneckBlock
        prev_w = self.stem_w
        self.stages = nn.ModuleList()
        for i, (d, w, s, bm, gw) in enumerate(self.stage_params):
            # Ensure group width is at least 1
            gw = max(1, gw or 1)
            stage = AnyStage(prev_w, w, s, d, block_fun, bm, gw, self.se_r)
            self.stages.append(stage)
            prev_w = w

        self._compute_feature_info()
        self._initialize_weights()

    def _compute_feature_info(self):
        """Computes information about features extracted at each stage."""
        self.feature_info = []
        current_stride = 2 # Stem has stride 2
        self.feature_info.append({'num_chs': self.stem_w, 'reduction': current_stride, 'module': 'stem'})
        prev_w = self.stem_w
        # Iterate through the actual stages created
        for i, stage_cfg in enumerate(self.stage_params):
            d, w, s, bm, gw = stage_cfg
            # Stride is applied at the beginning of the stage
            current_stride *= s
            self.feature_info.append({'num_chs': w, 'reduction': current_stride, 'module': f's{i+1}'})
            prev_w = w # Not strictly needed here but good practice

    def _initialize_weights(self):
        """Initializes weights for the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal initialization
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                std = math.sqrt(2.0 / fan_out) if fan_out > 0 else 0.0
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize gamma to 1 (or 0 for final BN) and beta to 0
                is_final_bn = hasattr(m, "final_bn") and m.final_bn
                nn.init.constant_(m.weight, 0. if is_final_bn else 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                # Initialization for linear layers
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        """Forward pass, extracting features at each stage."""
        # Input x assumed to be already resized to the target network input size (e.g., 480x720)
        features = OrderedDict()
        x = self.stem(x)
        features['stem'] = x # Stride 2 features
        # Iterate through the nn.ModuleList of stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[f's{i+1}'] = x # Features after s1 (stride 4), s2 (stride 8), etc.
        return features

# --- Classe RegNet qui utilise AnyNetFeatureExtractor ---
class RegNetBackbone(AnyNetFeatureExtractor):
    """RegNet backbone model."""
    def __init__(self, cfg, **kwargs):
        # Generate parameters from RegNet config
        b_ws, num_s, _, _ = generate_regnet(cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH'], cfg.get('Q', 8))
        if not b_ws:
            raise ValueError("Generated block widths (b_ws) are empty. Check RegNet parameters.")

        # Convert block parameters to stage parameters
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        if not ws:
            # Handle case where no stages are generated (e.g., DEPTH is very small)
            num_s = 0
            warnings.warn("No stages generated from RegNet parameters (ws is empty).")

        # Generate group widths and bottleneck multipliers per stage
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [cfg['BOT_MUL'] for _ in range(num_s)]

        # Adjust widths and groups for compatibility if stages exist
        if num_s > 0 and ws and gws:
            ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        elif num_s > 0:
            warnings.warn("ws or gws were empty before adjustment, skipping.")

        # Ensure all lists match the actual number of stages
        ds = ds[:num_s]
        ws = ws[:num_s]
        bms = bms[:num_s]
        gws = gws[:num_s]

        # Define strides per stage (typically 2 for all)
        ss = [2 for _ in range(num_s)]
        # Get SE ratio if specified (for RegNetY variants)
        se_r = cfg.get('SE_R', None)

        # Prepare arguments for the parent AnyNetFeatureExtractor class
        model_kwargs = {
            "stem_w": cfg.get('STEM_W', 32),
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            **kwargs # Pass any additional kwargs
        }

        # Call the parent constructor
        super().__init__(**model_kwargs)

# --- Configs RegNet ---
regnet_1600M_config = {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'BOT_MUL': 1, 'SE_R': 0.25}
model_paths = {'regnet_1600m': None} # Path to local prepared ImageNet weights (optional)

# --- Fonction de création du backbone ---
def create_regnet_backbone(config, pretrained=False, ckpt_path=None, **kwargs):
    """Creates the RegNet backbone, optionally loading pretrained weights."""
    model = RegNetBackbone(config, **kwargs)
    if pretrained and ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading pretrained backbone weights from: {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            # Handle common checkpoint formats
            if 'model_state' in state_dict: state_dict = state_dict['model_state']
            elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            # Remove 'module.' prefix if saved from DataParallel
            if all(k.startswith('module.')): state_dict = {k[7:]: v for k, v in state_dict.items()}
            # Remove classification head keys if present
            keys_to_remove = [k for k in state_dict if k.startswith('head.') or k.startswith('fc.')]
            if keys_to_remove:
                print(f"Removing classification head weights: {keys_to_remove}")
                for k in keys_to_remove: state_dict.pop(k, None)

            # Load into model (non-strict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing: print(f"Warning: Missing keys when loading backbone: {missing}")
            # Filter unexpected keys to ignore potential head keys if not removed above
            unexpected = [k for k in unexpected if not (k.startswith('head.') or k.startswith('fc.'))]
            if unexpected: print(f"Warning: Unexpected keys when loading backbone: {unexpected}")
            print("Pretrained backbone weights loaded successfully (strict=False).")
        except Exception as e:
            print(f"Error loading backbone weights from {ckpt_path}: {e}")
    elif pretrained and ckpt_path:
        print(f"Warning: Pretrained backbone checkpoint path '{ckpt_path}' not found.")
    elif pretrained:
        print("Warning: 'pretrained=True' but no 'ckpt_path' provided for backbone.")
    return model

# --- Neck BiFPN ---
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False, act=nn.ReLU):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.act = act(inplace=True) if act and hasattr(act, 'inplace') else (act() if act else nn.Identity())


    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BiFPNNode(nn.Module):
    """A node for weighted feature fusion in BiFPN."""
    def __init__(self, channels, num_inputs, eps=1e-4):
        super().__init__()
        self.num_inputs = num_inputs
        self.eps = eps
        # Learnable weights for weighted fusion
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        # ReLU for positivity, applied during forward pass
        self.relu = nn.ReLU(inplace=False)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs, f"BiFPNNode expected {self.num_inputs} inputs, got {len(inputs)}"
        # Ensure weights are positive and normalize
        weights = self.relu(self.weights)
        weights = weights / (torch.sum(weights, dim=0, keepdim=True) + self.eps) # Keepdim for broadcasting
        # Weighted fusion: stack, multiply, sum
        x = torch.stack(inputs, dim=-1)
        x = (weights * x).sum(dim=-1)
        return x

class BiFPNLayer(nn.Module):
    """A single BiFPN layer performing top-down and bottom-up fusion."""
    def __init__(self, channels, levels, eps=1e-4, act=nn.ReLU):
        super().__init__()
        self.channels = channels
        self.levels = levels # Number of FPN levels processed by this layer
        self.eps = eps
        self.act = act

        # Convolutions applied after each fusion node
        self.convs = nn.ModuleList([DepthwiseSeparableConv(channels, channels, act=act) for _ in range(levels)])

        # Fusion nodes for top-down path (N-1 nodes)
        self.fuse_td = nn.ModuleList([BiFPNNode(channels, 2, eps) for _ in range(levels - 1)])

        # Fusion nodes for bottom-up path (N nodes)
        # First and last BU nodes have 2 inputs, intermediate ones have 3
        self.fuse_bu = nn.ModuleList([
            BiFPNNode(channels, 2 if i == 0 or i == levels - 1 else 3, eps)
            for i in range(levels)
        ])

        # Resizing operations
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # MaxPool for downsampling is common, but interpolate might give more control
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _resize_like(self, source, target):
        """Resizes source tensor to match spatial dimensions of target tensor."""
        if source.shape[-2:] != target.shape[-2:]:
            return F.interpolate(source, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return source

    def forward(self, inputs):
        """
        Args:
            inputs (list[torch.Tensor]): List of features from different levels,
                                         ordered from highest resolution (e.g., P3)
                                         to lowest resolution (e.g., P5).
        Returns:
            list[torch.Tensor]: List of fused features at the same levels.
        """
        num_levels = len(inputs)
        if num_levels != self.levels:
             raise ValueError(f"BiFPNLayer expected {self.levels} input levels, got {num_levels}")

        # --- Top-Down Path ---
        td_features = [None] * num_levels
        # Initialize highest level (lowest resolution)
        td_features[-1] = inputs[-1]
        # Iterate from second-highest level down to the lowest
        for i in range(num_levels - 2, -1, -1):
            target_feature = inputs[i]
            feature_to_upsample = td_features[i+1]
            upsampled_feature = self.upsample(feature_to_upsample)
            # Ensure spatial dimensions match before fusion
            upsampled_feature = self._resize_like(upsampled_feature, target_feature)
            # Fuse original input and upsampled feature
            fused_node = self.fuse_td[i]([target_feature, upsampled_feature])
            # Apply convolution after fusion
            td_features[i] = self.convs[i](fused_node)

        # --- Bottom-Up Path ---
        out_features = [None] * num_levels
        # Initialize lowest level (highest resolution)
        out_features[0] = td_features[0]
        # Iterate from second-lowest level up to the highest
        for i in range(1, num_levels):
            target_feature_td = td_features[i] # Feature from TD path at this level
            feature_to_downsample = out_features[i-1] # Feature from the level below
            downsampled_feature = self.downsample(feature_to_downsample)
            # Ensure spatial dimensions match the TD feature at this level
            downsampled_feature = self._resize_like(downsampled_feature, target_feature_td)

            # Fuse based on position in the path
            if i == num_levels - 1: # Last BU node (highest level): 2 inputs
                target_feature_orig = inputs[i] # Original input feature
                # Ensure downsampled feature matches original feature size
                downsampled_feature = self._resize_like(downsampled_feature, target_feature_orig)
                fused_node = self.fuse_bu[i]([target_feature_orig, downsampled_feature])
            else: # Intermediate BU nodes: 3 inputs
                target_feature_orig = inputs[i] # Original input feature
                # Ensure all features match spatial size (use original as reference)
                downsampled_feature = self._resize_like(downsampled_feature, target_feature_orig)
                target_feature_td = self._resize_like(target_feature_td, target_feature_orig)
                fused_node = self.fuse_bu[i]([target_feature_orig, target_feature_td, downsampled_feature])

            # Apply convolution after fusion
            # Use the same conv index as the TD path for weight sharing (common practice)
            out_features[i] = self.convs[i](fused_node)

        return out_features

class BiFPNNeck(nn.Module):
    """Complete BiFPN Neck module."""
    def __init__(self, backbone_channels, fpn_channels, num_repeats, levels_to_use, activation=nn.ReLU, eps=1e-4):
        super().__init__()
        self.levels_to_use = levels_to_use # e.g., ['s2', 's3', 's4']
        self.num_levels_in = len(levels_to_use)
        self.fpn_channels = fpn_channels

        # Check if all requested levels are provided by the backbone
        missing = [lvl for lvl in levels_to_use if lvl not in backbone_channels]
        if missing:
            raise ValueError(f"BiFPNNeck: Backbone channels missing for levels: {missing}. Provided: {list(backbone_channels.keys())}")

        # Lateral convolutions (1x1) to adjust backbone channels
        self.lat_convs = nn.ModuleList([
            nn.Conv2d(backbone_channels[lvl], fpn_channels, kernel_size=1)
            for lvl in levels_to_use
        ])

        # Stacked BiFPN layers
        self.bifpn = nn.ModuleList([
            BiFPNLayer(fpn_channels, self.num_levels_in, eps, act=activation)
            for _ in range(num_repeats)
        ])

        # Determine output P-level names based on input backbone stages
        # Assumes s1->P2, s2->P3, s3->P4, s4->P5, etc.
        stage_to_p_level_map = {'s1':2, 's2':3, 's3':4, 's4':5, 's5':6, 's6':7}
        self.out_lvl_names = []
        min_p, max_p = float('inf'), float('-inf')
        for lvl in self.levels_to_use:
            p_idx = stage_to_p_level_map.get(lvl)
            if p_idx is None:
                try: # Attempt to deduce from index 'sX' -> P(X+1)
                    p_idx = int(lvl[1:]) + 1
                    warnings.warn(f"BiFPNNeck: Stage {lvl} not in map, P-level {p_idx} deduced from index.")
                except (ValueError, IndexError):
                    raise ValueError(f"Cannot determine P-level for backbone stage: {lvl}.")
            self.out_lvl_names.append(f"P{p_idx}")
            min_p, max_p = min(min_p, p_idx), max(max_p, p_idx)

        # Optional check for contiguous P-levels
        expected_p = list(range(min_p, max_p + 1))
        actual_p = sorted([int(name[1:]) for name in self.out_lvl_names])
        if actual_p != expected_p:
            warnings.warn(f"BiFPNNeck: Generated P levels {self.out_lvl_names} are not contiguous.")
        print(f"BiFPNNeck initialized. Output P-level names: {self.out_lvl_names}")

    def forward(self, features):
        """
        Args:
            features (dict[str, torch.Tensor]): Dictionary of backbone features
                                                 (e.g., {'s2': tensor, 's3': tensor,...})
        Returns:
            OrderedDict[str, torch.Tensor]: Dictionary of FPN features
                                            (e.g., {'P3': tensor, 'P4': tensor,...})
        """
        # Apply lateral convolutions to selected backbone features
        lat_feats = []
        for i, lvl in enumerate(self.levels_to_use):
            if lvl not in features:
                raise KeyError(f"BiFPNNeck: Backbone level '{lvl}' not found in input features {list(features.keys())}")
            lat_feats.append(self.lat_convs[i](features[lvl]))

        # Pass features through BiFPN layers
        for layer in self.bifpn:
            lat_feats = layer(lat_feats)

        # Return features in an ordered dictionary with P-level names
        return OrderedDict(zip(self.out_lvl_names, lat_feats))


# --- Têtes Seg/Depth (MODIFIÉES pour output_target_size=(480, 720) par défaut) ---
class SemanticSegmentationHead(nn.Module):
    """Segmentation head using FPN features."""
    def __init__(self, fpn_channels, num_classes, output_target_size=(480, 720), levels_to_use=['P3', 'P4', 'P5']): # *** MODIFIÉ ***
        super().__init__()
        self.levels_to_use = levels_to_use # Levels expected from FPN
        self.target_size = output_target_size # Final output HxW
        self.num_classes = num_classes

        # Internal dimension for feature adaptation
        embed_dim = max(fpn_channels // 2, 128)
        self.adapters = nn.ModuleDict()
        # Create adapter layers only for the levels specified at init
        for level in self.levels_to_use:
            self.adapters[level] = nn.Sequential(
                nn.Conv2d(fpn_channels, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )

        # Final classification layer - size depends on the number of levels used at init
        self.final_conv = nn.Conv2d(embed_dim * len(self.levels_to_use), num_classes, kernel_size=1)
        print(f"SegHead init: Uses {self.levels_to_use}. Final Conv expects {self.final_conv.in_channels} channels. Target output: {self.target_size}")

    def forward(self, features):
        """
        Args:
            features (dict[str, torch.Tensor]): FPN features (e.g., {'P3':..., 'P4':...}).
        Returns:
            dict[str, torch.Tensor or None]: Segmentation output or None if error.
        """
        adapted_features = []
        internal_target_size = None # Size for internal fusion
        processed_levels = [] # Keep track of levels actually used

        # Process only the levels that are available in the input features AND were expected at init
        for lvl in self.levels_to_use:
            if lvl in features:
                # Determine the internal fusion size from the highest-res available feature
                if internal_target_size is None:
                    internal_target_size = features[lvl].shape[-2:]
                # Apply adapter
                adapted = self.adapters[lvl](features[lvl])
                adapted_features.append(adapted)
                processed_levels.append(lvl)
            # else: warnings.warn(f"SegHead: Expected level {lvl} not found in input features.")

        if not adapted_features:
            print("Warning: SegHead: No usable FPN features found/adapted.")
            return {'segmentation': None}

        # Resize all adapted features to the internal target size before concatenation
        resized_features = []
        for feat in adapted_features:
            if feat.shape[-2:] != internal_target_size:
                feat = F.interpolate(feat, size=internal_target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)

        # Concatenate features from the levels that were actually processed
        final_features = torch.cat(resized_features, dim=1)

        # Check if the number of channels matches the final convolution layer
        if self.final_conv.in_channels != final_features.shape[1]:
            warnings.warn(f"SegHead channel mismatch! FinalConv expects {self.final_conv.in_channels}, "
                          f"but got {final_features.shape[1]} (from levels {processed_levels}). "
                          f"Check head's levels_to_use config vs. FPN output.")
            return {'segmentation': None} # Avoid runtime error

        # Final classification
        output = self.final_conv(final_features)

        # Upsample to the final target output size
        if output is not None:
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)

        return {'segmentation': output}

class DepthEstimationHead(nn.Module):
    """Depth estimation head using FPN features."""
    def __init__(self, fpn_channels, output_target_size=(480, 720), levels_to_use=['P3', 'P4', 'P5']): # *** MODIFIÉ ***
        super().__init__()
        self.levels_to_use = levels_to_use
        self.target_size = output_target_size

        embed_dim = max(fpn_channels // 2, 128)
        self.adapters = nn.ModuleDict()
        for level in self.levels_to_use:
            self.adapters[level] = nn.Sequential(
                nn.Conv2d(fpn_channels, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
        # Final depth prediction layer (1 channel)
        self.final_conv = nn.Conv2d(embed_dim * len(self.levels_to_use), 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True) # Ensure positive depth
        print(f"DepthHead init: Uses {self.levels_to_use}. Final Conv expects {self.final_conv.in_channels} channels. Target output: {self.target_size}")

    def forward(self, features):
        adapted_features = []
        internal_target_size = None
        processed_levels = []

        for lvl in self.levels_to_use:
            if lvl in features:
                if internal_target_size is None:
                    internal_target_size = features[lvl].shape[-2:]
                adapted = self.adapters[lvl](features[lvl])
                adapted_features.append(adapted)
                processed_levels.append(lvl)
            # else: warnings.warn(f"DepthHead: Expected level {lvl} not found.")

        if not adapted_features:
            print("Warning: DepthHead: No usable FPN features found/adapted.")
            return {'depth': None}

        resized_features = []
        for feat in adapted_features:
            if feat.shape[-2:] != internal_target_size:
                feat = F.interpolate(feat, size=internal_target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)

        final_features = torch.cat(resized_features, dim=1)

        if self.final_conv.in_channels != final_features.shape[1]:
            warnings.warn(f"DepthHead channel mismatch! FinalConv expects {self.final_conv.in_channels}, "
                          f"but got {final_features.shape[1]} (from levels {processed_levels}).")
            return {'depth': None}

        output = self.final_conv(final_features)
        output = self.relu(output) # Apply ReLU for positive depth

        if output is not None:
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)

        return {'depth': output}

# --- Modèle MultiTaskNet ---
class MultiTaskNet(nn.Module):
    """Combines Backbone, Neck, and Task Heads."""
    def __init__(self, backbone_config, backbone_pretrained, backbone_ckpt_path,
                 neck_config, segmentation_config, depth_config):
        super().__init__()
        print("Initializing MultiTaskNet...")
        # 1. Backbone
        self.backbone = create_regnet_backbone(backbone_config, backbone_pretrained, backbone_ckpt_path)
        available_bb_features = {info['module']: info['num_chs'] for info in self.backbone.feature_info}

        # 2. Neck
        # Determine which backbone levels the neck can actually use
        backbone_channels_for_neck = {}
        levels_actually_used_by_neck = []
        for lvl in neck_config['levels_to_use']:
            if lvl in available_bb_features:
                backbone_channels_for_neck[lvl] = available_bb_features[lvl]
                levels_actually_used_by_neck.append(lvl)
            else:
                warnings.warn(f"MultiTaskNet Init: Backbone level '{lvl}' requested by Neck not found in {list(available_bb_features.keys())}. Skipping for Neck.")
        if not levels_actually_used_by_neck:
            raise ValueError("MultiTaskNet Init: No usable backbone levels found for the Neck configuration!")
        # Update neck config to use only available levels
        neck_config_updated = {**neck_config, 'levels_to_use': levels_actually_used_by_neck}
        self.neck = BiFPNNeck(backbone_channels=backbone_channels_for_neck, **neck_config_updated)
        fpn_channels = neck_config_updated['fpn_channels']
        produced_fpn_levels = self.neck.out_lvl_names # Levels actually produced by the neck

        # 3. Heads
        # Configure heads to use levels produced by the neck
        def filter_head_levels(requested_levels, available_levels):
            """Selects levels for the head based on availability."""
            used = [lvl for lvl in requested_levels if lvl in available_levels]
            if not used and available_levels:
                # Fallback: use the highest-resolution level produced by the neck
                used = [available_levels[0]]
                warnings.warn(f"Head Config: No requested levels {requested_levels} found in FPN levels {available_levels}. Using fallback: {used}")
            elif not used:
                # This should not happen if available_levels is not empty
                raise ValueError(f"Head Config: No FPN levels available ({available_levels}) to satisfy request for {requested_levels}")
            return used

        seg_levels = filter_head_levels(segmentation_config.get('levels_to_use', produced_fpn_levels), produced_fpn_levels)
        depth_levels = filter_head_levels(depth_config.get('levels_to_use', produced_fpn_levels), produced_fpn_levels)
        segmentation_cfg_updated = {**segmentation_config, 'levels_to_use': seg_levels}
        depth_cfg_updated = {**depth_config, 'levels_to_use': depth_levels}

        self.segmentation_head = SemanticSegmentationHead(fpn_channels=fpn_channels, **segmentation_cfg_updated)
        self.depth_head = DepthEstimationHead(fpn_channels=fpn_channels, **depth_cfg_updated)
        print("MultiTaskNet Initialized Successfully.")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor, assumed to be at network input resolution (e.g., 480x720).
        Returns:
            dict[str, torch.Tensor]: Dictionary containing 'segmentation' and 'depth' outputs.
        """
        backbone_features = self.backbone(x)
        fpn_features = self.neck(backbone_features)
        seg_out = self.segmentation_head(fpn_features)
        depth_out = self.depth_head(fpn_features)

        # Combine outputs, ensuring keys exist even if value is None
        outputs = {
            'segmentation': seg_out.get('segmentation') if seg_out else None,
            'depth': depth_out.get('depth') if depth_out else None
        }
        # Filter out None values if desired, but returning keys might be better for consistency
        # outputs = {k: v for k, v in outputs.items() if v is not None}
        return outputs

# --- Fonction de Résumé ---
def print_model_summary(model):
    """Prints a summary of the model parameters."""
    print("-" * 80)
    print(f"Summary for {model.__class__.__name__}:")
    print("-" * 80)
    total_params = 0
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad) if module else 0

    parts = {
        'Backbone': getattr(model, 'backbone', None),
        'Neck': getattr(model, 'neck', None),
        'Segmentation Head': getattr(model, 'segmentation_head', None),
        'Depth Head': getattr(model, 'depth_head', None)
    }
    for name, part_module in parts.items():
        params = count_params(part_module)
        print(f"{name}: {params:,} parameters")
        total_params += params

    print("-" * 40)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 80)


# --- Bloc d'exécution (Test avec résolution 480x720 In/Out) ---
if __name__ == '__main__':
    print("Testing backbone_module.py with Input=(480, 720), Head Output=(480, 720)...")
    NETWORK_INPUT_RES = (480, 720) # *** RESOLUTION ENTREE RESEAU ***
    HEAD_OUTPUT_RES = (480, 720)   # *** RESOLUTION SORTIE TETES ***

    backbone_cfg = regnet_1600M_config
    backbone_ckpt = None # Mettre chemin vers poids ImageNet si préparé

    neck_cfg = dict(fpn_channels=128, num_repeats=1, levels_to_use=['s2', 's3', 's4'], activation=nn.ReLU)
    NUM_CLASSES_SEGMENTATION = 8

    # Déduire les niveaux FPN produits pour configurer les têtes
    stage_map={'s1':2,'s2':3,'s3':4,'s4':5}
    produced_fpn_levels=sorted([f"P{stage_map.get(lvl,int(lvl[1:])+1)}" for lvl in neck_cfg['levels_to_use']],key=lambda x:int(x[1:]))

    segmentation_cfg = dict(num_classes=NUM_CLASSES_SEGMENTATION, output_target_size=HEAD_OUTPUT_RES, levels_to_use=produced_fpn_levels)
    depth_cfg = dict(output_target_size=HEAD_OUTPUT_RES, levels_to_use=produced_fpn_levels)

    print(f"\nTest Instantiation: Input={NETWORK_INPUT_RES}, Head Output={HEAD_OUTPUT_RES}")
    try:
        model = MultiTaskNet(backbone_cfg, False, backbone_ckpt, neck_cfg, segmentation_cfg, depth_cfg)
        print_model_summary(model)

        print("\nTesting Forward Pass...")
        dummy_input = torch.randn(1, 3, NETWORK_INPUT_RES[0], NETWORK_INPUT_RES[1]) # Batch size 1
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)

        print("\nForward Pass Output Shapes:")
        ok = True
        expected_tasks = {'segmentation', 'depth'}
        found_tasks = set(outputs.keys())

        # Vérifier si toutes les tâches attendues sont présentes
        if not expected_tasks.issubset(found_tasks):
             print(f"ERROR: Tasks mismatch! Expected {expected_tasks}, got {found_tasks}")
             ok = False

        # Vérifier les shapes des tenseurs retournés
        for task, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {task}: {tensor.shape}")
                if task in ['segmentation', 'depth']:
                     if tensor.shape[-2:] != HEAD_OUTPUT_RES:
                         print(f"  ERROR: {task} output shape {tensor.shape[-2:]} != expected {HEAD_OUTPUT_RES}")
                         ok = False
            elif tensor is None:
                 print(f"  {task}: None") # OK si une tête n'a rien retourné (e.g., channel mismatch warning)
                 # Consider if None is acceptable or should be an error
                 # ok = False # Décommenter si None n'est pas une sortie valide
            else:
                 print(f"  {task}: Unexpected type {type(tensor)}")
                 ok = False

        if ok:
            print("\nTest PASSED!")
        else:
            print("\nTest FAILED!")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()