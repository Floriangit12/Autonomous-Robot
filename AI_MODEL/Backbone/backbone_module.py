# backbone_module_quant_friendly.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
import warnings
import os
import copy # Pour deepcopy si on ne veut pas modifier le modèle original lors de la fusion

# --- Import optionnel pour poids torchvision ---
try:
    from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
    TORCHVISION_REGNET_AVAILABLE = True
    print("Torchvision RegNetY-1.6GF et ses poids sont disponibles.")
except ImportError:
    TORCHVISION_REGNET_AVAILABLE = False
    print("Avertissement: torchvision.models.regnet_y_1_6gf non trouvé. "
          "Le chargement de poids pré-entraînés via torchvision est désactivé.")
# --- Fin Import ---

# =============================================================================
# SECTION 1 : Utilitaires et Blocs de Construction RegNet (adaptés pour fusion)
# =============================================================================

def quantize_float(value: float, quantum: int) -> int:
    if quantum == 0: return int(value)
    return int(round(value / quantum) * quantum)

def adjust_widths_groups_compatibility(widths: list[int], bottleneck_multipliers: list[float], group_widths: list[int]) -> tuple[list[int], list[int]]:
    widths_bottleneck = [int(w * bm) for w, bm in zip(widths, bottleneck_multipliers)]
    min_len = min(len(group_widths), len(widths_bottleneck))
    _group_widths = [min(group_widths[i], widths_bottleneck[i]) if widths_bottleneck[i] > 0 else group_widths[i] for i in range(min_len)] # Assurer que widths_bottleneck[i] > 0 pour min
    _widths_bottleneck = widths_bottleneck[:min_len]
    _widths_bottleneck = [quantize_float(wb, gw) if gw > 0 else wb for wb, gw in zip(_widths_bottleneck, _group_widths)]
    _bms = bottleneck_multipliers[:len(_widths_bottleneck)]
    if len(_bms) < len(_widths_bottleneck) and _bms:
        _bms.extend([_bms[-1]] * (len(_widths_bottleneck) - len(_bms)))
    elif not _bms and _widths_bottleneck:
        raise ValueError("bottleneck_multipliers ne peut être vide si widths_bottleneck ne l'est pas.")
    final_widths = [int(wb / bm) if bm != 0 else 0 for wb, bm in zip(_widths_bottleneck, _bms)]
    return final_widths, _group_widths

def get_stages_from_block_specs(block_widths: list[int], block_resolutions_ids: list[int]) -> tuple[list[int], list[int]]:
    widths_padded = block_widths + [0]; res_ids_padded = block_resolutions_ids + [0]
    prev_widths_padded = [0] + block_widths; prev_res_ids_padded = [0] + block_resolutions_ids
    stage_transitions = [w != pw or r_id != pr_id for w, pw, r_id, pr_id in zip(widths_padded, prev_widths_padded, res_ids_padded, prev_res_ids_padded)]
    stage_widths = [w for w, transition in zip(block_widths, stage_transitions[:-1]) if transition]
    transition_indices = [i for i, transition in enumerate(stage_transitions) if transition]
    stage_depths = []
    if len(transition_indices) >= 2: stage_depths = np.diff(transition_indices).tolist()
    return stage_widths, stage_depths

def generate_regnet_block_widths(width_slope: float, width_initial: int, width_mult: float, depth: int, q: int = 8) -> tuple[list[int], int, int, list[float]]:
    if not (width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0):
        raise ValueError(f"Paramètres RegNet invalides: wa={width_slope}, w0={width_initial}, wm={width_mult}, q={q}")
    if depth <= 0: return [], 0, 0, []
    u_values = np.arange(depth) * width_slope + width_initial; epsilon = 1e-8
    log_input = np.maximum(u_values / width_initial, epsilon)
    if width_mult <= 1: raise ValueError("width_mult (w_m) doit être > 1 pour np.log(width_mult)")
    block_strides_exp = np.round(np.log(log_input) / np.log(width_mult))
    quantized_block_widths = width_initial * np.power(width_mult, block_strides_exp)
    quantized_block_widths = np.round(quantized_block_widths / q) * q
    unique_widths = np.unique(quantized_block_widths); num_stages = len(unique_widths)
    max_possible_stages = 0
    if block_strides_exp.size > 0: max_possible_stages = int(block_strides_exp.max().item() + 1)
    final_block_widths = quantized_block_widths.astype(int).tolist(); continuous_block_widths = u_values.tolist()
    return final_block_widths, num_stages, max_possible_stages, continuous_block_widths

class SimpleStemImageNet(nn.Module):
    """Stem Conv-BN-ReLU, optimisé pour la fusion."""
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False) # Changé pour la fusion/quantification
        # Regrouper pour la fusion explicite si nécessaire, mais PyTorch peut les trouver
        # self.block = nn.Sequential(self.conv, self.bn, self.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.block(x)
        return self.relu(self.bn(self.conv(x))) # L'ordre est important pour fuse_modules

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, se_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_excitation = nn.Sequential(
            nn.Conv2d(input_channels, se_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=False), # Changé
            nn.Conv2d(se_channels, input_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc_excitation(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 bottleneck_multiplier: float, group_width: int, se_ratio: float or None):
        super().__init__()
        width_bottleneck = int(round(out_channels * bottleneck_multiplier)); _group_width = max(1, group_width)
        num_groups = max(1, width_bottleneck // _group_width) if width_bottleneck > 0 else 1
        width_bottleneck = max(num_groups, int(round(width_bottleneck / num_groups) * num_groups)) if num_groups > 0 else 1
        width_bottleneck = max(width_bottleneck, 1)

        # Bloc 1: Conv-BN-ReLU
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, width_bottleneck, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(width_bottleneck),
            nn.ReLU(inplace=False) # Changé
        )
        # Bloc 2: ConvGroup-BN-ReLU
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(width_bottleneck, width_bottleneck, kernel_size=3, stride=stride, padding=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(width_bottleneck),
            nn.ReLU(inplace=False) # Changé
        )
        if se_ratio and se_ratio > 0:
            se_channels = max(1, int(round(in_channels * se_ratio)))
            self.se_block = SqueezeExcitation(width_bottleneck, se_channels)
        else:
            self.se_block = nn.Identity()
        # Bloc 3: Conv-BN (ReLU est après la connexion résiduelle)
        self.conv3 = nn.Conv2d(width_bottleneck, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn3.final_bn = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.se_block(x)
        x = self.bn3(self.conv3(x))
        return x

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 bottleneck_multiplier: float = 1.0, group_width: int = 1, se_ratio: float or None = None):
        super().__init__()
        self.needs_projection = (in_channels != out_channels) or (stride != 1)
        if self.needs_projection:
            # Bloc de projection: Conv-BN
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.transform = BottleneckTransform(in_channels, out_channels, stride, bottleneck_multiplier, group_width, se_ratio)
        self.relu = nn.ReLU(inplace=False) # Changé

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.projection(x) if self.needs_projection else x
        transformed_x = self.transform(x)
        output = shortcut + transformed_x
        return self.relu(output)

class RegNetStage(nn.Module): # Structure inchangée, dépend des blocs internes
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int,
                 block_constructor: callable, bottleneck_multiplier: float,
                 group_width: int, se_ratio: float or None):
        super().__init__()
        self.blocks = nn.Sequential()
        current_in_channels = in_channels
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            _in_channels = current_in_channels if i == 0 else out_channels
            if not callable(block_constructor): raise TypeError(f"block_constructor doit être callable, reçu {type(block_constructor)}")
            block = block_constructor(_in_channels, out_channels, block_stride, bottleneck_multiplier, group_width, se_ratio)
            self.blocks.add_module(f"block{i + 1}", block)
            current_in_channels = out_channels
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.blocks(x)

class AnyNetFeatureExtractor(nn.Module): # Structure inchangée, dépend des blocs internes
    def __init__(self, stem_channels: int, stage_depths: list[int], stage_widths: list[int],
                 stage_strides: list[int], bottleneck_multipliers: list[float],
                 group_widths: list[int], se_ratio: float or None):
        super().__init__()
        if not (stage_depths and stage_widths and stage_strides): num_stages = 0
        else:
            num_stages = len(stage_depths)
            if not (len(stage_widths) == num_stages and len(stage_strides) == num_stages and \
                    len(bottleneck_multipliers) == num_stages and len(group_widths) == num_stages):
                raise ValueError("Listes de paramètres d'étage doivent avoir la même longueur.")
        self.stem = SimpleStemImageNet(out_channels=stem_channels)
        self.stages = nn.ModuleList()
        current_channels = stem_channels
        for i in range(num_stages):
            stage = RegNetStage(current_channels, stage_widths[i], stage_strides[i], stage_depths[i],
                                ResidualBottleneckBlock, bottleneck_multipliers[i], group_widths[i], se_ratio)
            self.stages.append(stage); current_channels = stage_widths[i]
        self._compute_feature_info(stem_channels, stage_widths, stage_strides); self._initialize_weights()
    def _compute_feature_info(self, stem_channels: int, stage_widths: list[int], stage_strides: list[int]):
        self.feature_info = []; current_stride = 2
        self.feature_info.append({'num_chs': stem_channels, 'reduction': current_stride, 'module': 'stem'})
        for i, (width, stride) in enumerate(zip(stage_widths, stage_strides)):
            current_stride *= stride
            self.feature_info.append({'num_chs': width, 'reduction': current_stride, 'module': f's{i + 1}'})
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                std_dev = math.sqrt(2.0 / fan_out) if fan_out > 0 else 0.01
                nn.init.normal_(m.weight, mean=0.0, std=std_dev)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                is_final_bn = hasattr(m, "final_bn") and m.final_bn
                nn.init.constant_(m.weight, 0.0 if is_final_bn else 1.0); nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = OrderedDict(); x = self.stem(x); features['stem'] = x
        for i, stage in enumerate(self.stages): x = stage(x); features[f's{i + 1}'] = x
        return features

class RegNetBackbone(AnyNetFeatureExtractor): # Structure inchangée, dépend des blocs internes
    def __init__(self, regnet_config: dict, **kwargs):
        block_widths_gen, _, _, _ = generate_regnet_block_widths(regnet_config['WA'], regnet_config['W0'], regnet_config['WM'], regnet_config['DEPTH'], regnet_config.get('Q', 8))
        if not block_widths_gen: raise ValueError("Génération des largeurs de bloc RegNet vide.")
        stage_widths_derived, stage_depths_derived = get_stages_from_block_specs(block_widths_gen, block_widths_gen)
        num_actual_stages = len(stage_widths_derived)
        if num_actual_stages == 0:
            stage_strides = []; bottleneck_multipliers_derived = []; group_widths_derived = []
        else:
            stage_strides = [2] * num_actual_stages
            _group_widths_cfg = [regnet_config.get('GROUP_W', 1)] * num_actual_stages
            _bottleneck_multipliers_cfg = [regnet_config.get('BOT_MUL', 1.0)] * num_actual_stages
            stage_widths_derived, group_widths_derived = adjust_widths_groups_compatibility(stage_widths_derived, _bottleneck_multipliers_cfg, _group_widths_cfg)
            bottleneck_multipliers_derived = _bottleneck_multipliers_cfg
        se_ratio_cfg = regnet_config.get('SE_R', None); stem_channels_cfg = regnet_config.get('STEM_W', 32)
        super().__init__(stem_channels_cfg, stage_depths_derived, stage_widths_derived, stage_strides, bottleneck_multipliers_derived, group_widths_derived, se_ratio_cfg, **kwargs)

def create_regnet_backbone(config: dict, pretrained: bool = False) -> RegNetBackbone: # Logique de chargement inchangée
    model = RegNetBackbone(config); weights_source_message = "Initialisation aléatoire"
    if pretrained:
        if TORCHVISION_REGNET_AVAILABLE:
            try:
                torchvision_weights = RegNet_Y_1_6GF_Weights.IMAGENET1K_V2
                model_tv = regnet_y_1_6gf(weights=torchvision_weights); torchvision_state_dict = model_tv.state_dict()
                keys_to_remove = [k for k in torchvision_state_dict if k.startswith('fc.')]
                if keys_to_remove: [torchvision_state_dict.pop(k, None) for k in keys_to_remove]
                if all(k.startswith('module.') for k in torchvision_state_dict.keys()): torchvision_state_dict = {k[len('module.'):]: v for k, v in torchvision_state_dict.items()}
                missing_keys, unexpected_keys = model.load_state_dict(torchvision_state_dict, strict=False)
                if missing_keys: print(f"Avertissement: Clés manquantes backbone: {missing_keys}")
                unexpected_keys = [k for k in unexpected_keys if not (k.startswith('head.') or k.startswith('fc.'))]
                if unexpected_keys: print(f"Avertissement: Clés inattendues backbone: {unexpected_keys}")
                weights_source_message = "Torchvision RegNetY-1.6GF (ImageNet)"
            except Exception as e: print(f"Échec chargement torchvision: {e}"); weights_source_message = "Init. aléatoire (échec torchvision)"
        else: weights_source_message = "Init. aléatoire (torchvision non disponible)"
    print(f"Backbone RegNet créé. Poids: {weights_source_message}"); return model

# =============================================================================
# SECTION 2 : Couche de Cou (Neck) BiFPN (adapté pour fusion)
# =============================================================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False, activation_fn_type: type = nn.ReLU): # type pour instanciation
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        
        # Regrouper Pointwise-BN-Activation pour la fusion
        _act_fn = activation_fn_type if activation_fn_type is None else activation_fn_type(inplace=False) # Toujours False pour fusion/quant
        if activation_fn_type is None: _act_fn = nn.Identity()

        self.pointwise_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            _act_fn
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_block(x)
        return x

class BiFPNFusionNode(nn.Module): # ReLU inplace=False
    def __init__(self, num_inputs: int, num_channels: int, epsilon: float = 1e-4):
        super().__init__(); self.num_inputs = num_inputs; self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.relu_weights = nn.ReLU(inplace=False) # Changé
    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.num_inputs: raise ValueError(f"Attendu {self.num_inputs} entrées, reçu {len(inputs)}")
        normalized_weights = self.relu_weights(self.weights)
        normalized_weights = normalized_weights / (torch.sum(normalized_weights, dim=0, keepdim=True) + self.epsilon)
        x_stacked = torch.stack(inputs, dim=-1); fused_output = (normalized_weights * x_stacked).sum(dim=-1)
        return fused_output

class BiFPNLayer(nn.Module): # Structure inchangée, dépend des blocs internes
    def __init__(self, num_channels: int, num_levels: int, epsilon: float = 1e-4, activation_fn_type: type = nn.ReLU):
        super().__init__(); self.num_channels=num_channels; self.num_levels=num_levels; self.epsilon=epsilon
        self.post_fusion_convs = nn.ModuleList([DepthwiseSeparableConv(num_channels,num_channels,activation_fn_type=activation_fn_type) for _ in range(num_levels)])
        self.top_down_fusion_nodes = nn.ModuleList([BiFPNFusionNode(2,num_channels,epsilon) for _ in range(num_levels-1)])
        self.bottom_up_fusion_nodes = nn.ModuleList([BiFPNFusionNode((2 if i==0 or i==num_levels-1 else 3),num_channels,epsilon) for i in range(num_levels)])
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest'); self.downsample = nn.MaxPool2d(3,2,1)
    def _resize_if_needed(self,s,t): return F.interpolate(s,size=t.shape[-2:],mode='bilinear',align_corners=False) if s.shape[-2:]!=t.shape[-2:] else s
    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(inputs)!=self.num_levels: raise ValueError(f"Attendu {self.num_levels} niveaux, reçu {len(inputs)}")
        td_features=[None]*self.num_levels; td_features[-1]=inputs[-1]
        for i in range(self.num_levels-2,-1,-1):
            tgt_in=inputs[i]; feat_above=td_features[i+1]; up_feat=self.upsample(feat_above); up_feat=self._resize_if_needed(up_feat,tgt_in)
            fused=self.top_down_fusion_nodes[i]([tgt_in,up_feat]); td_features[i]=self.post_fusion_convs[i](fused)
        out_features=[None]*self.num_levels; out_features[0]=td_features[0]
        for i in range(1,self.num_levels):
            tgt_td=td_features[i]; feat_below=out_features[i-1]; down_feat=self.downsample(feat_below); orig_in=inputs[i]
            down_feat=self._resize_if_needed(down_feat,orig_in); tgt_td_r=self._resize_if_needed(tgt_td,orig_in)
            if i==self.num_levels-1: fused_input=[orig_in,down_feat]
            else: fused_input=[orig_in,tgt_td_r,down_feat]
            fused=self.bottom_up_fusion_nodes[i](fused_input); out_features[i]=self.post_fusion_convs[i](fused)
        return out_features

class BiFPNNeck(nn.Module): # Structure inchangée, dépend des blocs internes
    def __init__(self, backbone_feature_channels: dict[str,int], fpn_target_channels: int, num_bifpn_repeats: int,
                 backbone_levels_to_use: list[str], activation_fn_type: type = nn.ReLU, epsilon: float=1e-4):
        super().__init__(); self.backbone_levels_to_use=backbone_levels_to_use; self.num_fpn_levels_in=len(backbone_levels_to_use)
        self.fpn_target_channels=fpn_target_channels
        missing=[lvl for lvl in backbone_levels_to_use if lvl not in backbone_feature_channels]
        if missing: raise ValueError(f"Niveaux BB manquants: {missing}. Dispo: {list(backbone_feature_channels.keys())}")
        self.lateral_convs=nn.ModuleList([nn.Conv2d(backbone_feature_channels[lvl],fpn_target_channels,1) for lvl in backbone_levels_to_use])
        self.bifpn_layers=nn.ModuleList([BiFPNLayer(fpn_target_channels,self.num_fpn_levels_in,epsilon,activation_fn_type=activation_fn_type) for _ in range(num_bifpn_repeats)])
        self.output_level_names=self._generate_output_level_names(backbone_levels_to_use)
        print(f"BiFPNNeck P-levels: {self.output_level_names}")
    def _generate_output_level_names(self,bb_lvls): # simplified
        return [f"P{i+3}" for i in range(len(bb_lvls))] # Ex: s2,s3,s4 -> P3,P4,P5
    def forward(self, bb_features: OrderedDict[str,torch.Tensor]) -> OrderedDict[str,torch.Tensor]:
        lat_f=[]; proc_lvls=[]
        for i,lvl in enumerate(self.backbone_levels_to_use):
            if lvl in bb_features: lat_f.append(self.lateral_convs[i](bb_features[lvl])); proc_lvls.append(lvl)
            else: warnings.warn(f"Niveau BB '{lvl}' non trouvé.")
        if not lat_f: return OrderedDict()
        if len(lat_f)!=self.num_fpn_levels_in: warnings.warn(f"BiFPNNeck attendait {self.num_fpn_levels_in} niveaux, reçu {len(lat_f)}.")
        fpn_outs=lat_f
        for layer in self.bifpn_layers: fpn_outs=layer(fpn_outs)
        return OrderedDict(zip(self.output_level_names[:len(fpn_outs)],fpn_outs))

# =============================================================================
# SECTION 3 : Têtes de Prédiction (adaptées pour fusion)
# =============================================================================
class BasePredictionHead(nn.Module):
    def __init__(self, fpn_input_channels: int, output_target_size: tuple[int,int], fpn_levels_to_use: list[str],
                 embedding_dim_ratio: float=0.5, min_embedding_dim: int=128):
        super().__init__()
        if not fpn_levels_to_use: raise ValueError("fpn_levels_to_use ne peut être vide.")
        self.fpn_levels_to_use=fpn_levels_to_use; self.output_target_size=output_target_size
        self.embedding_dim=max(min_embedding_dim,int(fpn_input_channels*embedding_dim_ratio))
        self.adapters=nn.ModuleDict()
        for level_name in self.fpn_levels_to_use:
            # Séquence Conv-BN-ReLU pour chaque adaptateur -> fusionnable
            self.adapters[level_name]=nn.Sequential(
                nn.Conv2d(fpn_input_channels,self.embedding_dim,3,1,1,bias=False),
                nn.BatchNorm2d(self.embedding_dim),
                nn.ReLU(inplace=False) # Changé
            )
        self.final_conv_in_channels = self.embedding_dim * len(self.fpn_levels_to_use) # Attendu
        print(f"{self.__class__.__name__} init: FPN levels {self.fpn_levels_to_use}. Adapters out_ch: {self.embedding_dim}. FinalConv in_ch (attendu): {self.final_conv_in_channels}. Target out: {self.output_target_size}")
    def _process_features(self, fpn_f: OrderedDict[str,torch.Tensor]) -> tuple[torch.Tensor or None, str or None]:
        adapted_f=[]; int_size=None; proc_lvls=[]
        for lvl_n in self.fpn_levels_to_use:
            if lvl_n in fpn_f:
                feat=fpn_f[lvl_n]
                if int_size is None: int_size=feat.shape[-2:]
                adapted_f.append(self.adapters[lvl_n](feat)); proc_lvls.append(lvl_n)
            else: warnings.warn(f"{self.__class__.__name__}: FPN level '{lvl_n}' non trouvé.")
        if not adapted_f: warnings.warn(f"{self.__class__.__name__}: Aucune caractéristique FPN adaptée."); return None,None
        resized_f=[F.interpolate(f,size=int_size,mode='bilinear',align_corners=False) if f.shape[-2:]!=int_size else f for f in adapted_f]
        concat_f=torch.cat(resized_f,dim=1)
        if self.final_conv_in_channels != concat_f.shape[1]: # Comparer avec le nombre de canaux réel des adaptateurs utilisés
            actual_final_conv_in_channels = self.embedding_dim * len(proc_lvls)
            if actual_final_conv_in_channels == 0: # Aucun niveau traité
                 warnings.warn(f"{self.__class__.__name__}: Aucun niveau FPN n'a été traité, impossible de continuer.")
                 return None, "No FPN levels processed"
            warnings.warn(f"{self.__class__.__name__}: Incohérence canaux final_conv! Attendu (basé sur config init): {self.final_conv_in_channels}, "
                          f"Reçu (basé sur niveaux traités {proc_lvls}): {concat_f.shape[1]}. "
                          f"Cela se produit si certains `fpn_levels_to_use` n'étaient pas dans `fpn_features`. "
                          "La tête pourrait échouer ou donner des résultats incorrects. "
                          "IMPORTANT: La convolution finale sera reconstruite pour correspondre aux canaux réels.")
            # Reconstruire la couche de convolution finale pour correspondre aux canaux réels
            # Cela rend la tête plus flexible mais peut indiquer un problème de configuration en amont.
            if hasattr(self, 'final_conv') and self.final_conv.in_channels != actual_final_conv_in_channels:
                original_out_channels = self.final_conv.out_channels
                original_kernel_size = self.final_conv.kernel_size
                # Garder le même device et dtype que le modèle
                device = self.final_conv.weight.device
                dtype = self.final_conv.weight.dtype
                self.final_conv = nn.Conv2d(actual_final_conv_in_channels, original_out_channels, kernel_size=original_kernel_size).to(device=device, dtype=dtype)
                print(f"    {self.__class__.__name__}: final_conv reconstruit avec in_channels={actual_final_conv_in_channels}, out_channels={original_out_channels}")

        return concat_f,None

class SemanticSegmentationHead(BasePredictionHead):
    def __init__(self, fpn_input_channels: int, num_classes: int, output_target_size: tuple[int,int]=(480,720), fpn_levels_to_use: list[str]=['P3','P4','P5']):
        super().__init__(fpn_input_channels,output_target_size,fpn_levels_to_use)
        self.num_classes=num_classes; self.final_conv=nn.Conv2d(self.final_conv_in_channels,num_classes,1)
        print(f"SemanticSegmentationHead: final_conv out_ch: {num_classes}")
    def forward(self, fpn_f: OrderedDict[str,torch.Tensor]) -> dict[str,torch.Tensor or None]:
        proc_f,err=self._process_features(fpn_f)
        if err or proc_f is None: warnings.warn(f"SegHead: Échec traitement feats ({err})."); return {'segmentation':None}
        logits=self.final_conv(proc_f); out=F.interpolate(logits,size=self.output_target_size,mode='bilinear',align_corners=False)
        return {'segmentation':out}
class DepthEstimationHead(BasePredictionHead):
    def __init__(self, fpn_input_channels: int, output_target_size: tuple[int,int]=(480,720), fpn_levels_to_use: list[str]=['P3','P4','P5']):
        super().__init__(fpn_input_channels,output_target_size,fpn_levels_to_use)
        self.final_conv=nn.Conv2d(self.final_conv_in_channels,1,1)
        self.relu_out=nn.ReLU(inplace=False) # Changé
        print(f"DepthEstimationHead: final_conv out_ch: 1")
    def forward(self, fpn_f: OrderedDict[str,torch.Tensor]) -> dict[str,torch.Tensor or None]:
        proc_f,err=self._process_features(fpn_f)
        if err or proc_f is None: warnings.warn(f"DepthHead: Échec traitement feats ({err})."); return {'depth':None}
        raw_depth=self.final_conv(proc_f); act_depth=self.relu_out(raw_depth)
        out=F.interpolate(act_depth,size=self.output_target_size,mode='bilinear',align_corners=False)
        return {'depth':out}

# =============================================================================
# SECTION 4 : Modèle Multi-Tâches (HydraNet)
# =============================================================================
class MultiTaskHydraNet(nn.Module): # Structure inchangée, dépend des blocs internes
    def __init__(self, backbone_name: str, backbone_config: dict, backbone_pretrained: bool,
                 neck_fpn_channels: int, neck_num_repeats: int, neck_backbone_levels_to_use: list[str],
                 segmentation_num_classes: int, head_fpn_levels_to_use: list[str],
                 head_output_resolution: tuple[int,int]=(480,720)):
        super().__init__(); print("Initialisation MultiTaskHydraNet...")
        if backbone_name.lower()=='regnet': self.backbone=create_regnet_backbone(backbone_config,pretrained=backbone_pretrained)
        else: raise ValueError(f"Backbone non supporté: {backbone_name}.")
        bb_out_info={info['module']:info['num_chs'] for info in self.backbone.feature_info}; print(f"Backbone '{backbone_name}' init. Outs: {bb_out_info}")
        actual_neck_in_lvls=[]; neck_in_ch_dict={}
        for lvl in neck_backbone_levels_to_use:
            if lvl in bb_out_info: actual_neck_in_lvls.append(lvl); neck_in_ch_dict[lvl]=bb_out_info[lvl]
            else: warnings.warn(f"Niveau BB '{lvl}' pour Neck non dispo. Ignoré.")
        if not actual_neck_in_lvls: raise ValueError("Aucun niveau BB utilisable pour Neck!")
        self.neck=BiFPNNeck(neck_in_ch_dict,neck_fpn_channels,neck_num_repeats,actual_neck_in_lvls,activation_fn_type=nn.ReLU)
        fpn_prod_lvls=self.neck.output_level_names; print(f"Neck BiFPN init. FPN levels produits: {fpn_prod_lvls}")
        actual_head_in_lvls=[lvl for lvl in head_fpn_levels_to_use if lvl in fpn_prod_lvls]
        if not actual_head_in_lvls:
            warnings.warn(f"Aucun FPN levels ({head_fpn_levels_to_use}) pour têtes par Neck ({fpn_prod_lvls}). "
                          f"Fallback: {fpn_prod_lvls[0] if fpn_prod_lvls else 'aucun'}")
            actual_head_in_lvls=[fpn_prod_lvls[0]] if fpn_prod_lvls else []
        if not actual_head_in_lvls: raise ValueError("Neck n'a produit aucun niveau utilisable pour têtes.")
        self.segmentation_head=SemanticSegmentationHead(neck_fpn_channels,segmentation_num_classes,head_output_resolution,actual_head_in_lvls)
        self.depth_head=DepthEstimationHead(neck_fpn_channels,head_output_resolution,actual_head_in_lvls)
        print("MultiTaskHydraNet initialisé.")
    def forward(self,x:torch.Tensor) -> dict[str,torch.Tensor or None]:
        bb_f=self.backbone(x); fpn_f=self.neck(bb_f)
        seg_o=self.segmentation_head(fpn_f); depth_o=self.depth_head(fpn_f)
        return {'segmentation':seg_o.get('segmentation'),'depth':depth_o.get('depth')}

# =============================================================================
# SECTION 5 : Utilitaires et Exécution de Test (avec fusion)
# =============================================================================
def print_model_summary(model: nn.Module, model_name: str = "Modèle"): # Inchangé
    print("-" * 80); print(f"Résumé: {model_name} ({model.__class__.__name__})"); print("-" * 80); total_params=0
    def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad) if m else 0
    components={'Backbone':getattr(model,'backbone',None),'Neck':getattr(model,'neck',None),
                'SegHead':getattr(model,'segmentation_head',None),'DepthHead':getattr(model,'depth_head',None)}
    for n,c_mod in components.items():
        if c_mod: params=count_params(c_mod); print(f"  {n:<20}: {params:>12,d} params"); total_params+=params
        else: print(f"  {n:<20}: Non présent")
    print("-" * 40); print(f"  Total Params Entraînables: {total_params:>12,d}"); print("=" * 80)

REGNET_Y_1_6GF_APPROX_CONFIG = {'WA':36.44,'W0':80,'WM':2.24,'DEPTH':18,'GROUP_W':24,'BOT_MUL':1.0,'STEM_W':32,'SE_R':0.25,'Q':8}

if __name__ == '__main__':
    print("Test du module HydraNet (backbone_module_quant_friendly.py)...")
    INPUT_RESOLUTION=(480,720); OUTPUT_RESOLUTION=(480,720)
    BACKBONE_CHOICE='regnet'; BACKBONE_CFG=REGNET_Y_1_6GF_APPROX_CONFIG; USE_PRETRAINED_BACKBONE=True
    NECK_BACKBONE_LEVELS=['s2','s3','s4']; FPN_CHANNELS=128; NUM_BIFPN_REPEATS=2
    HEAD_FPN_LEVELS=['P3','P4','P5']; NUM_SEG_CLASSES=19
    
    print_config = False # Mettre à True pour voir la config détaillée
    if print_config:
        print(f"\n--- Configuration ---"); print(f"Input Res: {INPUT_RESOLUTION}, Output Res: {OUTPUT_RESOLUTION}")
        print(f"Backbone: {BACKBONE_CHOICE}, Pretrained: {USE_PRETRAINED_BACKBONE}, Config: {BACKBONE_CFG}")
        print(f"Neck: BiFPN, FPN Chan: {FPN_CHANNELS}, Repeats: {NUM_BIFPN_REPEATS}, BB Levels: {NECK_BACKBONE_LEVELS}")
        print(f"Heads: FPN Levels: {HEAD_FPN_LEVELS}, Seg Classes: {NUM_SEG_CLASSES}"); print("--- End Config ---\n")

    try:
        hydra_model = MultiTaskHydraNet(
            BACKBONE_CHOICE,BACKBONE_CFG,USE_PRETRAINED_BACKBONE,FPN_CHANNELS,NUM_BIFPN_REPEATS,
            NECK_BACKBONE_LEVELS,NUM_SEG_CLASSES,HEAD_FPN_LEVELS,OUTPUT_RESOLUTION
        )
        print_model_summary(hydra_model, "HydraNet Original")

        # --- Application de la Fusion ---
        # Pour la quantification Post-Training (PTQ) ou la préparation pour QAT,
        # il est courant de fusionner les modules.
        # La fusion est généralement effectuée sur un modèle en mode `eval()`.
        # Pour QAT, la fusion est souvent faite après `prepare_qat`.
        
        print("\nTentative de fusion des modules pour la préparation à la quantification...")
        # Créer une copie si vous voulez garder l'original non fusionné
        # model_to_fuse = copy.deepcopy(hydra_model) 
        model_to_fuse = hydra_model # Modifier le modèle en place
        model_to_fuse.eval() # Important: mettre en mode eval pour la fusion PTQ

        # torch.quantization.fuse_modules est récursif. 
        # Il va chercher les motifs Conv-BN, Conv-BN-ReLU, etc. dans tout le modèle.
        # Les modifications apportées (utilisation de nn.Sequential, ReLU(inplace=False))
        # aident cette fonction à identifier correctement les modules à fusionner.
        # Pour QAT, on utiliserait `torch.quantization.fuse_modules_qat` après `prepare_qat`.
        
        # Lister explicitement les modules à fusionner peut être très complexe pour un grand modèle.
        # Il est préférable de structurer les sous-modules pour qu'ils soient auto-détectables.
        # Ex. SimpleStemImageNet : on pourrait spécifier [['conv', 'bn', 'relu']] pour son instance.
        # Mais `fuse_modules` sur le modèle entier devrait les trouver si bien structurés.

        torch.quantization.fuse_modules(model_to_fuse, inplace=True)
        print("Fusion des modules terminée (si des motifs fusables ont été trouvés).")
        print_model_summary(model_to_fuse, "HydraNet Après Fusion Potentielle")
        # Vous devriez voir une réduction du nombre de paramètres si les BN ont été fusionnées (leurs paramètres sont absorbés).

        print("\nTest de la passe avant du modèle fusionné...")
        dummy_input_tensor = torch.randn(1,3,INPUT_RESOLUTION[0],INPUT_RESOLUTION[1])
        model_to_fuse.eval() # S'assurer qu'il est en eval pour l'inférence
        with torch.no_grad(): outputs = model_to_fuse(dummy_input_tensor)
        
        print("Passe avant (modèle fusionné) terminée.")
        print("Formes des sorties (modèle fusionné):")
        if outputs.get('segmentation') is not None: print(f"  Segmentation: {outputs['segmentation'].shape}")
        else: print("  Segmentation: None")
        if outputs.get('depth') is not None: print(f"  Profondeur: {outputs['depth'].shape}")
        else: print("  Profondeur: None")

    except Exception as e: print("\n--- ÉCHEC SCRIPT ---"); import traceback; traceback.print_exc()