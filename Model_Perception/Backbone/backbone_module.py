# backbone_module.py (Corrected for UnboundLocalError & Reformatted)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
import warnings
import os

# --- Import optionnel pour poids torchvision ---
try:
    from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
    TORCHVISION_REGNET_AVAILABLE = True
    print("Torchvision RegNetY-1.6GF found.")
except ImportError:
    TORCHVISION_REGNET_AVAILABLE = False
    print("Warning: torchvision.models.regnet_y_1_6gf not found. Pretrained weights via torchvision disabled.")
# --- Fin Import ---


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
    # Add sentinel values to handle boundaries
    ws_padded = ws_in + [0]
    rs_padded = rs + [0]
    ws_prev_padded = [0] + ws_in
    rs_prev_padded = [0] + rs

    ts_temp = zip(ws_padded, ws_prev_padded, rs_padded, rs_prev_padded)
    # Determine stage transitions (change in width or resolution ID)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]

    # Extract stage widths (width at the start of the stage)
    s_ws = [w for w, t in zip(ws_in, ts[:-1]) if t] # Use ws_in here

    # Calculate stage depths based on transition indices
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
        # Ensure log_input[valid_indices] is always > 0 before taking log
        log_vals = np.log(log_input[valid_indices])
        # Ensure w_m > 1 before taking log
        if w_m <= 1: raise ValueError("w_m must be > 1")
        ks[valid_indices] = np.round(log_vals / np.log(w_m))

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
        gw = max(1, gw or 1) # Ensure group width is positive
        num_gs = max(1, w_b // gw) if w_b > 0 else 1 # Ensure num_gs is positive
        # Adjust w_b to be divisible by num_gs and non-zero
        w_b = max(num_gs, int(round(w_b / num_gs) * num_gs)) if num_gs > 0 else 1
        w_b = max(w_b, 1) # Ensure w_b is at least 1

        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r and se_r > 0:
            w_se = max(1, int(round(w_in * se_r))) # Ensure w_se >= 1
            self.se = SE(w_b, w_se)
        else:
            self.se = nn.Identity()
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out)
        self.c_bn.final_bn = True # Marker for potential special BN initialization

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
            self.proj, self.bn = None, None # Explicitly set to None
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
            if not callable(block_fun): raise TypeError(f"block_fun must be callable")
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
        required_args = ["stem_w", "ds", "ws", "ss"]
        for arg in required_args:
            if arg not in kwargs or not kwargs.get(arg):
                 if arg == "ds" and not kwargs.get("ws"): continue
                 raise ValueError(f"Missing/empty required arg '{arg}' for AnyNetFeatureExtractor")

        self.stem_w=kwargs["stem_w"]; self.ds=kwargs["ds"]; self.ws=kwargs["ws"]; self.ss=kwargs["ss"]
        self.bms=kwargs.get("bms",[1.0]*len(self.ds)); self.gws=kwargs.get("gws",[1]*len(self.ds)); self.se_r=kwargs.get("se_r")
        if self.ds:
            exp_len=len(self.ds)
            for n,lst in [("ws",self.ws),("ss",self.ss),("bms",self.bms),("gws",self.gws)]:
                 if len(lst)!=exp_len: raise ValueError(f"Len mismatch: {n}({len(lst)})!=ds({exp_len})")
        self.stage_params=list(zip(self.ds,self.ws,self.ss,self.bms,self.gws))
        self.stem=SimpleStemIN(3,self.stem_w); block_fun=ResBottleneckBlock; prev_w=self.stem_w; self.stages=nn.ModuleList()
        for i,(d,w,s,bm,gw) in enumerate(self.stage_params): gw=max(1,gw or 1); stage=AnyStage(prev_w,w,s,d,block_fun,bm,gw,self.se_r); self.stages.append(stage); prev_w=w
        self._compute_feature_info(); self._initialize_weights() # Init aléatoire d'abord

    def _compute_feature_info(self):
        self.feature_info=[{'num_chs':self.stem_w,'reduction':2,'module':'stem'}]; current_stride=2; prev_w=self.stem_w
        for i,(d,w,s,bm,gw) in enumerate(self.stage_params): current_stride*=s; self.feature_info.append({'num_chs':w,'reduction':current_stride,'module':f's{i+1}'}); prev_w=w

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): fan_out=(m.kernel_size[0]*m.kernel_size[1]*m.out_channels); std=math.sqrt(2./fan_out) if fan_out>0 else 0; nn.init.normal_(m.weight,0,std); nn.init.constant_(m.bias,0.) if m.bias is not None else None
            elif isinstance(m,nn.BatchNorm2d): zero=hasattr(m,"final_bn") and m.final_bn; nn.init.constant_(m.weight,0. if zero else 1.); nn.init.constant_(m.bias,0.)
            elif isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.01); nn.init.constant_(m.bias,0.) if m.bias is not None else None

    def forward(self, x):
        features=OrderedDict(); x=self.stem(x); features['stem']=x
        for i,stage in enumerate(self.stages): x=stage(x); features[f's{i+1}']=x
        return features

# --- Classe RegNet qui utilise AnyNetFeatureExtractor ---
class RegNetBackbone(AnyNetFeatureExtractor):
    """RegNet backbone model."""
    def __init__(self, cfg, **kwargs):
        b_ws,num_s,_,_=generate_regnet(cfg['WA'],cfg['W0'],cfg['WM'],cfg['DEPTH'],cfg.get('Q',8))
        if not b_ws: raise ValueError("Generated block widths (b_ws) are empty.")

        # *** CORRECTION: Assign ws, ds BEFORE checking if ws is empty ***
        ws, ds = get_stages_from_blocks(b_ws, b_ws)

        if not ws: # Now check if ws is empty AFTER assignment
            num_s = 0
            warnings.warn("No stages generated from RegNet parameters (ws is empty).")
            # Initialize lists as empty if no stages
            gws = []
            bms = []
        else:
            # If stages exist, proceed as before
            num_s = len(ws) # Use length of ws to determine num_s reliably
            gws = [cfg.get('GROUP_W', 1) for _ in range(num_s)]
            bms = [cfg.get('BOT_MUL', 1) for _ in range(num_s)]
            if not gws: # Should not happen if num_s > 0, but safety check
                 warnings.warn("gws is empty despite ws not being empty.")
            else:
                 ws, gws = adjust_ws_gs_comp(ws, bms, gws)

        # Ensure lists match the actual number of stages derived from ws
        ds = ds[:num_s]
        ws = ws[:num_s]
        bms = bms[:num_s]
        gws = gws[:num_s]

        ss = [2 for _ in range(num_s)]
        se_r = cfg.get('SE_R', None)
        model_kwargs = {"stem_w": cfg.get('STEM_W', 32), "ss": ss, "ds": ds, "ws": ws, "bms": bms, "gws": gws, "se_r": se_r, **kwargs}

        # Check required arguments (allow empty ds if num_s is 0)
        required = ["stem_w", "ds", "ws", "ss"]
        missing = [k for k in required if k not in model_kwargs or not model_kwargs.get(k)]
        if any(k in missing for k in ["stem_w", "ws", "ss"]) or (num_s > 0 and "ds" in missing):
            raise ValueError(f"Missing required args for AnyNetFeatureExtractor: {missing}. Check config/generation.")

        super().__init__(**model_kwargs)


# --- Configs RegNet ---
regnet_1600M_config = {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'BOT_MUL': 1, 'SE_R': 0.25}

# --- Fonction de création du backbone (MODIFIÉE pour charger depuis Torchvision par défaut) ---
def create_regnet_backbone(config, pretrained=False, **kwargs):
    """Creates the RegNet backbone. If pretrained=True, attempts to load ImageNet weights via Torchvision."""
    # 1. Instantiate custom architecture (random weights)
    model = RegNetBackbone(config, **kwargs)
    loaded_weights_source = "Random Initialization"

    if pretrained:
        # 2. Try loading from Torchvision
        if TORCHVISION_REGNET_AVAILABLE:
            print("Attempting to load RegNetY-1.6GF weights from torchvision...")
            try:
                weights = RegNet_Y_1_6GF_Weights.DEFAULT
                model_tv = regnet_y_1_6gf(weights=weights)
                loaded_state_dict = model_tv.state_dict()
                print("Torchvision weights downloaded/loaded.")

                # Clean state dict (remove head, prefix)
                keys_to_remove = [k for k in loaded_state_dict if k.startswith('fc.')]
                if keys_to_remove: print(f"Removing FC head: {keys_to_remove}"); [loaded_state_dict.pop(k, None) for k in keys_to_remove]
                if all(k.startswith('module.')): loaded_state_dict = {k[7:]: v for k, v in loaded_state_dict.items()}

                # Load into custom model
                print("Loading weights into custom RegNetBackbone...")
                missing, unexpected = model.load_state_dict(loaded_state_dict, strict=False)
                if missing: print(f"Warning: Missing keys loading backbone: {missing}")
                unexpected = [k for k in unexpected if not (k.startswith('head.') or k.startswith('fc.'))]
                if unexpected: print(f"Warning: Unexpected keys loading backbone: {unexpected}")
                print("Pretrained backbone weights loaded from Torchvision (strict=False).")
                loaded_weights_source = "Torchvision"
            except Exception as e:
                print(f"Failed to load weights from torchvision: {e}. Proceeding with random init.")
                loaded_weights_source = "Random Init (torchvision failed)"
        else:
             print("Warning: Torchvision RegNet unavailable. Proceeding with random init.")
             loaded_weights_source = "Random Init (torchvision unavailable)"

    print(f"RegNetBackbone created with weights: {loaded_weights_source}")
    return model


# --- Neck BiFPN ---
# (Identique à la version précédente, avec correction TypeError)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False, act=nn.ReLU):
        super().__init__(); self.dw=nn.Conv2d(in_ch,in_ch,k,s,p,groups=in_ch,bias=bias); self.pw=nn.Conv2d(in_ch,out_ch,1,1,0,bias=bias); self.bn=nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3);
        if act is None: self.act = nn.Identity()
        elif isinstance(act, type): self.act = act(inplace=True) if hasattr(act,'__constants__') and 'inplace' in act.__constants__ else act()
        else: self.act = act
    def forward(self, x): x=self.dw(x); x=self.pw(x); x=self.bn(x); return self.act(x)
class BiFPNNode(nn.Module):
    def __init__(self, channels, num_inputs, eps=1e-4): super().__init__(); self.num_inputs=num_inputs; self.eps=eps; self.w=nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True); self.relu=nn.ReLU(False)
    def forward(self, inputs): assert len(inputs)==self.num_inputs; w=self.relu(self.w); w=w/(torch.sum(w,dim=0,keepdim=True)+self.eps); x=torch.stack(inputs,dim=-1); x=(w*x).sum(dim=-1); return x
class BiFPNLayer(nn.Module):
    def __init__(self, channels, levels, eps=1e-4, act=nn.ReLU):
        super().__init__(); self.channels=channels; self.levels=levels; self.eps=eps; self.act=act
        self.convs=nn.ModuleList([DepthwiseSeparableConv(channels,channels,act=act) for _ in range(levels)]) # Corrected arg name
        self.fuse_td=nn.ModuleList([BiFPNNode(channels,2,eps) for _ in range(levels-1)])
        self.fuse_bu=nn.ModuleList([BiFPNNode(channels,2 if i==0 or i==levels-1 else 3,eps) for i in range(levels)])
        self.up=nn.Upsample(scale_factor=2,mode='nearest'); self.down=nn.MaxPool2d(3,2,1)
    def _resize_like(self,src,tgt): return F.interpolate(src,size=tgt.shape[-2:],mode='bilinear',align_corners=False) if src.shape[-2:]!=tgt.shape[-2:] else src
    def forward(self,inputs):
        nl=len(inputs); assert nl==self.levels; td=[None]*nl; td[-1]=inputs[-1]
        for i in range(nl-2,-1,-1): tgt=inputs[i]; feat_up=td[i+1]; up=self.up(feat_up); up=self._resize_like(up,tgt); fused=self.fuse_td[i]([tgt,up]); td[i]=self.convs[i](fused)
        out=[None]*nl; out[0]=td[0]
        for i in range(1,nl):
            tgt_td=td[i]; src_down=out[i-1]; down=self.down(src_down); down=self._resize_like(down,tgt_td); tgt_orig=inputs[i]
            if i==nl-1: down=self._resize_like(down,tgt_orig); fused=self.fuse_bu[i]([tgt_orig,down])
            else: down=self._resize_like(down,tgt_orig); tgt_td=self._resize_like(tgt_td,tgt_orig); fused=self.fuse_bu[i]([tgt_orig,tgt_td,down])
            out[i]=self.convs[i](fused); return out
class BiFPNNeck(nn.Module):
    def __init__(self, backbone_channels, fpn_channels, num_repeats, levels_to_use, activation=nn.ReLU, eps=1e-4):
        super().__init__(); self.levels_to_use=levels_to_use; self.num_levels_in=len(levels_to_use); self.fpn_channels=fpn_channels
        missing=[lvl for lvl in levels_to_use if lvl not in backbone_channels];
        if missing: raise ValueError(f"BiFPNNeck BB chans missing: {missing}")
        self.lat_convs=nn.ModuleList([nn.Conv2d(backbone_channels[lvl],fpn_channels,1) for lvl in levels_to_use])
        self.bifpn=nn.ModuleList([BiFPNLayer(fpn_channels,self.num_levels_in,eps,act=activation) for _ in range(num_repeats)]) # Pass act
        map={'s1':2,'s2':3,'s3':4,'s4':5,'s5':6,'s6':7}; self.out_lvl_names=[]; min_p,max_p=float('inf'),float('-inf')
        for lvl in self.levels_to_use:
            p_idx=map.get(lvl)
            if p_idx is None:
                try: p_idx=int(lvl[1:])+1; warnings.warn(f"{lvl}: P-lvl {p_idx} deduced.")
                except(ValueError,IndexError): raise ValueError(f"Cannot map {lvl}")
            self.out_lvl_names.append(f"P{p_idx}"); min_p,max_p=min(min_p,p_idx),max(max_p,p_idx)
        exp=list(range(min_p,max_p+1)); act=sorted([int(n[1:]) for n in self.out_lvl_names])
        if act!=exp: warnings.warn(f"P levels {self.out_lvl_names} not contiguous.")
        print(f"BiFPNNeck output P-levels: {self.out_lvl_names}")
    def forward(self, features):
        lat=[self.lat_convs[i](features[lvl]) for i,lvl in enumerate(self.levels_to_use) if lvl in features]
        if len(lat)!=len(self.levels_to_use): warnings.warn("Mismatch BB features for Neck.")
        if not lat: return OrderedDict()
        for layer in self.bifpn: lat=layer(lat)
        return OrderedDict(zip(self.out_lvl_names[:len(lat)], lat))


# --- Têtes Seg/Depth (Sortie 480x720) ---
class SemanticSegmentationHead(nn.Module):
    def __init__(self, fpn_channels, num_classes, output_target_size=(480, 720), levels_to_use=['P3', 'P4', 'P5']): # *** MODIFIÉ ***
        super().__init__(); self.levels_to_use=levels_to_use; self.target_size=output_target_size; self.num_classes=num_classes
        embed_dim=max(fpn_channels//2,128); self.adapters=nn.ModuleDict()
        self.used_levels_on_init=levels_to_use
        for lvl in self.used_levels_on_init: self.adapters[lvl]=nn.Sequential(nn.Conv2d(fpn_channels,embed_dim,3,1,1,bias=False),nn.BatchNorm2d(embed_dim),nn.ReLU(True))
        self.final_conv=nn.Conv2d(embed_dim*len(self.used_levels_on_init),num_classes,1)
        print(f"SegHead init: Uses {self.used_levels_on_init}. Conv in={self.final_conv.in_channels}. Target out={self.target_size}")
    def forward(self, features):
        adapted=[]; int_size=None; proc_lvls=[]
        for lvl in self.used_levels_on_init:
            if lvl in features:
                if int_size is None: int_size=features[lvl].shape[-2:]
                adapted.append(self.adapters[lvl](features[lvl])); proc_lvls.append(lvl)
        if not adapted: print("Warn: SegHead no features."); return {'segmentation': None}
        resized=[F.interpolate(f,size=int_size,mode='bilinear',align_corners=False) if f.shape[-2:]!=int_size else f for f in adapted]
        final_f=torch.cat(resized,dim=1)
        if self.final_conv.in_channels!=final_f.shape[1]: warnings.warn(f"SegHead chan mismatch! Conv={self.final_conv.in_channels}, got {final_f.shape[1]} (lvls {proc_lvls})."); return {'segmentation': None}
        out=self.final_conv(final_f);
        if out is not None: out=F.interpolate(out,size=self.target_size,mode='bilinear',align_corners=False)
        return {'segmentation': out}
class DepthEstimationHead(nn.Module):
    def __init__(self, fpn_channels, output_target_size=(480, 720), levels_to_use=['P3', 'P4', 'P5']): # *** MODIFIÉ ***
        super().__init__(); self.levels_to_use=levels_to_use; self.target_size=output_target_size
        embed_dim=max(fpn_channels//2,128); self.adapters=nn.ModuleDict()
        self.used_levels_on_init=levels_to_use
        for lvl in self.used_levels_on_init: self.adapters[lvl]=nn.Sequential(nn.Conv2d(fpn_channels,embed_dim,3,1,1,bias=False),nn.BatchNorm2d(embed_dim),nn.ReLU(True))
        self.final_conv=nn.Conv2d(embed_dim*len(self.used_levels_on_init),1,1); self.relu=nn.ReLU(True)
        print(f"DepthHead init: Uses {self.used_levels_on_init}. Conv in={self.final_conv.in_channels}. Target out={self.target_size}")
    def forward(self, features):
        adapted=[]; int_size=None; proc_lvls=[]
        for lvl in self.used_levels_on_init:
            if lvl in features:
                if int_size is None: int_size=features[lvl].shape[-2:]
                adapted.append(self.adapters[lvl](features[lvl])); proc_lvls.append(lvl)
        if not adapted: print("Warn: DepthHead no features."); return {'depth': None}
        resized=[F.interpolate(f,size=int_size,mode='bilinear',align_corners=False) if f.shape[-2:]!=int_size else f for f in adapted]
        final_f=torch.cat(resized,dim=1)
        if self.final_conv.in_channels!=final_f.shape[1]: warnings.warn(f"DepthHead chan mismatch! Conv={self.final_conv.in_channels}, got {final_f.shape[1]} (lvls {proc_lvls})."); return {'depth': None}
        out=self.final_conv(final_f); out=self.relu(out);
        if out is not None: out=F.interpolate(out,size=self.target_size,mode='bilinear',align_corners=False)
        return {'depth': out}

# --- Modèle MultiTaskNet ---
class MultiTaskNet(nn.Module):
    def __init__(self, backbone_config, backbone_pretrained, # Reste booléen
                 neck_config, segmentation_config, depth_config):
        super().__init__()
        print("Initializing MultiTaskNet...")
        # *** Utiliser backbone_pretrained flag pour create_regnet_backbone ***
        # *** ckpt_path n'est plus utilisé ici pour les poids ImageNet ***
        self.backbone = create_regnet_backbone(backbone_config, pretrained=backbone_pretrained)
        available_bb={i['module']:i['num_chs'] for i in self.backbone.feature_info}; bb_ch_neck={}; neck_lvls=[]
        for lvl in neck_config['levels_to_use']:
            if lvl in available_bb: bb_ch_neck[lvl]=available_bb[lvl]; neck_lvls.append(lvl)
            else: warnings.warn(f"BB lvl {lvl} for Neck missing. Skipping.")
        if not neck_lvls: raise ValueError("No usable BB levels for Neck!")
        neck_cfg_upd={**neck_config,'levels_to_use':neck_lvls}; self.neck=BiFPNNeck(bb_ch_neck,**neck_cfg_upd); fpn_ch=neck_cfg_upd['fpn_channels']; prod_fpn_lvls=self.neck.out_lvl_names
        def filter_levels(req,avail): used=[l for l in req if l in avail]; return used if used else([avail[0]] if avail else[])
        seg_lvls=filter_levels(segmentation_config.get('levels_to_use',prod_fpn_lvls),prod_fpn_lvls); seg_cfg_upd={**segmentation_config,'levels_to_use':seg_lvls}
        depth_lvls=filter_levels(depth_config.get('levels_to_use',prod_fpn_lvls),prod_fpn_lvls); depth_cfg_upd={**depth_config,'levels_to_use':depth_lvls}
        self.segmentation_head=SemanticSegmentationHead(fpn_ch,**seg_cfg_upd); self.depth_head=DepthEstimationHead(fpn_ch,**depth_cfg_upd)
        print("MultiTaskNet Initialized.");
    def forward(self, x): # x: (B, 3, 480, 720)
        bb_f=self.backbone(x); fpn_f=self.neck(bb_f); seg_o=self.segmentation_head(fpn_f); depth_o=self.depth_head(fpn_f)
        outs={'segmentation':seg_o.get('segmentation') if seg_o else None, 'depth':depth_o.get('depth') if depth_o else None}
        return outs

# --- Fonction de Résumé ---
def print_model_summary(model):
    print("-" * 80); print(f"Summary: {model.__class__.__name__}"); print("-" * 80); total=0
    def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad) if hasattr(m,'parameters') and m else 0
    parts={'Backbone': getattr(model,'backbone',None),'Neck': getattr(model,'neck',None),'SegHead': getattr(model,'segmentation_head',None),'DepthHead': getattr(model,'depth_head',None)}
    for n,p in parts.items(): c=count(p); print(f"{n}: {c:,} params"); total+=c;
    print("-" * 40); print(f"Total Params: {total:,}"); print("=" * 80)

# --- Bloc d'exécution (Test et Export du Backbone seul) ---
if __name__ == '__main__':
    print("Testing backbone_module.py: Init & Export Backbone...")
    # --- Configuration pour le test ---
    NETWORK_INPUT_RES = (480, 720) # Résolution d'entrée réseau
    HEAD_OUTPUT_RES = (480, 720)   # Résolution de sortie des têtes
    BACKBONE_PRETRAINED = True     # *** Tenter de charger les poids ImageNet ***
    # *** Chemin où sauvegarder le state_dict du backbone SEUL ***
    EXPORT_BACKBONE_PATH = "./regnet_1600m_pretrained_backbone_exported.pth" # Nom du fichier exporté

    backbone_cfg = regnet_1600M_config
    neck_cfg = dict(fpn_channels=128, num_repeats=1, levels_to_use=['s2', 's3', 's4'], activation=nn.ReLU)
    NUM_CLASSES_SEGMENTATION = 8

    map={'s1':2,'s2':3,'s3':4,'s4':5}; prod_fpn_lvls=sorted([f"P{map.get(lvl,int(lvl[1:])+1)}" for lvl in neck_cfg['levels_to_use']],key=lambda x:int(x[1:]))
    segmentation_cfg = dict(num_classes=NUM_CLASSES_SEGMENTATION, output_target_size=HEAD_OUTPUT_RES, levels_to_use=prod_fpn_lvls)
    depth_cfg = dict(output_target_size=HEAD_OUTPUT_RES, levels_to_use=prod_fpn_lvls)

    print(f"\nTest Instantiation: Input={NETWORK_INPUT_RES}, Head Output={HEAD_OUTPUT_RES}, Pretrained={BACKBONE_PRETRAINED}")
    try:
        # Instancier le modèle complet (ce qui crée le backbone avec les poids si pretrained=True)
        model = MultiTaskNet(
            backbone_config=backbone_cfg,
            backbone_pretrained=BACKBONE_PRETRAINED,
            # backbone_ckpt_path=None, # Plus nécessaire comme argument principal ici
            neck_config=neck_cfg,
            segmentation_config=segmentation_cfg,
            depth_config=depth_cfg
        )
        print_model_summary(model) # Afficher le résumé du modèle complet

        # --- Export du Backbone ---
        if hasattr(model, 'backbone') and model.backbone is not None:
             print(f"\nExporting backbone state_dict to: {EXPORT_BACKBONE_PATH}")
             # Créer le dossier si nécessaire
             os.makedirs(os.path.dirname(EXPORT_BACKBONE_PATH), exist_ok=True)
             backbone_to_export = model.backbone
             torch.save(backbone_to_export.state_dict(), EXPORT_BACKBONE_PATH)
             print("Backbone state_dict exported successfully.")
        else:
             print("Could not find backbone attribute to export.")

        # --- Test Forward (Optionnel mais utile) ---
        print("\nTesting Forward Pass...")
        dummy_input = torch.randn(1, 3, NETWORK_INPUT_RES[0], NETWORK_INPUT_RES[1])
        model.eval();
        with torch.no_grad(): outputs = model(dummy_input)
        print("Forward pass completed.")
        print("Output shapes:", {k: v.shape if isinstance(v, torch.Tensor) else type(v) for k, v in outputs.items()})


    except Exception as e:
        print("\n--- SCRIPT FAILED ---"); import traceback; traceback.print_exc()
