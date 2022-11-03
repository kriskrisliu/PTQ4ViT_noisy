from glob import glob
from timm.models.layers import config
from torch.nn.modules import module
from test_vit import *
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
from utils.net_wrap import wrap_certain_modules_in_net
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time

def test_all(name, cfg_modifier=lambda x: x, calib_size=32, config_name="PTQ4ViT"):
    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)

    net = get_net(name)
    net.eval()

    print(f"model: {name} \n")
    print(f"calibration size: {calib_size} \n")
    print(f"bit settings: {quant_cfg.bit} \n")
    print(f"config: {config_name} \n")
    print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
    print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
    print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
    
    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)

    g=datasets.ViTImageNetLoaderGenerator(args.data_path,'imagenet',32,32,16, kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=calib_size)
    _names = ['n_G_A', 'n_V_A', 'n_H_A', 'n_G_B', 'n_V_B', 'n_H_B', 'crb_groups_A', 'crb_groups_B', 'crb_rows_A', 'crb_cols_A', 'crb_rows_B', 'crb_cols_B']

    # add timing
    calib_start_time = time.time()
    if args.fs:
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()
    calib_end_time = time.time()

    if args.fs:
        # import ipdb; ipdb.set_trace()
        savings = {}
        for name0,module in net.named_modules():
            if hasattr(module,'w_interval'):
                savings[name0] = {"w_interval":module.w_interval, "a_interval":module.a_interval}
            if hasattr(module,'A_interval'):
                # pad_groups_A': 0, 'pad_groups_B': 0, 'pad_rows_A': 0, 'pad_rows_B': 0, 'pad_cols_A': 0, 'pad_cols_B': 0
                savings[name0] = {"A_interval":module.A_interval, "B_interval":module.B_interval,
                                  "pad_groups_A":module.pad_groups_A,"pad_groups_B":module.pad_groups_B,
                                  "pad_rows_A":module.pad_rows_A,"pad_rows_B":module.pad_rows_B,
                                  "pad_cols_A":module.pad_cols_A,"pad_cols_B":module.pad_cols_B}
                for _n in _names:
                    savings[name0][_n] = getattr(module,_n)
        torch.save(savings, f'{name}.pth')
        acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])
        print(f"accuracy: {acc} \n\n")
        print("-"*80)
        net0 = net
    else:
        savings = torch.load(f"{name}.pth")
        # import ipdb; ipdb.set_trace()
        net0 = get_net(name)
        wrapped_modules0=net_wrap.wrap_modules_in_net(net0,quant_cfg)
        for name0,module in net0.named_modules():
            if hasattr(module,'w_interval'):
                # print(name0)
                setattr(module,"w_interval", savings[name0]["w_interval"])
                setattr(module,"a_interval", savings[name0]["a_interval"])
            elif hasattr(module,'A_interval'):
                # print(name0)
                setattr(module,"A_interval", savings[name0]["A_interval"])
                setattr(module,"B_interval", savings[name0]["B_interval"])
                setattr(module,"pad_groups_A", savings[name0]["pad_groups_A"])
                setattr(module,"pad_groups_B", savings[name0]["pad_groups_B"])
                setattr(module,"pad_rows_A", savings[name0]["pad_rows_A"])
                setattr(module,"pad_rows_B", savings[name0]["pad_rows_B"])
                setattr(module,"pad_cols_A", savings[name0]["pad_cols_A"])
                setattr(module,"pad_cols_B", savings[name0]["pad_cols_B"])
                for _n in _names:
                    setattr(module,_n, savings[name0][_n])
            else:
                continue
            module.mode = "quant_forward"
            module.calibrated = True
    net0.eval()
    #-----------------------
    # noise search
    global feats
    from quant_config import extra_config
    noise_best_dict = {}
    fixed_noise_dict = {}
    static_best_dict = {}
    criterion = torch.nn.MSELoss()
    config = extra_config.get(f"{name}-6bit-noise",{})
    if config == {}:
        for n0,m0 in net0.named_modules():
            if isinstance(m0, MinMaxQuantLinear):
                config[n0]=0.0
    for iii, (_n,_num) in enumerate(config.items()):
        # if _n.endswith("fc2"):
        #     continue
        for __n,__m in net0.named_modules():
            if __n==_n:
                h = __m.register_forward_hook(hook)
                break
        acc = test_classification(net0,calib_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"],disable=True)
        module0 = feats["module"]
        input = feats["input"][0]
        output = feats["output"]
        w_sim,bias_sim = module0.quant_weight_bias()
        a_qmax = module0.a_qmax
        a_interval = module0.a_interval.flatten()[0]
        output_fp = F.linear(torch.clamp(input,-a_qmax*a_interval, (a_qmax-1)*a_interval), w_sim,bias_sim)
        raw_grad = module0.raw_grad.cuda()#.reshape_as(tensor_raw)
        # static ---------------------------------------#
        loss_lowest = 999999
        static_best = 0.
        for ii in [ii for ii in range(151)]+[-ii for ii in range(1,151)]:
            scale = ii/100.*a_interval
            x_sim=module0.quant_input(input + scale)
            out=F.linear(x_sim - scale, w_sim, bias_sim)
            if args.metric=="mse":
                loss = (criterion(out, output_fp)*1000).detach().cpu()
            else:
                loss = (raw_grad * (output_fp - out)) ** 2
                loss = loss.mean()
            # 
            if loss<loss_lowest:
                loss_lowest = loss
                static_best = scale
            # print(f"[{iii}/{len(config)}]-[{ii}/150] layer={_n}, lossx1000={loss:.4f}, static={scale:.4f}, "
            #       "calib_size={calib_size}")
        static_best_dict[_n] = float(static_best)#round(float(static_best),4)
        load_static(model=net0,layername=_n,static=static_best)
        # static ---------------------------------------#
        # static_best = 0.0
        # print("*"*80)
        
        # noise ---------------------------------------#
        noise = torch.rand(size=(1,w_sim.size(-1))).type_as(w_sim)
        noise = (noise-0.5)*2
        fixed_noise_dict[_n] = noise.clone()
        loss_lowest = 999999
        scale_best = 0.
        for ii in range(201):
            scale = ii/100.*a_interval
            x_sim=module0.quant_input(input + noise*scale+static_best)
            out=F.linear(x_sim - noise*scale-static_best, w_sim, bias_sim)
            # import ipdb; ipdb.set_trace()
            if args.metric=="mse":
                loss = (criterion(out, output_fp)*1000).detach().cpu()
            else:
                loss = (raw_grad * (output_fp - out)) ** 2
                loss = loss.mean()
            # loss = (raw_grad * (output_fp - out)) ** 2
            # loss = loss.mean()
            # loss = (criterion(out, output_fp)*1000).detach().cpu()
            if loss<loss_lowest:
                loss_lowest = loss
                scale_best = scale
            # print(f"[{iii}/{len(config)}]-[{ii}/150] layer={_n}, loss={loss:.4e}, noise={scale:.4f}, calib_size={calib_size}")
        noise_best_dict[_n] = float(scale_best)#round(float(scale_best),4)
        load_noise(model=net0,layername=_n,noiseScale=scale_best,noise=noise)
        # noise ---------------------------------------#
        # print("\'static_best_dict\':",static_best_dict)
        # print("\'noise_best_dict\':",noise_best_dict)
        print("-"*20,f"[{iii}/{len(config)}] layer={_n}, noise_best={scale_best:.4f}, calib_acc={acc*100:.2f}")
        h.remove()
    print("\'static_best_dict\':",static_best_dict)
    print("\'noise_best_dict\':",noise_best_dict)
    # import ipdb; ipdb.set_trace()
    
    
    # --------------------
    # eval calib
    # for _n,_num in noise_best_dict.items():
    #     for __n,__m in net0.named_modules():
    #         if __n==_n:
    #             setattr(__m,'noiseScale',_num)
    #             setattr(__m,'noise', fixed_noise_dict.get(_n))
    #             break
    # acc = test_classification(net0,calib_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])


    # eval test
    acc = test_classification(net0,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])

    # acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])

    print(f"model: {name} \n")
    print(f"calibration size: {calib_size} \n")
    print(f"bit settings: {quant_cfg.bit} \n")
    print(f"config: {config_name} \n")
    print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
    print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
    print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
    print(f"calibration time: {(calib_end_time-calib_start_time)/60}min \n")
    print(f"accuracy: {acc} \n\n")

feats = {}
def hook(module,input,output):
    global feats
    feats = {"input":input, "output":output, 'module':module}
    return

def load_noise(model,layername,noiseScale,noise=None):
    for n,m in model.named_modules():
        if n==layername:
            setattr(m,"noiseScale",noiseScale)
            if noise != None:
                setattr(m,"noise",noise)
            break

def load_static(model,layername,static):
    for n,m in model.named_modules():
        if n==layername:
            setattr(m,"static",static)
            break

class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg

if __name__=='__main__':
    args = parse_args()

    names = [
        args.model,
        # "vit_tiny_patch16_224",
        # "vit_small_patch32_224",
        "vit_small_patch16_224",
        # "vit_base_patch16_224",
        # "vit_base_patch16_384",
        #
        # "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",
        #
        "swin_tiny_patch4_window7_224",
        # "swin_small_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        ]
    metrics = ["hessian"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    # calib_sizes = [32,128]
    calib_sizes = [args.calib_num]
    # bit_settings = [(8,8), (6,6)] # weight, activation
    bit_settings = [(6,6)] # weight, activation
    # config_names = ["PTQ4ViT", "BasePTQ"]
    linear_type = {"linear":"BasePTQ", "nonlinear":"PTQ4ViT"}
    config_names = [linear_type[args.method]]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in \
            product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        cfg_list.append({
            "name": name,
            "cfg_modifier":cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size":calib_size,
            "config_name": config_name
        })
    
    if args.multiprocess:
        multiprocess(test_all, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            test_all(**cfg)