#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from argparse import ArgumentParser, Namespace

MACRO_ENABLE_SAVE_PROBABILITY_PLOTS_ARG = False
DEFAULT_SAVE_PROBABILITY_PLOTS = False


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.eval = False
        self.codeft = False
        self.no_simulate = False
        self.oct_merge = "mean"
        self.codebook_size=2048
        self.batch_size=262144
        self.steps=100
        self.raht=True
        self.percent=0.66
        self.per_channel_quant=False
        self.per_block_quant=True
        self.bit_packing=True  # 位打包功能，默认开启（将被手动参数覆盖）
        self.clamp_color=True 
        self.meson_count=False 
        self.f_count=False
        self.debug=False
        self.lseg=-1
        self.csv_path=''
        self.depth=12
        self.num_bits=8
        self.lambda_sparsity=5e-7  # PCS25的稀疏性损失权重
        self.quant_type="lsq"  # 量化器类型: "lsq" 或 "vanilla"
        self.encode="deflate" # 熵编码方式: "deflate"、"ans" 或 "laplace"
        self.lambda_rate=0.001  # 使用ANS熵编码时的R权重
        self.rate_grad_diag=False
        self.rate_grad_diag_interval=10
        self.rate_grad_diag_step=1e-4
        self.ans_subgroup_count=1
        self.export_ans_offline_fit=False
        self.export_ans_offline_fit_steps=100
        self.export_ans_offline_fit_plot_interval=100
        self.export_ans_offline_fit_main_lr=1e-3
        self.export_ans_offline_fit_aux_lr=1e-3
        self.save_probability_plots=(
            DEFAULT_SAVE_PROBABILITY_PLOTS if MACRO_ENABLE_SAVE_PROBABILITY_PLOTS_ARG else False
        )
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.save_imp = False
        self.use_indexed = False  # Changed to False since we no longer use VQ indexing
        self.depth_count = False
        self.save_mode = 'euler'
        self.scene_imp = ""
        self.not_update_rot = False
        self.skip_quant_rot = False
        self.hyper_config = "universal"
        self.save_ft_type=""
        self.n_block=66
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.posititon_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.finetune_lr_scale = 1.0
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def get_combined_args_render(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    # try:
    #     cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
    #     print("Looking for config file in", cfgfilepath)
    #     with open(cfgfilepath) as cfg_file:
    #         print("Config file found: {}".format(cfgfilepath))
    #         cfgfile_string = cfg_file.read()
    # except TypeError:
    #     print("Config file not found at")
    #     pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
