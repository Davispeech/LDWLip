import torch
import torch.nn as nn


def get_submodule_by_name(model, module_name: str):
    """
    根据 '点分路径' 返回模块对象
    e.g.
        module_name = "encoder.layers.0.feed_forward"
        返回 model.encoder.layers[0].feed_forward
    """
    attrs = module_name.split(".")
    submodule = model
    for a in attrs:
        if a.isdigit():  # 如果是数字，表示索引
            submodule = submodule[int(a)]
        else:
            submodule = getattr(submodule, a)
    return submodule


def inject_lora_to_model(model, params):
    # [3, LoRALinear, [8, 16], ["feed_forward_macaron"]]
    # _, lora_name, lora_param, target_modules = params
    lora_name = params[1]
    lora_param = params[2]
    target_modules = params[3]

    if lora_name == "LoRALinear":
        from .LoRALinear import LoRALinear

        for name, module in model.named_modules():
            # print(name)
            # if any(t in name.split('.') for t in target_modules):  # target_modules=("feed_forward_macaron")
            if any(t in name for t in target_modules):  # target_modules=("feed_forward_macaron")
                if hasattr(module, "w_1") and hasattr(module, "w_2"):
                    if lora_name == "LoRALinear":
                        module.w_1 = LoRALinear(module.w_1, lora_param[0], lora_param[1])
                        module.w_2 = LoRALinear(module.w_2, lora_param[0], lora_param[1])
                if hasattr(module, "linear_q") and hasattr(module, "linear_v"):
                    if lora_name == "LoRALinear":
                        module.linear_q = LoRALinear(module.linear_q, lora_param[0], lora_param[1])
                        module.linear_v = LoRALinear(module.linear_v, lora_param[0], lora_param[1])
    elif lora_name in ("LoRAMoE", "LoRAMoEPaper"):
        from .LoRALinear import LoRAMoE, LoRAMoEPaper

        wrapper = LoRAMoE if lora_name == "LoRAMoE" else LoRAMoEPaper
        for name, module in model.named_modules():
            if any(t in name for t in target_modules):
                if hasattr(module, "w_1") and hasattr(module, "w_2"):
                    module.w_1 = wrapper(module.w_1, lora_param[0], lora_param[1], lora_param[2])
                    module.w_2 = wrapper(module.w_2, lora_param[0], lora_param[1], lora_param[2])
                if hasattr(module, "linear_q") and hasattr(module, "linear_v"):
                    module.linear_q = wrapper(module.linear_q, lora_param[0], lora_param[1], lora_param[2])
                    module.linear_v = wrapper(module.linear_v, lora_param[0], lora_param[1], lora_param[2])
    elif lora_name == "LoRAConv":
        from .LoRAConv import LoRAConv

        def inject_conv_recursively(parent_module):
            for name, child in parent_module.named_children():
                # 如果模块名命中 target_modules 关键字
                if any(t in name for t in target_modules):
                    if isinstance(child, nn.modules.conv._ConvNd):
                        if not isinstance(child, LoRAConv):
                            # 真正替换父模块中的子模块
                            setattr(parent_module, name, LoRAConv(child, lora_param[0], lora_param[1]))
                            # 确保 LoRA 参数可训练
                            for p in getattr(parent_module, name).parameters():
                                p.requires_grad = True
                # 递归处理子模块
                inject_conv_recursively(child)
        inject_conv_recursively(model)
        last = None
        for child in model.modules():
            if isinstance(child, LoRAConv):
                child.record_corr = False
                last = child
        if last is not None:
            last.record_corr = True


