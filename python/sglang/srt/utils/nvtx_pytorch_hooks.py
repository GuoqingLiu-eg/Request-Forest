# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
import torch.cuda.nvtx as nvtx


class PytHooks(object):


    def __init__(self):

        super().__init__()
        self.module_to_name_map = {}

    @staticmethod
    def print_tensor(tensor_obj, prefix, tensor_list=None):

        if tensor_list is None:
            tensor_list = []

        if isinstance(tensor_obj, list) or isinstance(tensor_obj, tuple):
            for ten in tensor_obj:
                tensor_list = PytHooks.print_tensor(ten, prefix, tensor_list)
        elif isinstance(tensor_obj, torch.Tensor):
            tensor_dims = list(tensor_obj.size())
            tensor_list.append(tensor_dims)
        return tensor_list

    def process_layer_params(self, module_obj):

        param_info = {}

        if (
            isinstance(module_obj, torch.nn.Conv1d)
            or isinstance(module_obj, torch.nn.Conv2d)
            or isinstance(module_obj, torch.nn.Conv3d)
        ):
            conv_params = {}
            conv_params["in_chan"] = module_obj.in_channels
            conv_params["out_chan"] = module_obj.out_channels
            conv_params["filter_dim"] = module_obj.kernel_size
            conv_params["stride"] = module_obj.stride
            conv_params["padding"] = module_obj.padding
            conv_params["dilation"] = module_obj.dilation
            conv_params["transposed"] = module_obj.transposed
            conv_params["output_padding"] = module_obj.output_padding
            conv_params["groups"] = module_obj.groups
            conv_params["padding_mode"] = module_obj.padding_mode
            param_info = conv_params
        elif (
            isinstance(module_obj, torch.nn.ConvTranspose1d)
            or isinstance(module_obj, torch.nn.ConvTranspose2d)
            or isinstance(module_obj, torch.nn.ConvTranspose3d)
        ):
            convtranspose_params = {}
            convtranspose_params["in_chan"] = module_obj.in_channels
            convtranspose_params["out_chan"] = module_obj.out_channels
            convtranspose_params["filter_dim"] = module_obj.kernel_size
            convtranspose_params["stride"] = module_obj.stride
            convtranspose_params["padding"] = module_obj.padding
            convtranspose_params["dilation"] = module_obj.dilation
            convtranspose_params["transposed"] = module_obj.transposed
            convtranspose_params["output_padding"] = module_obj.output_padding
            convtranspose_params["groups"] = module_obj.groups
            convtranspose_params["padding_mode"] = module_obj.padding_mode
            param_info = convtranspose_params
        elif (
            isinstance(module_obj, torch.nn.MaxPool1d)
            or isinstance(module_obj, torch.nn.MaxPool2d)
            or isinstance(module_obj, torch.nn.MaxPool3d)
        ):

            def _handle_int_or_tuple(parameter):
                if isinstance(parameter, tuple):
                    return list(parameter)
                elif isinstance(parameter, int):
                    return [parameter, parameter]

            pooling_params = {}
            pooling_params["filter_dim"] = _handle_int_or_tuple(module_obj.kernel_size)
            pooling_params["stride"] = _handle_int_or_tuple(module_obj.stride)
            pooling_params["padding"] = _handle_int_or_tuple(module_obj.padding)
            pooling_params["dilation"] = _handle_int_or_tuple(module_obj.dilation)
            param_info = pooling_params
        elif (
            isinstance(module_obj, torch.nn.AvgPool1d)
            or isinstance(module_obj, torch.nn.AvgPool2d)
            or isinstance(module_obj, torch.nn.AvgPool3d)
        ):
            pooling_params = {}
            pooling_params["filter_dim"] = [
                module_obj.kernel_size,
                module_obj.kernel_size,
            ]
            pooling_params["stride"] = [module_obj.stride, module_obj.stride]
            pooling_params["padding"] = [module_obj.padding, module_obj.padding]
            pooling_params["ceil_mode"] = module_obj.ceil_mode
            pooling_params["count_include_pad"] = module_obj.count_include_pad
            param_info = pooling_params
        elif (
            isinstance(module_obj, torch.nn.AdaptiveAvgPool1d)
            or isinstance(module_obj, torch.nn.AdaptiveAvgPool2d)
            or isinstance(module_obj, torch.nn.AdaptiveAvgPool3d)
        ):
            pooling_params = {}
            pooling_params["output_size"] = [
                module_obj.output_size,
                module_obj.output_size,
            ]
            param_info = pooling_params
        elif isinstance(module_obj, torch.nn.Linear):
            param_info["in_features"] = module_obj.in_features
            param_info["out_features"] = module_obj.out_features
        elif (
            isinstance(module_obj, torch.nn.BatchNorm1d)
            or isinstance(module_obj, torch.nn.BatchNorm2d)
            or isinstance(module_obj, torch.nn.BatchNorm3d)
        ):
            param_info["num_features"] = module_obj.num_features
            param_info["epsilon"] = module_obj.eps
            param_info["momentum"] = module_obj.momentum
        elif isinstance(module_obj, torch.nn.ReLU):
            param_info["in_place"] = module_obj.inplace
        elif isinstance(module_obj, torch.nn.Dropout):
            param_info["p"] = module_obj.p
            param_info["in_place"] = module_obj.inplace
        elif isinstance(module_obj, torch.nn.Embedding):
            param_info["num_embeddings"] = module_obj.num_embeddings
            param_info["embedding_dim"] = module_obj.embedding_dim
        elif isinstance(
            module_obj,
            (
                torch.nn.Upsample,
                torch.nn.UpsamplingNearest2d,
                torch.nn.UpsamplingBilinear2d,
            ),
        ):
            param_info["scale_factor"] = module_obj.scale_factor

        return param_info

    def module_fwd_hook(self, module_obj, in_tensor, out_tensor):

        nvtx.range_pop()
        return

    def module_fwd_pre_hook(self, module_obj, in_tensor):

        marker_dict = {}
        module_name = self.module_to_name_map.get(module_obj, "unknown")
        marker_dict["Module"] = module_name


        module_params = module_obj.named_parameters(recurse=False)
        for idx, (param_name, param_obj) in enumerate(module_params):
            if idx == 0:
                marker_dict["TrainableParams"] = {}
            marker_dict["TrainableParams"][param_name] = list(param_obj.size())

        in_tensor_list = PytHooks.print_tensor(in_tensor, "Input")
        if in_tensor_list:
            marker_dict["Inputs"] = in_tensor_list

        param_info = self.process_layer_params(module_obj)
        if param_info:
            marker_dict["StaticParams"] = param_info

        nvtx.range_push("{}".format(marker_dict))

        return

    def register_hooks(self, network_model, module_prefix="top"):


        skip_types = (
            torch.nn.Identity,
            torch.nn.Dropout,
            torch.nn.Dropout1d,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
        )

        for name, module in network_model.named_modules(prefix=module_prefix):

            if isinstance(module, skip_types):
                continue

            module.register_forward_pre_hook(self.module_fwd_pre_hook)
            module.register_forward_hook(self.module_fwd_hook)
            if module not in self.module_to_name_map:
                self.module_to_name_map[module] = name
            else:
                raise ValueError("Module instance {} is not unique ".format(module))
        return
