#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer
from typing import Optional, List


class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            lora_C=None,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_C = lora_C
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros(1, (int(in_features / 2)))
            )

            self.lora_B = nn.Parameter(
                self.weight.new_zeros((int(out_features / 2), 1))
            )
            self.lora_C = nn.Parameter(torch.zeros(1, 1, 2, 2), requires_grad=True)
            nn.init.normal_(self.lora_C, mean=0.0, std=0.02)
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                lora_AB = self.lora_A.T @ self.lora_B.T  # (in_features/8, out_features/8)
                # 使用lora_C作为卷积核进行反卷积
                lora_output = F.conv_transpose2d(lora_AB.unsqueeze(0).unsqueeze(0), self.lora_C, stride=2, padding=0)
                self.weight.data -= T(
                    lora_output.squeeze()
                ) * self.scaling / (self.ranknum + 1e-5)
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                lora_AB = self.lora_A.T @ self.lora_B.T  # (in_features/8, out_features/8)

                # 使用lora_C作为卷积核进行反卷积
                lora_output = F.conv_transpose2d(lora_AB.unsqueeze(0).unsqueeze(0), self.lora_C, stride=2, padding=0)
                self.weight.data += T(
                    lora_output.squeeze()
                ) * self.scaling / (self.ranknum + 1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                lora_AB = self.lora_A.T @ self.lora_B.T  # (in_features/8, out_features/8)
                # 使用lora_C作为卷积核进行反卷积
                lora_output = F.conv_transpose2d(lora_AB.unsqueeze(0).unsqueeze(0), self.lora_C, stride=2, padding=0)
                result += (self.lora_dropout(x) @ lora_output.squeeze()) \
                          * self.scaling / (self.ranknum + 1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """

    def __init__(
            self, model,
            lora_r: int,
            target_rank: int,
            init_warmup: int,
            final_warmup: int,
            mask_interval: int,
            total_step: Optional[int] = None,
            target_total_rank: Optional[int] = None,
            tb_writter=None,
            tb_writter_loginterval: int = 500,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.total_step = total_step

        self.model = model
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        # self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval

    def set_total_step(self, total_step: int):
        # Set total step number 
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def schedule_threshold(self, step: int):
        initial_warmup = self.initial_warmup
        self.global_step = step
        if step <= initial_warmup:
            # Initial warmup
            curr_rank = 0
            mask_ind = False
        else:
            curr_rank = 0
            mask_ind = False
        return curr_rank, mask_ind

    def update_and_mask(self, model, global_step):
        curr_rank = self.schedule_threshold(global_step)
        mask_threshold = None
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step % self.log_interval == 0:
            with torch.no_grad():
                regu_loss = []
                for n, p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov - I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s" % n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss) / len(regu_loss), self.global_step
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`.
    regu_loss, num_param = 0., 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
    return regu_weight * regu_loss / num_param

