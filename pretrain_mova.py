# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pretrain or continue training an xLLM-compatible MoVA GPT model."""

import argparse
from functools import partial

from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain
from model_provider import model_provider
from mova_builders import mova_builder
from pretrain_gpt import forward_step, get_embedding_ranks, train_valid_test_datasets_provider


def add_mova_args(parser):
    """Add value-router arguments without conflating them with FFN MoE routing."""

    # These are architecture defaults for the native xLLM MoVA checkpoint.
    # parser.set_defaults is required here: Megatron's args_defaults only fills
    # arguments whose parser value is None, while both dropout flags default to
    # 0.1 in the shared GPT parser. Explicit CLI values still take precedence.
    parser.set_defaults(attention_dropout=0.0, hidden_dropout=0.0)
    group = parser.add_argument_group(title="mixture-of-value attention")
    group.add_argument("--mova-num-value-experts", type=int, default=0)
    group.add_argument("--mova-router-topk", type=int, default=1)
    group.add_argument(
        "--mova-router-score-function", choices=("softmax", "sigmoid"), default="sigmoid"
    )
    group.add_argument("--mova-router-topk-scaling-factor", type=float, default=1.0)
    group.add_argument("--mova-router-enable-expert-bias", action="store_true")
    group.add_argument("--mova-router-bias-update-rate", type=float, default=1.0e-3)
    group.add_argument("--mova-router-aux-loss-coeff", type=float, default=0.0)
    group.add_argument(
        "--mova-router-load-balancing-type", choices=("none", "aux_loss"), default="none"
    )
    group.add_argument("--mova-num-dense-layers", type=int, default=0)
    group.add_argument("--mova-norm-num-groups", type=int, default=1)
    group.add_argument(
        "--mova-attention-gate-function", choices=("softplus", "silu"), default="softplus"
    )
    group.add_argument(
        "--mova-value-backend", choices=("sequential", "grouped_gemm"), default="grouped_gemm"
    )
    group.add_argument(
        "--xllm-router-compatibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match xLLM's BF16 router GEMM followed by FP32 top-k scoring.",
    )
    group.add_argument(
        "--xllm-router-gemm-partitions",
        type=int,
        default=1,
        help="Original xLLM model-parallel partitions used by router GEMMs.",
    )
    return parser


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, mova_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_mova_args,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
