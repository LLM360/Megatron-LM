# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Convert a sharded xLLM MoVA Hugging Face export to MCore ``torch_dist``.

This is intentionally a topology-neutral, single-rank conversion.  The source
is validated up front and mmap-loaded one layer at a time; the resulting BF16
distributed checkpoint can be resharded into a production TP/PP/EP topology.
"""

import gc
import os
from pathlib import Path

import torch
from convert_xllm_mova import REFERENCE_XLLM_COMMIT, _apply_source_architecture

from megatron.core.models.gpt.mova_checkpoint import (
    load_xllm_mova_global_weights,
    load_xllm_mova_layer,
)
from megatron.core.models.gpt.mova_hf_checkpoint import HFXllmMoVACheckpoint
from megatron.core.transformer.mova import MoVATransformerConfig
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args
from megatron.training.checkpointing import get_checkpoint_name, save_checkpoint
from megatron.training.initialize import initialize_megatron
from mova_builders import mova_builder
from pretrain_mova import add_mova_args


def add_hf_conversion_args(parser):
    parser = add_mova_args(parser)
    group = parser.add_argument_group(title="xLLM MoVA Hugging Face conversion")
    group.add_argument("--xllm-hf-checkpoint", type=Path, required=True)
    group.add_argument("--mcore-output", type=Path, required=True)
    group.add_argument(
        "--xllm-source-router-gemm-partitions",
        type=int,
        required=True,
        help="Model-parallel partitions used by router GEMMs in the original xLLM run.",
    )
    group.add_argument(
        "--xllm-source-router-bias-update-rate",
        type=float,
        required=True,
        help="Loss-free router-bias update rate used in the original xLLM run.",
    )
    return parser


def main() -> None:
    parsed_args = parse_args(extra_args_provider=add_hf_conversion_args)
    reader = HFXllmMoVACheckpoint(parsed_args.xllm_hf_checkpoint)
    source_config = reader.source_config(
        router_gemm_partitions=parsed_args.xllm_source_router_gemm_partitions,
        router_bias_update_rate=parsed_args.xllm_source_router_bias_update_rate,
    )
    _apply_source_architecture(parsed_args, source_config)
    initialize_megatron(parsed_args=parsed_args)
    args = get_args()
    # No tokenizer participates in conversion. NullTokenizer reserves an EOD
    # id, so explicitly retain the exact source vocabulary tensor shape.
    args.padded_vocab_size = source_config["model"]["vocab_size"]
    if torch.distributed.get_world_size() != 1:
        raise ValueError("Topology-neutral conversion must run with exactly one rank")

    release_path = Path(get_checkpoint_name(args.save, 0, release=True, return_base_dir=True))
    if release_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {release_path}")

    config = core_transformer_config_from_args(args, MoVATransformerConfig)
    model = mova_builder(args, pre_process=True, post_process=True, config=config)
    layers_by_id = {layer.layer_number - 1: layer for layer in model.decoder.layers}
    if set(layers_by_id) != set(range(reader.num_layers)):
        raise RuntimeError("Topology-neutral model does not contain every source layer")

    print_rank_0(
        f"Converting complete HF export with xLLM reference {REFERENCE_XLLM_COMMIT}; "
        f"layers={reader.num_layers}, source_router_partitions="
        f"{args.xllm_router_gemm_partitions}, source_router_bias_update_rate="
        f"{args.mova_router_bias_update_rate}"
    )
    globals_loaded = False
    for layer_idx in range(reader.num_layers):
        include_globals = (
            layer_idx == reader.num_layers - 1
            and reader.layer_shard(layer_idx) == reader.global_shard
        )
        source_state = reader.load_layer(layer_idx, include_globals=include_globals)
        load_xllm_mova_layer(layers_by_id[layer_idx], source_state, layer_idx)
        if include_globals:
            load_xllm_mova_global_weights(model, source_state)
            globals_loaded = True
        del source_state
        gc.collect()
        print_rank_0(f"Loaded HF MoVA layer {layer_idx + 1}/{reader.num_layers}")

    if not globals_loaded:
        source_state = reader.load_globals()
        load_xllm_mova_global_weights(model, source_state)
        del source_state
        gc.collect()

    save_checkpoint(
        iteration=0,
        model=[model],
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
        release=True,
    )
    print_rank_0(f"Converted release checkpoint written under {os.fspath(args.save)}")


if __name__ == "__main__":
    main()
