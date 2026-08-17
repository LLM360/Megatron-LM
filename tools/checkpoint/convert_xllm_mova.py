# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Convert a native xLLM MoVA checkpoint into a reshardable MCore checkpoint.

The converter deliberately runs with TP=PP=EP=1 on one CPU-rich GPU node. It
builds the target model in BF16 CPU memory, loads full xLLM tensors in bounded
layer chunks, and writes a release-format ``torch_dist`` checkpoint. MCore can
then reshard that checkpoint into the production TP/PP/EP layout at load time.
"""

import gc
import os
import sys
from pathlib import Path

import torch

from megatron.core.models.gpt.mova_checkpoint import (
    load_xllm_mova_global_weights,
    load_xllm_mova_layer,
)
from megatron.core.transformer.mova import MoVATransformerConfig
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args
from megatron.training.checkpointing import get_checkpoint_name, save_checkpoint
from megatron.training.initialize import initialize_megatron
from mova_builders import mova_builder
from pretrain_mova import add_mova_args

REFERENCE_XLLM_COMMIT = "5494c84cc7a2a66e5d6439bf9aa07454097ce9be"


def add_conversion_args(parser):
    parser = add_mova_args(parser)
    group = parser.add_argument_group(title="native xLLM MoVA conversion")
    group.add_argument("--xllm-checkpoint", type=Path, required=True)
    group.add_argument("--xllm-repo", type=Path, required=True)
    group.add_argument("--mcore-output", type=Path, required=True)
    group.add_argument(
        "--xllm-layers-per-load",
        type=int,
        default=0,
        help="Layers retained per source read; zero loads all layers in one pass.",
    )
    return parser


def _validate_source_model(model_cfg: dict) -> None:
    required = {
        "arch": "transformer",
        "apply_rmsnorm": True,
        "swiglu": True,
        "apply_attn_gate": True,
        "qknorm": False,
        "two_hop_residual": False,
        "scale_emb": False,
        "rescale_nffn": False,
    }
    for key, expected in required.items():
        actual = model_cfg.get(key)
        if actual != expected:
            raise ValueError(f"Unsupported xLLM {key}={actual!r}; expected {expected!r}")
    if model_cfg.get("num_values", 0) <= 0 or model_cfg.get("num_experts", 0) <= 0:
        raise ValueError("The source must contain both MoVA and FFN MoE experts")
    if not 0 <= model_cfg.get("num_dense_layers", -1) < model_cfg.get("num_layers", 0):
        raise ValueError("The source must contain a valid dense prefix and sparse MoVA layers")
    if model_cfg.get("num_shared_experts") != 1:
        raise ValueError("The current mapping requires exactly one shared FFN expert")
    output_size = model_cfg.get("output_size", -1)
    if output_size not in (-1, model_cfg.get("vocab_size")):
        raise ValueError("The current mapping requires output_size == vocab_size")
    for key in ("dropout", "hidden_dropout", "attention_dropout"):
        if model_cfg.get(key, 0.0) != 0.0:
            raise ValueError(f"Exact conversion currently requires {key}=0")
    head_dim = model_cfg.get("head_dim") or (model_cfg["model_dim"] // model_cfg["num_heads"])
    rope_head_dim = model_cfg.get("rope_head_dim") or head_dim
    if rope_head_dim != head_dim:
        raise ValueError("The current MoVA mapping requires full-head RoPE")


def _apply_source_architecture(args, source_config: dict) -> None:
    model_cfg = source_config["model"]
    _validate_source_model(model_cfg)
    head_dim = model_cfg.get("head_dim") or (model_cfg["model_dim"] // model_cfg["num_heads"])

    args.num_layers = model_cfg["num_layers"]
    args.hidden_size = model_cfg["model_dim"]
    args.num_attention_heads = model_cfg["num_heads"]
    args.group_query_attention = True
    args.num_query_groups = model_cfg["num_kv_heads"]
    args.kv_channels = head_dim
    args.ffn_hidden_size = model_cfg["ffn_hidden_dim"]
    args.num_experts = model_cfg["num_experts"]
    args.moe_ffn_hidden_size = model_cfg["expert_inter_dim"]
    args.moe_router_topk = model_cfg["num_activated_experts"]
    args.moe_shared_expert_intermediate_size = (
        model_cfg["num_shared_experts"] * model_cfg["expert_inter_dim"]
    )
    args.moe_router_score_function = model_cfg["moe_router_score_func"]
    args.moe_router_topk_scaling_factor = model_cfg["moe_router_scaling_factor"]
    # xLLM rounds the GEMM in the activation dtype, then performs scoring and
    # top-k in FP32. The compatibility path enforces that operation order;
    # retain fp32 here as the checkpoint's explicit router-scoring contract.
    args.moe_router_dtype = "fp32"
    args.moe_router_enable_expert_bias = model_cfg["moe_router_bias"]
    args.moe_router_bias_update_rate = model_cfg["moe_router_bias_update_rate"]
    source_balance_type = source_config.get("moe_router_load_balancing_type")
    source_aux_coeff = source_config.get("moe_aux_loss_coeff", 0.0)
    if source_balance_type not in (None, "dot"):
        raise ValueError(
            "Only xLLM's dot router-balancing objective has an exact MCore mapping; "
            f"got {source_balance_type!r}"
        )
    num_sparse_layers = model_cfg["num_layers"] - model_cfg["num_dense_layers"]
    # xLLM averages the MoVA and FFN router losses, then averages over
    # sparse layers. MCore attaches the coefficient at each router.
    router_aux_coeff = (
        source_aux_coeff / (2 * num_sparse_layers) if source_balance_type == "dot" else 0.0
    )
    args.moe_router_load_balancing_type = "aux_loss" if source_balance_type == "dot" else "none"
    args.moe_aux_loss_coeff = router_aux_coeff

    args.mova_num_value_experts = model_cfg["num_values"]
    args.mova_router_topk = model_cfg["num_activated_values"]
    args.mova_router_score_function = model_cfg["moe_router_score_func"]
    args.mova_router_topk_scaling_factor = model_cfg["moe_router_scaling_factor"]
    args.mova_router_enable_expert_bias = model_cfg["moe_router_bias"]
    args.mova_router_bias_update_rate = model_cfg["moe_router_bias_update_rate"]
    args.mova_router_load_balancing_type = "aux_loss" if source_balance_type == "dot" else "none"
    args.mova_router_aux_loss_coeff = router_aux_coeff
    args.mova_num_dense_layers = model_cfg["num_dense_layers"]
    args.mova_norm_num_groups = model_cfg["layernorm_num_groups"]
    args.mova_attention_gate_function = model_cfg["attn_gate_func"]
    args.mova_value_backend = "grouped_gemm"
    args.xllm_router_compatibility = True
    args.xllm_router_gemm_partitions = source_config["model_parallel_size"]

    args.swiglu = True
    args.add_bias_linear = False
    args.attention_output_gate = True
    args.normalization = "RMSNorm"
    args.norm_epsilon = model_cfg["norm_eps"]
    args.apply_layernorm_1p = True
    args.position_embedding_type = "rope"
    args.rotary_base = int(model_cfg["rope_base"])
    args.rotary_percent = 1.0
    args.rotary_interleaved = True
    args.max_position_embeddings = source_config["seq_len"]
    args.seq_length = source_config["seq_len"]
    args.attention_dropout = 0.0
    args.hidden_dropout = 0.0

    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = model_cfg["vocab_size"]
    args.make_vocab_size_divisible_by = 1
    args.untie_embeddings_and_output_weights = True

    # Produce a topology-neutral source checkpoint. The distributed checkpoint
    # carries expert and tensor sharding metadata for later production resharding.
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 1
    args.expert_model_parallel_size = 1
    args.expert_tensor_parallel_size = 1
    args.sequence_parallel = False
    args.transformer_impl = "local"
    args.moe_grouped_gemm = False
    args.moe_use_legacy_grouped_gemm = False
    args.use_cpu_initialization = True
    args.perform_initialization = False
    args.bf16 = True
    args.fp16 = False

    args.ckpt_format = "torch_dist"
    args.save = os.fspath(args.mcore_output)
    args.no_save_optim = True
    args.no_save_rng = True
    args.async_save = False
    args.ckpt_fully_parallel_save = False
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.train_iters = 1
    args.save_interval = 1


def _import_xllm_checkpoint_reader(xllm_repo: Path):
    repo = xllm_repo.resolve()
    if not (repo / "xllm_bridges/huggingface/native_checkpoint.py").is_file():
        raise FileNotFoundError(f"xLLM native checkpoint reader not found under {repo}")
    sys.path.insert(0, str(repo))
    # pylint: disable-next=import-outside-toplevel
    from xllm_bridges.huggingface.native_checkpoint import (
        load_native_xllm_state_dict,
        read_xllm_config,
    )

    return load_native_xllm_state_dict, read_xllm_config


def main() -> None:
    parsed_args = parse_args(extra_args_provider=add_conversion_args)
    load_source, read_source_config = _import_xllm_checkpoint_reader(parsed_args.xllm_repo)
    source_config = read_source_config(str(parsed_args.xllm_checkpoint))
    _apply_source_architecture(parsed_args, source_config)
    initialize_megatron(parsed_args=parsed_args)
    args = get_args()
    # NullTokenizer reserves its own EOD id and therefore changes the padded
    # vocabulary calculation. Conversion must build the exact source tensor
    # shape; no tokenizer is used for checkpoint I/O.
    args.padded_vocab_size = source_config["model"]["vocab_size"]
    if torch.distributed.get_world_size() != 1:
        raise ValueError("Conversion is topology-neutral and must run with exactly one rank")

    release_path = Path(get_checkpoint_name(args.save, 0, release=True, return_base_dir=True))
    if release_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {release_path}")

    config = core_transformer_config_from_args(args, MoVATransformerConfig)
    model = mova_builder(args, pre_process=True, post_process=True, config=config)
    num_layers = config.num_layers
    layers_per_load = args.xllm_layers_per_load or num_layers
    if not 0 < layers_per_load <= num_layers:
        raise ValueError("xllm_layers_per_load must be zero or in [1, num_layers]")

    print_rank_0(
        f"Converting xLLM MoVA reference {REFERENCE_XLLM_COMMIT}; "
        f"{num_layers} layers in chunks of {layers_per_load}"
    )
    layers_by_id = {layer.layer_number - 1: layer for layer in model.decoder.layers}
    for layer_l in range(0, num_layers, layers_per_load):
        layer_r = min(layer_l + layers_per_load, num_layers)
        source_state = load_source(str(args.xllm_checkpoint), source_config, layer_l, layer_r)
        for layer_idx in range(layer_l, layer_r):
            load_xllm_mova_layer(layers_by_id[layer_idx], source_state, layer_idx)
        if layer_r == num_layers:
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
