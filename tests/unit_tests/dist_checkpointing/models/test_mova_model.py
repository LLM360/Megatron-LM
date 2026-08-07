# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed-checkpoint resharding tests for MoVA GPT models."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.mova_layer_specs import get_mova_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mova import MoVATransformerConfig
from tests.unit_tests.dist_checkpointing.models.common import (
    common_test_parallel_reconfiguration_e2e,
)
from tests.unit_tests.test_utilities import Utils


def _initialize_mova_model(seed, backend, vocab_size=32, **config_overrides):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    value_backend, moe_grouped_gemm = backend
    defaults = dict(
        num_layers=4,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        ffn_hidden_size=32,
        num_moe_experts=4,
        moe_ffn_hidden_size=8,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_shared_expert_intermediate_size=8,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        attention_output_gate=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        mova_num_value_experts=4,
        mova_router_topk=2,
        mova_router_score_function="sigmoid",
        mova_router_topk_scaling_factor=2.5,
        mova_router_enable_expert_bias=True,
        mova_num_dense_layers=1,
        mova_norm_num_groups=2,
        mova_attention_gate_function="softplus",
        mova_value_backend=value_backend,
        rotary_interleaved=True,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    defaults.update(config_overrides)
    defaults["sequence_parallel"] = defaults.get("tensor_model_parallel_size", 1) > 1
    config = MoVATransformerConfig(**defaults)
    block_spec = get_mova_gpt_decoder_block_spec(
        config, use_transformer_engine=False, moe_grouped_gemm=moe_grouped_gemm
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=block_spec,
        vocab_size=vocab_size,
        max_sequence_length=16,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        position_embedding_type="rope",
        rotary_base=10000,
        share_embeddings_and_output_weights=False,
    )


@pytest.mark.skipif(Utils.world_size < 8, reason="requires eight distributed ranks")
def test_mova_checkpoint_reshards_tp_pp_ep_and_changes_ffn_backend(tmp_path_dist_ckpt):
    """Exercise the conversion topology and a representative production topology."""

    common_test_parallel_reconfiguration_e2e(
        _initialize_mova_model,
        tmp_path_dist_ckpt,
        src_tp_pp=(1, 1),
        dest_tp_pp=(2, 2),
        src_layer_spec_fn=("grouped_gemm", False),
        dst_layer_spec_fn=("grouped_gemm", True),
        use_fpsl=False,
        load_order="tp-cp-ep-dp-pp",
        store_order="tp-cp-ep-dp-pp",
        src_tp_pp_kwargs={"expert_model_parallel_size": 1, "expert_tensor_parallel_size": 1},
        dst_tp_pp_kwargs={"expert_model_parallel_size": 2, "expert_tensor_parallel_size": 1},
        src_model_init_kwargs={"expert_model_parallel_size": 1, "expert_tensor_parallel_size": 1},
        dst_model_init_kwargs={"expert_model_parallel_size": 2, "expert_tensor_parallel_size": 1},
        metadata={"singleton_local_shards": False},
    )
