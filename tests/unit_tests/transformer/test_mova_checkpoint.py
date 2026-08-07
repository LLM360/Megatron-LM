# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint-layout tests for the native xLLM to MCore MoVA mapping."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.training.checkpointing as checkpointing
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.mova_checkpoint import _pack_gqa_projection, load_xllm_mova_state_dict
from megatron.core.models.gpt.mova_layer_specs import get_mova_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mova import MoVATransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config() -> MoVATransformerConfig:
    return MoVATransformerConfig(
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
        mova_value_backend="sequential",
        rotary_interleaved=True,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )


def _make_source(config: MoVATransformerConfig, vocab_size: int) -> dict[str, torch.Tensor]:
    source = {}
    cursor = 1

    def add(key, shape):
        nonlocal cursor
        numel = int(torch.tensor(shape).prod())
        value = torch.arange(cursor, cursor + numel, dtype=torch.float32).view(shape)
        source[key] = value / 1000.0
        cursor += numel

    hidden = config.hidden_size
    query_width = config.num_attention_heads * config.kv_channels
    value_width = config.num_query_groups * config.kv_channels
    for layer_idx in range(config.num_layers):
        add(f"layers.{layer_idx}.norm.weight", (hidden,))
        if layer_idx < config.mova_num_dense_layers:
            prefix = f"layers.{layer_idx}.attention"
            add(f"{prefix}.wq.weight", (query_width, hidden))
            add(f"{prefix}.wk.weight", (value_width, hidden))
            add(f"{prefix}.wv.weight", (value_width, hidden))
            add(f"{prefix}.wr.weight", (query_width, hidden))
            add(f"{prefix}.wo.weight", (hidden, query_width))
            prefix = f"layers.{layer_idx}.nffn"
            add(f"{prefix}.norm.weight", (hidden,))
            add(f"{prefix}.fc1.weight", (config.ffn_hidden_size, hidden))
            add(f"{prefix}.fc2.weight", (hidden, config.ffn_hidden_size))
            add(f"{prefix}.fc3.weight", (config.ffn_hidden_size, hidden))
            continue

        prefix = f"layers.{layer_idx}.mova"
        add(f"{prefix}.wq.weight", (query_width, hidden))
        add(f"{prefix}.wk.weight", (value_width, hidden))
        add(f"{prefix}.wr.weight", (query_width, hidden))
        add(f"{prefix}.wo.weight", (hidden, query_width))
        add(f"{prefix}.router.weight", (config.mova_num_value_experts, hidden))
        add(f"{prefix}.router.bias", (config.mova_num_value_experts,))
        add(f"{prefix}.wv.weight", (config.mova_num_value_experts, value_width, hidden))
        prefix = f"layers.{layer_idx}.moe"
        add(f"{prefix}.norm.weight", (hidden,))
        add(f"{prefix}.router.weight", (config.num_moe_experts, hidden))
        add(f"{prefix}.router.bias", (config.num_moe_experts,))
        add(
            f"{prefix}.experts.weight1",
            (config.num_moe_experts, config.moe_ffn_hidden_size, hidden),
        )
        add(
            f"{prefix}.experts.weight2",
            (config.num_moe_experts, hidden, config.moe_ffn_hidden_size),
        )
        add(
            f"{prefix}.experts.weight3",
            (config.num_moe_experts, config.moe_ffn_hidden_size, hidden),
        )
        add(f"{prefix}.fc1.weight", (config.moe_ffn_hidden_size, hidden))
        add(f"{prefix}.fc2.weight", (hidden, config.moe_ffn_hidden_size))
        add(f"{prefix}.fc3.weight", (config.moe_ffn_hidden_size, hidden))

    add("embed.weight", (vocab_size, hidden))
    add("output.final_norm.weight", (hidden,))
    add("output.output.weight", (vocab_size, hidden))
    return source


def _make_reference_source(
    config: MoVATransformerConfig, vocab_size: int
) -> dict[str, torch.Tensor]:
    """Return small, non-degenerate weights suitable for forward parity."""

    source = _make_source(config, vocab_size)
    for key_index, key in enumerate(sorted(source)):
        value = source[key]
        phase = torch.arange(value.numel(), dtype=torch.float32).view_as(value)
        scale = 0.015 if value.ndim > 1 else 0.025
        source[key] = (torch.sin(phase * 0.173 + key_index * 0.37) * scale).requires_grad_(
            not key.endswith("router.bias")
        )
    return source


def _group_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, num_groups: int, eps: float
) -> torch.Tensor:
    grouped = hidden.float().view(*hidden.shape[:-1], num_groups, -1)
    grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    return grouped.view_as(hidden) * (weight.float() + 1.0)


def _apply_interleaved_rope(
    hidden: torch.Tensor, positions: torch.Tensor, base: float
) -> torch.Tensor:
    head_dim = hidden.shape[-1]
    inv_freq = torch.pow(
        torch.tensor(base, dtype=hidden.dtype, device=hidden.device),
        -torch.arange(0, head_dim, 2, dtype=hidden.dtype, device=hidden.device) / head_dim,
    )
    angles = positions.to(hidden.dtype).unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)
    paired = hidden.view(*hidden.shape[:-1], head_dim // 2, 2)
    even, odd = paired.unbind(dim=-1)
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return rotated.flatten(start_dim=-2)


def _route(
    hidden: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, topk: int, scale: float
) -> torch.Tensor:
    scores = torch.sigmoid(F.linear(hidden.float(), weight.float()))
    selected = torch.topk(scores + expert_bias.float(), topk, dim=-1).indices
    selected_scores = torch.gather(scores, -1, selected)
    if topk > 1:
        selected_scores = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    selected_scores = selected_scores * scale
    return torch.zeros_like(scores).scatter(-1, selected, selected_scores)


def _reference_attention(
    hidden: torch.Tensor,
    source: dict[str, torch.Tensor],
    prefix: str,
    config: MoVATransformerConfig,
    positions: torch.Tensor,
    rotary_base: float,
    mova: bool,
) -> torch.Tensor:
    batch, seq_len, _ = hidden.shape
    head_dim = config.kv_channels
    query = F.linear(hidden, source[f"{prefix}.wq.weight"])
    key = F.linear(hidden, source[f"{prefix}.wk.weight"])
    gate = F.softplus(F.linear(hidden, source[f"{prefix}.wr.weight"]), beta=math.log(2.0))
    if mova:
        dense_scores = _route(
            hidden,
            source[f"{prefix}.router.weight"],
            source[f"{prefix}.router.bias"],
            config.mova_router_topk,
            config.mova_router_topk_scaling_factor,
        )
        expert_values = torch.stack(
            [F.silu(F.linear(hidden, weight)) for weight in source[f"{prefix}.wv.weight"]], dim=-2
        )
        value = (expert_values * dense_scores.unsqueeze(-1)).sum(dim=-2)
    else:
        value = F.linear(hidden, source[f"{prefix}.wv.weight"])

    query = query.view(batch, seq_len, config.num_attention_heads, head_dim)
    key = key.view(batch, seq_len, config.num_query_groups, head_dim)
    value = value.view(batch, seq_len, config.num_query_groups, head_dim)
    query = _apply_interleaved_rope(query, positions, rotary_base)
    key = _apply_interleaved_rope(key, positions, rotary_base)
    heads_per_group = config.num_attention_heads // config.num_query_groups
    key = key.repeat_interleave(heads_per_group, dim=2)
    value = value.repeat_interleave(heads_per_group, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(head_dim)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden.device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    context = torch.einsum("bhqk,bkhd->bqhd", probabilities, value).reshape(batch, seq_len, -1)
    context = context * gate
    return F.linear(context, source[f"{prefix}.wo.weight"])


def _reference_dense_mlp(
    hidden: torch.Tensor, source: dict[str, torch.Tensor], prefix: str
) -> torch.Tensor:
    activated = F.silu(F.linear(hidden, source[f"{prefix}.fc1.weight"]))
    linear = F.linear(hidden, source[f"{prefix}.fc3.weight"])
    return F.linear(activated * linear, source[f"{prefix}.fc2.weight"])


def _reference_sparse_mlp(
    hidden: torch.Tensor,
    source: dict[str, torch.Tensor],
    prefix: str,
    config: MoVATransformerConfig,
) -> torch.Tensor:
    dense_scores = _route(
        hidden,
        source[f"{prefix}.router.weight"],
        source[f"{prefix}.router.bias"],
        config.moe_router_topk,
        config.moe_router_topk_scaling_factor,
    )
    expert_outputs = []
    weight1 = source[f"{prefix}.experts.weight1"]
    weight2 = source[f"{prefix}.experts.weight2"]
    weight3 = source[f"{prefix}.experts.weight3"]
    for expert_idx in range(config.num_moe_experts):
        activated = F.silu(F.linear(hidden, weight1[expert_idx]))
        linear = F.linear(hidden, weight3[expert_idx])
        expert_outputs.append(F.linear(activated * linear, weight2[expert_idx]))
    routed = (torch.stack(expert_outputs, dim=-2) * dense_scores.unsqueeze(-1)).sum(dim=-2)
    shared = _reference_dense_mlp(hidden, source, prefix)
    return routed + shared


def _reference_model(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    source: dict[str, torch.Tensor],
    config: MoVATransformerConfig,
    rotary_base: float,
) -> torch.Tensor:
    hidden = F.embedding(tokens, source["embed.weight"])
    for layer_idx in range(config.num_layers):
        normalized = _group_rms_norm(
            hidden,
            source[f"layers.{layer_idx}.norm.weight"],
            config.mova_norm_num_groups,
            config.layernorm_epsilon,
        )
        if layer_idx < config.mova_num_dense_layers:
            attention_prefix = f"layers.{layer_idx}.attention"
            mlp_prefix = f"layers.{layer_idx}.nffn"
            hidden = hidden + _reference_attention(
                normalized, source, attention_prefix, config, positions, rotary_base, mova=False
            )
            mlp_input = _group_rms_norm(
                hidden,
                source[f"{mlp_prefix}.norm.weight"],
                config.mova_norm_num_groups,
                config.layernorm_epsilon,
            )
            hidden = hidden + _reference_dense_mlp(mlp_input, source, mlp_prefix)
            continue

        attention_prefix = f"layers.{layer_idx}.mova"
        mlp_prefix = f"layers.{layer_idx}.moe"
        hidden = hidden + _reference_attention(
            normalized, source, attention_prefix, config, positions, rotary_base, mova=True
        )
        mlp_input = _group_rms_norm(
            hidden,
            source[f"{mlp_prefix}.norm.weight"],
            config.mova_norm_num_groups,
            config.layernorm_epsilon,
        )
        hidden = hidden + _reference_sparse_mlp(mlp_input, source, mlp_prefix, config)

    hidden = _group_rms_norm(
        hidden,
        source["output.final_norm.weight"],
        config.mova_norm_num_groups,
        config.layernorm_epsilon,
    )
    return F.linear(hidden, source["output.output.weight"])


def _mock_checkpoint_args(monkeypatch, checkpoint_args):
    state = {"args": checkpoint_args, "checkpoint_version": 3.0, "iteration": 0}
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: (state, "unused", False, "torch_dist"),
    )
    return SimpleNamespace(
        load="unused",
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
        attention_output_gate=False,
        moe_router_score_function="softmax",
        moe_grouped_gemm=True,
        mova_num_value_experts=0,
    )


def test_checkpoint_arg_loading_keeps_standard_checkpoint_behavior(monkeypatch):
    checkpoint_args = SimpleNamespace(
        mova_num_value_experts=0,
        attention_output_gate=True,
        moe_router_score_function="sigmoid",
        moe_grouped_gemm=False,
    )
    args = _mock_checkpoint_args(monkeypatch, checkpoint_args)

    loaded, _ = checkpointing.load_args_from_checkpoint(args)

    assert loaded.attention_output_gate is False
    assert loaded.moe_router_score_function == "softmax"
    assert loaded.moe_grouped_gemm is False


def test_checkpoint_arg_loading_preserves_mova_launch_backend(monkeypatch):
    checkpoint_args = SimpleNamespace(
        num_experts=100,
        mova_num_value_experts=64,
        attention_output_gate=True,
        moe_router_score_function="sigmoid",
        moe_grouped_gemm=False,
    )
    args = _mock_checkpoint_args(monkeypatch, checkpoint_args)

    loaded, _ = checkpointing.load_args_from_checkpoint(args)

    assert loaded.attention_output_gate is True
    assert loaded.moe_router_score_function == "sigmoid"
    assert loaded.moe_grouped_gemm is True
    assert loaded.mova_num_value_experts == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoVACheckpoint:
    @pytest.fixture(scope="function", autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(1234)
        yield
        Utils.destroy_model_parallel()

    def test_native_xllm_mapping_covers_all_parameter_families(self):
        config = _config()
        block_spec = get_mova_gpt_decoder_block_spec(
            config, use_transformer_engine=False, moe_grouped_gemm=False
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=32,
            max_sequence_length=16,
            position_embedding_type="rope",
            rotary_base=10000,
            share_embeddings_and_output_weights=False,
        ).cuda()
        source = _make_source(config, vocab_size=32)
        load_xllm_mova_state_dict(model, source)

        dense = model.decoder.layers[0]
        dense_prefix = "layers.0.attention"
        expected_qkv = _pack_gqa_projection(
            source[f"{dense_prefix}.wq.weight"],
            source[f"{dense_prefix}.wr.weight"],
            source[f"{dense_prefix}.wk.weight"],
            source[f"{dense_prefix}.wv.weight"],
            num_query_groups=config.num_query_groups,
            key_name="dense",
        )
        torch.testing.assert_close(dense.self_attention.linear_qkv.weight, expected_qkv.cuda())
        torch.testing.assert_close(
            dense.self_attention.linear_proj.weight, source[f"{dense_prefix}.wo.weight"].cuda()
        )
        torch.testing.assert_close(
            dense.mlp.linear_fc1.weight,
            torch.cat(
                (source["layers.0.nffn.fc1.weight"], source["layers.0.nffn.fc3.weight"])
            ).cuda(),
        )
        torch.testing.assert_close(
            dense.mlp.linear_fc2.weight, source["layers.0.nffn.fc2.weight"].cuda()
        )

        sparse = model.decoder.layers[1]
        mova_prefix = "layers.1.mova"
        expected_qkg = _pack_gqa_projection(
            source[f"{mova_prefix}.wq.weight"],
            source[f"{mova_prefix}.wr.weight"],
            source[f"{mova_prefix}.wk.weight"],
            None,
            num_query_groups=config.num_query_groups,
            key_name="mova",
        )
        torch.testing.assert_close(sparse.self_attention.linear_qkg.weight, expected_qkg.cuda())
        torch.testing.assert_close(
            sparse.self_attention.value_projection.router.weight,
            source[f"{mova_prefix}.router.weight"].cuda(),
        )
        torch.testing.assert_close(
            sparse.self_attention.value_projection.router.expert_bias,
            source[f"{mova_prefix}.router.bias"].cuda(),
        )
        for expert_idx, expert in enumerate(sparse.self_attention.value_projection.experts.experts):
            torch.testing.assert_close(
                expert.weight, source[f"{mova_prefix}.wv.weight"][expert_idx].cuda()
            )

        moe_prefix = "layers.1.moe"
        torch.testing.assert_close(
            sparse.mlp.router.weight, source[f"{moe_prefix}.router.weight"].cuda()
        )
        torch.testing.assert_close(
            sparse.mlp.router.expert_bias, source[f"{moe_prefix}.router.bias"].cuda()
        )
        for expert_idx, expert in enumerate(sparse.mlp.experts.local_experts):
            torch.testing.assert_close(
                expert.linear_fc1.weight,
                torch.cat(
                    (
                        source[f"{moe_prefix}.experts.weight1"][expert_idx],
                        source[f"{moe_prefix}.experts.weight3"][expert_idx],
                    )
                ).cuda(),
            )
            torch.testing.assert_close(
                expert.linear_fc2.weight, source[f"{moe_prefix}.experts.weight2"][expert_idx].cuda()
            )
        torch.testing.assert_close(
            sparse.mlp.shared_experts.linear_fc1.weight,
            torch.cat(
                (source[f"{moe_prefix}.fc1.weight"], source[f"{moe_prefix}.fc3.weight"])
            ).cuda(),
        )
        torch.testing.assert_close(
            sparse.mlp.shared_experts.linear_fc2.weight, source[f"{moe_prefix}.fc2.weight"].cuda()
        )

        torch.testing.assert_close(
            model.embedding.word_embeddings.weight, source["embed.weight"].cuda()
        )
        torch.testing.assert_close(
            model.decoder.final_layernorm.weight, source["output.final_norm.weight"].cuda()
        )
        torch.testing.assert_close(model.output_layer.weight, source["output.output.weight"].cuda())

    def test_converted_model_matches_independent_xllm_math(self):
        config = _config()
        rotary_base = 10000.0
        block_spec = get_mova_gpt_decoder_block_spec(
            config, use_transformer_engine=False, moe_grouped_gemm=False
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=32,
            max_sequence_length=16,
            position_embedding_type="rope",
            rotary_base=rotary_base,
            share_embeddings_and_output_weights=False,
        ).cuda()
        source = _make_reference_source(config, vocab_size=32)
        load_xllm_mova_state_dict(model, source)
        reference_source = {
            key: value.detach().clone().cuda().requires_grad_(value.requires_grad)
            for key, value in source.items()
        }

        tokens = torch.tensor([[1, 7, 3, 12, 5, 9]], dtype=torch.long)
        positions = torch.arange(tokens.shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device="cuda"),
            diagonal=1,
        ).view(1, 1, tokens.shape[1], tokens.shape[1])

        logits = model(tokens.cuda(), positions.cuda(), attention_mask)
        reference = _reference_model(
            tokens.cuda(), positions.cuda(), reference_source, config, rotary_base
        )
        # Both paths run on CUDA so this checks the independently implemented
        # xLLM math without conflating device-specific reduction differences.
        torch.testing.assert_close(logits, reference, rtol=2.0e-3, atol=5.0e-6)

        grad = torch.cos(torch.arange(reference.numel(), dtype=torch.float32)).view_as(reference)
        logits.backward(grad.cuda())
        reference.backward(grad.cuda())

        gradient_rtol = 5.0e-3
        gradient_atol = 5.0e-5
        torch.testing.assert_close(
            model.embedding.word_embeddings.weight.grad.cpu(),
            reference_source["embed.weight"].grad.cpu(),
            rtol=gradient_rtol,
            atol=gradient_atol,
        )
        dense = model.decoder.layers[0]
        dense_prefix = "layers.0.attention"
        dense_qkv_grad = _pack_gqa_projection(
            reference_source[f"{dense_prefix}.wq.weight"].grad,
            reference_source[f"{dense_prefix}.wr.weight"].grad,
            reference_source[f"{dense_prefix}.wk.weight"].grad,
            reference_source[f"{dense_prefix}.wv.weight"].grad,
            num_query_groups=config.num_query_groups,
            key_name="dense gradient",
        )
        torch.testing.assert_close(
            dense.self_attention.linear_qkv.weight.grad.cpu(),
            dense_qkv_grad.cpu(),
            rtol=gradient_rtol,
            atol=gradient_atol,
        )

        sparse = model.decoder.layers[1]
        mova_prefix = "layers.1.mova"
        torch.testing.assert_close(
            sparse.self_attention.value_projection.router.weight.grad.cpu(),
            reference_source[f"{mova_prefix}.router.weight"].grad.cpu(),
            rtol=gradient_rtol,
            atol=gradient_atol,
        )
        for expert_idx, expert in enumerate(sparse.self_attention.value_projection.experts.experts):
            torch.testing.assert_close(
                expert.weight.grad.cpu(),
                reference_source[f"{mova_prefix}.wv.weight"].grad[expert_idx].cpu(),
                rtol=gradient_rtol,
                atol=gradient_atol,
            )

        moe_prefix = "layers.1.moe"
        first_expert = sparse.mlp.experts.local_experts[0]
        expected_fc1_grad = torch.cat(
            (
                reference_source[f"{moe_prefix}.experts.weight1"].grad[0],
                reference_source[f"{moe_prefix}.experts.weight3"].grad[0],
            )
        )
        torch.testing.assert_close(
            first_expert.linear_fc1.weight.grad.cpu(),
            expected_fc1_grad.cpu(),
            rtol=gradient_rtol,
            atol=gradient_atol,
        )
        output_grad = model.output_layer.weight.grad.cpu()
        reference_output_grad = reference_source["output.output.weight"].grad.cpu()
        output_grad_error = (output_grad - reference_output_grad).abs()
        output_grad_rel_l2 = output_grad_error.norm() / reference_output_grad.norm().clamp_min(
            1.0e-12
        )
        # This wgrad sums every token and is sensitive to cancellation in the
        # independently computed hidden states. Bound both aggregate and worst
        # error rather than applying relative tolerance around zeros.
        assert output_grad_rel_l2 < 5.0e-3
        assert output_grad_error.max() < 1.0e-3

    def test_missing_required_tensor_fails_with_source_name(self):
        config = _config()
        block_spec = get_mova_gpt_decoder_block_spec(
            config, use_transformer_engine=False, moe_grouped_gemm=False
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=32,
            max_sequence_length=16,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
        ).cuda()
        source = _make_source(config, vocab_size=32)
        del source["layers.0.attention.wq.weight"]
        with pytest.raises(KeyError, match="layers.0.attention.wq.weight"):
            load_xllm_mova_state_dict(model, source)
