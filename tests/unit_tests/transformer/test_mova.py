# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Correctness tests for Mixture-of-Value Attention."""

import importlib
import math
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import transformer_engine_torch as tex

import megatron.core.parallel_state as parallel_state
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.mova_layer_specs import (
    get_k2mova_36b_config,
    get_mova_gpt_decoder_block_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.mova import (
    GroupedGemmMoVAValueExperts,
    GroupRMSNorm,
    MoVASelfAttention,
    MoVASelfAttentionSubmodules,
    MoVATransformerConfig,
    MoVAValueProjection,
    MoVAValueProjectionSubmodules,
    SequentialMoVAValueExperts,
    SequentialMoVAValueExpertsSubmodules,
    SoftplusGatedSelfAttention,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.training.training import num_floating_point_operations
from mova_builders import mova_builder
from tests.unit_tests.test_utilities import Utils

finalize_model_grads_module = importlib.import_module(
    "megatron.core.distributed.finalize_model_grads"
)


def _small_config(**overrides) -> MoVATransformerConfig:
    values = dict(
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
        mova_router_aux_loss_coeff=0.0,
        mova_num_dense_layers=1,
        mova_norm_num_groups=2,
        mova_attention_gate_function="softplus",
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )
    values.update(overrides)
    return MoVATransformerConfig(**values)


def _value_projection_spec(value_backend: str = "sequential") -> ModuleSpec:
    backend = LocalSpecProvider()
    expert_module = (
        GroupedGemmMoVAValueExperts
        if value_backend == "grouped_gemm"
        else SequentialMoVAValueExperts
    )
    return ModuleSpec(
        module=MoVAValueProjection,
        submodules=MoVAValueProjectionSubmodules(
            experts=ModuleSpec(
                module=expert_module,
                submodules=SequentialMoVAValueExpertsSubmodules(
                    linear=backend.column_parallel_linear()
                ),
            )
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mova_norm_num_groups", 0),
        ("mova_router_topk_scaling_factor", 0.0),
        ("mova_router_bias_update_rate", math.inf),
        ("mova_router_aux_loss_coeff", -1.0),
        ("mova_num_dense_layers", 4),
    ),
)
def test_mova_config_rejects_invalid_numerical_contract(field, value):
    with pytest.raises(ValueError):
        _small_config(**{field: value})


def test_mova_flop_accounting_includes_routed_values_and_dense_prefix():
    args = SimpleNamespace(
        is_hybrid_model=False,
        group_query_attention=True,
        num_query_groups=2,
        num_attention_heads=4,
        kv_channels=4,
        mova_num_value_experts=8,
        mova_router_topk=3,
        mova_num_dense_layers=1,
        mtp_num_layers=None,
        num_layers=3,
        hidden_size=16,
        attention_output_gate=True,
        seq_length=32,
        swiglu=True,
        ffn_hidden_size=64,
        moe_ffn_hidden_size=12,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=12,
        padded_vocab_size=128,
    )
    batch_size = 2
    query_size = 16
    key_value_size = 8
    dense_attention = 6 * (
        16 * (2 * query_size + 2 * key_value_size + query_size) + query_size * args.seq_length
    )
    mova_attention = 6 * (
        16
        * (
            2 * query_size
            + key_value_size
            + query_size
            + args.mova_router_topk * key_value_size
            + args.mova_num_value_experts
        )
        + query_size * args.seq_length
    )
    dense_mlp = 12 * 16 * 64 * 1.5
    sparse_mlp = 12 * 16 * 1.5 * (12 * 2 + 12)
    logits = 6 * 16 * 128
    expected = (
        batch_size
        * args.seq_length
        * (dense_attention + dense_mlp + 2 * (mova_attention + sparse_mlp) + logits)
    )

    assert num_floating_point_operations(args, batch_size) == expected


def test_mova_builder_rejects_unimplemented_mtp_composition():
    args = SimpleNamespace(use_legacy_models=False, yaml_cfg=None, spec=None, mtp_num_layers=1)
    with pytest.raises(ValueError, match="does not currently support MTP"):
        mova_builder(args, pre_process=True, post_process=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoVA:
    @pytest.fixture(scope="function", autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("use_torch_rms_norm", (False, True))
    def test_group_rms_norm_forward_and_backward(self, use_torch_rms_norm):
        config = _small_config(mova_use_torch_rms_norm=use_torch_rms_norm)
        norm = GroupRMSNorm(config, config.hidden_size, eps=1.0e-6).cuda()
        norm.weight.data.normal_(mean=0.0, std=0.1)

        hidden = torch.randn(3, 2, config.hidden_size, device="cuda", requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        reference_weight = norm.weight.detach().clone().requires_grad_(True)

        output = norm(hidden)
        grouped = reference_hidden.float().view(3, 2, config.mova_norm_num_groups, -1)
        reference = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + norm.eps)
        reference = reference.view_as(reference_hidden) * (reference_weight.float() + 1.0)

        grad = torch.randn_like(output)
        output.backward(grad)
        reference.backward(grad)

        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        torch.testing.assert_close(norm.weight.grad, reference_weight.grad)

    @pytest.mark.parametrize("use_torch_rms_norm", (False, True))
    def test_group_rms_norm_rounds_offset_scale_in_bf16(self, use_torch_rms_norm):
        config = _small_config(
            hidden_size=4,
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=4,
            mova_norm_num_groups=1,
            mova_use_torch_rms_norm=use_torch_rms_norm,
            params_dtype=torch.bfloat16,
        )
        norm = GroupRMSNorm(config, config.hidden_size, eps=1.0e-6).cuda()
        norm.weight.data.copy_(
            torch.tensor([-0.0634765625, -0.125, 0.25, -0.5], dtype=torch.bfloat16, device="cuda")
        )
        hidden = torch.tensor(
            [[[-0.0869140625, -0.25, 0.5, 1.0]]], dtype=torch.bfloat16, device="cuda"
        )

        output = norm(hidden)
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(dim=-1, keepdim=True) + norm.eps
        )
        effective_weight = norm.weight.detach() + torch.ones(
            (), dtype=torch.bfloat16, device="cuda"
        )
        expected = (normalized * effective_weight.float()).to(torch.bfloat16)

        assert torch.equal(output, expected)

    def test_group_rms_norm_supports_sharded_meta_initialization(self):
        config = _small_config(
            init_model_with_meta_device=True,
            use_cpu_initialization=False,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        with torch.device("meta"):
            norm = GroupRMSNorm(config, config.hidden_size, eps=1.0e-6)

        assert norm.weight.is_meta
        norm.to_empty(device=torch.cuda.current_device())
        norm.weight.data.fill_(1.0)
        norm.reset_parameters()

        assert torch.count_nonzero(norm.weight) == 0
        assert norm.weight.sequence_parallel

    @pytest.mark.parametrize("moe_router_fusion", (False, True))
    def test_routed_value_projection_matches_reference(self, moe_router_fusion):
        config = _small_config(moe_router_fusion=moe_router_fusion)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        value_projection = MoVAValueProjection(
            config=config,
            submodules=_value_projection_spec().submodules,
            input_size=config.hidden_size,
            output_size=config.num_query_groups * config.kv_channels,
            layer_number=2,
            pg_collection=pg_collection,
        ).cuda()
        value_projection.router.weight.data.normal_(mean=0.0, std=0.2)
        value_projection.router.expert_bias.copy_(
            torch.tensor([0.03, -0.02, 0.01, 0.0], device="cuda")
        )
        for expert in value_projection.experts.experts:
            expert.weight.data.normal_(mean=0.0, std=0.2)

        hidden = torch.randn(5, 2, config.hidden_size, device="cuda", requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        reference_router_weight = (
            value_projection.router.weight.detach().clone().requires_grad_(True)
        )
        reference_expert_weights = [
            expert.weight.detach().clone().requires_grad_(True)
            for expert in value_projection.experts.experts
        ]

        output = value_projection(hidden)

        logits = F.linear(reference_hidden.float(), reference_router_weight.float())
        scores = torch.sigmoid(logits)
        _, selected = torch.topk(
            scores + value_projection.router.expert_bias, config.mova_router_topk, dim=-1
        )
        selected_scores = torch.gather(scores, -1, selected)
        selected_scores = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        selected_scores = selected_scores * config.mova_router_topk_scaling_factor
        expert_outputs = torch.stack(
            [F.silu(F.linear(reference_hidden, weight)) for weight in reference_expert_weights],
            dim=-2,
        )
        expert_mask = F.one_hot(selected, num_classes=config.mova_num_value_experts).sum(dim=-2)
        dense_scores = torch.zeros_like(scores).scatter(-1, selected, selected_scores)
        reference = (expert_outputs * dense_scores.unsqueeze(-1)).sum(dim=-2)

        assert torch.equal(expert_mask.bool(), dense_scores.ne(0))
        torch.testing.assert_close(output, reference, rtol=2.0e-5, atol=2.0e-6)

        grad = torch.randn_like(output)
        output.backward(grad)
        reference.backward(grad)

        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=3.0e-5, atol=3.0e-6)
        torch.testing.assert_close(
            value_projection.router.weight.grad,
            reference_router_weight.grad,
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        for expert, reference_weight in zip(
            value_projection.experts.experts, reference_expert_weights
        ):
            torch.testing.assert_close(
                expert.weight.grad, reference_weight.grad, rtol=3.0e-5, atol=3.0e-6
            )

    def test_grouped_value_experts_match_sequential_backend(self):
        config = _small_config(bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=True)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        backend = LocalSpecProvider()
        submodules = SequentialMoVAValueExpertsSubmodules(linear=backend.column_parallel_linear())
        kwargs = dict(
            config=config,
            submodules=submodules,
            input_size=config.hidden_size,
            output_size=config.num_query_groups * config.kv_channels,
            num_experts=config.mova_num_value_experts,
            pg_collection=pg_collection,
        )
        sequential = SequentialMoVAValueExperts(**kwargs).cuda()
        grouped = GroupedGemmMoVAValueExperts(**kwargs).cuda()
        with torch.no_grad():
            for expert_id, expert in enumerate(sequential.experts):
                grouped.weight[expert_id].copy_(expert.weight.T)

        tokens_per_expert = torch.tensor([3, 0, 4, 2], dtype=torch.long)
        num_tokens = int(tokens_per_expert.sum())
        hidden = torch.randn(
            num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grouped_hidden = hidden.detach().clone().requires_grad_(True)

        sequential_output = sequential(hidden, tokens_per_expert)
        grouped_output = grouped(grouped_hidden, tokens_per_expert)
        torch.testing.assert_close(grouped_output, sequential_output, rtol=1.0e-2, atol=1.0e-2)

        grad = torch.randn_like(sequential_output)
        sequential_output.backward(grad)
        grouped_output.backward(grad)
        torch.testing.assert_close(grouped_hidden.grad, hidden.grad, rtol=2.0e-2, atol=2.0e-2)
        for expert_id, expert in enumerate(sequential.experts):
            torch.testing.assert_close(
                grouped.weight.grad[expert_id], expert.weight.grad.T, rtol=2.0e-2, atol=2.0e-2
            )

    def test_grouped_value_experts_support_sharded_meta_initialization(self):
        config = _small_config(
            bf16=True,
            params_dtype=torch.bfloat16,
            init_model_with_meta_device=True,
            use_cpu_initialization=False,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        submodules = SequentialMoVAValueExpertsSubmodules(
            linear=LocalSpecProvider().column_parallel_linear()
        )
        with torch.device("meta"):
            experts = GroupedGemmMoVAValueExperts(
                config=config,
                submodules=submodules,
                input_size=config.hidden_size,
                output_size=config.num_query_groups * config.kv_channels,
                num_experts=config.mova_num_value_experts,
                pg_collection=pg_collection,
            )

        assert experts.weight.is_meta
        experts.to_empty(device=torch.cuda.current_device())
        model_parallel_cuda_manual_seed(4321)
        experts.reset_parameters()

        assert torch.isfinite(experts.weight).all()
        assert torch.count_nonzero(experts.weight) > 0
        assert experts.weight.tensor_model_parallel
        assert experts.weight.partition_dim == 1
        assert experts.weight.allreduce

    def test_attention_forward_matches_gqa_reference(self):
        config = _small_config()
        backend = LocalSpecProvider()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        attention = MoVASelfAttention(
            config=config,
            submodules=MoVASelfAttentionSubmodules(
                linear_qkg=backend.column_parallel_linear(),
                value_projection=_value_projection_spec(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
            ),
            layer_number=2,
            pg_collection=pg_collection,
        ).cuda()
        for parameter in attention.parameters():
            parameter.data.normal_(mean=0.0, std=0.15)
        attention.value_projection.router.expert_bias.zero_()

        seq_len, batch_size = 5, 2
        hidden = torch.randn(seq_len, batch_size, config.hidden_size, device="cuda")
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"), diagonal=1
        ).view(1, 1, seq_len, seq_len)

        query, key, value, gate = attention.get_query_key_value_tensors(hidden)
        repeated_key = key.repeat_interleave(
            config.num_attention_heads // config.num_query_groups, dim=2
        )
        repeated_value = value.repeat_interleave(
            config.num_attention_heads // config.num_query_groups, dim=2
        )
        scores = torch.einsum("tbhd,sbhd->bhts", query, repeated_key)
        scores = scores * (config.kv_channels**-0.5)
        scores = scores.masked_fill(causal_mask, -10000.0)
        probabilities = torch.softmax(scores, dim=-1)
        reference = torch.einsum("bhts,sbhd->tbhd", probabilities, repeated_value)
        reference = reference.reshape(seq_len, batch_size, -1)
        reference = reference * F.softplus(gate, beta=math.log(2.0)).reshape_as(reference)
        reference = F.linear(reference, attention.linear_proj.weight)

        output, bias = attention(hidden, causal_mask)
        assert bias is None
        torch.testing.assert_close(output, reference, rtol=3.0e-5, atol=3.0e-6)

    def test_tiny_gpt_forward_and_backward(self):
        config = _small_config(rotary_interleaved=True)
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
        tokens = torch.randint(0, 32, (2, 6), device="cuda")
        positions = torch.arange(6, device="cuda").unsqueeze(0).expand_as(tokens)
        attention_mask = torch.triu(
            torch.ones(6, 6, dtype=torch.bool, device="cuda"), diagonal=1
        ).view(1, 1, 6, 6)

        logits = model(tokens, positions, attention_mask)
        assert logits.shape == (2, 6, 32)
        assert torch.isfinite(logits).all()
        logits.float().square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

    def test_layer_layout_and_exact_architecture_defaults(self):
        config = _small_config()
        block_spec = get_mova_gpt_decoder_block_spec(
            config, use_transformer_engine=False, moe_grouped_gemm=False
        )
        assert len(block_spec.layer_specs) == config.num_layers
        assert (
            block_spec.layer_specs[0].submodules.self_attention.module is SoftplusGatedSelfAttention
        )
        assert all(
            layer.submodules.self_attention.module is MoVASelfAttention
            for layer in block_spec.layer_specs[1:]
        )

        exact = get_k2mova_36b_config()
        assert (exact.num_layers, exact.mova_num_dense_layers) == (48, 3)
        assert (exact.hidden_size, exact.num_attention_heads, exact.num_query_groups) == (
            2560,
            32,
            8,
        )
        assert (exact.mova_num_value_experts, exact.mova_router_topk) == (64, 4)
        assert (exact.num_moe_experts, exact.moe_router_topk) == (100, 8)
        assert exact.layernorm_epsilon == 1.0e-6
        assert exact.moe_router_dtype == "fp32"
        assert exact.moe_router_load_balancing_type == "none"
        assert exact.moe_aux_loss_coeff == 0.0
        assert exact.mova_router_load_balancing_type == "none"
        assert exact.mova_router_aux_loss_coeff == 0.0
        assert exact.attention_dropout == 0.0
        assert exact.hidden_dropout == 0.0
        assert exact.rotary_interleaved
        assert exact.xllm_router_compatibility
        assert exact.hetereogenous_dist_checkpoint
        assert not exact.heterogeneous_block_specs

    def test_dense_prefix_layout_is_preserved_with_virtual_pipeline_parallelism(self):
        config = _small_config(
            num_layers=48,
            mova_num_dense_layers=3,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=4,
            pipeline_dtype=torch.float32,
        )

        attention_modules = []
        for vp_stage in range(config.virtual_pipeline_model_parallel_size):
            for pp_rank in range(config.pipeline_model_parallel_size):
                block_spec = get_mova_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=False,
                    moe_grouped_gemm=False,
                    vp_stage=vp_stage,
                    pp_rank=pp_rank,
                )
                assert len(block_spec.layer_specs) == 6
                attention_modules.extend(
                    layer.submodules.self_attention.module for layer in block_spec.layer_specs
                )

        assert attention_modules[:3] == [SoftplusGatedSelfAttention] * 3
        assert attention_modules[3:] == [MoVASelfAttention] * 45

    def test_xllm_bf16_router_gemm_precedes_fp32_scoring(self):
        config = _small_config(
            xllm_router_compatibility=True,
            xllm_router_gemm_partitions=2,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
        )
        router = TopKRouter(
            config=config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.weight.data.copy_(
            torch.sin(
                torch.arange(router.weight.numel(), device="cuda", dtype=torch.float32) * 0.173
            )
            .view_as(router.weight)
            .to(torch.bfloat16)
            * 0.03
        )
        router.expert_bias.copy_(torch.tensor([0.002, -0.001, 0.003, -0.004], device="cuda"))
        hidden = (
            torch.cos(
                torch.arange(8 * config.hidden_size, device="cuda", dtype=torch.float32) * 0.097
            )
            .view(4, 2, config.hidden_size)
            .to(torch.bfloat16)
        )

        logits = router.gating(hidden)
        xllm_logits = sum(
            F.linear(hidden_part, weight_part).float()
            for hidden_part, weight_part in zip(
                hidden.chunk(config.xllm_router_gemm_partitions, dim=-1),
                router.weight.chunk(config.xllm_router_gemm_partitions, dim=-1),
            )
        )
        full_bf16_logits = F.linear(hidden, router.weight).float()
        fp32_logits = F.linear(hidden.float(), router.weight.float())
        torch.testing.assert_close(logits, xllm_logits, rtol=0.0, atol=0.0)
        assert not torch.equal(logits, full_bf16_logits)
        assert not torch.equal(logits, fp32_logits)

        probabilities, routing_map = router(hidden)
        scores = torch.sigmoid(logits).view(-1, config.num_moe_experts)
        selected = torch.topk(scores + router.expert_bias, config.moe_router_topk, dim=-1).indices
        selected_scores = torch.gather(scores, -1, selected)
        selected_scores = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        selected_scores = selected_scores * config.moe_router_topk_scaling_factor
        expected_probabilities = torch.zeros_like(scores).scatter(-1, selected, selected_scores)
        expected_map = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, selected, True)

        torch.testing.assert_close(routing_map, expected_map)
        torch.testing.assert_close(probabilities, expected_probabilities, rtol=2.0e-3, atol=2.0e-4)

    def test_xllm_aux_loss_counts_bias_adjusted_router_choices(self):
        config = _small_config(
            xllm_router_compatibility=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1.0,
        )
        router = TopKRouter(
            config=config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0], device="cuda"))
        captured = {}

        def capture_aux_loss(self, probs, scores_for_aux_loss, routing_map):
            del self, scores_for_aux_loss
            captured["routing_map"] = routing_map.detach().clone()
            return probs

        router._apply_aux_loss = MethodType(capture_aux_loss, router)
        logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0]], [[4.0, 3.0, 2.0, 1.0]]], device="cuda")

        _, actual_map = router.routing(logits)

        assert torch.equal(captured["routing_map"], actual_map)
        assert actual_map[:, 3].all()


def test_router_bias_update_supports_heterogeneous_router_shapes(monkeypatch):
    class FakeRouter(torch.nn.Module):
        def __init__(self, num_experts, update_rate):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(num_experts))
            self.register_buffer("local_tokens_per_expert", torch.ones(num_experts))
            self.expert_bias_update_rate = update_rate

    model = torch.nn.Module()
    model.value_router_a = FakeRouter(64, 1.0e-3)
    model.value_router_b = FakeRouter(64, 1.0e-3)
    model.mlp_router = FakeRouter(100, 2.0e-3)
    calls = []

    def fake_update(tokens_per_expert, expert_bias, update_rate):
        calls.append((tokens_per_expert.shape, expert_bias.shape, update_rate))
        return expert_bias + update_rate

    monkeypatch.setattr(finalize_model_grads_module, "get_updated_expert_bias", fake_update)
    config = SimpleNamespace(moe_router_bias_update_rate=9.0e-3)
    finalize_model_grads_module._update_router_expert_bias([model], config)

    assert calls == [
        (torch.Size([2, 64]), torch.Size([2, 64]), 1.0e-3),
        (torch.Size([1, 100]), torch.Size([1, 100]), 2.0e-3),
    ]
    torch.testing.assert_close(model.value_router_a.expert_bias, torch.full((64,), 1.0e-3))
    torch.testing.assert_close(model.mlp_router.expert_bias, torch.full((100,), 2.0e-3))


def test_mova_cli_uses_native_dropout_defaults():
    """Keep continuation defaults exact while preserving explicit CLI overrides."""

    import argparse

    from pretrain_mova import add_mova_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dropout", type=float, default=0.1)
    add_mova_args(parser)

    defaults = parser.parse_args([])
    assert defaults.attention_dropout == 0.0
    assert defaults.hidden_dropout == 0.0
    assert not defaults.mova_use_torch_rms_norm
    assert defaults.xllm_router_compatibility
    assert defaults.xllm_router_gemm_partitions == 1

    explicit = parser.parse_args(["--attention-dropout", "0.2", "--hidden-dropout", "0.3"])
    assert explicit.attention_dropout == 0.2
    assert explicit.hidden_dropout == 0.3

    disabled = parser.parse_args(["--no-xllm-router-compatibility"])
    assert not disabled.xllm_router_compatibility

    partitioned = parser.parse_args(["--xllm-router-gemm-partitions", "2"])
    assert partitioned.xllm_router_gemm_partitions == 2

    torch_rms_norm = parser.parse_args(["--mova-use-torch-rms-norm"])
    assert torch_rms_norm.mova_use_torch_rms_norm


@pytest.mark.skipif(Utils.world_size < 2, reason="requires two distributed ranks")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoVATensorParallel:
    @pytest.fixture(scope="function", autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(2, 1)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ("value_backend", "moe_router_fusion"),
        (("sequential", False), ("grouped_gemm", False), ("grouped_gemm", True)),
    )
    def test_routed_value_projection_tp2_matches_dense_reference(
        self, value_backend, moe_router_fusion
    ):
        grouped = value_backend == "grouped_gemm"
        dtype = torch.bfloat16 if grouped else torch.float32
        config = _small_config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            mova_router_enable_expert_bias=True,
            mova_value_backend=value_backend,
            moe_router_fusion=moe_router_fusion,
            bf16=grouped,
            params_dtype=dtype,
            pipeline_dtype=dtype,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        value_projection = MoVAValueProjection(
            config=config,
            submodules=_value_projection_spec(value_backend).submodules,
            input_size=config.hidden_size,
            output_size=config.num_query_groups * config.kv_channels,
            layer_number=2,
            pg_collection=pg_collection,
        ).cuda()
        tp_rank = torch.distributed.get_rank(group=pg_collection.tp)
        output_size = config.num_query_groups * config.kv_channels
        output_per_rank = output_size // 2

        router_weight = torch.arange(
            config.mova_num_value_experts * config.hidden_size, device="cuda", dtype=torch.float32
        ).view(config.mova_num_value_experts, config.hidden_size)
        router_weight = (torch.sin(router_weight * 0.07) * 0.2).to(dtype)
        reference_router_weight = router_weight.detach().clone().requires_grad_(True)
        expert_weights = torch.arange(
            config.mova_num_value_experts * output_size * config.hidden_size,
            device="cuda",
            dtype=torch.float32,
        ).view(config.mova_num_value_experts, output_size, config.hidden_size)
        expert_weights = (torch.cos(expert_weights * 0.03) * 0.1).to(dtype).requires_grad_(True)
        expert_bias = torch.tensor([0.03, -0.02, 0.01, 0.0], device="cuda")
        with torch.no_grad():
            value_projection.router.weight.copy_(router_weight)
            value_projection.router.expert_bias.copy_(expert_bias)
            if grouped:
                input_per_rank = config.hidden_size // 2
                input_slice = slice(tp_rank * input_per_rank, (tp_rank + 1) * input_per_rank)
                value_projection.experts.weight.copy_(
                    expert_weights[:, :, input_slice].transpose(1, 2)
                )
            else:
                output_slice = slice(tp_rank * output_per_rank, (tp_rank + 1) * output_per_rank)
                for expert_id, expert in enumerate(value_projection.experts.experts):
                    expert.weight.copy_(expert_weights[expert_id, output_slice])

        full_hidden = torch.arange(
            6 * 2 * config.hidden_size, device="cuda", dtype=torch.float32
        ).view(6, 2, config.hidden_size)
        full_hidden = torch.sin(full_hidden * 0.05).to(dtype)
        local_hidden = full_hidden.chunk(2, dim=0)[tp_rank].detach().clone().requires_grad_(True)
        reference_hidden = full_hidden.detach().clone().requires_grad_(True)

        output = value_projection(local_hidden)
        gathered_outputs = [torch.empty_like(output) for _ in range(2)]
        torch.distributed.all_gather(
            gathered_outputs, output.detach().contiguous(), group=pg_collection.tp
        )
        gathered_output = torch.cat(gathered_outputs, dim=-1)

        logits = F.linear(reference_hidden.float(), reference_router_weight.float())
        scores = torch.sigmoid(logits)
        _, selected = torch.topk(scores + expert_bias, config.mova_router_topk, dim=-1)
        selected_scores = torch.gather(scores, -1, selected)
        selected_scores = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        selected_scores = selected_scores * config.mova_router_topk_scaling_factor
        dense_scores = torch.zeros_like(scores).scatter(-1, selected, selected_scores)
        expert_outputs = torch.stack(
            [F.silu(F.linear(reference_hidden, weight)) for weight in expert_weights], dim=-2
        )
        reference = (expert_outputs * dense_scores.to(expert_outputs.dtype).unsqueeze(-1)).sum(
            dim=-2
        )
        rtol, atol = (2.0e-2, 2.0e-2) if grouped else (3.0e-5, 3.0e-6)
        torch.testing.assert_close(gathered_output, reference, rtol=rtol, atol=atol)

        output.float().square().sum().backward()
        reference.float().square().sum().backward()
        reference_local_grad = reference_hidden.grad.chunk(2, dim=0)[tp_rank]
        grad_rtol, grad_atol = (3.0e-2, 3.0e-2) if grouped else (8.0e-5, 8.0e-6)
        torch.testing.assert_close(
            local_hidden.grad, reference_local_grad, rtol=grad_rtol, atol=grad_atol
        )
        router_grad = value_projection.router.weight.grad.detach().clone()
        torch.distributed.all_reduce(router_grad, group=pg_collection.tp)
        torch.testing.assert_close(
            router_grad, reference_router_weight.grad, rtol=grad_rtol, atol=grad_atol
        )
        if grouped:
            torch.testing.assert_close(
                value_projection.experts.weight.grad,
                expert_weights.grad[:, :, input_slice].transpose(1, 2),
                rtol=grad_rtol,
                atol=grad_atol,
            )
        else:
            for expert_id, expert in enumerate(value_projection.experts.experts):
                torch.testing.assert_close(
                    expert.weight.grad,
                    expert_weights.grad[expert_id, output_slice],
                    rtol=grad_rtol,
                    atol=grad_atol,
                )


def _build_te_mova_attention(context_parallel_size: int) -> MoVASelfAttention:
    config = _small_config(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=32,
        mova_num_dense_layers=0,
        mova_value_backend="grouped_gemm",
        context_parallel_size=context_parallel_size,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
    )
    block_spec = get_mova_gpt_decoder_block_spec(
        config, use_transformer_engine=True, moe_grouped_gemm=False
    )
    attention_spec = block_spec.layer_specs[0].submodules.self_attention
    return build_module(attention_spec, config=config, layer_number=1).cuda()


def _packed_seq_params(cu_seqlens: list[int]) -> PackedSeqParams:
    cumulative = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    max_seqlen = int((cumulative[1:] - cumulative[:-1]).max())
    return PackedSeqParams(
        cu_seqlens_q=cumulative,
        cu_seqlens_kv=cumulative,
        cu_seqlens_q_padded=cumulative,
        cu_seqlens_kv_padded=cumulative,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )


@pytest.mark.skipif(Utils.world_size < 2, reason="requires two distributed ranks")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mova_cp2_packed_varlen_matches_single_rank_forward_and_backward():
    """Validate the production combination of CP and packed FlashAttention."""

    seed = 2468
    sequence_length = 64
    batch_size = 1
    # Unequal document lengths; each remains divisible by 2 * CP so TE can
    # apply its load-balanced THD partitioning without padding.
    cu_seqlens = [0, 8, 20, 40, 64]
    full_hidden = (
        torch.sin(
            torch.arange(sequence_length * batch_size * 64, dtype=torch.float32, device="cuda")
            * 0.013
        )
        .view(sequence_length, batch_size, 64)
        .to(torch.bfloat16)
    )

    try:
        Utils.initialize_model_parallel(1, 1, context_parallel_size=1)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        baseline_attention = _build_te_mova_attention(context_parallel_size=1)
        baseline_hidden = full_hidden.detach().clone().requires_grad_(True)
        baseline_output, baseline_bias = baseline_attention(
            baseline_hidden, attention_mask=None, packed_seq_params=_packed_seq_params(cu_seqlens)
        )
        assert baseline_bias is None
        baseline_output.float().sum().backward()
        baseline_grad = baseline_hidden.grad.detach()
        baseline_output = baseline_output.detach()

        del baseline_attention
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        parallel_attention = _build_te_mova_attention(context_parallel_size=2)
        cp_group = parallel_state.get_context_parallel_group()
        cumulative = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
        cp_rank = torch.distributed.get_rank(group=cp_group)
        cp_size = torch.distributed.get_world_size(group=cp_group)
        partitioned_indices = tex.thd_get_partitioned_indices(
            cumulative, sequence_length, cp_size, cp_rank
        )
        local_hidden = full_hidden.index_select(0, partitioned_indices)
        local_hidden = local_hidden.detach().clone().requires_grad_(True)
        parallel_output, parallel_bias = parallel_attention(
            local_hidden, attention_mask=None, packed_seq_params=_packed_seq_params(cu_seqlens)
        )
        assert parallel_bias is None
        parallel_output.float().sum().backward()

        expected_output = baseline_output.index_select(0, partitioned_indices)
        expected_grad = baseline_grad.index_select(0, partitioned_indices)
        output_error = (parallel_output.float() - expected_output.float()).abs()
        output_rel_l2 = output_error.norm() / expected_output.float().norm().clamp_min(1.0e-6)
        grad_error = (local_hidden.grad.float() - expected_grad.float()).abs()
        grad_rel_l2 = grad_error.norm() / expected_grad.float().norm().clamp_min(1.0e-6)
        assert output_error.max() < 3.0e-2
        assert output_rel_l2 < 1.0e-2
        # CP FlashAttention changes BF16 reduction order in backward. Bound
        # both the worst element and aggregate error instead of using a large
        # elementwise relative tolerance around near-zero gradients.
        assert grad_error.max() < 1.0e-1
        assert grad_rel_l2 < 3.0e-2
    finally:
        Utils.destroy_model_parallel()
