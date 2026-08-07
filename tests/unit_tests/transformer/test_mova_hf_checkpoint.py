# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for strict, streaming xLLM MoVA Hugging Face checkpoint reads."""

import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.mova_hf_checkpoint import (
    HF_COMPLETION_MARKER,
    HF_CONFIG_NAME,
    HF_INDEX_NAME,
    HFXllmMoVACheckpoint,
    expected_hf_xllm_mova_layer_keys,
    hf_xllm_config_to_source_config,
    normalize_hf_xllm_mova_global_state,
    normalize_hf_xllm_mova_layer_state,
)
from tools.checkpoint.convert_xllm_mova import _apply_source_architecture


def _hf_config() -> dict:
    return {
        "model_type": "xllm",
        "apply_attn_gate": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_gate_func": "softplus",
        "decoder_sparse_step": 1,
        "hidden_act": "silu",
        "hidden_size": 8,
        "head_dim": 4,
        "intermediate_size": 16,
        "layernorm_num_groups": 2,
        "max_position_embeddings": 128,
        "mlp_only_layers": [0],
        "moe_gate_bias": True,
        "moe_intermediate_size": 4,
        "norm_topk_prob": True,
        "num_attention_heads": 2,
        "num_dense_layers": 1,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_shared_experts": 1,
        "num_values": 2,
        "num_values_per_tok": 1,
        "query_key_norm": False,
        "rms_norm_eps": 1.0e-6,
        "rope_head_dim": 4,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "router_aux_loss_coef": 0.001,
        "router_scaling_factor": 2.5,
        "router_score_func": "sigmoid",
        "sliding_window": None,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "vocab_size": 32,
    }


def _state_for_keys(keys: set[str], start: int = 0) -> dict[str, torch.Tensor]:
    config = _hf_config()
    state = {}
    for index, key in enumerate(sorted(keys)):
        value = start + index
        if key.endswith("self_attn.q_proj.weight"):
            shape = (config["num_attention_heads"] * config["head_dim"], config["hidden_size"])
            state[key] = torch.arange(value, value + shape[0] * shape[1]).reshape(shape)
        elif key.endswith("self_attn.k_proj.weight"):
            shape = (config["num_key_value_heads"] * config["head_dim"], config["hidden_size"])
            state[key] = torch.arange(value, value + shape[0] * shape[1]).reshape(shape)
        else:
            state[key] = torch.tensor([value], dtype=torch.float32)
    return state


def _unpermute_reference(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    config = _hf_config()
    return (
        weight.reshape(num_heads, 2, config["head_dim"] // 2, config["hidden_size"])
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def _write_checkpoint(tmp_path, *, include_done: bool = True):
    config = _hf_config()
    layer_0_keys = expected_hf_xllm_mova_layer_keys(config, 0)
    layer_1_keys = expected_hf_xllm_mova_layer_keys(config, 1)
    global_keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    shard_0 = "pytorch_model-00001-of-00002.bin"
    shard_1 = "pytorch_model-00002-of-00002.bin"
    state_0 = _state_for_keys(layer_0_keys)
    state_1 = _state_for_keys(layer_1_keys | global_keys, start=1000)
    torch.save(state_0, tmp_path / shard_0)
    torch.save(state_1, tmp_path / shard_1)
    (tmp_path / HF_CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")
    weight_map = {key: shard_0 for key in layer_0_keys}
    weight_map.update({key: shard_1 for key in layer_1_keys | global_keys})
    (tmp_path / HF_INDEX_NAME).write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}), encoding="utf-8"
    )
    if include_done:
        (tmp_path / HF_COMPLETION_MARKER).touch()
    return config, state_0, state_1


def test_hf_config_translation_preserves_mova_architecture():
    source = hf_xllm_config_to_source_config(
        _hf_config(),
        router_gemm_partitions=2,
        router_bias_update_rate=1.0e-3,
        router_load_balancing_type="dot",
        router_aux_loss_coeff=1.0e-4,
    )
    model = source["model"]
    assert source["seq_len"] == 128
    assert source["model_parallel_size"] == 2
    assert source["moe_router_load_balancing_type"] == "dot"
    # The xLLM HF exporter currently writes its bridge default (0.001), not
    # the native training value. Conversion must use the explicit source run.
    assert source["moe_aux_loss_coeff"] == 1.0e-4
    assert (model["num_layers"], model["num_dense_layers"]) == (2, 1)
    assert (model["num_heads"], model["num_kv_heads"], model["head_dim"]) == (2, 1, 4)
    assert (model["num_values"], model["num_activated_values"]) == (2, 1)
    assert (model["num_experts"], model["num_activated_experts"]) == (3, 2)
    assert model["moe_router_bias_update_rate"] == 1.0e-3


def test_native_aux_loss_is_distributed_across_sparse_routers(tmp_path):
    source = hf_xllm_config_to_source_config(
        _hf_config(),
        router_gemm_partitions=2,
        router_bias_update_rate=1.0e-3,
        router_load_balancing_type="dot",
        router_aux_loss_coeff=1.0e-4,
    )
    args = SimpleNamespace(mcore_output=tmp_path)

    _apply_source_architecture(args, source)

    # One sparse layer contributes one MoVA and one FFN router loss.
    assert args.moe_aux_loss_coeff == pytest.approx(5.0e-5)
    assert args.mova_router_aux_loss_coeff == pytest.approx(5.0e-5)


def test_hf_config_translation_rejects_unrepresented_forward_contract():
    config = _hf_config()
    config["use_sliding_window"] = True
    with pytest.raises(ValueError, match="use_sliding_window"):
        hf_xllm_config_to_source_config(
            config,
            router_gemm_partitions=2,
            router_bias_update_rate=1.0e-3,
            router_load_balancing_type="dot",
            router_aux_loss_coeff=1.0e-4,
        )

    config = _hf_config()
    config["num_dense_layers"] = config["num_hidden_layers"]
    config["mlp_only_layers"] = list(range(config["num_hidden_layers"]))
    with pytest.raises(ValueError, match="num_dense_layers"):
        hf_xllm_config_to_source_config(
            config,
            router_gemm_partitions=2,
            router_bias_update_rate=1.0e-3,
            router_load_balancing_type="dot",
            router_aux_loss_coeff=1.0e-4,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("router_scaling_factor", float("nan")), ("router_aux_loss_coef", float("inf"))),
)
def test_hf_config_translation_rejects_nonfinite_router_values(field, value):
    config = _hf_config()
    config[field] = value
    with pytest.raises(ValueError, match=field):
        hf_xllm_config_to_source_config(
            config,
            router_gemm_partitions=2,
            router_bias_update_rate=1.0e-3,
            router_load_balancing_type="dot",
            router_aux_loss_coeff=1.0e-4,
        )


def test_hf_config_translation_rejects_nonfinite_bias_update_rate():
    with pytest.raises(ValueError, match="router_bias_update_rate"):
        hf_xllm_config_to_source_config(
            _hf_config(),
            router_gemm_partitions=2,
            router_bias_update_rate=float("nan"),
            router_load_balancing_type="dot",
            router_aux_loss_coeff=1.0e-4,
        )


@pytest.mark.parametrize(
    ("balance_type", "aux_coeff", "match"),
    (
        ("global", 1.0e-4, "must be none or dot"),
        (None, 1.0e-4, "must be zero"),
        ("dot", 0.0, "must be positive"),
        ("dot", float("nan"), "must be non-negative"),
    ),
)
def test_hf_config_translation_validates_native_aux_loss(balance_type, aux_coeff, match):
    with pytest.raises(ValueError, match=match):
        hf_xllm_config_to_source_config(
            _hf_config(),
            router_gemm_partitions=2,
            router_bias_update_rate=1.0e-3,
            router_load_balancing_type=balance_type,
            router_aux_loss_coeff=aux_coeff,
        )


def test_hf_layer_and_global_names_normalize_to_native_contract():
    config = _hf_config()
    dense_source = _state_for_keys(expected_hf_xllm_mova_layer_keys(config, 0))
    dense = normalize_hf_xllm_mova_layer_state(dense_source, config, 0)
    torch.testing.assert_close(
        dense["layers.0.attention.wq.weight"],
        _unpermute_reference(
            dense_source["model.layers.0.self_attn.q_proj.weight"], config["num_attention_heads"]
        ),
    )
    assert dense["layers.0.nffn.fc2.weight"] is dense_source["model.layers.0.mlp.down_proj.weight"]

    sparse_source = _state_for_keys(expected_hf_xllm_mova_layer_keys(config, 1))
    sparse = normalize_hf_xllm_mova_layer_state(sparse_source, config, 1)
    torch.testing.assert_close(
        sparse["layers.1.mova.wk.weight"],
        _unpermute_reference(
            sparse_source["model.layers.1.self_attn.k_proj.weight"], config["num_key_value_heads"]
        ),
    )
    assert sparse["layers.1.mova.wv.weight"].shape == (2, 1)
    assert sparse["layers.1.moe.experts.weight1"].shape == (3, 1)
    assert sparse["layers.1.moe.experts.weight2"].shape == (3, 1)
    assert sparse["layers.1.moe.experts.weight3"].shape == (3, 1)
    torch.testing.assert_close(
        sparse["layers.1.mova.wv.weight"][0],
        sparse_source["model.layers.1.self_attn.v_experts.0.weight"],
    )

    global_source = _state_for_keys(
        {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    )
    globals_ = normalize_hf_xllm_mova_global_state(global_source)
    assert globals_["embed.weight"] is global_source["model.embed_tokens.weight"]
    assert globals_["output.output.weight"] is global_source["lm_head.weight"]


def test_hf_checkpoint_reader_validates_index_and_streams_one_shard(tmp_path):
    config, state_0, state_1 = _write_checkpoint(tmp_path)
    reader = HFXllmMoVACheckpoint(tmp_path)
    assert reader.num_layers == 2
    assert reader.layer_shard(0) != reader.layer_shard(1)
    assert reader.layer_shard(1) == reader.global_shard
    with pytest.raises(IndexError, match="outside"):
        reader.layer_shard(-1)

    dense = reader.load_layer(0)
    torch.testing.assert_close(
        dense["layers.0.attention.wq.weight"],
        _unpermute_reference(
            state_0["model.layers.0.self_attn.q_proj.weight"], config["num_attention_heads"]
        ),
    )
    sparse = reader.load_layer(1, include_globals=True)
    torch.testing.assert_close(
        sparse["layers.1.mova.router.bias"], state_1["model.layers.1.self_attn.v_router.bias"]
    )
    torch.testing.assert_close(sparse["output.final_norm.weight"], state_1["model.norm.weight"])
    source_config = reader.source_config(
        router_gemm_partitions=2,
        router_bias_update_rate=1.0e-3,
        router_load_balancing_type="dot",
        router_aux_loss_coeff=1.0e-4,
    )
    assert source_config["model"]["vocab_size"] == config["vocab_size"]


def test_hf_checkpoint_reader_requires_completion_marker(tmp_path):
    _write_checkpoint(tmp_path, include_done=False)
    with pytest.raises(FileNotFoundError, match="completion marker"):
        HFXllmMoVACheckpoint(tmp_path)


def test_hf_checkpoint_reader_rejects_index_drift(tmp_path):
    _write_checkpoint(tmp_path)
    index_path = tmp_path / HF_INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"].pop("model.layers.0.self_attn.q_proj.weight")
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        HFXllmMoVACheckpoint(tmp_path)


def test_hf_checkpoint_reader_rejects_shard_contents_not_in_index(tmp_path):
    _write_checkpoint(tmp_path)
    reader = HFXllmMoVACheckpoint(tmp_path)
    shard_path = tmp_path / reader.layer_shard(0)
    state = torch.load(shard_path, weights_only=True)
    state["not.in.index"] = torch.ones(1)
    torch.save(state, shard_path)
    with pytest.raises(ValueError, match="disagrees with its index"):
        reader.load_layer(0)
