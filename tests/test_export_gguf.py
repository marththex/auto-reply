"""Behavior of the adapter -> GGUF export filtering (text-only deployment)."""

from autoreply.training.export_gguf import build_metadata, keep_tensor, rename_tensor


class TestTensorFiltering:
    def test_language_model_tensors_are_kept(self):
        name = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        assert keep_tensor(name)

    def test_tower_and_embed_tensors_are_dropped(self):
        for name in (
            "base_model.model.model.audio_tower.layers.0.attention.q_proj.lora_A.weight",
            "base_model.model.model.vision_tower.blocks.3.mlp.fc1.lora_B.weight",
            "base_model.model.model.embed_audio.projection.lora_A.weight",
            "base_model.model.model.embed_vision.projection.lora_B.weight",
        ):
            assert not keep_tensor(name), name

    def test_rename_strips_language_model_segment(self):
        name = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        assert rename_tensor(name) == (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )

    def test_rename_leaves_other_names_untouched(self):
        name = "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"
        assert rename_tensor(name) == name


class TestMetadata:
    def test_metadata_records_provenance(self):
        meta = build_metadata(
            adapter_name="my-lora-v2-epoch3",
            source_checkpoint="checkpoints/checkpoint-117",
            eval_loss=1.958,
            base_gguf="gemma-4-e2b-base-q5.gguf",
        )
        assert meta["adapter_name"] == "my-lora-v2-epoch3"
        assert meta["source_checkpoint"] == "checkpoints/checkpoint-117"
        assert meta["eval_loss"] == 1.958
        assert meta["base_gguf"] == "gemma-4-e2b-base-q5.gguf"
        assert meta["exported_at"]  # ISO timestamp present
