from transformers import Qwen2Config


class Mineru2QwenConfig(Qwen2Config):
    model_type = "mineru2_qwen"

    def __init__(
        self,
        ignore_index=-100,
        image_aspect_ratio="square_anyres_max_9",
        image_grid_pinpoints="(1x1),...,(4x4)",
        image_token_index=151646,
        mm_hidden_size=1152,
        mm_patch_merge_type="spatial_unpad",
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_feature="full",
        mm_vision_select_layer=-2,
        mm_vision_tower="google/siglip-so400m-patch14-384",
        tie_word_embeddings=False,
        tokenizer_model_max_length=16384,
        tokenizer_padding_side="right",
        unfreeze_mm_vision_tower=True,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.image_token_index = image_token_index
        self.mm_hidden_size = mm_hidden_size
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_tower = mm_vision_tower
        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.tokenizer_padding_side = tokenizer_padding_side
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
