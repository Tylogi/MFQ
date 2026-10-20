"""Import-light AlphaQ profiles and UD limits shared by the CLI and allocator."""

ALPHAQ_PROFILES = (
    "NVQ1-S", "NVQ1-L",
    "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
    "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    "NINT4", "NINT5", "NINT6", "NINT8",
)

UD_MODELS = {
    'qwen4_exp_text': dict(
        config=dict(hidden_size=2560, num_hidden_layers=48, num_experts=512,
                    num_experts_per_tok=10, vocab_size=248320),
        parameters=176943899520, directory='', prefix='',
        caps=dict(IQ1_S=72546461344, IQ1_M=74538755776, Q2_K_XL=78869128864,
            IQ3_XXS=81961823936, Q3_K_XL=89986353824, IQ4_XS=93682584224,
            Q4_K_XL=111334654784, Q5_K_XL=158286406650, Q6_K_XL=169165382688)),
}
UD_TIERS = tuple(dict.fromkeys(tier for model in UD_MODELS.values() for tier in model['caps']))

__all__ = ["ALPHAQ_PROFILES", "UD_MODELS", "UD_TIERS"]
