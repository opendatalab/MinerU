import lmdeploy

lmdeploy_check_vl_llm = lmdeploy.archs.check_vl_llm


def custom_check_vl_llm(config: dict) -> bool:
    arch = config['architectures'][0]
    if arch in ["Mineru2QwenForCausalLM"]:
        return True
    return lmdeploy_check_vl_llm(config)


lmdeploy.archs.check_vl_llm = custom_check_vl_llm