import os
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

from mineru.utils.config_reader import get_local_models_dir
from mineru.utils.enum_class import ModelPath

def auto_download_and_get_model_root_path(relative_path: str, repo_mode='pipeline') -> str:
    """
    支持文件或目录的可靠下载。
    - 如果输入文件: 返回本地文件绝对路径
    - 如果输入目录: 返回本地缓存下与 relative_path 同结构的相对路径字符串
    :param repo_mode: 指定仓库模式，'pipeline' 或 'vlm'
    :param relative_path: 文件或目录相对路径
    :return: 本地文件绝对路径或相对路径
    """
    model_source = os.getenv('MINERU_MODEL_SOURCE', "huggingface")

    if model_source == 'local':
        local_models_config = get_local_models_dir()
        root_path = local_models_config.get(repo_mode, None)
        if not root_path:
            raise ValueError(f"Local path for repo_mode '{repo_mode}' is not configured.")
        return root_path

    # 建立仓库模式到路径的映射
    repo_mapping = {
        'pipeline': {
            'huggingface': ModelPath.pipeline_root_hf,
            'modelscope': ModelPath.pipeline_root_modelscope,
            'default': ModelPath.pipeline_root_hf
        },
        'vlm': {
            'huggingface': ModelPath.vlm_root_hf,
            'modelscope': ModelPath.vlm_root_modelscope,
            'default': ModelPath.vlm_root_hf
        }
    }

    if repo_mode not in repo_mapping:
        raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")

    # 如果没有指定model_source或值不是'modelscope'，则使用默认值
    repo = repo_mapping[repo_mode].get(model_source, repo_mapping[repo_mode]['default'])


    if model_source == "huggingface":
        snapshot_download = hf_snapshot_download
    elif model_source == "modelscope":
        snapshot_download = ms_snapshot_download
    else:
        raise ValueError(f"未知的仓库类型: {model_source}")

    cache_dir = None

    if repo_mode == 'pipeline':
        relative_path = relative_path.strip('/')
        cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path+"/*"])
    elif repo_mode == 'vlm':
        # VLM 模式下，根据 relative_path 的不同处理方式
        if relative_path == "/":
            cache_dir = snapshot_download(repo)
        else:
            relative_path = relative_path.strip('/')
            cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path+"/*"])

    if not cache_dir:
        raise FileNotFoundError(f"Failed to download model: {relative_path} from {repo}")
    return cache_dir


if __name__ == '__main__':
    path1 = "models/README.md"
    root = auto_download_and_get_model_root_path(path1)
    print("本地文件绝对路径:", os.path.join(root, path1))