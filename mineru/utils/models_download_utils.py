import os
import hashlib
import requests
from typing import List, Union
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from mineru.utils.enum_class import ModelPath


def _sha256sum(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
def get_file_from_repos(relative_path: str, repo_mode='pipeline') -> Union[str, str]:
    """
    支持文件或目录的可靠下载。
    - 如果输入文件: 返回本地文件绝对路径
    - 如果输入目录: 返回本地缓存下与 relative_path 同结构的相对路径字符串
    :param repo_mode: 指定仓库模式，'pipeline' 或 'vlm'
    :param relative_path: 文件或目录相对路径
    :return: 本地文件绝对路径或相对路径
    """
    model_source = os.getenv('MINERU_MODEL_SOURCE', None)

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

    input_clean = relative_path.strip('/')
    # 获取huggingface云端仓库文件树
    try:
        # 获取仓库信息,包含文件元数据
        info = model_info(repo, files_metadata=True)
        # 构建文件字典
        siblings_dict = {f.rfilename: f for f in info.siblings}
    except Exception as e:
        siblings_dict = {}
        print(f"[Warn] 获取 Huggingface 仓库结构失败，错误: {e}")
    # 1. 文件还是目录拓展
    if input_clean in siblings_dict and not siblings_dict[input_clean].rfilename.endswith("/"):
        is_file = True
        all_paths = [input_clean]
    else:
        is_file = False
        all_paths = [k for k in siblings_dict if k.startswith(input_clean + "/") and not k.endswith("/")]
    # 若获取不到siblings（如 Huggingface 失败，直接按输入处理）
    if not all_paths:
        is_file = os.path.splitext(input_clean)[1] != ""
        all_paths = [input_clean] if is_file else []
    cache_home = str(HUGGINGFACE_HUB_CACHE)
    # 判断主逻辑
    output_files = []
    # ---- Huggingface 分支 ----
    hf_ok = False
    for relpath in all_paths:
        ok = False
        if relpath in siblings_dict:
            meta = siblings_dict[relpath]
            sha256 = ""
            if meta.lfs:
                sha256 = meta.lfs.sha256
            try:
                # 不允许下载线上文件，只寻找本地文件
                file_path = hf_hub_download(repo_id=repo, filename=relpath, local_files_only=True)
                if sha256 and os.path.exists(file_path):
                    if _sha256sum(file_path) == sha256:
                        ok = True
                        output_files.append(file_path)
            except Exception as e:
                print(f"[Info] Huggingface {relpath} 获取失败: {e}")
            if not hf_ok:
                file_path = hf_hub_download(repo_id=repo, filename=relpath, force_download=False)
                print("file_path = ", file_path)
                if sha256 and _sha256sum(file_path) != sha256:
                    raise ValueError(f"Huggingface下载后校验失败: {relpath}")
                ok = True
                output_files.append(file_path)
            hf_ok = hf_ok and ok
    # ---- ModelScope 分支 ----
    for relpath in all_paths:
        if hf_ok:
            break
        if "/" in repo:
            org_name, model_name = repo.split("/", 1)
        else:
            org_name, model_name = "modelscope", repo
        # 目录结构: 缓存/home/modelscope-fallback/org/model/相对路径
        target_dir = os.path.join(cache_home, "modelscope-fallback", org_name, model_name, os.path.dirname(relpath))
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, os.path.basename(relpath))
        remote_len = 0
        sha256 = ""
        try:
            get_meta_url = f"https://www.modelscope.cn/api/v1/models/{org_name}/{model_name}/repo/raw?Revision=master&FilePath={relpath}&Needmeta=true"
            resp = requests.get(get_meta_url, timeout=15)
            if resp.ok:
                remote_len = resp.json()["Data"]["MetaContent"]["Size"]
                sha256 = resp.json()["Data"]["MetaContent"]["Sha256"]
        except Exception as e:
            print(f"[Info] modelscope {relpath} 获取失败: {e}")
        ok_local = False
        if remote_len > 0 and os.path.exists(local_path):
            if sha256 == _sha256sum(local_path):
                output_files.append(local_path)
                ok_local = True
        if not ok_local:
            try:
                modelscope_url = f"https://www.modelscope.cn/api/v1/models/{org_name}/{model_name}/repo?Revision=master&FilePath={relpath}"
                with requests.get(modelscope_url, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in resp.iter_content(1024*1024):
                            if chunk:
                                f.write(chunk)
                if remote_len == 0 or os.path.getsize(local_path) == remote_len:
                    output_files.append(local_path)
                    ok_local = True
            except Exception as e:
                print(f"[Error] ModelScope下载失败: {relpath} {e}")
    if not output_files:
        raise FileNotFoundError(f"{relative_path} 在 Huggingface 和 ModelScope 都未能获取")
    if is_file:
        return output_files[0]
    else:
        # 输入是文件，只返回路径字符串
        return os.path.dirname(os.path.abspath(output_files[0]))
if __name__ == '__main__':
    path1 = get_file_from_repos("models/README.md")
    print("本地文件绝对路径:", path1)
    path2 = get_file_from_repos("models/OCR/paddleocr_torch/")
    print("本地文件绝对路径:", path2)