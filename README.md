<div align="center">
  <img src="https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docs/images/MinerU-logo.png" width="220px" />
  <h1>MinerU Optimized Fork</h1>
  <p>
    基于 <a href="https://github.com/opendatalab/MinerU">opendatalab/MinerU</a> 的个人优化分支<br/>
    聚焦性能提升、工程化改进与可测试性
  </p>
</div>

---

## 这是什么？

本仓库是 MinerU 的一个**个人 Fork**，在原版基础上做了一系列面向工程落地与解析性能的优化。所有改动仅保留在本分支中，不会提交到上游仓库。

## 已完成的优化

| 优先级 | 优化项 | 主要改动 | 目标收益 |
|---|---|---|---|
| P1 | 表格 OCR 结果排序 | 用复合键 `(round(y/10), x)` 一次稳定排序，移除 O(n²) 冒泡修正 | 大表格排序 5-20 倍提速 |
| P2 | 重复计算与内存分配 | 提取循环外重复计算；padding 改用 `np.full`/`cv2.copyMakeBorder` | 降低 10-30% 临时内存分配 |
| P3 | 内联对象提取 | 用网格空间索引替代 O(n×m) 嵌套循环 | 复杂页面提取耗时降低 60-80% |
| P4 | 动态 VRAM 阈值 | `clean_vram` 根据 `batch_ratio` 动态计算阈值 | 减少无效显存清理 |
| P5 | 移除 `requests` 依赖 | 统一使用 `httpx`；从核心依赖中删除 `requests` | 精简依赖 |
| P6 | 单元测试与覆盖率 | 新增 `test_batch_analyze.py`（15 个用例），覆盖率阈值从 20% 提升至 50% | 增强核心函数可测试性 |
| P7 | 模型初始化错误处理 | `exit(1)` 改为抛出自定义 `ModelInitError` | 服务启动更鲁棒 |
| P8 | 表格 OCR 批处理 | 表格 det/rec 按分辨率分组走批处理分支 | 多表格 PDF 耗时降低 20-40% |
| P9 | 异步 HTTP 上游提交 | `submit_payload_to_upstream` 改用 `httpx.AsyncClient` | 提升并发路由性能 |
| P10 | 工程化基础设施 | CI lint 工作流、依赖分组（`[api]`）、reportlab 版本约束、rerun 指数退避等 | 提升工程质量 |

## 安装

```bash
# 克隆本仓库
git clone https://github.com/davidjirou/MinerU.git
cd MinerU

# 推荐通过 uv 安装
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[all]"
```

## 使用

基本用法与原版一致：

```bash
# GPU 环境
mineru -p <input_path> -o <output_path>

# CPU 环境
mineru -p <input_path> -o <output_path> -b pipeline
```

详细用法请参考原版文档：[MinerU Usage Guide](https://opendatalab.github.io/MinerU/usage/)

## 测试

```bash
# 运行新增单元测试
pytest tests/unittest/test_batch_analyze.py -v

# 运行覆盖率检查
python tests/clean_coverage.py
coverage run
python tests/get_coverage.py
```

## 与原版的差异说明

- 本分支核心解析结果与原版保持一致，仅对内部实现做了性能与工程化优化。
- 移除了 `requests`，统一使用 `httpx`。
- `fastapi`、`uvicorn`、`python-multipart` 被移动到 `[api]` 可选依赖组。
- 模型初始化失败不再直接退出进程，而是抛出异常，方便上层降级处理。

## 许可证

与上游一致，采用 [MinerU Open Source License](https://github.com/davidjirou/MinerU/blob/master/LICENSE.md)。

## 致谢

感谢 [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 团队开源的优秀文档解析工具。
