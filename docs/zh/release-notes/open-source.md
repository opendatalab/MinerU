# 更新记录
- 2025/09/26 2.5.4 发布
  - 🎉🎉 MinerU2.5[技术报告](https://arxiv.org/abs/2509.22186)现已发布，欢迎阅读全面了解其模型架构、训练策略、数据工程和评测结果。
  - 修复部分`pdf`文件被识别成`ai`文件导致无法解析的问题

- 2025/09/20 2.5.3 发布
  - 依赖版本范围调整，使得Turing及更早架构显卡可以使用vLLM加速推理MinerU2.5模型。
  - `pipeline`后端对torch 2.8.0的一些兼容性修复。
  - 降低vLLM异步后端默认的并发数，降低服务端压力以避免高压导致的链接关闭问题。
  - 更多兼容性相关内容详见[公告](https://github.com/opendatalab/MinerU/discussions/3547)

- 2025/09/19 2.5.2 发布
  我们正式发布 MinerU2.5，当前最强文档解析多模态大模型。仅凭 1.2B 参数，MinerU2.5 在 OmniDocBench 文档解析评测中，精度已全面超越 Gemini2.5-Pro、GPT-4o、Qwen2.5-VL-72B等顶级多模态大模型，并显著领先于主流文档解析专用模型（如 dots.ocr, MonkeyOCR, PP-StructureV3 等）。
  模型已发布至[HuggingFace](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)和[ModelScope](https://modelscope.cn/models/opendatalab/MinerU2.5-2509-1.2B)平台，欢迎大家下载使用！
  - 核心亮点
    - 极致能效，性能SOTA: 以 1.2B 的轻量化规模，实现了超越百亿乃至千亿级模型的SOTA性能，重新定义了文档解析的能效比。
    - 先进架构，全面领先: 通过 “两阶段推理” (解耦布局分析与内容识别) 与 原生高分辨率架构 的结合，在布局分析、文本识别、公式识别、表格识别及阅读顺序五大方面均达到 SOTA 水平。
  - 关键能力提升
    - 布局检测: 结果更完整，精准覆盖页眉、页脚、页码等非正文内容；同时提供更精准的元素定位与更自然的格式还原（如列表、参考文献）。
    - 表格解析: 大幅优化了对旋转表格、无线/少线表、以及长难表格的解析能力。
    - 公式识别: 显著提升中英混合公式及复杂长公式的识别准确率，大幅改善数学类文档解析能力。

  此外，伴随vlm 2.5的发布，我们对仓库做出一些调整：
  - vlm后端升级至2.5版本，支持MinerU2.5模型，不再兼容MinerU2.0-2505-0.9B模型，最后一个支持2.0模型的版本为mineru-2.2.2。
  - vlm推理相关代码已移至[mineru_vl_utils](https://github.com/opendatalab/mineru-vl-utils),降低与mineru主仓库的耦合度，便于后续独立迭代。
  - vlm加速推理框架从`sglang`切换至`vllm`,并实现对vllm生态的完全兼容，使得用户可以在任何支持vllm框架的平台上使用MinerU2.5模型并加速推理。
  - 由于vlm模型的重大升级，支持更多layout type，因此我们对解析的中间文件`middle.json`和结果文件`content_list.json`的结构做出一些调整，请参考[文档](https://opendatalab.github.io/MinerU/zh/reference/output_files/)了解详情。

  其他仓库优化：
  - 移除对输入文件的后缀名白名单校验，当输入文件为PDF文档或图片时，对文件的后缀名不再有要求，提升易用性。

<details>
  <summary>历史日志</summary>

  <details>
    <summary>2025/09/10 2.2.2 发布</summary>
    <ul>
      <li>修复新的表格识别模型在部分表格解析失败时影响整体解析任务的问题</li>
    </ul>
  </details>  

  <details>
    <summary>2025/09/08 2.2.1 发布</summary>
    <ul>
      <li>修复使用模型下载命令时，部分新增模型未下载的问题</li>
    </ul>
  </details>  

  <details>
    <summary>2025/09/05 2.2.0 发布</summary>
    <ul>
      <li>
        主要更新
        <ul>
          <li>在这个版本我们重点提升了表格的解析精度，通过引入新的<a href="https://github.com/RapidAI/TableStructureRec">有线表识别模型</a>和全新的混合表格结构解析算法，显著提升了<code>pipeline</code>后端的表格识别能力。</li>
          <li>另外我们增加了对跨页表格合并的支持，这一功能同时支持<code>pipeline</code>和<code>vlm</code>后端，进一步提升了表格解析的完整性和准确性。</li>
        </ul>
      </li>
      <li>
        其他更新
        <ul>
          <li><code>pipeline</code>后端增加270度旋转的表格解析能力，现已支持0/90/270度三个方向的表格解析</li>
          <li><code>pipeline</code>增加对泰文、希腊文的ocr能力支持，并更新了英文ocr模型至最新，英文识别精度提升11%，泰文识别模型精度 82.68%，希腊文识别模型精度 89.28%（by PPOCRv5）</li>
          <li>在输出的<code>content_list.json</code>中增加了<code>bbox</code>字段(映射至0-1000范围内)，方便用户直接获取每个内容块的位置信息</li>
          <li>移除<code>pipeline_old_linux</code>安装可选项，不再支持老版本的Linux系统如<code>Centos 7</code>等，以便对<code>uv</code>的<code>sync</code>/<code>run</code>等命令进行更好的支持</li>
        </ul>
      </li>
    </ul>
  </details>

  <details>
    <summary>2025/08/01 2.1.10 发布</summary>
    <ul>
      <li>修复<code>pipeline</code>后端因block覆盖导致的解析结果与预期不符 #3232</li>
    </ul>
  </details>

  <details>
    <summary>2025/07/30 2.1.9 发布</summary>
    <ul>
      <li><code>transformers</code> 4.54.1 版本适配</li>
    </ul>
  </details>

  <details>
    <summary>2025/07/28 2.1.8 发布</summary>
    <ul>
      <li><code>sglang</code> 0.4.9.post5 版本适配</li>
    </ul>
  </details>

  <details>
    <summary>2025/07/27 2.1.7 发布</summary>
    <ul>
      <li><code>transformers</code> 4.54.0 版本适配</li>
    </ul>
  </details>

  <details>
    <summary>2025/07/26 2.1.6 发布</summary>
    <ul>
      <li>修复<code>vlm</code>后端解析部分手写文档时的表格异常问题</li>
      <li>修复文档旋转时可视化框位置漂移问题 #3175</li>
    </ul>
  </details>

  <details>
    <summary>2025/07/24 2.1.5 发布</summary>
    <ul>
      <li><code>sglang</code> 0.4.9 版本适配，同步升级dockerfile基础镜像为sglang 0.4.9.post3</li>
    </ul>
  </details>

  <details>
    <summary>2025/07/23 2.1.4 发布</summary>
    <ul>
      <li><strong>bug修复</strong>
        <ul>
          <li>修复<code>pipeline</code>后端中<code>MFR</code>步骤在某些情况下显存消耗过大的问题 #2771</li>
          <li>修复某些情况下<code>image</code>/<code>table</code>与<code>caption</code>/<code>footnote</code>匹配不准确的问题 #3129</li>
        </ul>
      </li>
    </ul>
  </details>

  <details>
    <summary>2025/07/16 2.1.1 发布</summary>
    <ul>
      <li><strong>bug修复</strong>
        <ul>
          <li>修复<code>pipeline</code>在某些情况可能发生的文本块内容丢失问题 #3005</li>
          <li>修复<code>sglang-client</code>需要安装<code>torch</code>等不必要的包的问题 #2968</li>
          <li>更新<code>dockerfile</code>以修复linux字体缺失导致的解析文本内容不完整问题 #2915</li>
        </ul>
      </li>
      <li><strong>易用性更新</strong>
        <ul>
          <li>更新<code>compose.yaml</code>，便于用户直接启动<code>sglang-server</code>、<code>mineru-api</code>、<code>mineru-gradio</code>服务</li>
          <li>启用全新的<a href="https://opendatalab.github.io/MinerU/zh/">在线文档站点</a>，简化readme，提供更好的文档体验</li>
        </ul>
      </li>
    </ul>
  </details>

  <details>
    <summary>2025/07/05 2.1.0 发布</summary>
    <p>这是 MinerU 2 的第一个大版本更新，包含了大量新功能和改进，包含众多性能优化、体验优化和bug修复，具体更新内容如下：</p>
    <ul>
      <li><strong>性能优化：</strong>
        <ul>
          <li>大幅提升某些特定分辨率（长边2000像素左右）文档的预处理速度</li>
          <li>大幅提升<code>pipeline</code>后端批量处理大量页数较少（&lt;10）文档时的后处理速度</li>
          <li><code>pipeline</code>后端的layout分析速度提升约20%</li>
        </ul>
      </li>
      <li><strong>体验优化：</strong>
        <ul>
          <li>内置开箱即用的<code>fastapi服务</code>和<code>gradio webui</code>，详细使用方法请参考<a href="https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuisglang-clientserver">文档</a></li>
          <li><code>sglang</code>适配<code>0.4.8</code>版本，大幅降低<code>vlm-sglang</code>后端的显存要求，最低可在<code>8G显存</code>(Turing及以后架构)的显卡上运行</li>
          <li>对所有命令增加<code>sglang</code>的参数透传，使得<code>sglang-engine</code>后端可以与<code>sglang-server</code>一致，接收<code>sglang</code>的所有参数</li>
          <li>支持基于配置文件的功能扩展，包含<code>自定义公式标识符</code>、<code>开启标题分级功能</code>、<code>自定义本地模型目录</code>，详细使用方法请参考<a href="https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#mineru_1">文档</a></li>
        </ul>
      </li>
      <li><strong>新特性：</strong>
        <ul>
          <li><code>pipeline</code>后端更新 PP-OCRv5 多语种文本识别模型，支持法语、西班牙语、葡萄牙语、俄语、韩语等 37 种语言的文字识别，平均精度涨幅超30%。<a href="https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html">详情</a></li>
          <li><code>pipeline</code>后端增加对竖排文本的有限支持</li>
        </ul>
      </li>
    </ul>
  </details>

  <details>
    <summary>2025/06/20 2.0.6发布</summary>
    <ul>
      <li>修复<code>vlm</code>模式下，某些偶发的无效块内容导致解析中断问题</li>
      <li>修复<code>vlm</code>模式下，某些不完整的表结构导致的解析中断问题</li>
    </ul>
  </details>
  
  <details>
    <summary>2025/06/17 2.0.5发布</summary>
    <ul>
      <li>修复了<code>sglang-client</code>模式下依然需要下载模型的问题</li>
      <li>修复了<code>sglang-client</code>模式需要依赖<code>torch</code>等实际运行不需要的包的问题</li>
      <li>修复了同一进程内尝试通过多个url启动多个<code>sglang-client</code>实例时，只有第一个生效的问题</li>
    </ul>
  </details>
  
  <details>
    <summary>2025/06/15 2.0.3发布</summary>
    <ul>
      <li>修复了当下载模型类型设置为<code>all</code>时，配置文件出现键值更新错误的问题</li>
      <li>修复了命令行模式下公式和表格功能开关不生效导致功能无法关闭的问题</li>
      <li>修复了<code>sglang-engine</code>模式下，0.4.7版本sglang的兼容性问题</li>
      <li>更新了sglang环境下部署完整版MinerU的Dockerfile和相关安装文档</li>
    </ul>
  </details>
  
  <details>
    <summary>2025/06/13 2.0.0发布</summary>
    <ul>
      <li><strong>全新架构</strong>：MinerU 2.0 在代码结构和交互方式上进行了深度重构，显著提升了系统的易用性、可维护性与扩展能力。
        <ul>
          <li><strong>去除第三方依赖限制</strong>：彻底移除对 <code>pymupdf</code> 的依赖，推动项目向更开放、合规的开源方向迈进。</li>
          <li><strong>开箱即用，配置便捷</strong>：无需手动编辑 JSON 配置文件，绝大多数参数已支持命令行或 API 直接设置。</li>
          <li><strong>模型自动管理</strong>：新增模型自动下载与更新机制，用户无需手动干预即可完成模型部署。</li>
          <li><strong>离线部署友好</strong>：提供内置模型下载命令，支持完全断网环境下的部署需求。</li>
          <li><strong>代码结构精简</strong>：移除数千行冗余代码，简化类继承逻辑，显著提升代码可读性与开发效率。</li>
          <li><strong>统一中间格式输出</strong>：采用标准化的 <code>middle_json</code> 格式，兼容多数基于该格式的二次开发场景，确保生态业务无缝迁移。</li>
        </ul>
      </li>
      <li><strong>全新模型</strong>：MinerU 2.0 集成了我们最新研发的小参数量、高性能多模态文档解析模型，实现端到端的高速、高精度文档理解。
        <ul>
          <li><strong>小模型，大能力</strong>：模型参数不足 1B，却在解析精度上超越传统 72B 级别的视觉语言模型（VLM）。</li>
          <li><strong>多功能合一</strong>：单模型覆盖多语言识别、手写识别、版面分析、表格解析、公式识别、阅读顺序排序等核心任务。</li>
          <li><strong>极致推理速度</strong>：在单卡 NVIDIA 4090 上通过 <code>sglang</code> 加速，达到峰值吞吐量超过 10,000 token/s，轻松应对大规模文档处理需求。</li>
          <li><strong>在线体验</strong>：您可以在<a href="https://mineru.net/OpenSourceTools/Extractor">MinerU.net</a>、<a href="https://huggingface.co/spaces/opendatalab/MinerU">Hugging Face</a>, 以及<a href="https://www.modelscope.cn/studios/OpenDataLab/MinerU">ModelScope</a>体验我们的全新VLM模型</li>
        </ul>
      </li>
      <li><strong>不兼容变更说明</strong>:为提升整体架构合理性与长期可维护性，本版本包含部分不兼容的变更：
        <ul>
          <li>Python 包名从 <code>magic-pdf</code> 更改为 <code>mineru</code>，命令行工具也由 <code>magic-pdf</code> 改为 <code>mineru</code>，请同步更新脚本与调用命令。</li>
          <li>出于对系统模块化设计与生态一致性的考虑，MinerU 2.0 已不再内置 LibreOffice 文档转换模块。如需处理 Office 文档，建议通过独立部署的 LibreOffice 服务先行转换为 PDF 格式，再进行后续解析操作。</li>
        </ul>
      </li>
    </ul>
  </details>
  <details>
  <summary>2025/05/24 1.3.12 发布</summary>
  <ul>
      <li>增加ppocrv5模型的支持，将<code>ch_server</code>模型更新为<code>PP-OCRv5_rec_server</code>，<code>ch_lite</code>模型更新为<code>PP-OCRv5_rec_mobile</code>（需更新模型）
        <ul>
          <li>在测试中，发现ppocrv5(server)对手写文档效果有一定提升，但在其余类别文档的精度略差于v4_server_doc，因此默认的ch模型保持不变，仍为<code>PP-OCRv4_server_rec_doc</code>。</li>
          <li>由于ppocrv5强化了手写场景和特殊字符的识别能力，因此您可以在日繁混合场景以及手写文档场景下手动选择使用ppocrv5模型</li>
          <li>您可通过lang参数<code>lang='ch_server'</code>(python api)或<code>--lang ch_server</code>(命令行)自行选择相应的模型：
            <ul>
              <li><code>ch</code> ：<code>PP-OCRv4_rec_server_doc</code>（默认）（中英日繁混合/1.5w字典）</li>
              <li><code>ch_server</code> ：<code>PP-OCRv5_rec_server</code>（中英日繁混合+手写场景/1.8w字典）</li>
              <li><code>ch_lite</code> ：<code>PP-OCRv5_rec_mobile</code>（中英日繁混合+手写场景/1.8w字典）</li>
              <li><code>ch_server_v4</code> ：<code>PP-OCRv4_rec_server</code>（中英混合/6k字典）</li>
              <li><code>ch_lite_v4</code> ：<code>PP-OCRv4_rec_mobile</code>（中英混合/6k字典）</li>
            </ul>
          </li>
        </ul>
      </li>
      <li>增加手写文档的支持，通过优化layout对手写文本区域的识别，现已支持手写文档的解析
        <ul>
          <li>默认支持此功能，无需额外配置</li>
          <li>可以参考上述说明，手动选择ppocrv5模型以获得更好的手写文档解析效果</li>
        </ul>
      </li>
      <li><code>huggingface</code>和<code>modelscope</code>的demo已更新为支持手写识别和ppocrv5模型的版本，可自行在线体验</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/29 1.3.10 发布</summary>
  <ul>
      <li>支持使用自定义公式标识符，可通过修改用户目录下的<code>magic-pdf.json</code>文件中的<code>latex-delimiter-config</code>项实现。</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/27 1.3.9 发布</summary>
  <ul>
      <li>优化公式解析功能，提升公式渲染的成功率</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/23 1.3.8 发布</summary>
  <ul>
      <li><code>ocr</code>默认模型(<code>ch</code>)更新为<code>PP-OCRv4_server_rec_doc</code>（需更新模型）
        <ul>
          <li><code>PP-OCRv4_server_rec_doc</code>是在<code>PP-OCRv4_server_rec</code>的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力。</li>
          <li><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html#_3">PP-OCRv4_server_rec_doc/PP-OCRv4_server_rec/PP-OCRv4_mobile_rec 性能对比</a></li>
          <li>经验证，<code>PP-OCRv4_server_rec_doc</code>模型在<code>中英日繁</code>单种语言或多种语言混合场景均有明显精度提升，且速度与<code>PP-OCRv4_server_rec</code>相当，适合绝大部分场景使用。</li>
          <li><code>PP-OCRv4_server_rec_doc</code>在小部分纯英文场景可能会发生单词粘连问题，<code>PP-OCRv4_server_rec</code>则在此场景下表现更好，因此我们保留了<code>PP-OCRv4_server_rec</code>模型，用户可通过增加参数<code>lang='ch_server'</code>(python api)或<code>--lang ch_server</code>(命令行)调用。</li>
        </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/22 1.3.7 发布</summary>
  <ul>
      <li>修复表格解析模型初始化时lang参数失效的问题</li>
      <li>修复在<code>cpu</code>模式下ocr和表格解析速度大幅下降的问题</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/16 1.3.4 发布</summary>
  <ul>
      <li>通过移除一些无用的块，小幅提升了ocr-det的速度</li>
      <li>修复部分情况下由footnote导致的页面内排序错误</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/12 1.3.2 发布</summary>
  <ul>
      <li>修复了windows系统下，在python3.13环境安装时一些依赖包版本不兼容的问题</li>
      <li>优化批量推理时的内存占用</li>
      <li>优化旋转90度表格的解析效果</li>
      <li>优化财报样本中超大表格的解析效果</li>
      <li>修复了在未指定OCR语言时，英文文本区域偶尔出现的单词黏连问题（需要更新模型）</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/08 1.3.1 发布</summary>
  <ul>
      <li>修复了一些兼容问题
        <ul>
          <li>支持python 3.13</li>
          <li>为部分过时的linux系统（如centos7）做出最后适配，并不再保证后续版本的继续支持，<a href="https://github.com/opendatalab/MinerU/issues/1004">安装说明</a></li>
        </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/03 1.3.0 发布</summary>
  <ul>
      <li>安装与兼容性优化
        <ul>
          <li>通过移除layout中<code>layoutlmv3</code>的使用，解决了由<code>detectron2</code>导致的兼容问题</li>
          <li>torch版本兼容扩展到2.2~2.6(2.5除外)</li>
          <li>cuda兼容支持11.8/12.4/12.6/12.8（cuda版本由torch决定），解决部分用户50系显卡与H系显卡的兼容问题</li>
          <li>python兼容版本扩展到3.10~3.12，解决了在非3.10环境下安装时自动降级到0.6.1的问题</li>
          <li>优化离线部署流程，部署成功后不需要联网下载任何模型文件</li>
        </ul>
      </li>
      <li>性能优化
        <ul>
          <li>通过支持多个pdf文件的batch处理（<a href="demo/batch_demo.py">脚本样例</a>），提升了批量小文件的解析速度 (与1.0.1版本相比，公式解析速度最高提升超过1400%，整体解析速度最高提升超过500%)</li>
          <li>通过优化mfr模型的加载和使用，降低了显存占用并提升了解析速度(需重新执行<a href="docs/how_to_download_models_zh_cn.md">模型下载流程</a>以获得模型文件的增量更新)</li>
          <li>优化显存占用，最低仅需6GB即可运行本项目</li>
          <li>优化了在mps设备上的运行速度</li>
        </ul>
      </li>
      <li>解析效果优化
        <ul>
          <li>mfr模型更新到<code>unimernet(2503)</code>，解决多行公式中换行丢失的问题</li>
        </ul>
      </li>
      <li>易用性优化
        <ul>
          <li>通过使用<code>paddleocr2torch</code>，完全替代<code>paddle</code>框架以及<code>paddleocr</code>在项目中的使用，解决了<code>paddle</code>和<code>torch</code>的冲突问题，和由于<code>paddle</code>框架导致的线程不安全问题</li>
          <li>解析过程增加实时进度条显示，精准把握解析进度，让等待不再痛苦</li>
        </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/03/03 1.2.1 发布，修复了一些问题</summary>
  <ul>
      <li>修复在字母与数字的全角转半角操作时对标点符号的影响</li>
      <li>修复在某些情况下caption的匹配不准确问题</li>
      <li>修复在某些情况下的公式span丢失问题</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/02/24 1.2.0 发布，这个版本我们修复了一些问题，提升了解析的效率与精度：</summary>
  <ul>
      <li>性能优化
          <ul>
              <li>auto模式下pdf文档的分类速度提升</li>
          </ul>
      </li>
      <li>解析优化
          <ul>
              <li>优化对包含水印文档的解析逻辑，显著提升包含水印文档的解析效果</li>
              <li>改进了单页内多个图像/表格与caption的匹配逻辑，提升了复杂布局下图文匹配的准确性</li>
          </ul>
      </li>
      <li>问题修复
          <ul>
              <li>修复在某些情况下图片/表格span被填充进textblock导致的异常</li>
              <li>修复在某些情况下标题block为空的问题</li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/01/22 1.1.0 发布，在这个版本我们重点提升了解析的精度与效率：</summary>
  <ul>
      <li>模型能力升级（需重新执行 <a href="https://github.com/opendatalab/MinerU/docs/how_to_download_models_zh_cn.md">模型下载流程</a> 以获得模型文件的增量更新）
          <ul>
              <li>布局识别模型升级到最新的 `doclayout_yolo(2501)` 模型，提升了layout识别精度</li>
              <li>公式解析模型升级到最新的 `unimernet(2501)` 模型，提升了公式识别精度</li>
          </ul>
      </li>
      <li>性能优化
          <ul>
              <li>在配置满足一定条件（显存16GB+）的设备上，通过优化资源占用和重构处理流水线，整体解析速度提升50%以上</li>
          </ul>
      </li>
      <li>解析效果优化
          <ul>
              <li>在线demo（<a href="https://mineru.net/OpenSourceTools/Extractor">mineru.net</a> / <a href="https://huggingface.co/spaces/opendatalab/MinerU">huggingface</a> / <a href="https://www.modelscope.cn/studios/OpenDataLab/MinerU">modelscope</a>）上新增标题分级功能（测试版本，默认开启），支持对标题进行分级，提升文档结构化程度</li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/01/10 1.0.1 发布，这是我们的第一个正式版本，在这个版本中，我们通过大量重构带来了全新的API接口和更广泛的兼容性，以及全新的自动语言识别功能：</summary>
  <ul>
      <li>全新API接口
          <ul>
              <li>对于数据侧API，我们引入了Dataset类，旨在提供一个强大而灵活的数据处理框架。该框架当前支持包括图像（.jpg及.png）、PDF、Word（.doc及.docx）、以及PowerPoint（.ppt及.pptx）在内的多种文档格式，确保了从简单到复杂的数据处理任务都能得到有效的支持。</li>
              <li>针对用户侧API，我们将MinerU的处理流程精心设计为一系列可组合的Stage阶段。每个Stage代表了一个特定的处理步骤，用户可以根据自身需求自由地定义新的Stage，并通过创造性地组合这些阶段来定制专属的数据处理流程。</li>
          </ul>
      </li>
      <li>更广泛的兼容性适配
          <ul>
              <li>通过优化依赖环境和配置项，确保在ARM架构的Linux系统上能够稳定高效运行。</li>
              <li>深度适配华为昇腾NPU加速，积极响应信创要求，提供自主可控的高性能计算能力，助力人工智能应用平台的国产化应用与发展。 <a href="https://github.com/opendatalab/MinerU/docs/README_Ascend_NPU_Acceleration_zh_CN.md">NPU加速教程</a></li>
          </ul>
      </li>
      <li>自动语言识别
          <ul>
              <li>通过引入全新的语言识别模型， 在文档解析中将 `lang` 配置为 `auto`，即可自动选择合适的OCR语言模型，提升扫描类文档解析的准确性。</li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2024/11/22 0.10.0发布，通过引入混合OCR文本提取能力，</summary>
  <ul>
      <li>在公式密集、span区域不规范、部分文本使用图像表现等复杂文本分布场景下获得解析效果的显著提升</li>
      <li>同时具备文本模式内容提取准确、速度更快与OCR模式span/line区域识别更准的双重优势</li>
  </ul>
  </details>
  
  <details>
  <summary>2024/11/15 0.9.3发布，为表格识别功能接入了<a href="https://github.com/RapidAI/RapidTable">RapidTable</a>,单表解析速度提升10倍以上，准确率更高，显存占用更低</summary>
  </details>
  
  <details>
  <summary>2024/11/06 0.9.2发布，为表格识别功能接入了<a href="https://huggingface.co/U4R/StructTable-InternVL2-1B">StructTable-InternVL2-1B</a>模型</summary>
  </details>
  
  <details>
  <summary>2024/10/31 0.9.0发布，这是我们进行了大量代码重构的全新版本，解决了众多问题，提升了性能，降低了硬件需求，并提供了更丰富的易用性：</summary>
  <ul>
      <li>重构排序模块代码，使用 <a href="https://github.com/ppaanngggg/layoutreader">layoutreader</a> 进行阅读顺序排序，确保在各种排版下都能实现极高准确率</li>
      <li>重构段落拼接模块，在跨栏、跨页、跨图、跨表情况下均能实现良好的段落拼接效果</li>
      <li>重构列表和目录识别功能，极大提升列表块和目录块识别的准确率及对应文本段落的解析效果</li>
      <li>重构图、表与描述性文本的匹配逻辑，大幅提升 caption 和 footnote 与图表的匹配准确率，并将描述性文本的丢失率降至接近0</li>
      <li>增加 OCR 的多语言支持，支持 84 种语言的检测与识别，语言支持列表详见 <a href="https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/blog/multi_languages.html#5">OCR 语言支持列表</a></li>
      <li>增加显存回收逻辑及其他显存优化措施，大幅降低显存使用需求。开启除表格加速外的全部加速功能(layout/公式/OCR)的显存需求从16GB降至8GB，开启全部加速功能的显存需求从24GB降至10GB</li>
      <li>优化配置文件的功能开关，增加独立的公式检测开关，无需公式检测时可大幅提升速度和解析效果</li>
      <li>集成 <a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit 1.0</a>
          <ul>
              <li>加入自研的 `doclayout_yolo` 模型，在相近解析效果情况下比原方案提速10倍以上，可通过配置文件与 `layoutlmv3` 自由切换</li>
              <li>公式解析升级至 `unimernet 0.2.1`，在提升公式解析准确率的同时，大幅降低显存需求</li>
              <li>因 `PDF-Extract-Kit 1.0` 更换仓库，需要重新下载模型，步骤详见 <a href="https://github.com/opendatalab/MinerU/docs/how_to_download_models_zh_cn.md">如何下载模型</a></li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2024/09/27 0.8.1发布，修复了一些bug，同时提供了<a href="https://opendatalab.com/OpenSourceTools/Extractor/PDF/">在线demo</a>的<a href="https://github.com/opendatalab/MinerU/projects/web_demo/README_zh-CN.md">本地化部署版本</a>和<a href="https://github.com/opendatalab/MinerU/projects/web/README_zh-CN.md">前端界面</a></summary>
  </details>
  
  <details>
  <summary>2024/09/09 0.8.0发布，支持Dockerfile快速部署，同时上线了huggingface、modelscope demo</summary>
  </details>
  
  <details>
  <summary>2024/08/30 0.7.1发布，集成了paddle tablemaster表格识别功能</summary>
  </details>
  
  <details>
  <summary>2024/08/09 0.7.0b1发布，简化安装步骤提升易用性，加入表格识别功能</summary>
  </details>
  
  <details>
  <summary>2024/08/01 0.6.2b1发布，优化了依赖冲突问题和安装文档</summary>
  </details>
  
  <details>
  <summary>2024/07/05 首次开源</summary>
  </details>
</details>
  
  <details>
  <summary>2025/04/27 1.3.9 发布</summary>
  <ul>
      <li>优化公式解析功能，提升公式渲染的成功率</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/23 1.3.8 发布</summary>
  <ul>
      <li><code>ocr</code>默认模型(<code>ch</code>)更新为<code>PP-OCRv4_server_rec_doc</code>（需更新模型）
        <ul>
          <li><code>PP-OCRv4_server_rec_doc</code>是在<code>PP-OCRv4_server_rec</code>的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力。</li>
          <li><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html#_3">PP-OCRv4_server_rec_doc/PP-OCRv4_server_rec/PP-OCRv4_mobile_rec 性能对比</a></li>
          <li>经验证，<code>PP-OCRv4_server_rec_doc</code>模型在<code>中英日繁</code>单种语言或多种语言混合场景均有明显精度提升，且速度与<code>PP-OCRv4_server_rec</code>相当，适合绝大部分场景使用。</li>
          <li><code>PP-OCRv4_server_rec_doc</code>在小部分纯英文场景可能会发生单词粘连问题，<code>PP-OCRv4_server_rec</code>则在此场景下表现更好，因此我们保留了<code>PP-OCRv4_server_rec</code>模型，用户可通过增加参数<code>lang='ch_server'</code>(python api)或<code>--lang ch_server</code>(命令行)调用。</li>
        </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/22 1.3.7 发布</summary>
  <ul>
      <li>修复表格解析模型初始化时lang参数失效的问题</li>
      <li>修复在<code>cpu</code>模式下ocr和表格解析速度大幅下降的问题</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/16 1.3.4 发布</summary>
  <ul>
      <li>通过移除一些无用的块，小幅提升了ocr-det的速度</li>
      <li>修复部分情况下由footnote导致的页面内排序错误</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/12 1.3.2 发布</summary>
  <ul>
      <li>修复了windows系统下，在python3.13环境安装时一些依赖包版本不兼容的问题</li>
      <li>优化批量推理时的内存占用</li>
      <li>优化旋转90度表格的解析效果</li>
      <li>优化财报样本中超大表格的解析效果</li>
      <li>修复了在未指定OCR语言时，英文文本区域偶尔出现的单词黏连问题（需要更新模型）</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/08 1.3.1 发布</summary>
  <ul>
      <li>修复了一些兼容问题
        <ul>
          <li>支持python 3.13</li>
          <li>为部分过时的linux系统（如centos7）做出最后适配，并不再保证后续版本的继续支持，<a href="https://github.com/opendatalab/MinerU/issues/1004">安装说明</a></li>
        </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/04/03 1.3.0 发布</summary>
  <ul>
      <li>安装与兼容性优化
        <ul>
          <li>通过移除layout中<code>layoutlmv3</code>的使用，解决了由<code>detectron2</code>导致的兼容问题</li>
          <li>torch版本兼容扩展到2.2~2.6(2.5除外)</li>
          <li>cuda兼容支持11.8/12.4/12.6/12.8（cuda版本由torch决定），解决部分用户50系显卡与H系显卡的兼容问题</li>
          <li>python兼容版本扩展到3.10~3.12，解决了在非3.10环境下安装时自动降级到0.6.1的问题</li>
          <li>优化离线部署流程，部署成功后不需要联网下载任何模型文件</li>
        </ul>
      </li>
      <li>性能优化
        <ul>
          <li>通过支持多个pdf文件的batch处理（<a href="demo/batch_demo.py">脚本样例</a>），提升了批量小文件的解析速度 (与1.0.1版本相比，公式解析速度最高提升超过1400%，整体解析速度最高提升超过500%)</li>
          <li>通过优化mfr模型的加载和使用，降低了显存占用并提升了解析速度(需重新执行<a href="docs/how_to_download_models_zh_cn.md">模型下载流程</a>以获得模型文件的增量更新)</li>
          <li>优化显存占用，最低仅需6GB即可运行本项目</li>
          <li>优化了在mps设备上的运行速度</li>
        </ul>
      </li>
      <li>解析效果优化
        <ul>
          <li>mfr模型更新到<code>unimernet(2503)</code>，解决多行公式中换行丢失的问题</li>
        </ul>
      </li>
      <li>易用性优化
        <ul>
          <li>通过使用<code>paddleocr2torch</code>，完全替代<code>paddle</code>框架以及<code>paddleocr</code>在项目中的使用，解决了<code>paddle</code>和<code>torch</code>的冲突问题，和由于<code>paddle</code>框架导致的线程不安全问题</li>
          <li>解析过程增加实时进度条显示，精准把握解析进度，让等待不再痛苦</li>
        </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/03/03 1.2.1 发布，修复了一些问题</summary>
  <ul>
      <li>修复在字母与数字的全角转半角操作时对标点符号的影响</li>
      <li>修复在某些情况下caption的匹配不准确问题</li>
      <li>修复在某些情况下的公式span丢失问题</li>
  </ul>
  </details>
  
  <details>
  <summary>2025/02/24 1.2.0 发布，这个版本我们修复了一些问题，提升了解析的效率与精度：</summary>
  <ul>
      <li>性能优化
          <ul>
              <li>auto模式下pdf文档的分类速度提升</li>
          </ul>
      </li>
      <li>解析优化
          <ul>
              <li>优化对包含水印文档的解析逻辑，显著提升包含水印文档的解析效果</li>
              <li>改进了单页内多个图像/表格与caption的匹配逻辑，提升了复杂布局下图文匹配的准确性</li>
          </ul>
      </li>
      <li>问题修复
          <ul>
              <li>修复在某些情况下图片/表格span被填充进textblock导致的异常</li>
              <li>修复在某些情况下标题block为空的问题</li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/01/22 1.1.0 发布，在这个版本我们重点提升了解析的精度与效率：</summary>
  <ul>
      <li>模型能力升级（需重新执行 <a href="https://github.com/opendatalab/MinerU/docs/how_to_download_models_zh_cn.md">模型下载流程</a> 以获得模型文件的增量更新）
          <ul>
              <li>布局识别模型升级到最新的 `doclayout_yolo(2501)` 模型，提升了layout识别精度</li>
              <li>公式解析模型升级到最新的 `unimernet(2501)` 模型，提升了公式识别精度</li>
          </ul>
      </li>
      <li>性能优化
          <ul>
              <li>在配置满足一定条件（显存16GB+）的设备上，通过优化资源占用和重构处理流水线，整体解析速度提升50%以上</li>
          </ul>
      </li>
      <li>解析效果优化
          <ul>
              <li>在线demo（<a href="https://mineru.net/OpenSourceTools/Extractor">mineru.net</a> / <a href="https://huggingface.co/spaces/opendatalab/MinerU">huggingface</a> / <a href="https://www.modelscope.cn/studios/OpenDataLab/MinerU">modelscope</a>）上新增标题分级功能（测试版本，默认开启），支持对标题进行分级，提升文档结构化程度</li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2025/01/10 1.0.1 发布，这是我们的第一个正式版本，在这个版本中，我们通过大量重构带来了全新的API接口和更广泛的兼容性，以及全新的自动语言识别功能：</summary>
  <ul>
      <li>全新API接口
          <ul>
              <li>对于数据侧API，我们引入了Dataset类，旨在提供一个强大而灵活的数据处理框架。该框架当前支持包括图像（.jpg及.png）、PDF、Word（.doc及.docx）、以及PowerPoint（.ppt及.pptx）在内的多种文档格式，确保了从简单到复杂的数据处理任务都能得到有效的支持。</li>
              <li>针对用户侧API，我们将MinerU的处理流程精心设计为一系列可组合的Stage阶段。每个Stage代表了一个特定的处理步骤，用户可以根据自身需求自由地定义新的Stage，并通过创造性地组合这些阶段来定制专属的数据处理流程。</li>
          </ul>
      </li>
      <li>更广泛的兼容性适配
          <ul>
              <li>通过优化依赖环境和配置项，确保在ARM架构的Linux系统上能够稳定高效运行。</li>
              <li>深度适配华为昇腾NPU加速，积极响应信创要求，提供自主可控的高性能计算能力，助力人工智能应用平台的国产化应用与发展。 <a href="https://github.com/opendatalab/MinerU/docs/README_Ascend_NPU_Acceleration_zh_CN.md">NPU加速教程</a></li>
          </ul>
      </li>
      <li>自动语言识别
          <ul>
              <li>通过引入全新的语言识别模型， 在文档解析中将 `lang` 配置为 `auto`，即可自动选择合适的OCR语言模型，提升扫描类文档解析的准确性。</li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2024/11/22 0.10.0发布，通过引入混合OCR文本提取能力，</summary>
  <ul>
      <li>在公式密集、span区域不规范、部分文本使用图像表现等复杂文本分布场景下获得解析效果的显著提升</li>
      <li>同时具备文本模式内容提取准确、速度更快与OCR模式span/line区域识别更准的双重优势</li>
  </ul>
  </details>
  
  <details>
  <summary>2024/11/15 0.9.3发布，为表格识别功能接入了<a href="https://github.com/RapidAI/RapidTable">RapidTable</a>,单表解析速度提升10倍以上，准确率更高，显存占用更低</summary>
  </details>
  
  <details>
  <summary>2024/11/06 0.9.2发布，为表格识别功能接入了<a href="https://huggingface.co/U4R/StructTable-InternVL2-1B">StructTable-InternVL2-1B</a>模型</summary>
  </details>
  
  <details>
  <summary>2024/10/31 0.9.0发布，这是我们进行了大量代码重构的全新版本，解决了众多问题，提升了性能，降低了硬件需求，并提供了更丰富的易用性：</summary>
  <ul>
      <li>重构排序模块代码，使用 <a href="https://github.com/ppaanngggg/layoutreader">layoutreader</a> 进行阅读顺序排序，确保在各种排版下都能实现极高准确率</li>
      <li>重构段落拼接模块，在跨栏、跨页、跨图、跨表情况下均能实现良好的段落拼接效果</li>
      <li>重构列表和目录识别功能，极大提升列表块和目录块识别的准确率及对应文本段落的解析效果</li>
      <li>重构图、表与描述性文本的匹配逻辑，大幅提升 caption 和 footnote 与图表的匹配准确率，并将描述性文本的丢失率降至接近0</li>
      <li>增加 OCR 的多语言支持，支持 84 种语言的检测与识别，语言支持列表详见 <a href="https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/blog/multi_languages.html#5">OCR 语言支持列表</a></li>
      <li>增加显存回收逻辑及其他显存优化措施，大幅降低显存使用需求。开启除表格加速外的全部加速功能(layout/公式/OCR)的显存需求从16GB降至8GB，开启全部加速功能的显存需求从24GB降至10GB</li>
      <li>优化配置文件的功能开关，增加独立的公式检测开关，无需公式检测时可大幅提升速度和解析效果</li>
      <li>集成 <a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit 1.0</a>
          <ul>
              <li>加入自研的 `doclayout_yolo` 模型，在相近解析效果情况下比原方案提速10倍以上，可通过配置文件与 `layoutlmv3` 自由切换</li>
              <li>公式解析升级至 `unimernet 0.2.1`，在提升公式解析准确率的同时，大幅降低显存需求</li>
              <li>因 `PDF-Extract-Kit 1.0` 更换仓库，需要重新下载模型，步骤详见 <a href="https://github.com/opendatalab/MinerU/docs/how_to_download_models_zh_cn.md">如何下载模型</a></li>
          </ul>
      </li>
  </ul>
  </details>
  
  <details>
  <summary>2024/09/27 0.8.1发布，修复了一些bug，同时提供了<a href="https://opendatalab.com/OpenSourceTools/Extractor/PDF/">在线demo</a>的<a href="https://github.com/opendatalab/MinerU/projects/web_demo/README_zh-CN.md">本地化部署版本</a>和<a href="https://github.com/opendatalab/MinerU/projects/web/README_zh-CN.md">前端界面</a></summary>
  </details>
  
  <details>
  <summary>2024/09/09 0.8.0发布，支持Dockerfile快速部署，同时上线了huggingface、modelscope demo</summary>
  </details>
  
  <details>
  <summary>2024/08/30 0.7.1发布，集成了paddle tablemaster表格识别功能</summary>
  </details>
  
  <details>
  <summary>2024/08/09 0.7.0b1发布，简化安装步骤提升易用性，加入表格识别功能</summary>
  </details>
  
  <details>
  <summary>2024/08/01 0.6.2b1发布，优化了依赖冲突问题和安装文档</summary>
  </details>
  
  <details>
  <summary>2024/07/05 首次开源</summary>
  </details>
</details>
