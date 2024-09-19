import { EXTRACTOR_TYPE_LIST } from "@/types/extract-task-type";
import odlLogo from "@/assets/pdf/odl-logo.svg";
import labelLLMLogo from "@/assets/pdf/label-llm.svg";
import labelULogo from "@/assets/pdf/labelU.svg";

export default {
  "extractor.side.tabList": [
    {
      label: "PDF文档提取",
      type: EXTRACTOR_TYPE_LIST.pdf,
    },
    // {
    //   label: "公式检测与识别",
    //   type: EXTRACTOR_TYPE_LIST.formula,
    // },
  ],
  "extractor.side.guide_list": [
    {
      type: "odl",
      icon: odlLogo,
      title: "OpenDataLab",
      desc: "涵盖海量优质、多模态数据集",
      goToText: "立即前往",
      link: "https://opendatalab.com",
    },
    {
      type: "labelU",
      icon: labelULogo,
      title: "Label U 标注工具",
      desc: "轻量级开源标注工具",
      goToText: "github",
      link: "https://github.com/opendatalab/labelU",
    },
    {
      type: "labelLLM",
      icon: labelLLMLogo,
      title: "LabelLLM 标注工具",
      desc: "专攻于大模型的对话标注",
      goToText: "github",
      link: "https://github.com/opendatalab/LabelLLM",
    },
  ],
};
