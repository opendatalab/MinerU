import { EXTRACTOR_TYPE_LIST } from "@/types/extract-task-type";
import odlLogo from "@/assets/pdf/odl-logo.svg";
import labelLLMLogo from "@/assets/pdf/label-llm.svg";
import labelULogo from "@/assets/pdf/labelU.svg";

export default {
  "extractor.side.tabList": [
    {
      label: "PDF Extraction",
      type: EXTRACTOR_TYPE_LIST.pdf,
    },
    // {
    //   label: "Formula Extraction",
    //   type: EXTRACTOR_TYPE_LIST.formula,
    // },
  ],
  "extractor.side.guide_list": [
    {
      type: "odl",
      icon: odlLogo,
      title: "OpenDataLab",
      desc: "Covers a huge amount of high-quality, multimodal datasets",
      goToText: "Go Now",
      link: "https://opendatalab.com",
    },
    {
      type: "labelU",
      icon: labelULogo,
      title: "Label U Labeling Tool",
      desc: "Lightweight open source annotation tools",
      goToText: "github",
      link: "https://github.com/opendatalab/labelU",
    },
    {
      type: "labelLLM",
      icon: labelLLMLogo,
      [`zh-CN-title`]: "LabelLLM Labeling Tool",
      title: "LabelLLM Labeling Tool",
      desc: "Specializing in dialogue annotation for large language models",
      goToText: "github",
      link: "https://github.com/opendatalab/LabelLLM",
    },
  ],
};
