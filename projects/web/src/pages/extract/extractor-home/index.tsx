import extractorPdfIcon from "@/assets/pdf/extractor-pdf.svg";
import extractorTableIcon from "@/assets/pdf/extractor-table.svg";
import extractorFormulaIcon from "@/assets/pdf/extractor-formula.svg";
import style from "./index.module.scss";
import cls from "classnames";
import { EXTRACTOR_TYPE_LIST } from "@/types/extract-task-type";
import { useNavigate } from "react-router-dom";
import { useIntl } from "react-intl";

const ITEM_LIST = [
  {
    id: 1,
    icon: extractorPdfIcon,
    [`zh-CN-title`]: "PDF文档提取",
    [`en-US-title`]: "PDF Document Extraction",
    type: EXTRACTOR_TYPE_LIST.pdf,
    [`zh-CN-desc`]:
      "支持文本/扫描型 pdf 解析，识别各类版面元素并转换为多模态 Markdown 格式",
    [`en-US-desc`]:
      "Supports text/scanned PDF parsing, identifies various layout elements and converts them into multimodal Markdown format",
  },
  {
    id: 2,
    icon: extractorFormulaIcon,
    [`zh-CN-title`]: "公式检测与识别",
    [`en-US-title`]: "Formula Detection and Recognition",
    type: EXTRACTOR_TYPE_LIST.formula,
    [`zh-CN-desc`]:
      "对行内、行间公式进行检测，对数学公式进行识别并转换为 Latex 格式",
    [`en-US-desc`]:
      "Detect formulas within and between lines, identify mathematical formulas and convert them into Latex format",
  },
  {
    id: 3,
    icon: extractorTableIcon,
    [`zh-CN-title`]: "表格识别",
    [`en-US-title`]: "Table recognition",
    type: EXTRACTOR_TYPE_LIST.table,
    [`zh-CN-desc`]: "对表格进行检测并转换为 Markdown 格式",
    [`en-US-desc`]: "Detect and convert tables to Markdown format",
    comingSoon: true,
  },
] as Record<string, any>;

const ExtractorHome = () => {
  const navigate = useNavigate();
  const { formatMessage, locale } = useIntl();
  const defaultLocale = "zh-CN";

  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="mb-4 font-semibold text-base !text-[2rem] leading-[3rem]">
        {formatMessage({ id: "extractor.home.title" })}
      </div>
      <div className="mb-12 text-[1.25rem] leading-[2rem]">
        {formatMessage({ id: "extractor.home.subTitle" })}
      </div>

      <div className="flex mx-[5.25rem]">
        {ITEM_LIST?.map((i: Record<string, any>) => {
          return (
            <div
              className={cls(
                style.item,
                i?.comingSoon &&
                  style?.[`itemComingSoon_${locale || defaultLocale}`],
                "mx-4 basis-4/12"
              )}
              key={i.desc}
              onClick={() => {
                if (i?.comingSoon) return;
                navigate(`/OpenSourceTools/Extractor/${i?.type}`);
              }}
            >
              <div className={style.itemContent}>
                <div className="text-center leading-[1.5rem] flex items-center justify-center">
                  <img
                    src={i.icon}
                    alt={""}
                    className="w-[2.25rem] h-[2.25rem]"
                  />
                  <span className="text-[#121316]/[0.8] ml-2  text-[20px] font-semibold">
                    {i?.[`${locale}-title`]}
                  </span>
                </div>
                <div className="text-center text-[#121316]/[0.6] mt-4 text-[14px] leading-[1.5rem]">
                  {i?.[`${locale}-desc`]}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="absolute bottom-[1.5rem] text-[13px] text-[#121316]/[0.35] text-center leading-[20px] max-w-[64rem]">
        {formatMessage({
          id: "extractor.law",
        })}
      </div>
    </div>
  );
};
export default ExtractorHome;
