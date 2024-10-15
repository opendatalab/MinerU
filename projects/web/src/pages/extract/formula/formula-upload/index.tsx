import { useIntl } from "react-intl";
import IconFont from "@/components/icon-font";
import { useState } from "react";
import cls from "classnames";
import { ExtractorUploadButton } from "../../components/pdf-upload-button";
import UploadBg from "@/assets/imgs/online.experience/file-upload-bg.svg";
import style from "./index.module.scss";
import { SubmitRes } from "@/api/extract";
import { ADD_TASK_LIST } from "@/constant/event";
import { FORMULA_TYPE } from "@/types/extract-task-type";
import { useNavigate } from "react-router-dom";

const FORMULA_ITEM_LIST = [
  {
    type: FORMULA_TYPE.detect,
    [`zh-CN-name`]: "公式检测",
    [`en-US-name`]: "Formula Detection",
  },
  {
    type: FORMULA_TYPE.extract,
    [`zh-CN-name`]: "公式识别",
    [`en-US-name`]: "Formula Recognition",
  },
];

const FormulaUpload = () => {
  const navigate = useNavigate();
  const { formatMessage, locale } = useIntl();
  const [formulaType, setFormulaType] = useState(FORMULA_TYPE.detect);

  const afterUploadSuccess = (data: SubmitRes) => {
    navigate(`/OpenSourceTools/Extractor/formula/${data?.id}`);
    setTimeout(() => {
      document.dispatchEvent(
        new CustomEvent(ADD_TASK_LIST, {
          detail: data,
        })
      );
    }, 10);
  };
  const afterAsyncCheck = () => {
    return Promise.resolve(true);
  };

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center ">
      <div
        className="absolute top-10 left-8  hover:!text-[#0D53DE] cursor-pointer"
        onClick={() => navigate("/OpenSourceTools/Extractor")}
      >
        <IconFont type="icon-fanhui" className="mr-2" />
        <span>{formatMessage({ id: "extractor.home" })}</span>
      </div>
      <div className="translate-y-[-60px] flex flex-col items-center ">
        <div className="mb-[2.25rem]">
          {FORMULA_ITEM_LIST.map((i) => {
            return (
              <span
                key={i.type}
                onClick={() => setFormulaType(i?.type)}
                className={cls(
                  "relative text-[1.5rem] text-[#121316] cursor-pointer mx-[1.5rem]",
                  formulaType === i?.type && "!text-[#0D53DE] font-semibold"
                )}
              >
                {i?.[`${locale || "zh-CN"}-name` as "en-US-name"]}
                {formulaType === i?.type && (
                  <span className="absolute bottom-[-0.75rem] right-[50%] translate-x-[50%] w-[3rem] bg-[#0D53DE] rounded-[2px] h-[0.25rem]"></span>
                )}
              </span>
            );
          })}
        </div>

        <div className="text-[1.25rem] text-[#121316]/[0.8] mb-[3rem] text-center w-max-[50rem]">
          {formatMessage({
            id:
              formulaType === "extract"
                ? "extractor.formula.title2"
                : "extractor.formula.title",
          })}
        </div>
        <ExtractorUploadButton
          accept="image/png, image/jpg, .png ,.jpg"
          afterUploadSuccess={afterUploadSuccess}
          taskType="extract"
          afterAsyncCheck={afterAsyncCheck}
          extractType={
            formulaType === FORMULA_TYPE.extract
              ? "formula-extract"
              : "formula-detect"
          }
          className={style.textBtn}
          showIcon={false}
          text={
            <div
              className={cls(
                style.uploadSection,
                "border-[1px] border-dashed border-[#0D53DE] rounded-xl flex flex-col items-center justify-center"
              )}
            >
              <img src={UploadBg} className="mb-4" />

              <span
                className={cls(style.uploadText, "text-[18px] leading-[20px]")}
              >
                {formatMessage({ id: "extractor.formula.upload.text" })}
              </span>
              <span className={cls(style.uploadDescText)}>
                {formatMessage({ id: "extractor.formula.upload.accept" })}
              </span>
              <div>
                <span className={cls(style.linearText, "cursor-pointer")}>
                  {formatMessage({
                    id: "extractor.formula.upload.try",
                  })}
                </span>
              </div>
            </div>
          }
        ></ExtractorUploadButton>
      </div>
      <div className="absolute bottom-[1.5rem] text-[13px] text-[#121316]/[0.35] text-center leading-[20px] max-w-[64rem]">
        {formatMessage({
          id: "extractor.law",
        })}
      </div>
    </div>
  );
};

export default FormulaUpload;
