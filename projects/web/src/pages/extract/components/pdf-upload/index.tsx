import UploadBg from "@/assets/imgs/online.experience/file-upload-bg.svg";
import style from "./index.module.scss";
import { ExtractorUploadButton } from "../pdf-upload-button";
import { useNavigate } from "react-router-dom";
import cls from "classnames";

import { SubmitRes } from "@/api/extract";
import { Checkbox, Popover } from "antd";

import { useIntl } from "react-intl";
import IconFont from "@/components/icon-font";

import { ADD_TASK_LIST } from "@/constant/event";
import { useState } from "react";

const PdfUpload = () => {
  const navigate = useNavigate();

  const { formatMessage } = useIntl();

  const [checked, setChecked] = useState(false);

  const afterUploadSuccess = (data: SubmitRes) => {
    navigate(`/OpenSourceTools/Extractor/PDF/${data?.id}`);
    setTimeout(() => {
      document.dispatchEvent(
        new CustomEvent(ADD_TASK_LIST, {
          detail: data,
        })
      );
    }, 10);
  };

  const afterAsyncCheck = async () => {
    return Promise.resolve(true);
  };

  return (
    <div className="w-full h-full flex flex-col relative items-center relative">
      <div className="w-full h-full flex flex-col  relative justify-center items-center translate-y-[-60px] z-0">
        <div className="mb-6 text-[1.5rem] text-[#121316] font-semibold">
          {formatMessage({ id: "extractor.pdf.title" })}
        </div>
        <div className="mb-12 text-[1.25rem] text-center text-[#121316]/[0.8] leading-[1.5rem]  max-w-[48rem]">
          {formatMessage({ id: "extractor.pdf.subTitle" })}
        </div>
        <ExtractorUploadButton
          accept=".pdf"
          taskType="pdf"
          afterUploadSuccess={afterUploadSuccess}
          afterAsyncCheck={afterAsyncCheck}
          extractType={"pdf"}
          isOcr={checked}
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
                {formatMessage({ id: "extractor.common.upload" })}
              </span>
              <span
                className={cls(style.uploadDescText, "!mb-0 flex items-center")}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
              >
                <Checkbox
                  className="mr-1"
                  checked={checked}
                  onClick={() => setChecked(!checked)}
                />
                {formatMessage({ id: "extractor.pdf.ocr" })}
                <Popover
                  content={
                    <div className="max-w-[20rem]">
                      {formatMessage({
                        id: "extractor.pdf.ocr.popover",
                      })}
                    </div>
                  }
                  placement="right"
                  showArrow={false}
                  overlayClassName={style.customPopover}
                >
                  <IconFont
                    type="icon-QuestionCircleOutlined"
                    className="text-[#121316]/[0.6] ml-1 text-[16px] hover:text-[#0D53DE]"
                  />
                </Popover>
              </span>
              {/* <span className={cls(style.uploadDescText)}>
                {formatMessage({ id: "extractor.common.pdf.upload.tip" })}
              </span> */}
            </div>
          }
          className={style.textBtn}
          showIcon={false}
        />
      </div>
      <div className="absolute bottom-[1.5rem] text-[13px] text-[#121316]/[0.35] text-center leading-[20px] max-w-[64rem]">
        {formatMessage({
          id: "extractor.law",
        })}
      </div>
    </div>
  );
};
export default PdfUpload;
