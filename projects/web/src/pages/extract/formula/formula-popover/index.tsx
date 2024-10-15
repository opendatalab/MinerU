import React, { ReactNode } from "react";
import { Popover } from "antd";
import IconFont from "@/components/icon-font";
import { useIntl } from "react-intl";
import style from "./index.module.scss";

interface IFormulaPopoverProps {
  type: string;
  text?: string | ReactNode;
}

const FormulaPopover = ({ type, text }: IFormulaPopoverProps) => {
  const { formatMessage } = useIntl();
  const content = (
    <div className="flex flex-col w-[20rem] items-center">
      {/* 顺序反了 */}
      {formatMessage({
        id:
          type === "detect"
            ? "extractor.formula.popover.extract"
            : "extractor.formula.popover.detect",
      })}
      <img
        className="w-full mt-4"
        src={
          type === "extract"
            ? "https://static.openxlab.org.cn/opendatalab/assets/pdf/svg/extract-formula-extract.svg"
            : "https://static.openxlab.org.cn/opendatalab/assets/pdf/svg/extract-formula-detect.svg"
        }
        alt="formula-popover"
      />
    </div>
  );
  return (
    <span className={""}>
      <Popover
        content={content}
        placement="right"
        showArrow={false}
        overlayClassName={style.formulaPopover}
      >
        <span className="group inline-flex items-center">
          {text}
          <IconFont
            type="icon-QuestionCircleOutlined"
            className="text-[#121316]/[0.6] text-[15px] mt-[2px] leading-[1rem] group-hover:text-[#0D53DE]"
          />
        </span>
      </Popover>
    </span>
  );
};

export default FormulaPopover;
