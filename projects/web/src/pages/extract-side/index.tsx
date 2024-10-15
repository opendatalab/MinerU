import DarkLogo from "@/assets/svg/logo.svg";

import { useLocation, useNavigate, useParams } from "react-router-dom";
import commonStyles from "./index.module.scss";

import { EXTRACTOR_TYPE_LIST } from "@/types/extract-task-type";

import extractorPdfIcon from "@/assets/pdf/extractor-pdf.svg";
import extractorTableIcon from "@/assets/pdf/extractor-table.svg";
import extractorFormulaIcon from "@/assets/pdf/extractor-formula.svg";
import { useIntl } from "react-intl";
import cls from "classnames";
import ExtractorGuide from "@/pages/extract/components/extractor-guide";
import ExtractorQueue from "@/pages/extract/components/extractor-queue";
import ExtractorLang from "@/pages/extract/components/extractor-lang";

interface IExtractorSideProps {
  className?: string;
}

interface TabItem {
  label: string;
  type: string;
}

export const ExtractorSide = ({ className = "" }: IExtractorSideProps) => {
  const navigate = useNavigate();

  const params = useParams();
  const location = useLocation();
  const { messages } = useIntl();

  const menuClass =
    "px-2 py-2.5 mb-1 text-[0.875rem] text-[#121316]/[0.8] font-semibold rounded h-10 flex items-center cursor-pointer hover:bg-[#0d53de]/[0.05]";

  const handleMenuClick = (type: string) => {
    navigate(`/OpenSourceTools/Extractor/${type}`);
  };
  const goToOpenSource = () => {
    navigate("/OpenSourceTools/Extractor/");
  };

  const tabList =
    (messages?.["extractor.side.tabList"] as unknown[] as TabItem[]) || [];

  const getIconStyle = (type: string) => {
    const activeClassName = "!bg-[#0d53de]/[0.05] !text-[#0D53DE]";
    const path = location.pathname;
    const regex = /\/Extractor\/([^/]+)(\/|$)/;

    const match = params?.jobID ? "" : path.match(regex)?.[1] || "/";

    const getIcon = () => {
      switch (type) {
        case EXTRACTOR_TYPE_LIST.pdf:
          return extractorPdfIcon;
        case EXTRACTOR_TYPE_LIST.table:
          return extractorTableIcon;
        case EXTRACTOR_TYPE_LIST.formula:
          return extractorFormulaIcon;
      }
    };
    return {
      icon: getIcon(),
      tabClassName: match === type ? activeClassName : "",
    };
  };

  return (
    <div
      className={cls(
        `w-[240px] min-w-[240px] h-full px-4  py-6 flex flex-col justify-start border-r-[1px] border-y-0 border-l-0 border-solid border-[#EBECF0] select-none`,
        commonStyles.linearBlue,
        className
      )}
    >
      <div className={""}>
        <div className="h-[2rem] mb-6 flex justify-between items-center">
          <img
            className="h-full cursor-pointer"
            src={DarkLogo}
            alt=""
            onClick={goToOpenSource}
          />
          <ExtractorGuide />
        </div>

        {/* tab-list */}
        <div className="mb-2">
          {tabList.map((i) => (
            <div
              key={i.type}
              className={cls(menuClass, getIconStyle(i.type)?.tabClassName)}
              onClick={() => handleMenuClick(i.type)}
            >
              <img src={getIconStyle(i.type).icon} className="mr-2 w-6 h-6" />
              {i.label}
            </div>
          ))}
        </div>
      </div>
      <div className="bg-[#0d53de]/[0.08] w-full h-[1px] mt-2 mb-4"></div>
      <ExtractorQueue className="flex-1 overflow-y-auto mb-6" />
      <ExtractorLang className="absolute bottom-6" />
    </div>
  );
};
