import { Popover } from "antd";
import guideToolsSvg from "@/assets/pdf/guideTools.svg";
import style from "./index.module.scss";
import { useIntl } from "react-intl";
import IconFont from "@/components/icon-font";
import { windowOpen } from "@/utils/windowOpen";

interface GuideItem {
  type: string;
  icon: string;
  "zh-CN-title": string;
  title: string;
  desc: string;
  goToText: string;
  link: string;
}

const ExtractorGuide = () => {
  const { formatMessage, messages } = useIntl();

  const EXTRACTOR_GUIDE_ITEM_LIST = (messages?.["extractor.side.guide_list"] ||
    []) as unknown as GuideItem[];

  const content = (
    <div>
      <div className="text-[1.25rem] font-semibold mt-3 mb-2  ml-4">
        {formatMessage({
          id: "extractor.guide.title",
        })}
      </div>
      <hgroup>
        {EXTRACTOR_GUIDE_ITEM_LIST?.map((i) => {
          return (
            <div
              key={i.type}
              className="flex p-4 items-center cursor-pointer hover:bg-[#F4F5F9] rounded group h-[6.5rem]"
              onClick={() => windowOpen(i.link)}
            >
              <img
                src={i.icon}
                alt=""
                className="w-[1.5rem] h-[1.5rem] transition-all mr-[0.75rem]"
              />
              <div className="">
                <div className="font-semibold transition-all text-[1rem]">
                  {i.title}
                </div>
                <div className="text-base text-[13px] text-[#121316]/[0.6] transition-all ">
                  {i.desc}
                </div>
                <div className="h-0 mt-2 overflow-hidden !text-[13px] text-[#121316]/[0.8] transition-all group-hover:h-auto">
                  {i.goToText}
                  <IconFont type="icon-ArrowRightOutlined" className="ml-1" />
                </div>
              </div>
            </div>
          );
        })}
      </hgroup>
    </div>
  );
  return (
    <Popover
      overlayClassName={style.extractorGuide}
      content={content}
      showArrow={false}
      placement="right"
    >
      <img
        className="w-[1.32rem] h-[1.32rem] p-0.5 hover:rotate-45 transition-all cursor-pointer rounded"
        src={guideToolsSvg}
        alt="guideToolsSvg"
      />
    </Popover>
  );
};

export default ExtractorGuide;
