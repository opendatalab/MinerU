import { useEffect, useRef, useState } from "react";
import { Tooltip } from "antd";

import cls from "classnames";
import styles from "./index.module.scss";
import { useDeepCompareEffect, useHover } from "ahooks";
import IconFont from "@/components/icon-font";
import { downloadFileUseAScript } from "@/utils/download";
import { MD_DRIVE_PDF } from "@/constant/event";
import { useIntl } from "react-intl";
import LazyUrlMarkdown from "../url-markdown";
import exitFullScreenSvg from "@/assets/pdf/exitFullScreen.svg";
import fullScreenSvg from "@/assets/pdf/fullScreen.svg";
import { MD_PREVIEW_TYPE } from "@/types/extract-task-type";
import _ from "lodash";
import { TaskIdResItem } from "@/api/extract";
import useMdStore from "@/store/mdStore";
import CodeMirror from "@/components/code-mirror";
import { useParams } from "react-router-dom";
import SaveStatus, { SaveStatusRef } from "@/components/SaveStatus";

interface IMdViewerProps {
  md?: string;
  className?: string;
  filename?: string;
  url?: string;
  taskInfo: TaskIdResItem;
  curPage: number;
  fullScreen?: boolean;
  setFullScreen?: (value?: boolean) => void;
}

const MdViewer: React.FC<IMdViewerProps> = ({
  fullScreen,
  setFullScreen,
  taskInfo,
  className = "",
  curPage,
}) => {
  const mdViewerPef = useRef<HTMLDivElement>(null);
  const url = taskInfo?.fullMdLink || "";
  const containerRef = useRef<HTMLDivElement>(null);
  const isHovering = useHover(containerRef);
  const { formatMessage } = useIntl();
  const [displayType, setDisplayType] = useState(MD_PREVIEW_TYPE.preview);
  const params = useParams();
  const {
    setAllMdContentWithAnchor,
    allMdContentWithAnchor,
    setMdUrlArr,
    mdContents,
    updateMdContent,
  } = useMdStore();
  const [lineWrap, setLineWrap] = useState(false);

  const threshold = 562 - 427;
  const statusRef = useRef<SaveStatusRef>(null);

  const menuList = [
    {
      name: formatMessage({ id: "extractor.markdown.preview" }),
      code: MD_PREVIEW_TYPE.preview,
    },
    {
      name: formatMessage({ id: "extractor.markdown.code" }),
      code: MD_PREVIEW_TYPE.code,
    },
  ];

  const getVisibleFromType = (str: string, type: string) => {
    return str === type
      ? "relative w-full h-full"
      : "w-0 h-0 overflow-hidden hidden";
  };

  const pushMdViewerScroll = (scrollType?: "instant" | "smooth") => {
    const container = document.getElementById(`md-container`);
    // md渲染的时候用一个元素包括anchor
    const element =
      displayType === MD_PREVIEW_TYPE.preview
        ? document.getElementById(`md-anchor-${curPage - 1}`)?.parentElement
        : document.getElementById(`code-${curPage - 1}`);
    if (element && container) {
      container.scrollTo({
        top: element.offsetTop - 124,
        behavior: scrollType || "smooth",
      });
    }
  };

  useEffect(() => {
    if (isHovering) return;
    pushMdViewerScroll();
  }, [curPage, isHovering]);

  useEffect(() => {
    pushMdViewerScroll("instant");
  }, [displayType]);

  useEffect(() => {
    if (!isHovering) return;
    const handleScroll = () => {
      if (!containerRef.current) return;

      taskInfo?.markdownUrl?.forEach((page, index) => {
        const element =
          displayType === MD_PREVIEW_TYPE.preview
            ? document.getElementById(`md-anchor-${index}`)?.parentElement
            : document.getElementById(`code-${index}`);

        if (element) {
          const rect = element.getBoundingClientRect();

          if (rect.top <= threshold) {
            document.dispatchEvent(
              new CustomEvent(MD_DRIVE_PDF, {
                detail: index,
              })
            );
          }
        }
      });
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener("scroll", handleScroll);
    }

    return () => {
      if (container) {
        container?.removeEventListener("scroll", handleScroll);
      }
    };
  }, [taskInfo, isHovering, displayType]);

  useDeepCompareEffect(() => {
    if (taskInfo?.markdownUrl) {
      setMdUrlArr(taskInfo?.markdownUrl);
    }
    statusRef?.current?.reset();
  }, [taskInfo?.markdownUrl, params?.jobID]);

  const handleContentChange = (val: string, index: number) => {
    setAllMdContentWithAnchor(val);
    statusRef?.current?.triggerSave();
    if (taskInfo?.file_key) {
      updateMdContent(taskInfo.file_key!, index, val);
    }
  };

  return (
    <div className={cls(className)} ref={mdViewerPef}>
      <div
        className={cls(
          "h-[49px] px-6  border-0 border-solid border-b-[1px] border-[#EBECF0] w-full pl-[24px] flex justify-between   items-center"
        )}
      >
        <ul className="p-1 list-none mb-0 inline-block rounded-sm mr-auto  bg-[#F4F5F9] select-none">
          {menuList.map((item) => (
            <li
              key={item.code}
              className={`mx-[0.125rem] px-2 leading-[25px] inline-block rounded-sm text-[14px] cursor-pointer  text-color ${
                displayType === item.code && "bg-white text-primary"
              }`}
              onClick={() => setDisplayType(item.code)}
            >
              {item.name}
            </li>
          ))}
        </ul>
        <SaveStatus ref={statusRef} />

        {displayType === "code" && (
          <>
            <Tooltip
              title={
                fullScreen
                  ? formatMessage({ id: "extractor.button.lineWrap" })
                  : formatMessage({
                      id: "extractor.button.lineWrap",
                    })
              }
            >
              <IconFont
                type="icon-line-wrap"
                className={cls(
                  "text-lg text-[#464a53] leading-0  ml-[1rem] cursor-pointer hover:bg-[#F4F5F9] p-1 rounded",
                  lineWrap && "!text-[#0D53DE]"
                )}
                onClick={() => setLineWrap?.(!lineWrap)}
              />
            </Tooltip>
            <span className="w-[1px] h-[0.75rem] bg-[#D7D8DD] mx-[1rem]"></span>
          </>
        )}
        <Tooltip
          title={
            fullScreen
              ? formatMessage({ id: "extractor.button.exitFullScreen" })
              : formatMessage({
                  id: "extractor.button.fullScreen",
                })
          }
        >
          <span
            className="cursor-pointer w-[1.5rem] user-select-none flex items-center justify-center h-[1.5rem] hover:bg-[#F4F5F9] rounded "
            onClick={() => setFullScreen?.(!fullScreen)}
          >
            {!fullScreen ? (
              <img
                className=" w-[1.125rem] h-[1.125rem] "
                src={fullScreenSvg}
              />
            ) : (
              <img
                className=" w-[1.125rem] h-[1.125rem] "
                src={exitFullScreenSvg}
              />
            )}
          </span>
        </Tooltip>
        <span className="w-[1px] h-[0.75rem] bg-[#D7D8DD] ml-[1rem]"></span>
        <Tooltip title={formatMessage({ id: "extractor.button.download" })}>
          <IconFont
            type="icon-xiazai"
            className="text-lg text-[#464a53] leading-0  ml-[1rem] cursor-pointer hover:bg-[#F4F5F9] p-1 rounded"
            onClick={() =>
              downloadFileUseAScript(
                url,
                `${_(taskInfo?.fileName).split(".").slice(0, -1).join(".")}.md`
              )
            }
          />
        </Tooltip>
      </div>
      <div
        className={cls(
          "bg-white !h-[calc(100%-60px)] px-6 py-8 overflow-auto w-full max-w-[100%]",
          styles.scrollBar
        )}
        id="md-container"
        ref={containerRef}
      >
        <div
          className={cls(
            getVisibleFromType(displayType, MD_PREVIEW_TYPE.preview)
          )}
        >
          <LazyUrlMarkdown
            markdownClass={"relative"}
            content={allMdContentWithAnchor}
          />
        </div>
        <div
          className={cls(getVisibleFromType(displayType, MD_PREVIEW_TYPE.code))}
        >
          {taskInfo?.markdownUrl?.map((url: string, index: number) => {
            const md = mdContents[url]?.content || "";
            if (!md) return null;
            return (
              <div key={url} id={`code-${index}`} className="opacity-1 z-[-1]">
                <CodeMirror
                  value={md}
                  lineWrapping={lineWrap}
                  onChange={(val) => handleContentChange(val, index)}
                  editable
                  className="w-full h-full"
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
export default MdViewer;
