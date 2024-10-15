import ImageLayerViewer from "../../components/image-layer-viwer";
import exitFullScreenSvg from "@/assets/pdf/exitFullScreen.svg";
import fullScreenSvg from "@/assets/pdf/fullScreen.svg";
import { Tooltip } from "antd";
import { useIntl } from "react-intl";
import { TaskIdProgress, TaskIdResItem } from "@/api/extract";
import IconFont from "@/components/icon-font";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { useEffect, useMemo, useRef, useState } from "react";
import { message } from "antd";
import { MD_PREVIEW_TYPE } from "@/types/extract-task-type";
import CodeMirror from "@/components/code-mirror";
import LatexRenderer from "../../components/latex-renderer";
import { useParams } from "react-router-dom";
interface IImageOriginViewerProps {
  fullScreen?: boolean;
  setFullScreen?: (val: boolean) => void;
  taskInfo: TaskIdProgress & TaskIdResItem;
}

const FormulaDetailRight = ({
  fullScreen,
  setFullScreen,
  taskInfo,
}: IImageOriginViewerProps) => {
  const CONTROL_BAR_HEIGHT = 48;
  const { formatMessage } = useIntl();
  const [displayType, setDisplayType] = useState(MD_PREVIEW_TYPE.preview);
  const imageViewerRef = useRef<any>();
  const formulaType = taskInfo?.type;
  const params = useParams();
  const jobID = params?.jobID;
  const formulaLateX = useMemo(() => {
    return taskInfo?.content?.map((i: any) => i?.latex + "\\\\\n").join("");
  }, [taskInfo?.content]);

  const handleCopy = () => {};

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

  useEffect(() => {
    imageViewerRef?.current?.updateScaleAndPosition();
  }, [fullScreen]);

  useEffect(() => {
    setDisplayType(MD_PREVIEW_TYPE.preview);
  }, [jobID]);

  return (
    <div className="w-full h-full">
      <header
        className={`flex  border-solid border-0 !border-b-[1px] border-[#EBECF0] px-4 items-center h-[${CONTROL_BAR_HEIGHT}px]`}
      >
        {formulaType === "formula-extract" && (
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
            className="cursor-pointer ml-auto w-[1.5rem] select-none flex items-center justify-center h-[1.5rem] hover:bg-[#F4F5F9] rounded "
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
        {formulaType === "formula-extract" && (
          <div className="flex items-center">
            <span className="w-[1px] h-[0.75rem] bg-[#D7D8DD] mx-[1rem]"></span>
            <Tooltip title={formatMessage({ id: "common.copy" })}>
              <CopyToClipboard
                text={formulaLateX}
                onCopy={() => {
                  message.success(formatMessage({ id: "common.copySuccess" }));
                }}
              >
                <span>
                  <IconFont
                    type="icon-copy"
                    className="text-[#464a53] !text-[1.32rem] leading-0 cursor-pointer hover:bg-[#F4F5F9] rounded"
                    onClick={() => handleCopy()}
                  />
                </span>
              </CopyToClipboard>
            </Tooltip>
          </div>
        )}
      </header>
      {displayType === MD_PREVIEW_TYPE.preview ? (
        formulaType === "formula-extract" ? (
          <div className="w-full h-[calc(100%-48px)] flex items-center justify-center scrollbar-thin-layer overflow-auto">
            <LatexRenderer
              formula={formulaLateX}
              className="text-base sm:text-lg md:text-xl lg:text-2xl xl:text-3xl"
            />
          </div>
        ) : (
          <ImageLayerViewer
            imageHeight={taskInfo?.fileInfo?.height || 0}
            imageWidth={taskInfo?.fileInfo?.width || 0}
            layout={
              taskInfo?.type === "formula-extract"
                ? []
                : (taskInfo.content as any[])
            }
            className={"!h-[calc(100%-48px)]"}
            imageUrl={taskInfo?.url}
            ref={imageViewerRef}
          />
        )
      ) : (
        <div
          className={
            "!h-[calc(100%-48px)] flex items-center justify-center w-full px-4 scroll-thin overflow-auto"
          }
        >
          <CodeMirror className="w-full" value={formulaLateX} />
        </div>
      )}
    </div>
  );
};

export default FormulaDetailRight;
