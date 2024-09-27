import showLayerIcon from "@/assets/pdf/extractor-show-layer.svg";
import hiddenLayerIcon from "@/assets/pdf/extractor-hidden-layer.svg";

import { useRef, useState } from "react";
import IconFont from "@/components/icon-font";
import ImageLayerViewer, {
  ImageLayerViewerRef,
} from "../../components/image-layer-viwer";

import { useUpdate } from "ahooks";
import { TaskIdProgress, TaskIdResItem } from "@/api/extract";
import { Tooltip } from "antd";
import { useIntl } from "react-intl";

interface IFormulaDetailLeftProps {
  taskInfo: TaskIdProgress & TaskIdResItem;
}

const FormulaDetailLeft = ({ taskInfo }: IFormulaDetailLeftProps) => {
  const imageRef = useRef<ImageLayerViewerRef>(null);
  const { formatMessage } = useIntl();
  const [layerVisible, setLayerVisible] = useState(true);
  const update = useUpdate();

  if (!taskInfo?.fileInfo?.height || !taskInfo?.fileInfo?.width) {
    console.info(
      "formula extractor interface error: the picture size is invalid"
    );
  }
  return (
    <div className="w-full h-full">
      <div
        className={`flex border-solid border-0 !border-b-[1px] border-[#EBECF0] items-center px-4 h-[48px]`}
      >
        <Tooltip
          title={
            <>
              {layerVisible
                ? formatMessage({
                    id: "extractor.button.hiddenLayer",
                  })
                : formatMessage({
                    id: "extractor.button.showLayer",
                  })}
            </>
          }
        >
          <span
            className="ml-auto mr-2 cursor-pointer hover:bg-[#f4f5f9] w-6 text-center inline-block rounded leading-normal"
            onClick={() => setLayerVisible(!layerVisible)}
          >
            {taskInfo?.type === "formula-detect" ? null : layerVisible ? (
              <img src={hiddenLayerIcon} alt="Hide Layer" />
            ) : (
              <img src={showLayerIcon} alt="Show Layer" />
            )}
          </span>
        </Tooltip>
        {taskInfo?.type === "formula-detect" ? null : (
          <span className="w-[1px] leading-normal h-[12px] bg-[#D7D8DD] mr-1"></span>
        )}
        <div className="select-none w-[7.8rem] flex justify-center">
          <IconFont
            className="rounded mx-2 cursor-pointer hover:bg-[#F4F5F9]"
            type="icon-SubtractOutlined"
            onClick={() => {
              imageRef?.current?.zoomOut();
            }}
          />
          <span className="mx-2">
            {((imageRef?.current?.scale || 0) * 100 || 1).toFixed(0)}%
          </span>
          <IconFont
            className="rounded mx-2 cursor-pointer hover:bg-[#F4F5F9]"
            type="icon-PlusOutlined"
            onClick={() => {
              imageRef?.current?.zoomIn();
            }}
          />
        </div>
      </div>
      <ImageLayerViewer
        imageHeight={taskInfo?.fileInfo?.height || 0}
        imageWidth={taskInfo?.fileInfo?.width || 0}
        layout={taskInfo.content as any[]}
        ref={imageRef}
        onChange={() => {
          // imageRef?.current?.scale为了这个更新
          update();
        }}
        className={"!h-[calc(100%-48px)]"}
        layerVisible={
          taskInfo?.type === "formula-detect" ? false : layerVisible
        }
        imageUrl={taskInfo?.url}
      />
    </div>
  );
};

export default FormulaDetailLeft;
