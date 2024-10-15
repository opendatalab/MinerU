import cls from "classnames";

import { useEffect, useState } from "react";

import LoadingIcon from "../../components/loading-icon";

import { SubmitRes } from "@/api/extract";
import emptySvg from "@/assets/svg/empty.svg";

import { FormattedMessage } from "react-intl";

import FormulaDetailLeft from "../formula-detail-left";
import FormulaDetailRight from "../formula-detail-right";

import { useIntl } from "react-intl";
import { ExtractorUploadButton } from "../../components/pdf-upload-button";
import useExtractorJobProgress from "@/store/jobProgress";
interface IPdfExtractionProps {
  setUploadShow: (bool: boolean) => void;
  className?: string;
}

const FormulaDetail = ({ className = "" }: IPdfExtractionProps) => {
  const {
    taskInfo,
    queueLoading,
    interfaceError: compileError,
    refreshQueue,
    jobID,
  } = useExtractorJobProgress();

  const [fullScreen, setFullScreen] = useState<boolean>(false);
  const { formatMessage } = useIntl();

  const isQueueAndExtract = queueLoading;
  const hiddenQueuePage = !isQueueAndExtract ? "opacity-0 " : "";
  const hiddenResultPage = isQueueAndExtract ? "z-[-1] opacity-0" : "";

  const getLayoutClassName = (_fullScreen?: boolean) => {
    return {
      left: _fullScreen ? "w-0 overflow-hidden" : "w-[50%] max-w-[50%]",
      right: _fullScreen ? "w-full " : "w-[50%] max-w-[50%]",
    };
  };

  const afterUploadSuccess = (data: SubmitRes) => {
    refreshQueue();
  };
  const afterAsyncCheck = () => {
    return Promise.resolve(true);
  };

  useEffect(() => {
    setFullScreen(false);
  }, [jobID]);

  return (
    <>
      <div
        className={cls(
          "flex flex-col justify-center items-center h-[60px] w-[300px] bg-white h-full w-full  absolute top-0 left-0",
          hiddenQueuePage
        )}
      >
        <LoadingIcon className="w-12" color={"#0D53DE"} />

        <div className="text-base text-[#121316]/[0.8] mt-4">
          {taskInfo?.rank > 1 ? (
            <FormattedMessage
              id="extractor.common.extracting.queue"
              values={{
                id: taskInfo?.rank || 0,
              }}
            />
          ) : taskInfo.state === "done" || taskInfo?.state === "unknown" ? (
            formatMessage({
              id: "extractor.common.loading",
            })
          ) : (
            formatMessage({
              id: "extractor.common.extracting",
            })
          )}
        </div>
      </div>

      <div
        className={cls("h-full w-full relative", className, hiddenResultPage)}
      >
        {!compileError ? (
          <div className="w-full flex h-full">
            <div className={cls("h-full", getLayoutClassName(fullScreen).left)}>
              <FormulaDetailLeft taskInfo={taskInfo} />
            </div>
            <div
              className={cls(
                "!overflow-auto",
                getLayoutClassName(fullScreen).right
              )}
              style={{
                borderLeft: "1px solid #EBECF0",
              }}
            >
              <FormulaDetailRight
                fullScreen={fullScreen}
                setFullScreen={setFullScreen}
                taskInfo={taskInfo}
              />
            </div>
          </div>
        ) : (
          <div className="ml-[50%] translate-x-[-50%] !h-[calc(100%-70px)]  flex-1  flex items-center h-[110px] flex-col justify-center">
            <img src={emptySvg} alt="emptySvg" />
            <span className="text-[#121316]/[0.8] mt-2">
              {formatMessage({
                id: "extractor.failed",
              })}
            </span>
            <ExtractorUploadButton
              className="!mb-0 !w-[120px] !m-6"
              accept="image/png, image/jpg, .png ,.jpg"
              afterUploadSuccess={afterUploadSuccess}
              taskType="extract"
              afterAsyncCheck={afterAsyncCheck}
              extractType={taskInfo?.type}
              submitType="reUpload"
              showIcon={false}
              text={
                <span className="text-white">
                  {formatMessage({
                    id: "extractor.button.reUpload",
                  })}
                </span>
              }
            />
          </div>
        )}
      </div>
    </>
  );
};

export default FormulaDetail;
