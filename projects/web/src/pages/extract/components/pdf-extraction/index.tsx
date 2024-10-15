import cls from "classnames";
import {
  ExtractorUploadButton,
  LinearButton,
} from "../pdf-upload-button/index";
import { useState } from "react";
import MdViewer from "../md-viewer";
import PDFViewerMemo from "../pdf-viewer";

import LoadingIcon from "../../components/loading-icon";
import emptySvg from "@/assets/svg/empty.svg";

import { useIntl, FormattedMessage } from "react-intl";

import { useJobExtraction } from "@/store/jobProgress";
import { postReUploadExtractTask } from "@/api/extract";
import { useParams } from "react-router-dom";

interface IPdfExtractionProps {
  className?: string;
}

const PdfExtraction = ({ className = "" }: IPdfExtractionProps) => {
  const {
    refreshQueue,
    taskInfo,
    isLoading: queueLoading,
    isError: compileError,
  } = useJobExtraction();

  const [pdfState, setPdfState] = useState({
    page: 1,
  });
  const curPage = pdfState.page;

  const { jobID } = useParams();

  const { formatMessage } = useIntl();

  const [fullScreen, setFullScreen] = useState<boolean>(false);

  const afterUploadSuccess = () => {
    refreshQueue();
  };

  const isQueueAndExtract = queueLoading;
  const hiddenQueuePage = !isQueueAndExtract
    ? "!w-0 !h-0 overflow-hidden "
    : "";
  const hiddenResultPage = isQueueAndExtract ? "!w-0 !h-0 overflow-hidden" : "";

  const getLayoutClassName = (_fullScreen?: boolean) => {
    return {
      left: _fullScreen ? "!w-0  !h-0 overflow-hidden hidden" : "min-w-[50%]",
      right: _fullScreen ? "w-full " : "min-w-[50%]",
    };
  };

  const afterAsyncCheck = async () => {
    return Promise.resolve(true);
  };

  const getExtractionStatusText = (rank: number) => {
    switch (true) {
      case rank > 1:
        return (
          <FormattedMessage
            id="extractor.common.extracting.queue"
            values={{
              id: taskInfo?.rank || 0,
            }}
          />
        );
      case rank === 1:
        return formatMessage({ id: "extractor.common.extracting" });

      default:
        return "";
    }
  };

  return (
    <>
      <div
        className={cls(
          "flex flex-col items-center justify-center w-full h-full",
          hiddenQueuePage
        )}
      >
        <LoadingIcon className="w-12" color={"#0D53DE"} />

        <div className="text-base text-[#121316]/[0.8] mt-4 min-h-6">
          {getExtractionStatusText(taskInfo?.rank)}
        </div>
      </div>

      <div className={cls("h-full w-full", className, hiddenResultPage)}>
        {!compileError ? (
          <div
            className={cls(
              "flex h-[calc(100%-16px)] relative grid ",
              fullScreen ? "grid-cols-1" : "grid-cols-2"
            )}
          >
            <div className={cls(getLayoutClassName(fullScreen).left)}>
              <PDFViewerMemo
                taskInfo={taskInfo}
                onChange={(p) => setPdfState(p)}
              />
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
              <MdViewer
                curPage={curPage}
                taskInfo={taskInfo}
                className="!h-full w-full flex flex-col flex-s"
                fullScreen={fullScreen}
                setFullScreen={(bool?: boolean) => setFullScreen(!!bool)}
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
            <LinearButton
              onClick={async () => {
                await postReUploadExtractTask(String(jobID));
                refreshQueue();
              }}
              className="mt-4"
              text={formatMessage({
                id: "common.retry",
              })}
            />
          </div>
        )}
      </div>
    </>
  );
};

export default PdfExtraction;
