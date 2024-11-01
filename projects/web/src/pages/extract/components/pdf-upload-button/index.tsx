import { Button } from "antd";
import cls from "classnames";
import UploadingOutlined from "@/assets/imgs/online.experience/UploadingOutlined.svg";
import styles from "./index.module.scss";
import Upload from "@/components/upload";
import { ReactNode } from "react";
import {
  postExtractTask,
  postReUploadExtractTask,
  SubmitRes,
} from "@/api/extract";

import { useParams } from "react-router-dom";
import { ExtractTaskType } from "@/types/extract-task-type";

interface IPdfUploadButtonProps {
  afterUploadSuccess?: (submitTask: SubmitRes) => void;
  afterAsyncCheck?: () => Promise<boolean>;
  text?: string | ReactNode;
  className?: string;
  showIcon?: boolean;
  beforeUpload?: () => void;
  onUploadError?: () => void;
  accept: string;
  extractType: ExtractTaskType;
  taskType?: string;
  submitType?: "submit" | "reUpload";
  isOcr?: boolean;
}

interface ILinearButtonProps {
  className?: string;
  text?: string | ReactNode;
  onClick?: () => void;
}

export const LinearButton = ({
  className = "",
  onClick,
  text,
}: ILinearButtonProps) => {
  return (
    <button
      onClick={() => onClick?.()}
      className={cls(styles.linearBtn, className)}
    >
      {text}
    </button>
  );
};

export const ExtractorUploadButton = ({
  text = "上传PDF",
  className = "",
  afterAsyncCheck,
  afterUploadSuccess,
  beforeUpload: beforeLocalUpload,
  onUploadError,
  showIcon = true,
  accept,
  extractType,
  taskType,
  submitType,
  isOcr,
}: IPdfUploadButtonProps) => {
  const urlParams = useParams();
  const beforeUpload = async () => {
    beforeLocalUpload?.();
    const isCheck = await afterAsyncCheck?.();
    return isCheck;
  };
  const onChange = async (pdfFile: any) => {
    if (pdfFile?.file?.status === "done") {
      const res =
        submitType === "reUpload"
          ? await postReUploadExtractTask(String(urlParams?.jobID))
          : await postExtractTask({
              fileKey: pdfFile?.file?.response?.data?.data?.file_key,
              fileName: pdfFile?.file?.name,
              taskType: extractType,
              isOcr,
            });

      if (res) {
        if (!("error" in res)) {
          afterUploadSuccess?.({
            ...(res || {}),
            type: extractType,
          } as any);
        } else {
          onUploadError?.();
        }
      } else {
        onUploadError?.();
      }
    }
  };

  return (
    <>
      <Upload
        isDragger
        accept={accept}
        className={cls(styles.gradientBtn, "mb-4", className)}
        beforeUpload={beforeUpload}
        showUploadList={false}
        onChange={onChange}
        openRead={true}
        taskType={taskType}
      >
        <div className="flex justify-center items-center ">
          {showIcon && <img src={UploadingOutlined} className="mr-1" />}
          {text}
        </div>
      </Upload>
    </>
  );
};
