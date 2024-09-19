import LoadingAnimation from "@/components/loading-animation";
import { ExclamationCircleFilled } from "@ant-design/icons";
import cls from "classnames";

export const IframeLoading = ({
  filename,
  type,
  text,
  errorElement,
  classNameTitle = "",
  showHeader,
}: {
  filename?: string;
  type: "loading" | "error";
  text?: string;
  errorElement?: React.ReactElement;
  classNameTitle?: string;
  showHeader?: boolean;
}) => {
  return (
    <div className="flex flex-col h-full text-sm text-[#121316]/[0.8] whitespace-nowrap ">
      {showHeader && (
        <div
          className={cls(
            "h-[47px] border-0 border-solid border-b-[1px] border-[#EBECF0] w-full pl-[24px]",
            classNameTitle
          )}
        >
          {filename}
        </div>
      )}

      <div className="flex-1 flex justify-center items-center">
        {type === "error" ? (
          errorElement ? (
            errorElement
          ) : (
            <>
              <ExclamationCircleFilled
                style={{ color: "#FF8800" }}
                rotate={180}
              />
              <span className="ml-2.5">上传失败，请</span>
              <span className="text-[#0D53DE] ml-1 cursor-pointer">
                重新上传
              </span>
            </>
          )
        ) : (
          <>
            <LoadingAnimation />
            <span className="ml-2.5">{text || "PDF 上传中，请稍等..."}</span>
          </>
        )}
      </div>
    </div>
  );
};
