// cl 2022/4/21 18:22
import { customUploadToOss } from "@/api/oss";
import { Upload as AntdUpload } from "antd";
import { DraggerProps, UploadProps } from "antd/es/upload";
import React from "react";

interface IProps extends UploadProps, DraggerProps {
  isDragger?: boolean;
  openRead?: boolean;
  taskType?: string;
  changeOption?: (option: any) => any;
}

const Upload: React.FC<IProps> = (props) => {
  const { isDragger, openRead, taskType, changeOption, ...rest } = props;
  const Component = isDragger ? AntdUpload.Dragger : AntdUpload;

  return (
    <Component
      {...rest}
      customRequest={(options: any) =>
        customUploadToOss(changeOption ? changeOption?.(options) : options, {
          openRead: openRead || false,
          fileType: taskType || "pdf",
          uploadType: "local",
        })
      }
    >
      {props.children}
    </Component>
  );
};

export default Upload;
