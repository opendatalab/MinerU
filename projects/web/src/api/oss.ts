import http from "./http";
import { UploadFile } from "antd/es/upload/interface";
import { stringify } from "qs";
import { localUpload } from "./extract";

interface IUploadRes {
  downloadUrl: string;
  headers: any;
  key: string;
  previewUrl: string;
  putUrl: string;
}

/**
 * 获取上传链接的接口
 * openRead 是否返回公开链接
 */
export const uploadUrl = (
  fileName: string,
  openRead: boolean,
  fileType = ""
): Promise<IUploadRes> =>
  http
    .post<IUploadRes>(
      `/datasets/api/v2/file?${stringify({ fileName, openRead, fileType })}`
    )
    .then((res) => res.data.data);

export interface IFile extends UploadFile {
  url?: string;
  thumbUrl?: string;
  objectKey?: string;
}

// 获取 阿里云上传链接
export const uploadToOss = async (
  file: IFile,
  openRead: boolean,
  fileType = ""
): Promise<any> => {
  const { downloadUrl, previewUrl, key, putUrl, headers } = await uploadUrl(
    file.name,
    openRead,
    fileType
  );
  file.url = downloadUrl;
  file.thumbUrl = previewUrl;
  file.objectKey = key;
  // 上传文件
  await http.put(putUrl, file, { headers: { ...headers } });
  return {
    downloadUrl,
    objectKey: key,
    thumbUrl: previewUrl,
    url: downloadUrl,
  };
};

interface IUploadOptions {
  openRead: boolean;
  fileType: string;
  uploadType?: "local" | "oss";
}

// 可覆盖 antd customUpload 的上传逻辑
export const customUploadToOss = async (
  options: any,
  otherUploadOptions: IUploadOptions
) => {
  const { openRead, fileType, uploadType = "local" } = otherUploadOptions;
  const uploadFile = async () => {
    switch (uploadType) {
      case "oss":
        return await uploadToOss(options.file, openRead, fileType)
          .then((res) => {
            options?.onSuccess({ ...res, ...options.file });
            return res;
          })
          .catch((error) => {
            options?.onError(error);
            return {};
          });
      case "local":
        return await localUpload(options.file);
      default:
        throw new Error(`Unsupported upload type: ${uploadType}`);
    }
  };
  try {
    const res = await uploadFile();
    options?.onSuccess({ ...res, ...options.file });
  } catch (error) {
    options?.onError(error);
  }
};
