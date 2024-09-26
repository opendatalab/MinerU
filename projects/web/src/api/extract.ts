import axios from "./http";
import { message } from "antd";
import { ExtractTaskType } from "@/types/extract-task-type";
import { getLocale } from "@/utils/locale";

export interface PdfExtractTaskReq {
  fileKey: string;
  fileName: string;
  taskType: ExtractTaskType;
  isOcr?: boolean;
}

export interface SubmitRes {
  filename: string;
  url: string;
  id: string;
}

export const handleErrorMsg = (res: any) => {
  const isCN = getLocale() === "zh-CN";
  const msg = isCN ? res?.data?.error?.msg : res?.data?.error?.msgZh;
  message.error(msg);
  return { data: null, error: res.data?.error };
};

export const postExtractTask = (
  params: PdfExtractTaskReq
): Promise<SubmitRes | null> => {
  return axios
    .post<SubmitRes>(`/api/v2/extract/task/submit`, params)
    .then((res) => {
      if (!res?.data?.error) {
        return res.data.data;
      } else {
        handleErrorMsg(res);
        return null;
      }
    });
};

export const postReUploadExtractTask = (
  id: string
): Promise<SubmitRes | null> => {
  return axios
    .put<SubmitRes>(`/api/v2/extract/task/submit`, {
      id: Number(id),
    })
    .then((res) => {
      if (!res?.data?.error) {
        return res.data.data;
      } else {
        handleErrorMsg(res);
        return null;
      }
    });
};

export interface TaskIdProgress {
  state: "running" | "done" | "pending" | "failed" | "unknown";
  markdownUrl: string[] | [];
  fullMdLink: string;
  content: string[] | [];
  url: string;
  fileName: string;
  thumb: string;
  type: ExtractTaskType | "unknown";
  isTemplate?: boolean;
  fileInfo?: {
    pages: number;
    width: number;
    height: number;
  };
}

export const getExtractTaskIdProgress = async (
  jobID: string | number
): Promise<TaskIdProgress | null> => {
  return axios
    .get<TaskIdProgress>(`/api/v2/extract/task/progress?id=${jobID}`)
    .then((res) => {
      if (res?.data?.error) {
        handleErrorMsg(res);
      }
      return res.data.data;
    });
};

export interface TaskIdResItem {
  queues: number;
  rank: number;
  id?: number;
  url: string;
  fileName?: string;
  fullMdLink?: string;
  type: ExtractTaskType | "unknown";
  state: "running" | "done" | "pending" | "failed" | "unknown";
  markdownUrl: string[];
  file_key?: string;
}

export type TaskIdRes = TaskIdResItem[];

// Get ongoing tasks
export const getPdfExtractQueue = async (): Promise<TaskIdRes | null> => {
  return axios.get<TaskIdRes>(`/api/v2/extract/taskQueue`).then((res) => {
    if (!res?.data?.error) {
      return res.data.data;
    } else {
      handleErrorMsg(res);
      return null;
    }
  });
};

interface TaskHistoryResponse {
  list: TaskItem[];
  total: number;
  pageNo: number;
  pageSize: number;
}

interface TaskItem {
  fileName: string;
  id: string;
  type: string;
  thumb: string;
  state: string; // 提取状态
}

export const getExtractorHistory = ({
  pageNo,
  pageSize,
}: {
  pageNo?: number;
  pageSize?: number;
}): Promise<TaskHistoryResponse | null> => {
  return axios
    .get<TaskHistoryResponse>(
      `/api/v2/extract/list?pageNo=${pageNo || 1}&pageSize=${pageSize || 10}`
    )
    .then((res) => {
      if (!res?.data?.error) {
        return res.data.data;
      } else {
        handleErrorMsg(res);
        return null;
      }
    });
};

export const deleteExtractJob = (jobId: string) => {
  return axios.delete(`/api/v2/extract/task/${jobId}`);
};

interface UploadResponse {
  file_key: string;
  url: string;
}

export const localUpload = (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  return axios.post<UploadResponse>("/api/v2/analysis/upload_pdf", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};

export interface UpdateMarkdownRequest {
  file_key: string;
  data: {
    [pageNumber: string]: string;
  };
}

export interface UpdateMarkdownResponse {
  success: boolean;
  message?: string;
}

export const updateMarkdownContent = async (
  params: UpdateMarkdownRequest
): Promise<UpdateMarkdownResponse | null> => {
  return axios
    .put<UpdateMarkdownResponse>("/api/v2/extract/markdown", params)
    .then((res) => {
      if (!res?.data?.error) {
        return res.data.data;
      } else {
        handleErrorMsg(res);
        return null;
      }
    });
};
