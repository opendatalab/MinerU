import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";
import Cookies from "js-cookie";
import { message } from "antd";

interface ApiResponse<T> {
  code: number;
  msg: string;
  data: T;
}

interface ErrorResponse {
  code: number;
  msg: string;
  msgZh: string;
}

export interface ProcessedResponse<T> {
  data: T;
  error: null | Pick<ErrorResponse, "msg" | "msgZh">;
}

interface CustomAxiosInstance
  extends Omit<AxiosInstance, "get" | "post" | "put" | "delete"> {
  get<T, R = AxiosResponse<ProcessedResponse<T>>>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<R>;
  post<T, R = AxiosResponse<ProcessedResponse<T>>>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig
  ): Promise<R>;
  put<T, R = AxiosResponse<ProcessedResponse<T>>>(
    url: string,
    data?: unknown,
    config?: AxiosRequestConfig
  ): Promise<R>;
  delete<T, R = AxiosResponse<ProcessedResponse<T>>>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<R>;
}

const instance: CustomAxiosInstance = axios.create({
  baseURL: "",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

const processResponse = <T>(
  response: AxiosResponse<ApiResponse<T>>
): ProcessedResponse<T> => {
  if (response.data.code === 200) {
    return {
      data: response.data.data,
      error: null,
    };
  } else {
    return {
      data: response.data.data || ({} as T),
      error: {
        msg: response.data.msg,
        msgZh: (response.data as unknown as ErrorResponse).msgZh,
      },
    };
  }
};

instance.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

instance.interceptors.response.use(
  <T>(
    response: AxiosResponse<ApiResponse<T>>
  ): AxiosResponse<ProcessedResponse<T>> => {
    return { ...response, data: processResponse(response) };
  },
  (error) => {
    message.error(error?.response?.data?.msg || "Error");
    return Promise.reject(error);
  }
);

export default instance;
