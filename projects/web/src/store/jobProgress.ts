import {
  getExtractTaskIdProgress,
  getPdfExtractQueue,
  TaskIdResItem,
} from "@/api/extract";
import { create } from "zustand";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { UPDATE_TASK_LIST } from "@/constant/event";
import { useQuery } from "@tanstack/react-query";

interface ExtractorState {
  taskInfo: TaskIdResItem;
  queueLoading: boolean | null;
  interfaceError: boolean;
  setTaskInfo: (taskInfo: TaskIdResItem) => void;
  setQueueLoading: (loading: boolean | null) => void;
  setInterfaceError: (error: boolean) => void;
}

const defaultTaskInfo: TaskIdResItem = {
  id: 0,
  rank: 0,
  state: "pending",
  url: "",
  type: "unknown",
  queues: -1,
};

const useExtractorStore = create<ExtractorState>((set) => ({
  taskInfo: defaultTaskInfo,
  queueLoading: null,
  interfaceError: false,
  setTaskInfo: (taskInfo: any) => set({ taskInfo }),
  setQueueLoading: (loading) => set({ queueLoading: loading }),
  setInterfaceError: (error) => set({ interfaceError: error }),
}));

export const useJobExtraction = () => {
  const { jobID } = useParams<{ jobID: string }>();
  const {
    setTaskInfo,
    setQueueLoading,
    queueLoading,
    taskInfo,
    interfaceError,
    setInterfaceError,
  } = useExtractorStore();

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isPolling, setIsPolling] = useState(true);

  const stopTaskLoading = () => {
    setQueueLoading(false);
  };

  // Query for task progress
  const taskProgressQuery = useQuery({
    queryKey: ["taskProgress", jobID],
    queryFn: () => {
      setQueueLoading(true);
      setIsPolling(true);
      return getExtractTaskIdProgress(jobID!)
        .then((res) => {
          if (res?.state === "done" || res?.state === "failed") {
            stopTaskLoading();

            document.dispatchEvent(
              new CustomEvent("UPDATE_TASK_LIST", {
                detail: { state: res.state, id: jobID },
              })
            );
          }
          if (res) {
            setTaskInfo(res);
          }

          return res;
        })
        .catch(() => {
          stopTaskLoading();
          setTaskInfo({ state: "failed" });
        });
    },
    enabled: false,
  });

  // Query for queue status
  const queueStatusQuery = useQuery({
    queryKey: ["queueStatus", jobID],
    queryFn: async () => {
      setQueueLoading(true);
      const response = await getPdfExtractQueue(jobID).then((res) => {
        // setTaskInfo({ rand: "failed" });

        if (res) {
          const targetPendingRunningJob = res?.filter(
            (i) => String(i.id) === jobID
          )?.[0];

          if (targetPendingRunningJob) {
            setTaskInfo(targetPendingRunningJob);
          } else {
            setIsPolling(false);
            setQueueLoading(false);
            getExtractTaskIdProgress(jobID!).then((res) => {
              setTaskInfo(res as any);
            });
          }
        }

        return res;
      });
      return response;
    },

    enabled:
      isPolling &&
      (taskProgressQuery?.data?.state === "running" ||
        taskProgressQuery?.data?.state === "pending"),
    refetchInterval: 2000, // Poll every 2 seconds
  });

  useEffect(() => {
    if (taskProgressQuery.data?.state === "done") {
      stopTaskLoading();
      setInterfaceError(false);
      setIsPolling(false);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      } else {
        timeoutRef.current = setTimeout(() => {
          document.dispatchEvent(
            new CustomEvent(UPDATE_TASK_LIST, {
              detail: { state: "done", jobID },
            })
          );
        }, 10);
      }
    } else if (taskProgressQuery.data?.state === "failed") {
      stopTaskLoading();
      setInterfaceError(true);

      setIsPolling(false);

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      } else {
        timeoutRef.current = setTimeout(() => {
          document.dispatchEvent(
            new CustomEvent(UPDATE_TASK_LIST, {
              detail: { state: "failed", jobID },
            })
          );
        }, 10);
      }
    }
    // TIP这里得用taskInfo
  }, [taskProgressQuery.data]);

  const refreshQueue = () => {
    // stop last ID polling
    setIsPolling(false);
    setTaskInfo(defaultTaskInfo);
    taskProgressQuery.refetch();
  };

  useEffect(() => {
    if (jobID) {
      // stop last ID polling d
      setTaskInfo(defaultTaskInfo);
      taskProgressQuery.refetch();
    }
  }, [jobID]);

  return {
    taskInfo: taskInfo,
    isLoading: queueLoading,
    isError:
      interfaceError || taskProgressQuery.isError || queueStatusQuery.isError,
    refreshQueue,
  };
};
