import IconFont from "@/components/icon-font";
import { useIntl } from "react-intl";
import extractorQueueSvg from "@/assets/pdf/extractor-queue.svg";
import { useNavigate, useParams } from "react-router-dom";
import {
  EXTRACTOR_TYPE_LIST,
  ExtractTaskType,
} from "@/types/extract-task-type";
import cls from "classnames";
import { useLatest, useRequest } from "ahooks";
import { deleteExtractJob, getExtractorHistory } from "@/api/extract";
import { message, Popconfirm, Tooltip } from "antd";
import { useEffect } from "react";
import { ADD_TASK_LIST, UPDATE_TASK_LIST } from "@/constant/event";
import { findIndex } from "lodash";
import { TextTooltip } from "@/components/text-tooltip";

interface ExtractorQueueProps {
  className?: string;
}

const ExtractorQueue: React.FC<ExtractorQueueProps> = ({ className }) => {
  const { formatMessage, locale } = useIntl();
  const navigate = useNavigate();

  const params = useParams();

  const { data: taskList, mutate } = useRequest(() => {
    return getExtractorHistory({
      pageNo: 1,
      pageSize: 100,
    }).then((res) => {
      return res?.list?.filter((i) => !!i.id && !!i.type) || [];
    });
  });

  let timeout: NodeJS.Timeout | null = null;

  const activeClassName = "!bg-[#0d53de]/[0.05] !text-[#0D53DE]";
  const handleExtractor = (originType: ExtractTaskType, id: string) => {
    const type = originType?.split("-")[0];
    const detailType = originType?.split("-")[1];

    if (type === EXTRACTOR_TYPE_LIST.formula.toLowerCase()) {
      navigate(`/OpenSourceTools/Extractor/formula/${id}?type=${detailType}`);
    } else if (type === EXTRACTOR_TYPE_LIST.pdf.toLowerCase()) {
      navigate(`/OpenSourceTools/Extractor/PDF/${id}`);
    } else if (type === EXTRACTOR_TYPE_LIST.table.toLocaleLowerCase()) {
      navigate(`/OpenSourceTools/Extractor/table/${id}`);
    }
    return;
  };

  const cancel = (e?: React.MouseEvent<HTMLElement>) => {
    e?.stopPropagation();
    e?.preventDefault();
  };

  const confirm = (id: string) => {
    const deleteIndex = findIndex(taskList, (i) => i.id === id);
    const nextJob = taskList?.[deleteIndex + 1]
      ? taskList?.[deleteIndex + 1]
      : taskList?.[deleteIndex - 1];
    mutate(taskList?.filter((i) => i.id !== id));
    deleteExtractJob(id).then(() => {
      message.success(formatMessage({ id: "extractor.queue.delete.success" }));
    });
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => {
      if (nextJob?.id) {
        handleExtractor(nextJob?.type as any, nextJob?.id);
      } else {
        navigate("/OpenSourceTools/Extractor");
      }
    }, 10);
  };
  const taskListRef = useLatest(taskList);
  const handleAddList = ({ detail }: CustomEvent) => {
    const taskData = detail as any;
    mutate(
      [
        {
          fileName: taskData?.fileName,
          id: taskData?.id,
          type: taskData?.type,
          state: taskData?.state, // 提取状态
        } as any,
      ].concat(taskListRef?.current)
    );
  };

  useEffect(() => {
    const handleUpdateList = ({ detail }: CustomEvent) => {
      const taskData = detail as any;
      taskListRef?.current?.forEach((i) => {
        if (i.id === taskData?.id) {
          i.state = taskData?.state || taskData?.state;
        }
      });

      mutate(taskListRef?.current);
    };
    document.addEventListener(
      UPDATE_TASK_LIST,
      handleUpdateList as EventListener
    );
    document.addEventListener(ADD_TASK_LIST, handleAddList as EventListener);
    return () => {
      document.removeEventListener(
        UPDATE_TASK_LIST,
        handleUpdateList as EventListener
      );
      document.removeEventListener(
        ADD_TASK_LIST,
        handleAddList as EventListener
      );
    };
  }, []);

  useEffect(() => {
    mutate(taskListRef?.current);
  }, [locale]);

  return (
    <div className={cls("w-full flex flex-col mb-3", className)}>
      <header className="flex items-center px-2 py-[0.625rem] text-[#121316]/[0.8] text-[0.875rem] font-semibold">
        <img
          src={extractorQueueSvg}
          className="w-6 h-6 mr-2 "
          alt="extractorQueueSvg"
        />
        {formatMessage({
          id: "extractor.queue",
        })}
      </header>
      <hgroup className="overflow-auto flex-1 scrollbar-thin">
        {taskList?.map((i, index) => {
          return (
            <div
              className={cls(
                "group h-[2.5rem] flex items-center px-4 py-2.5 mb-1 text-[#121316]/[0.8] pl-10 text-sm rounded h-10 flex items-center cursor-pointer hover:bg-[#0d53de]/[0.05]",
                params?.jobID === String(i?.id) && activeClassName
              )}
              key={i?.fileName + index + i?.id}
              onClick={() => handleExtractor(i.type as any, i.id)}
            >
              <span className="truncate mr-2 max-w-[calc(100%-2rem)]">
                <TextTooltip trigger="hover" str={i?.fileName} />
              </span>
              <>
                {i?.state === "failed" && (
                  <Tooltip
                    title={formatMessage({
                      id: "extractor.error",
                    })}
                  >
                    <IconFont
                      type={"icon-attentionFilled"}
                      className="text-[#FF8800] mr-1"
                    />
                  </Tooltip>
                )}

                <Popconfirm
                  title={formatMessage({ id: "extractor.queue.delete" })}
                  description={<div className="my-4"></div>}
                  onConfirm={(e) => {
                    e?.stopPropagation();
                    e?.preventDefault();
                    confirm(i.id);
                  }}
                  onCancel={cancel}
                  okText={formatMessage({ id: "common.confirm" })}
                  cancelText={formatMessage({ id: "common.cancel" })}
                  okButtonProps={{
                    style: {
                      backgroundColor: "#F5483B",
                    },
                  }}
                >
                  <IconFont
                    onClick={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                    }}
                    type="icon-shanchu"
                    className="hidden ml-auto text-[1rem] text-[#121316]/[0.8] hover:text-[#0D53DE] group-hover:block"
                  />
                </Popconfirm>
              </>
            </div>
          );
        })}
      </hgroup>
    </div>
  );
};

export default ExtractorQueue;
