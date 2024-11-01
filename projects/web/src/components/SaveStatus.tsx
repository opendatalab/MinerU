import React, {
  useState,
  useEffect,
  useImperativeHandle,
  forwardRef,
} from "react";
import cls from "classnames";

interface SaveStatusProps {
  className?: string;
}

export interface SaveStatusRef {
  triggerSave: () => void;
  reset: () => void; // 新增的重置方法
}

const SaveStatus = forwardRef<SaveStatusRef, SaveStatusProps>(
  ({ className }, ref) => {
    const [lastSaveTime, setLastSaveTime] = useState<Date | null>(null);
    const [showSaved, setShowSaved] = useState(false);
    const [timeSinceLastSave, setTimeSinceLastSave] = useState(0);

    useImperativeHandle(ref, () => ({
      triggerSave: () => {
        setLastSaveTime(new Date());
        setShowSaved(true);
      },
      reset: () => {
        // 新增的重置方法
        setLastSaveTime(null);
        setShowSaved(false);
        setTimeSinceLastSave(0);
      },
    }));

    useEffect(() => {
      if (showSaved) {
        const timer = setTimeout(() => {
          setShowSaved(false);
        }, 10000);
        return () => clearTimeout(timer);
      }
    }, [showSaved]);

    useEffect(() => {
      const updateTimeSinceLastSave = () => {
        if (lastSaveTime) {
          const now = new Date();
          const diffInMinutes = Math.floor(
            (now.getTime() - lastSaveTime.getTime()) / 60000
          );
          if (diffInMinutes > 0) {
            setTimeSinceLastSave(diffInMinutes);
          }
        }
      };

      const timer = setInterval(updateTimeSinceLastSave, 60000);
      updateTimeSinceLastSave(); // 立即更新一次
      return () => clearInterval(timer);
    }, [lastSaveTime]);

    return (
      <div className={cls("flex items-center", className)}>
        {showSaved && (
          <span className="text-[#121316]/[0.6] text-[13px] leading-[24px]">
            已保存
          </span>
        )}
        {timeSinceLastSave > 0 && !showSaved && lastSaveTime && (
          <span className="text-[#121316]/[0.6] text-[13px] leading-[24px]">
            最近修改：{timeSinceLastSave} 分钟前
          </span>
        )}
        {(showSaved ||
          (timeSinceLastSave > 0 && !showSaved && lastSaveTime)) && (
          <span className="w-[1px] h-[0.75rem] bg-[#D7D8DD] ml-[1rem] block"></span>
        )}
      </div>
    );
  }
);

export default SaveStatus;
