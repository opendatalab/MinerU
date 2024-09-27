import React, {
  useState,
  useEffect,
  useImperativeHandle,
  forwardRef,
} from "react";

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
    const [timeSinceLastSave, setTimeSinceLastSave] = useState("");

    useImperativeHandle(ref, () => ({
      triggerSave: () => {
        setLastSaveTime(new Date());
        setShowSaved(true);
      },
      reset: () => {
        // 新增的重置方法
        setLastSaveTime(null);
        setShowSaved(false);
        setTimeSinceLastSave("");
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
            setTimeSinceLastSave(`${diffInMinutes} 分钟前`);
          }
        }
      };

      const timer = setInterval(updateTimeSinceLastSave, 60000);
      updateTimeSinceLastSave(); // 立即更新一次
      return () => clearInterval(timer);
    }, [lastSaveTime]);

    return (
      <div className={className}>
        {showSaved && <span>已保存</span>}
        {!showSaved && lastSaveTime && (
          <span>最近修改：{timeSinceLastSave}</span>
        )}
      </div>
    );
  }
);

export default SaveStatus;
