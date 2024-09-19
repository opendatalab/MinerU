import classNames from "classnames";
import style from "./index.module.scss";

const LoadingIcon = ({
  color,
  className,
}: {
  color: string;
  className?: string;
}) => {
  return (
    <div className={classNames(style.container, className)}>
      <div
        className={style.dotPulse}
        style={{ "--color": color || "grey" } as any}
      ></div>
    </div>
  );
};

export default LoadingIcon;
