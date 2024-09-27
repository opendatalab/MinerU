import githubSvg from "@/assets/pdf/github.svg";
import { windowOpen } from "@/utils/windowOpen";
import styles from "./index.module.scss";
import cls from "classnames";

const ExtractorRepo = () => {
  return (
    <div
      className={cls(styles.githubBtn)}
      onClick={() =>
        windowOpen("https://github.com/opendatalab/MinerU", "_blank")
      }
    >
      <span className="text-sm ">
        <img src={githubSvg} className="mr-2" />
        <span className="!text-[14px] ml-[0.5rem]">🎉</span>
      </span>
    </div>
  );
};

export default ExtractorRepo;
