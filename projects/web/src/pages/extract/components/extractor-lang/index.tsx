import LangChangeIcon from "@/assets/pdf/lang-change.svg";
import { useLanguageStore } from "@/store/languageStore";
import cls from "classnames";

interface ExtractorLangProps {
  className?: string;
}

const ExtractorLang: React.FC<ExtractorLangProps> = ({ className }) => {
  const { toggleLanguage } = useLanguageStore();
  const changeLang = () => {
    toggleLanguage?.();
  };
  return (
    <>
      <img
        onClick={() => changeLang()}
        src={LangChangeIcon}
        alt="LangChangeIcon"
        className={cls(
          "w-[1.5rem] h-[1.5rem] cursor-pointer object-cover hover:bg-[#0D53DE]/[0.1] rounded cursor-pointer",
          className
        )}
      />
    </>
  );
};

export default ExtractorLang;
