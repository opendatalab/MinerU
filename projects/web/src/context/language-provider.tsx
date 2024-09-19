import React from "react";
import { IntlProvider } from "react-intl";
import { useLanguageStore } from "@/store/languageStore";
import contentEn from "@/locale/en.json";
import contentZh from "@/locale/zh.json";
import sideEn from "@/locale/side/en.ts";
import sideZh from "@/locale/side/zh.ts";
import commonEn from "@/locale/common/en.json";
import commonZh from "@/locale/common/zh.json";
import { Language } from "@/constant";

const messages = {
  [Language.EN_US]: {
    ...contentEn,
    ...sideEn,
    ...commonEn,
  },
  [Language.ZH_CN]: {
    ...contentZh,
    ...sideZh,
    ...commonZh,
  },
};

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { language } = useLanguageStore();

  return (
    <IntlProvider
      messages={messages[language] as unknown as Record<string, string>}
      locale={language}
      defaultLocale="en"
    >
      {children}
    </IntlProvider>
  );
};
