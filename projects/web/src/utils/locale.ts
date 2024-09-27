export enum ELocal {
  "zh-CN" = "zh-CN",
  "en-US" = "en-US",
}
export const locale: { [key: string]: string } = {
  [ELocal["zh-CN"]]: "中文",
  [ELocal["en-US"]]: "En",
};
export const localeName: { [key: string]: string } = {
  [ELocal["zh-CN"]]: "nameZh",
  [ELocal["en-US"]]: "name",
};

export const getLocale = () => {
  return localStorage.getItem("umi_locale") || ELocal["zh-CN"];
};
