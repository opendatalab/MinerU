import React, { PropsWithChildren } from "react";
import ReactCodeMirror, { EditorView, Extension } from "@uiw/react-codemirror";
import { loadLanguage } from "@uiw/codemirror-extensions-langs";
import cls from "classnames";
import style from "./index.module.scss";
// import { scrollPastEnd } from "@codemirror/view";

interface IProps {
  className?: string;
  editable?: boolean;
  language?: "json" | "markdown" | "yaml";
  value: string;
  onChange?: (value: string) => void;
  lineWrapping?: boolean;
  onBeforeChange?: (editor: any, data: any, value: any) => void;
}

const CodeMirror: React.FC<PropsWithChildren<IProps>> = ({
  language = "markdown",
  value,
  className,
  onChange,
  lineWrapping,
  onBeforeChange,
  editable,
}) => {
  //   const noScrollPastEnd = scrollPastEnd();
  const extensions = [
    {
      ext: EditorView.lineWrapping,
      on: lineWrapping,
    },
    {
      ext: loadLanguage(language),
      on: true,
    },
  ]
    .map((i) => (i.on ? i.ext : null))
    .filter(Boolean) as Extension[];

  return (
    <ReactCodeMirror
      className={cls("odl-code-mirror", className, style.codeMirror)}
      value={value}
      theme="light"
      basicSetup={{
        lineNumbers: false,
        highlightActiveLineGutter: false,
        foldGutter: false,

        highlightActiveLine: false,
        // syntaxHighlighting: true,
      }}
      editable={editable}
      extensions={extensions}
      onChange={(v) => {
        onChange?.(v);
      }}
    />
  );
};
export default React.memo(CodeMirror);
