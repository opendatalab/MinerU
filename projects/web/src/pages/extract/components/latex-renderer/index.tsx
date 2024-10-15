import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import style from './index.module.scss';
import classNames from 'classnames';

interface LatexRendererProps {
  formula: string;
  className?: string;
  'aria-label'?: string;
  title?: string;
}

function LatexRenderer({ formula, className = '', 'aria-label': ariaLabel, title }: LatexRendererProps) {
  try {
    return (
      <div
        className={`${className} max-w-[100%] max-h-[100%] scrollbar-thin  ${style.customStyle}`}
        aria-label={ariaLabel}
      >
        <BlockMath math={formula} className="scrollbar-thin" />
      </div>
    );
  } catch (error) {
    console.error('Error rendering Latex:', error);
    return <div>Unable to render Latex formula.</div>;
  }
}

export default LatexRenderer;
