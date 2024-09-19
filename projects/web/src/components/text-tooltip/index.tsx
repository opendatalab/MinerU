import { Popover, Tooltip } from 'antd';
import React, { useRef, useState } from 'react';
import styles from './index.module.scss';

import { RefObject } from 'react';
import { useIsomorphicLayoutEffect, useMemoizedFn } from 'ahooks';

export function useResizeEffect<T extends HTMLElement>(effect: (target: T) => void, targetRef: RefObject<T>) {
  const fn = useMemoizedFn(effect);
  useIsomorphicLayoutEffect(() => {
    const target = targetRef.current;
    if (!target) return;
    if (window.ResizeObserver) {
      let animationFrame: number;
      const observer = new ResizeObserver(() => {
        animationFrame = window.requestAnimationFrame(() => fn(target));
      });
      observer.observe(target);
      return () => {
        window.cancelAnimationFrame(animationFrame);
        observer.disconnect();
      };
    } else {
      fn(target);
    }
  }, [targetRef]);
}

interface ITextTooltip {
  style?: React.CSSProperties;
  str: string;
  suffix?: React.ReactNode | string;
  trigger?: 'hover' | 'click';
  handleClick?: () => void;
}

export const TextTooltip = (props: ITextTooltip) => {
  const { style = {}, str, trigger = 'click', suffix, handleClick } = props;
  const rootRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [clickable, setClickable] = useState(false);
  function calcEllipsised() {
    // 没有被截断
    if (tooltipRef!?.current!?.scrollWidth > tooltipRef!?.current!?.clientWidth) {
      setClickable(true);
    } else {
      setClickable(false);
    }
  }
  useResizeEffect(calcEllipsised, rootRef);

  return (
    <Tooltip
      title={<div className="bg-black/[0.85] text-white p-[6px]">{str}</div>}
      trigger={clickable ? trigger : ('' as 'click')}
      overlayClassName={styles.textTooltip}
      style={{ width: '100%' }}
      zIndex={999999}
      placement="right"
      align={{
        offset: [72, 0]
      }}
    >
      <div style={{ width: '100%', ...style }} className="flex" ref={rootRef}>
        <div className="text-ellipsis overflow-hidden whitespace-nowrap" ref={tooltipRef}>
          <span onClick={() => handleClick?.()}>{str}</span>
        </div>
        {suffix}
      </div>
    </Tooltip>
  );
};
