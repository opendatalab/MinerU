import React, {
  useEffect,
  useRef,
  useState,
  useMemo,
  forwardRef,
  useImperativeHandle,
  useCallback,
} from "react";
import cls from "classnames";
import { isObjEqual } from "@/utils/render";
import { useSize } from "ahooks";

interface IImageLayersViewerProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  layout: Array<{
    category_id: number;
    poly: number[];
    score: number;
    latex?: string;
  }>;
  layerVisible?: boolean;
  disableZoom?: boolean;
  className?: string;
  onChange?: (data: { scale: number }) => void;
}

export interface ImageLayerViewerRef {
  containerRef: HTMLDivElement | null;
  zoomIn: () => void;
  zoomOut: () => void;
  scale: number;
  updateScaleAndPosition: () => void;
}

const ImageLayerViewer = forwardRef<
  ImageLayerViewerRef,
  IImageLayersViewerProps
>(
  (
    {
      imageUrl,
      imageHeight,
      imageWidth,
      onChange,
      layout,
      disableZoom,
      className = "",
      layerVisible = true,
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const imageCanvasRef = useRef<HTMLCanvasElement>(null);
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
    const rafRef = useRef<number | null>(null);
    const containerSize = useSize(containerRef);

    const [scale, setScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [padding, setPadding] = useState({ left: 0, top: 0 });

    const minZoom = 0.1;
    const maxZoom = 3;
    const zoomSensitivity = 0.001;
    const zoomStep = 0.1;

    const dpr = useMemo(() => window.devicePixelRatio || 1, []);

    const image = useMemo(() => {
      const img = new Image();
      img.src = imageUrl;
      return img;
    }, [imageUrl]);

    const calculateInitialScaleAndPosition = useCallback(() => {
      if (!containerRef.current)
        return { initialScale: 1, initialPosition: { x: 0, y: 0 } };
      const containerWidth = containerRef.current.clientWidth;
      const containerHeight = containerRef.current.clientHeight;

      const scaleX = containerWidth / imageWidth;
      const scaleY = containerHeight / imageHeight;
      const initialScale = Math.min(scaleX, scaleY, 1); // Ensure it doesn't scale up initially

      const scaledWidth = imageWidth * initialScale;
      const scaledHeight = imageHeight * initialScale;

      const initialPosition = {
        x: (containerWidth - scaledWidth) / 2,
        y: (containerHeight - scaledHeight) / 2,
      };

      return { initialScale, initialPosition };
    }, [imageWidth, imageHeight]);

    const updateScaleAndPosition = useCallback(() => {
      const { initialScale, initialPosition } =
        calculateInitialScaleAndPosition();
      setScale(initialScale);
      setPosition(initialPosition);
      setPadding({ left: 0, top: 0 });
    }, [calculateInitialScaleAndPosition]);

    useEffect(() => {
      updateScaleAndPosition();
    }, [imageWidth, imageHeight]);

    const drawImage = useCallback(() => {
      const ctx = imageCanvasRef.current?.getContext("2d");
      if (!ctx || !image.complete) return;

      const scaledWidth = imageWidth * scale;
      const scaledHeight = imageHeight * scale;

      ctx.canvas.width = scaledWidth * dpr;
      ctx.canvas.height = scaledHeight * dpr;
      ctx.canvas.style.width = `${scaledWidth}px`;
      ctx.canvas.style.height = `${scaledHeight}px`;

      ctx.scale(dpr, dpr);

      ctx.clearRect(0, 0, scaledWidth, scaledHeight);
      ctx.drawImage(image, 0, 0, scaledWidth, scaledHeight);
    }, [image, imageWidth, imageHeight, scale, dpr]);

    const drawLayout = useCallback(() => {
      const ctx = overlayCanvasRef.current?.getContext("2d");
      if (!ctx) return;

      const scaledWidth = imageWidth * scale;
      const scaledHeight = imageHeight * scale;

      ctx.canvas.width = scaledWidth * dpr;
      ctx.canvas.height = scaledHeight * dpr;
      ctx.canvas.style.width = `${scaledWidth}px`;
      ctx.canvas.style.height = `${scaledHeight}px`;

      ctx.scale(dpr, dpr);

      ctx.clearRect(0, 0, scaledWidth, scaledHeight);

      layout?.forEach((item) => {
        const [x1, y1, x2, y2, x3, y3, x4, y4] = item.poly.map(
          (coord) => coord * scale
        );

        switch (item.category_id) {
          case 9:
            ctx.fillStyle = "rgba(230, 113, 230, 0.4)";
            ctx.strokeStyle = "rgba(230, 113, 230, 1)";
            break;
          case 8:
            ctx.fillStyle = "rgba(240, 240, 124, 0.4)";
            ctx.strokeStyle = "rgba(240, 240, 124, 1)";
            break;
          case 13:
            ctx.fillStyle = "rgba(150, 232, 172, 0.4)";
            ctx.strokeStyle = "rgba(150, 232, 172, 1)";
            break;
          case 14:
            ctx.fillStyle = "rgba(230, 122, 171, 0.4)";
            ctx.strokeStyle = "rgba(230, 122, 171, 1)";
            break;
          default:
            ctx.fillStyle = "transparent";
            ctx.strokeStyle = "transparent";
        }

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x3, y3);
        ctx.lineTo(x4, y4);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      });
    }, [layout, scale, dpr]);

    const updateScale = useCallback(
      (newScale: number, clientX: number, clientY: number) => {
        if (containerRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          const containerWidth = rect.width;
          const containerHeight = rect.height;
          const x = clientX - rect.left;
          const y = clientY - rect.top;

          const prevScaledWidth = imageWidth * scale;
          const prevScaledHeight = imageHeight * scale;
          const newScaledWidth = imageWidth * newScale;
          const newScaledHeight = imageHeight * newScale;

          let newPosition = {
            x:
              position.x -
              ((x - position.x) * (newScaledWidth - prevScaledWidth)) /
                prevScaledWidth,
            y:
              position.y -
              ((y - position.y) * (newScaledHeight - prevScaledHeight)) /
                prevScaledHeight,
          };

          // Center the image if it's smaller than the container
          if (newScaledWidth < containerWidth) {
            newPosition.x = (containerWidth - newScaledWidth) / 2;
          }
          if (newScaledHeight < containerHeight) {
            newPosition.y = (containerHeight - newScaledHeight) / 2;
          }

          setScale(newScale);
          setPosition(newPosition);

          // Calculate new padding
          const newPadding = {
            left: Math.max(0, -newPosition.x),
            top: Math.max(0, -newPosition.y),
          };
          setPadding(newPadding);
        }
      },
      [scale, position, imageWidth, imageHeight]
    );

    const handleZoom = useCallback(
      (delta: number, clientX: number, clientY: number) => {
        const newScale = scale * Math.exp(-delta * zoomSensitivity);
        const boundedNewScale = Math.max(minZoom, Math.min(newScale, maxZoom));

        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
        }

        rafRef.current = requestAnimationFrame(() => {
          updateScale(boundedNewScale, clientX, clientY);
        });
      },
      [scale, updateScale]
    );

    const handleCenterZoom = useCallback(
      (zoomIn: boolean) => {
        const newScale = zoomIn
          ? scale * (1 + zoomStep)
          : scale / (1 + zoomStep);
        const boundedNewScale = Math.max(minZoom, Math.min(newScale, maxZoom));

        if (containerRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          const centerX = rect.width / 2;
          const centerY = rect.height / 2;

          updateScale(boundedNewScale, centerX, centerY);
        }
      },
      [scale, updateScale]
    );

    const zoomIn = useCallback(() => {
      handleCenterZoom(true);
    }, [handleCenterZoom]);

    const zoomOut = useCallback(() => {
      handleCenterZoom(false);
    }, [handleCenterZoom]);

    useImperativeHandle(
      ref,
      () => ({
        containerRef: containerRef.current,
        zoomIn,
        zoomOut,
        scale,
        updateScaleAndPosition,
      }),
      [zoomIn, zoomOut, scale]
    );

    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const handleWheel = (e: WheelEvent) => {
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          handleZoom(e.deltaY * 4.8, e.clientX, e.clientY);
        }
      };

      container.addEventListener("wheel", handleWheel, { passive: false });

      return () => {
        container.removeEventListener("wheel", handleWheel);
      };
    }, [handleZoom]);

    useEffect(() => {
      if (containerRef?.current) {
        containerRef.current?.scrollTo({
          left: padding.left,
          top: padding.top,
        });
      }
    }, [padding]);

    useEffect(() => {
      const draw = () => {
        drawImage();
        drawLayout();
      };

      if (image.complete) {
        draw();
      } else {
        image.onload = draw;
      }
    }, [image, drawImage, drawLayout]);

    useEffect(() => {
      if (overlayCanvasRef.current) {
        overlayCanvasRef.current.style.opacity = layerVisible ? "1" : "0";
      }
    }, [layerVisible]);

    useEffect(() => {
      onChange?.({ scale });
    }, [scale]);

    return (
      <div
        className={cls(
          className,
          "w-full h-full overflow-auto scrollbar-thin relative"
        )}
        ref={containerRef}
      >
        <div
          style={{
            paddingLeft: `${padding.left}px`,
            paddingTop: `${padding.top}px`,
          }}
        >
          <div
            className="absolute"
            style={{
              width: `${imageWidth * scale}px`,
              height: `${imageHeight * scale}px`,
              transform: `translate(${position.x}px, ${position.y}px)`,
            }}
          >
            <canvas
              ref={imageCanvasRef}
              style={{
                width: `${imageWidth * scale}px`,
                height: `${imageHeight * scale}px`,
              }}
            />
            <canvas
              ref={overlayCanvasRef}
              className="absolute top-0 left-0"
              style={{
                width: `${imageWidth * scale}px`,
                height: `${imageHeight * scale}px`,
              }}
            />
          </div>
        </div>
      </div>
    );
  }
);

export default React.memo(ImageLayerViewer, isObjEqual);
