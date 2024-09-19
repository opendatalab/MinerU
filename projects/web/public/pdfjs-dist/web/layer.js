var LAYER_CHANGE_ACTIVE_CLASS = 'layerChange-active'

// 获取 button 元素
var layerChangeButton = document.getElementById('layerChange');

var isLayerChangeButtonActive = () => {
  return layerChangeButton.classList.contains(LAYER_CHANGE_ACTIVE_CLASS)
}

var getLocale = () => {
  console.log('test-iframe-locale', localStorage.getItem('umi-locale') ,localStorage.getItem('locale') , '')
  return localStorage.getItem('umi_locale') || localStorage.getItem('locale') || 'zh-CN';
}

// 添加tooltip
function createHoverTooltip(element, tooltipText, id, options = {}) {
  const tooltip = document.createElement('div');
  tooltip.className = `tooltip-${id} tooltip`;
  tooltip.style.display = 'none';
  tooltip.style.zIndex = '999';
  document.body.appendChild(tooltip);

  const defaultOptions = {
    offset: { y: 10 },
    delay: 200,
    duration: 200,
  };

  const mergedOptions = { ...defaultOptions, ...options };

  let tooltipTimer;

  function positionTooltip() {
    const rect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const left = rect.left + (rect.width - tooltipRect.width) / 2;
    const top = rect.bottom + mergedOptions.offset.y;

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function showTooltip() {
    tooltip.textContent = tooltipText;
    tooltip.style.display = 'block';
    tooltip.style.opacity = '0';
    tooltip.style.transition = `opacity ${mergedOptions.duration}ms`;
    positionTooltip();
    setTimeout(() => {
      tooltip.style.opacity = '1';
    }, 10);
  }

  function hideTooltip() {
    tooltip.style.opacity = '0';
    setTimeout(() => {
      tooltip.style.display = 'none';
    }, mergedOptions.duration);
  }

  element.addEventListener('mouseenter', () => {
    tooltipTimer = setTimeout(showTooltip, mergedOptions.delay);
  });

  element.addEventListener('mouseleave', () => {
    clearTimeout(tooltipTimer);
    hideTooltip();
  });

  window.addEventListener('resize', positionTooltip);
  window.addEventListener('scroll', positionTooltip);

  // 返回对象，包含 updateTooltipText 函数
  return {
    updateTooltipText: (newText) => {
      tooltipText = newText;
      positionTooltip()
      if (tooltip.style.display !== 'none') {
        showTooltip();
        tooltipText = newText;
      }
    }
  };
}

const tooltipLayer = createHoverTooltip(layerChangeButton, getLocale() === 'zh-CN' ? '隐藏识别结果': 'Hide recognition results', 'layerChange', { offset: { y: 22 } });



window.addEventListener('storage', function(event) {
  // 检查事件是否与监听的键相关
  if (event.key === 'umi_locale'|| event.key === 'locale') {
    const text = isLayerChangeButtonActive() ?  event?.newValue === 'zh-CN' ? '隐藏识别结果': 'Hide recognition results': getLocale() === 'zh-CN' ? '显示识别结果': 'Display recognition results'
    tooltipLayer?.updateTooltipText(text)
  }
});


// 添加点击事件监听器
layerChangeButton.addEventListener('click', function() {
  // 检查当前 button 的选中状态
  if (isLayerChangeButtonActive()) {
    // 如果已经处于选中状态,则移除选中状态的 class
    const annotationLayerList =  document.getElementsByClassName('annotationLayer')
    Array?.from(annotationLayerList)?.forEach(element => {
      var extractLayer = element.querySelector('#extractLayer');
      if(extractLayer) {
        extractLayer.style.opacity = 0
      }
    });
    console.log('test-dd', annotationLayerList, typeof annotationLayerList)
    tooltipLayer?.updateTooltipText( getLocale() === 'zh-CN' ? '显示识别结果': 'Display recognition results')
    this.classList.remove(LAYER_CHANGE_ACTIVE_CLASS);
  } else {
    // 如果未处于选中状态,则添加选中状态的 class
    this.classList.add(LAYER_CHANGE_ACTIVE_CLASS);
    tooltipLayer?.updateTooltipText(getLocale() === 'zh-CN' ? '隐藏识别结果': 'Hide recognition results')
    const scale = 0.943 / 0.7071 * window?.PDFViewerApplication?.pdfViewer?._currentScale || 1
    window.renderExtractLayer(window.pdfExtractData, Number(0), scale)

    const annotationLayerList =  document.getElementsByClassName('annotationLayer')
    Array?.from(annotationLayerList)?.forEach(element => {
      var extractLayer = element.querySelector('#extractLayer');
      if(extractLayer) {
        extractLayer.style.opacity = 1
      }
    })
  }
});


// 获取显示消息的元素
const messageDisplay = document.getElementById('messageDisplay');
function removeDuplicates(arr) {
    return [...new Set(arr)];
}

window.addEventListener('error', function(event) {
    if (event.target && event.target.tagName === 'SCRIPT') {
        console.error("Script error detected: ", event);
    }
}, true);

// 添加消息监听器
window.addEventListener('message', function(event) {
      const receivedMessage = event.data;
      const data = receivedMessage?.data
      const type = receivedMessage?.type
      function setHasRenderAnimatedPage (num) {
        if(!window.hasRenderAnimatedPage ) {
          window.hasRenderAnimatedPage = []
        }
        if( typeof window.hasRenderAnimatedPage === 'object'){
          window.hasRenderAnimatedPage = removeDuplicates([...window.hasRenderAnimatedPage, num])
        }
      }
      let animatingBox = new Map()
      function renderExtractLayer(data, pageNum, scale) {
          // 判断按钮是开的还是关的
          if(!isLayerChangeButtonActive()) return;

          const bboxes = data?.[pageNum]?.bboxes || []

          function drawBoxes(boxes, scale) {
            if(animatingBox.get(pageNum)) return

            // const annotationLayer = document.querySelector('.canvasWrapper');
            const pageLayer =  document.getElementsByClassName('page')?.[pageNum]
            const annotationLayer = pageLayer.querySelectorAll('.annotationLayer')?.[0];
            // annotationLayer.removeAttribute('hidden');
            if(!annotationLayer) {
              // console.error('error: annotationLayer has not been rendered')
              return
            }
            const extractLayer = annotationLayer.querySelector('#extractLayer');

            // 因为pdfjs只会缓存8页的内容，所以采用每次切换移除重建canvas的方式
            if (extractLayer) {
              extractLayer?.remove();
            }
            annotationLayer.style.width = '100%';
            annotationLayer.style.height = '100%'
            annotationLayer.style.position = 'absolute';
            annotationLayer.style.top = 0;
            annotationLayer.style.left = 0;
            const computedLayer = document.querySelector('.canvasWrapper');
            const canvas = document.createElement('canvas');
            canvas.id = 'extractLayer'
            const w = pageLayer?.offsetWidth - 18;
            const h = pageLayer?.offsetHeight - 18;
            canvas.width =  true ? `${w}` : '100%';
            canvas.height = true ? `${h}`: '100%';
            canvas.style.width =  true ? `${w}px` : '100%';
            canvas.style.height =  true ? `${h}px`: '100%';
            canvas.style.position = 'absolute';
            canvas.style.top = 0;
            canvas.style.left = 0;
            annotationLayer.append(canvas)
            const ctx = canvas.getContext('2d');

              // 移除之前的画布内容
            // ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            // console.log('renderExtractLayer: draw boxes')

            function drawPartialRect(ctx, box, progress, scale) {
                const [x, y, x2, y2]= box.bbox.map((i) => (i * scale));
                const width = x2 - x;
                const height = y2 - y;
                const color = box?.color?.line
                const fillColor = box?.color?.fill
                ctx.beginPath();
                ctx.strokeStyle = color;

                // 左边竖线
                ctx.moveTo(x, y);
                ctx.lineTo(x, y2);

                // 上边横线
                ctx.moveTo(x, y);
                ctx.lineTo(x + width * (progress < 0 ? 0: progress), y);

                // 右边竖线 (只在进度完成时绘制)
                if (progress === 1) {
                    ctx.moveTo(x2, y);
                    ctx.lineTo(x2, y2);
                    ctx.fillStyle = fillColor;
                    ctx.fillRect(x, y, width, height);
                }

                // 下边横线
                ctx.moveTo(x, y2);
                ctx.lineTo(x + width * (progress < 0 ? 0: progress), y2);

                ctx.stroke();
            }

            function fillRect(ctx, box, scale) {
                const [x, y, x2, y2]= box.bbox.map((i) => (i * scale));
                const width = x2 - x;
                const height = y2 - y;
                const color = box?.color?.fill

                ctx.fillStyle = color;
                ctx.fillRect(x, y, width, height);
            }

            function animateBox(ctx, box, duration = 1000) {
                const startTime = performance.now();

                function animate(currentTime) {
                    const elapsedTime = currentTime - startTime;
                    const progress = Math.min(elapsedTime / duration, 1);
                    // ctx.clearRect(...box.bbox); // 清除之前的绘制
                    drawPartialRect(ctx, box, progress, scale);

                    if (progress < 1) {
                        requestAnimationFrame(animate);
                    }
                }

                requestAnimationFrame(animate);
            }

            async function animateAllBoxes() {
              // const [index, value] of array.entries()
                for (const [index, box] of boxes?.entries()) {
                  await animateBox(ctx, box, 600); // 动画时间改为500ms
                  await new Promise(resolve => setTimeout(resolve, 200)); // 每个框之间的延迟也减少到100ms
                }

                // 所有线框动画完成后，一次性填充所有矩形
                // ctx.clearRect(...box.bbox); // 清除之前的绘制
                // boxes.forEach(box => fillRect(ctx, box, scale));
                console.log("test-animate All animations completed and boxes filled");
                animatingBox.set(pageNum, false)
            }

            boxes.forEach((box, index) => {
              drawPartialRect(ctx, box,  1, scale);
            });


            canvas.style.width =  false ? `${w}px` : '100%';
            canvas.style.height =  false ? `${h}px`: '100%';
            ctx.restore();
        }
        !!bboxes?.length&&drawBoxes(bboxes, scale);

      }

      // init extractLayer data
      if(type === 'initExtractLayerData') {
        const scale = 0.943 / 0.7071 * window?.PDFViewerApplication?.pdfViewer?._currentScale || 1
        const currentPageNumber = window?.PDFViewerApplication?.pdfViewer?._currentPageNumber || 1
        window.pdfExtractData = data;
        window.renderExtractLayer = renderExtractLayer
        // window.renderExtractLayer(window.pdfExtractData, currentPageNumber - 1, scale)
        // use the picture view rather than outlined view
        window.renderExtractLayer(window.pdfExtractData, Number(0), scale)
        window?.PDFViewerApplication?.pdfSidebar?.switchView(1, false)
      }

      if(type === 'pageChange') {
          if(window.renderExtractLayer
          && window.pdfExtractData
        ) {
            const scale = 0.943 / 0.7071 * window?.PDFViewerApplication?.pdfViewer?._currentScale || 1
            const currentPageNumber = data  || 0
            window.renderExtractLayer(window.pdfExtractData, Number(0), scale)
            window.renderExtractLayer(window.pdfExtractData, Number(data), scale)
          } else if(!window.pdfExtractData) {
            // console.error('extract pdf render data has not been initialized')
          }
      }

      if( type === 'title') {
            const odlPdfTitle = document.getElementById("odl-pdf-title");
            odlPdfTitle.innerText = data;
      }

      if( type === 'setPage') {
            window?.PDFViewerApplication?.eventBus?.dispatch("pagenumberchanged", {
              value: data
            })
      }

      if( type === '') {

      }
});


