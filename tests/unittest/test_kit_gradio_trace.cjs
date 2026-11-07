// 执行真实父页面协调脚本，覆盖加载乱序、文档替换和消息身份。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const handlers = {};
const delivered = [];
// 为两个实际 iframe 接口提供最小可观测替身。
const frame = (name) => ({ contentWindow: { postMessage: (message, origin) => delivered.push({ name, message, origin }) } });
const html = frame("html");
const pdf = frame("pdf");
const window = {
    addEventListener: (name, callback) => {
        handlers[name] = callback;
    },
    __mineruI18n: { text: () => "来源页" }
};
const document = { querySelector: (selector) => (selector.includes("rendered") ? html : pdf) };
const location = { origin: "https://mineru.test" };
const script = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_trace.js"), "utf8");
vm.runInNewContext(`(${script})()`, { window, document, location });
// 统一构造来源消息，允许测试替换发送者与文档身份。
const emit = (source, data, origin = "null") =>
    handlers.message({ source: source.contentWindow, origin, data: { channel: "mineru-trace", sourceId: "source-one", documentId: "one", ...data } });
const entry = {
    id: "p2-b1",
    sources: [
        { page_idx: 2, block_index: 1, child_path: [], bbox: [0.1, 0.1, 0.8, 0.2] },
        { page_idx: 5, block_index: 2, child_path: [], bbox: [0.1, 0.1, 0.8, 0.2] },
        { page_idx: 5, block_index: 7, child_path: [], bbox: [0.1, 0.4, 0.8, 0.6] }
    ],
    depth: 0
};
// 正文先加载时首次 ready 会被丢弃，激活必须主动请求同一产物的快照。
emit(html, { type: "ready", entries: [entry], initialPageIdx: 2 });
window.__mineruTrace.activate("one", "source-one");
assert.equal(delivered.at(-1).name, "html");
assert.equal(delivered.at(-1).message.type, "request-ready");
emit(pdf, { type: "ready" }, location.origin);
const layoutRegions = [{ page_idx: 5, block_index: 2, child_path: [], type: "text", index: 2, bbox: [0.1, 0.1, 0.8, 0.2], color: [0.6, 0.05, 0.3] }];
emit(html, { type: "ready", entries: [entry], initialPageIdx: 2, layoutRegions });
assert.equal(delivered.at(-1).message.layoutRegions, layoutRegions);
emit(html, { type: "select", id: entry.id });
assert.equal(delivered.at(-1).message.pageIdx, 2);
assert.equal(delivered.at(-1).message.scroll, true);
emit(pdf, { type: "select", id: entry.id, pageIdx: 5 }, location.origin);
assert.equal(delivered.at(-2).message.pageIdx, 5);
assert.equal(delivered.at(-2).message.scroll, true);
assert.equal(delivered.at(-1).message.scroll, false);
// 同页合并后点击第二个来源框时，父窗口透传其实际块号和规范化页号。
const samePageKey = JSON.stringify([5, 7, []]);
emit(pdf, { type: "select", id: entry.id, pageIdx: 2, sourceKey: samePageKey }, location.origin);
assert.equal(delivered.at(-2).message.sourceKey, samePageKey);
assert.equal(delivered.at(-2).message.pageIdx, 5);
assert.equal(delivered.at(-1).message.sourceKey, samePageKey);
// 同一结果重复激活仍保留映射，可供重挂载的 PDF 查看器恢复。
window.__mineruTrace.activate("one", "source-one");
emit(pdf, { type: "ready" }, location.origin);
assert.equal(delivered.at(-1).message.entries.length, 1);
assert.equal(delivered.at(-1).message.layoutRegions, layoutRegions);
const count = delivered.length;
emit(frame("foreign"), { type: "select", id: entry.id });
emit(pdf, { type: "select", id: entry.id }, "https://foreign.test");
emit(html, { type: "select", id: "missing" });
assert.equal(delivered.length, count);
window.__mineruTrace.activate("two", "source-one");
const switched = delivered.length;
emit(html, { type: "ready", entries: [entry], initialPageIdx: 2 });
emit(pdf, { type: "select", id: entry.id }, location.origin);
assert.equal(delivered.length, switched);
emit(pdf, { type: "ready", documentId: "two" }, location.origin);
assert.equal(delivered.at(-1).message.type, "bind");
window.__mineruTrace.reset();
console.log("Trace identity, handshake and bidirectional routing passed");

// 提供可观测的 CSS 变量写入，验证主题色从真实脚本传到正文及来源控件。
const makeStyle = () => ({
    // 保存生产代码设置的自有样式属性。
    setProperty(name, value) { this[name] = value; }
});

// 执行完整正文脚本，验证滚动后的跨页按钮避开目录工具栏并受视口边界限制。
const checkHtmlNavigation = (
    offset, blockTop, expectedTop, configuredOrigin = location.origin, eventOrigin = location.origin
) => {
    const htmlHandlers = {};
    const targetOrigins = [];
    const navigation = {
        style: makeStyle(), offsetHeight: 30,
        setAttribute() {}, replaceChildren() {}, append() {}
    };
    const selected = {
        style: makeStyle(),
        dataset: {
            docvortexTraceId: entry.id,
            docvortexSources: JSON.stringify(entry.sources.map(source => ({ ...source, child_path: [] })))
        },
        setAttribute() {}, removeAttribute() {},
        getBoundingClientRect: () => ({ top: blockTop })
    };
    // 记录实际发送目标，验证 ready 与后续消息使用同一规范化来源。
    const htmlParent = { postMessage: (_message, origin) => targetOrigins.push(origin) };
    const context = {
        parent: htmlParent, navigator: { language: "en" }, innerHeight: 600, URL,
        document: {
            querySelectorAll: () => [selected],
            createElement: tag => tag === "nav" ? navigation : { setAttribute() {}, addEventListener() {} },
            body: { append() {} }, addEventListener() {}
        },
        window: { addEventListener: (name, callback) => { htmlHandlers[name] = callback; } },
        getComputedStyle: () => ({ getPropertyValue: () => offset }),
        // 此用例仅驱动滚动事件，尺寸观察器保留生产注册入口。
        ResizeObserver: class { observe() {} }
    };
    const source = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_trace_html.js"), "utf8");
    vm.runInNewContext(`(${source})(${JSON.stringify({
        pageLabels: ["Source page", "来源页"], documentId: "one", parentOrigin: configuredOrigin, initialPageIdx: 2,
        layoutRegions: [
            { ...entry.sources[0], child_path: [0], color: [0.2, 0.4, 0.9] },
            { ...entry.sources[1], color: [0.6, 0.05, 0.3] },
            { ...entry.sources[2], color: [0, 0.6, 0.1] }
        ]
    })})`, context);
    assert.deepEqual(targetOrigins, [eventOrigin]);
    htmlHandlers.message({ source: htmlParent, origin: eventOrigin, data: {
        channel: "mineru-trace", documentId: "one", type: "request-ready"
    } });
    assert.deepEqual(targetOrigins, [eventOrigin, eventOrigin]);
    // 同一父窗口传来的不同来源消息仍须拒绝，避免规范化放宽来源边界。
    htmlHandlers.message({ source: htmlParent, origin: "https://foreign.test", data: {
        channel: "mineru-trace", sourceId: "source-one", documentId: "one", type: "select", id: entry.id, pageIdx: 2, scroll: false
    } });
    assert.equal(navigation.hidden, true);
    htmlHandlers.message({ source: htmlParent, origin: eventOrigin, data: {
        channel: "mineru-trace", sourceId: "source-one", documentId: "one", type: "select", id: entry.id, pageIdx: 2, scroll: false
    } });
    htmlHandlers.scroll();
    assert.equal(navigation.hidden, false);
    assert.equal(navigation.style.top, `${expectedTop}px`);
    assert.equal(selected.style["--trace-color"], "rgb(51 102 230)");
    assert.equal(navigation.style["--trace-color"], selected.style["--trace-color"]);
    htmlHandlers.message({ source: htmlParent, origin: eventOrigin, data: {
        channel: "mineru-trace", sourceId: "source-one", documentId: "one", type: "select", id: entry.id,
        pageIdx: 5, sourceKey: samePageKey, scroll: false
    } });
    assert.equal(selected.style["--trace-color"], "rgb(0 153 26)");
    assert.equal(navigation.style["--trace-color"], selected.style["--trace-color"]);
    // 用户选择后已停用正文自动对齐，缩小视口仍须把来源控件移回可见范围。
    context.innerHeight = 200;
    htmlHandlers.resize();
    assert.equal(navigation.style.top, `${Math.min(expectedTop, 162)}px`);
};
checkHtmlNavigation("52px", -100, 52);
checkHtmlNavigation("", -100, 8);
checkHtmlNavigation("52px", 250, 250);
checkHtmlNavigation("52px", 700, 562);
checkHtmlNavigation("52px", -100, 52, "https://mineru.test:443", "https://mineru.test");
checkHtmlNavigation("52px", -100, 52, "HTTP://MINERU.TEST:80", "http://mineru.test");
checkHtmlNavigation("52px", -100, 52, "https://mineru.test:8443", "https://mineru.test:8443");
console.log("HTML source-page controls respect the outline toolbar and viewport");

// 执行 PDF 溯源入口，用放大后的真实边界语义检查横纵两个方向的定位。
// 只实现浮层使用的 DOM 接口，所有布局输入由测试显式给定。
class TraceNode {
    // 为每个节点保留事件、属性、子节点及可观测的 CSS 状态。
    constructor(tag = "div") {
        this.tagName = tag;
        this.children = [];
        this.dataset = {};
        this.style = makeStyle();
        this.attributes = {};
        this.handlers = {};
        this.isConnected = true;
        this.clientWidth = 500;
        this.clientHeight = 600;
        this.offsetWidth = 30;
        this.offsetHeight = 14;
        this.classes = new Set();
        this.classList = {
            // 模拟显式开关，供选中和悬停断言检查。
            toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name)
        };
    }
    // 属性统一保存为字符串，与浏览器元素行为一致。
    setAttribute(name, value) { this.attributes[name] = String(value); }
    // 记录实际事件处理器，测试通过相同入口驱动交互。
    addEventListener(name, handler) { this.handlers[name] = handler; }
    // 挂载节点时维护父子关系，支持重绘后的失效检测。
    append(...nodes) {
        for (const node of nodes) { node.parentElement = this; this.children.push(node); }
    }
    // 清空子节点时同步断开旧节点。
    replaceChildren() {
        for (const child of [...this.children]) child.remove();
    }
    // 删除时递归失效所有后代，模拟 PDF.js 页面回收。
    remove() {
        this.isConnected = false;
        this.replaceChildren();
        if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(node => node !== this);
    }
    // 标签底板插到文字前，便于验证真实绘制顺序。
    insertBefore(node, before) {
        node.parentElement = this;
        this.children.splice(this.children.indexOf(before), 0, node);
    }
    // 字宽仅用于确定短线阈值，浏览器验收另行检查实际字体几何。
    getComputedTextLength() { return this.textContent.length * 6; }
    // 页面查询仅支持生产溯源入口实际使用的画布和命中层。
    querySelector(selector) {
        return selector === "canvas" ? {} : this.children.find(node => node.className === selector.slice(1));
    }
    // 测试按页设置稳定几何，滚动和缩放直接更新该输入。
    getBoundingClientRect() { return this.rect; }
}
const pdfHandlers = {};
const pdfParent = { postMessage() {} };
const scrolls = [];
const jumps = [];
let boxRect = { left: 950, top: 800, width: 100, height: 80 };
const pdfBox = { traceIds: [entry.id], getBoundingClientRect: () => boxRect };
const container = {
    scrollTop: 300, scrollLeft: 200, clientWidth: 500, clientHeight: 600,
    getBoundingClientRect: () => ({ left: 100, top: 40 }),
    scrollTo: options => scrolls.push(options), addEventListener() {}
};
const pdfContext = {
    parent: pdfParent, location, AbortController, container,
    document: {
        createElement: () => ({ setAttribute() {}, addEventListener() {} }),
        createElementNS: (_namespace, name) => new TraceNode(name),
        body: new TraceNode(),
        getElementById: () => ({ append() {} })
    },
    requestAnimationFrame: () => 1, cancelAnimationFrame() {},
    ResizeObserver: class { observe() {} },
    window: { addEventListener: (name, callback) => { pdfHandlers[name] = callback; } },
    viewer: {
        pagesCount: 6, currentScale: 1.75,
        getPageView: () => ({ div: { querySelector: () => ({ children: [pdfBox] }) } }),
        getCachedPageViews: () => [],
        // 记录实际页跳转，验证重复握手不移动用户位置或改变缩放。
        scrollPageIntoView({ pageNumber }) { jumps.push(pageNumber); }
    },
    eventBus: { on() {}, off() {} }
};
const pdfScript = fs.readFileSync(path.join(__dirname, "../../mineru/resources/pdf_preview/trace.mjs"), "utf8");
vm.runInNewContext(`${pdfScript.replace("export function", "function")}
installTrace({ viewer, eventBus, container, sourceId: "source-one", messages: {} });`, pdfContext);
// 模拟已验证的父页面消息，复用生产 init/select 的完整分发路径。
const emitPdf = data => pdfHandlers.message({ source: pdfParent, origin: location.origin, data: {
    channel: "mineru-trace", sourceId: "source-one", documentId: "one", ...data
} });
emitPdf({ type: "bind" });
emitPdf({ type: "init", entries: [entry], initialPageIdx: 2 });
assert.deepEqual(jumps, [3]);
emitPdf({ type: "bind" });
emitPdf({ type: "init", entries: [entry], initialPageIdx: 5 });
assert.deepEqual(jumps, [3]);
assert.equal(pdfContext.viewer.currentScale, 1.75);
emitPdf({ type: "select", id: entry.id, pageIdx: 2, scroll: true });
assert.equal(scrolls.at(-1).top, 800);
assert.equal(scrolls.at(-1).left, 850);
// 已经横向滚动后再定位左侧及超宽区域，均把区域起始部分带回可见范围。
boxRect = { left: -50, top: 800, width: 100, height: 80 };
emitPdf({ type: "select", id: entry.id, pageIdx: 2, scroll: true });
assert.ok(scrolls.at(-1).left <= 0);
boxRect = { left: 950, top: 800, width: 700, height: 80 };
emitPdf({ type: "select", id: entry.id, pageIdx: 2, scroll: true });
assert.equal(scrolls.at(-1).left, 1034);
// 无版式框仍使用实际解析范围首跳；空页范围和 clear 不触发跳页。
emitPdf({ type: "bind", documentId: "two" });
emitPdf({ type: "init", documentId: "two", entries: [], initialPageIdx: 5 });
assert.equal(jumps.at(-1), 6);
const jumpCount = jumps.length;
emitPdf({ type: "clear", documentId: "two" });
emitPdf({ type: "bind", documentId: "three" });
emitPdf({ type: "init", documentId: "three", entries: [], initialPageIdx: null });
assert.equal(jumps.length, jumpCount);
console.log("PDF source navigation handles horizontal zoom overflow");

// 用同一入口挂载完整页面，覆盖跨栏后跨页的三段链及真实焦点优先级。
const mergeHandlers = {};
const mergeEvents = {};
const animationFrames = new Map();
let nextFrame = 0;
let resizeCallback;
const mergeBody = new TraceNode("body");
const mergeToolbar = new TraceNode();
const mergeContainer = new TraceNode();
Object.assign(mergeContainer, { clientLeft: 0, clientTop: 0, rect: { left: 100, top: 40 } });
const pageViews = [0, 1].map(index => {
    const div = new TraceNode();
    div.rect = { left: 100, top: 40 + index * 602, width: 500, height: 600 };
    return { id: [3, 6][index], div, pdfPage: {} };
});
const mergeContext = {
    parent: pdfParent, location, AbortController,
    document: {
        createElement: name => new TraceNode(name),
        createElementNS: (_namespace, name) => new TraceNode(name),
        getElementById: () => mergeToolbar, body: mergeBody
    },
    window: { addEventListener: (name, handler) => { mergeHandlers[name] = handler; } },
    requestAnimationFrame: callback => { animationFrames.set(++nextFrame, callback); return nextFrame; },
    cancelAnimationFrame: id => animationFrames.delete(id),
    ResizeObserver: class {
        // 保存生产观察器回调，用于验证尺寸更新和释放。
        constructor(callback) { resizeCallback = callback; }
        // 测试直接控制观察对象的几何。
        observe() {}
        // 释放后清除可调用入口。
        disconnect() { resizeCallback = null; }
    },
    container: mergeContainer,
    viewer: {
        pagesCount: 6,
        getPageView: index => pageViews.find(page => page.id === index + 1), getCachedPageViews: () => pageViews,
        scrollPageIntoView() {}
    },
    eventBus: {
        on: (name, callback) => { mergeEvents[name] = callback; },
        off: name => { delete mergeEvents[name]; }
    },
    messages: { trace_merged: "合并", trace_merged_description: "由 {count} 个原始内容块合并" }
};
vm.runInNewContext(`${pdfScript.replace("export function", "function")}
globalThis.disposeTrace = installTrace({ viewer, eventBus, container, sourceId: "source-one", messages });`, mergeContext);
// 消息经过生产身份校验，避免测试绕过初始化和选择流程。
const emitMerge = data => mergeHandlers.message({ source: pdfParent, origin: location.origin, data: {
    channel: "mineru-trace", sourceId: "source-one", documentId: "one", ...data
} });
// 一次刷新执行已排队的回调，并验证更新确实按帧合并。
const flushConnections = () => {
    assert.ok(animationFrames.size <= 1);
    const callbacks = [...animationFrames.values()];
    animationFrames.clear();
    for (const callback of callbacks) callback();
};
const chain = {
    id: "chain", depth: 1, sources: [
        { page_idx: 2, block_index: 4, child_path: [0], bbox: [0.1, 0.4, 0.48, 0.8] },
        { page_idx: 2, block_index: 7, child_path: [0], bbox: [0.52, 0.1, 0.9, 0.8] },
        { page_idx: 5, block_index: 1, child_path: [0], bbox: [0.1, 0.1, 0.48, 0.3] }
    ]
};
const single = { id: "single", depth: 0, sources: [{
    page_idx: 2, block_index: 9, child_path: [], bbox: [0.1, 0.85, 0.4, 0.95]
}] };
const parentChain = { id: "parent", depth: 0, sources: chain.sources.map(source => ({ ...source, child_path: [] })) };
const layoutRegionsForChain = chain.sources.map(source => ({ ...source, type: "text", index: 1, color: [0.6, 0.05, 0.3] }));
const mergeInit = { type: "init", entries: [parentChain, chain, single], initialPageIdx: 2, layoutRegions: layoutRegionsForChain };
emitMerge({ type: "bind" });
emitMerge(mergeInit);
flushConnections();
const overlay = mergeBody.children[0];
assert.equal(overlay.children.length, 0);
// 两个父子来源占据同框时只挂载一个命中区域，优先采用最深的合并节点。
const firstHit = pageViews[0].div.querySelector(".mineru-trace-layer");
const chainBox = firstHit.children.find(node => node.dataset.traceId === "chain");
const singleBox = firstHit.children.find(node => node.dataset.traceId === "single");
assert.equal(chainBox.attributes["aria-description"], "由 3 个原始内容块合并");
chainBox.handlers.pointerenter();
flushConnections();
assert.equal(overlay.dataset.traceId, "chain");
assert.equal(overlay.children.length, 2);
assert.deepEqual(overlay.children[0].children[0].attributes, { x1: "240", y1: "480", x2: "260", y2: "60" });
assert.deepEqual(overlay.children[1].children[0].attributes, { x1: "450", y1: "480", x2: "50", y2: "662" });
assert.equal(overlay.children[0].style["--trace-color"], "rgb(153 13 77)");
assert.equal(overlay.children[0].children.at(-1).textContent, "合并");
emitMerge({ type: "select", id: "chain", pageIdx: 2, scroll: false });
singleBox.handlers.pointerenter();
flushConnections();
assert.equal(overlay.children.length, 0);
singleBox.handlers.pointerleave();
flushConnections();
assert.equal(overlay.children.length, 2);
singleBox.handlers.focus();
flushConnections();
assert.equal(overlay.children.length, 0);
singleBox.handlers.blur();
mergeToolbar.children[0].handlers.click();
flushConnections();
assert.equal(overlay.children.length, 2);
// 端点由实时页框计算，滚动、缩放和尺寸变化共用同一个动画帧。
pageViews[0].div.rect = { left: 80, top: -60, width: 600, height: 720 };
mergeContainer.handlers.scroll();
mergeEvents.scalechanging();
resizeCallback();
flushConnections();
assert.deepEqual(overlay.children[0].children[0].attributes, { x1: "268", y1: "476", x2: "292", y2: "-28" });
pageViews[1].pdfPage = null;
mergeEvents.updateviewarea();
flushConnections();
assert.equal(overlay.children.length, 1);
pageViews[1].pdfPage = {};
mergeEvents.pagerendered({ pageNumber: 6 });
flushConnections();
assert.equal(overlay.children.length, 2);
// 重复来源不会多画线，几何无效的中间片段阻断两侧线段。
const duplicateChain = { ...chain, sources: [chain.sources[0], chain.sources[0], ...chain.sources.slice(1)] };
emitMerge({ ...mergeInit, entries: [duplicateChain] });
flushConnections();
assert.equal(overlay.children.length, 2);
const invalidChain = { ...chain, sources: [chain.sources[0], { ...chain.sources[1], bbox: [0.5, 0.1, 0.5, 0.8] }, chain.sources[2]] };
emitMerge({ ...mergeInit, entries: [invalidChain] });
flushConnections();
assert.equal(overlay.children.length, 0);
const shortChain = { ...chain, sources: [chain.sources[0], { ...chain.sources[1], bbox: [0.49, 0.81, 0.9, 0.9] }] };
emitMerge({ ...mergeInit, entries: [shortChain] });
flushConnections();
assert.equal(overlay.children.length, 1);
assert.deepEqual(overlay.children[0].children.map(node => node.tagName), ["line", "circle", "circle"]);
// 单来源与失败的合并目标都不产生连线，清理时撤销旧帧并解绑观察器。
emitMerge({ ...mergeInit, entries: [single] });
emitMerge({ type: "select", id: "single", pageIdx: 2 });
flushConnections();
assert.equal(overlay.children.length, 0);
mergeContainer.handlers.scroll();
emitMerge({ type: "clear" });
assert.equal(animationFrames.size, 0);
assert.equal(overlay.dataset.traceId, "");
mergeContext.disposeTrace();
assert.equal(overlay.isConnected, false);
assert.equal(resizeCallback, null);
assert.deepEqual(Object.keys(mergeEvents), []);
console.log("Merge chains preserve source order, corner geometry, interaction priority and lifecycle");
