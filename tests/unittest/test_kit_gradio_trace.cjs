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
    handlers.message({ source: source.contentWindow, origin, data: { channel: "mineru-trace", documentId: "one", ...data } });
const entry = {
    id: "p2-b1",
    sources: [
        { page_idx: 2, bbox: [0.1, 0.1, 0.8, 0.2] },
        { page_idx: 5, bbox: [0.1, 0.1, 0.8, 0.2] }
    ],
    depth: 0
};
window.__mineruTrace.activate("one");
emit(pdf, { type: "ready" }, location.origin);
emit(html, { type: "ready", entries: [entry], pageIndices: [2, 5] });
emit(html, { type: "select", id: entry.id });
assert.equal(delivered.at(-1).message.pageIdx, 2);
assert.equal(delivered.at(-1).message.scroll, true);
emit(pdf, { type: "select", id: entry.id, pageIdx: 5 }, location.origin);
assert.equal(delivered.at(-2).message.pageIdx, 5);
assert.equal(delivered.at(-2).message.scroll, true);
assert.equal(delivered.at(-1).message.scroll, false);
// 同一结果重复激活仍保留映射，可供重挂载的 PDF 查看器恢复。
window.__mineruTrace.activate("one");
emit(pdf, { type: "ready" }, location.origin);
assert.equal(delivered.at(-1).message.entries.length, 1);
const count = delivered.length;
emit(frame("foreign"), { type: "select", id: entry.id });
emit(pdf, { type: "select", id: entry.id }, "https://foreign.test");
emit(html, { type: "select", id: "missing" });
assert.equal(delivered.length, count);
window.__mineruTrace.activate("two");
const switched = delivered.length;
emit(html, { type: "ready", entries: [entry], pageIndices: [2, 5] });
emit(pdf, { type: "select", id: entry.id }, location.origin);
assert.equal(delivered.length, switched);
window.__mineruTrace.reset();
console.log("Trace identity, handshake and bidirectional routing passed");

// 执行完整正文脚本，验证滚动后的跨页按钮避开目录工具栏并受视口边界限制。
const checkHtmlNavigation = (
    offset, blockTop, expectedTop, configuredOrigin = location.origin, eventOrigin = location.origin
) => {
    const htmlHandlers = {};
    const targetOrigins = [];
    const navigation = {
        style: {}, offsetHeight: 30,
        setAttribute() {}, replaceChildren() {}, append() {}
    };
    const selected = {
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
        pageLabels: ["Source page", "来源页"], documentId: "one", parentOrigin: configuredOrigin, pageIndices: [2, 5]
    })})`, context);
    assert.deepEqual(targetOrigins, [eventOrigin]);
    // 同一父窗口传来的不同来源消息仍须拒绝，避免规范化放宽来源边界。
    htmlHandlers.message({ source: htmlParent, origin: "https://foreign.test", data: {
        channel: "mineru-trace", documentId: "one", type: "select", id: entry.id, pageIdx: 2, scroll: false
    } });
    assert.equal(navigation.hidden, true);
    htmlHandlers.message({ source: htmlParent, origin: eventOrigin, data: {
        channel: "mineru-trace", documentId: "one", type: "select", id: entry.id, pageIdx: 2, scroll: false
    } });
    htmlHandlers.scroll();
    assert.equal(navigation.hidden, false);
    assert.equal(navigation.style.top, `${expectedTop}px`);
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
const pdfHandlers = {};
const pdfParent = { postMessage() {} };
const scrolls = [];
let boxRect = { left: 950, top: 800, width: 100, height: 80 };
const pdfBox = { traceIds: [entry.id], getBoundingClientRect: () => boxRect };
const container = {
    scrollTop: 300, scrollLeft: 200, clientWidth: 500, clientHeight: 600,
    getBoundingClientRect: () => ({ left: 100, top: 40 }),
    scrollTo: options => scrolls.push(options)
};
const pdfContext = {
    parent: pdfParent, location, AbortController, container,
    document: {
        createElement: () => ({ setAttribute() {}, addEventListener() {} }),
        getElementById: () => ({ append() {} })
    },
    window: { addEventListener: (name, callback) => { pdfHandlers[name] = callback; } },
    viewer: {
        getPageView: () => ({ div: { querySelector: () => ({ children: [pdfBox] }) } }),
        getCachedPageViews: () => [], scrollPageIntoView() {}
    },
    eventBus: { on() {}, off() {} }
};
const pdfScript = fs.readFileSync(path.join(__dirname, "../../mineru/resources/pdf_preview/trace.mjs"), "utf8");
vm.runInNewContext(`${pdfScript.replace("export function", "function")}
installTrace({ viewer, eventBus, container, documentId: "one", messages: {} });`, pdfContext);
// 模拟已验证的父页面消息，复用生产 init/select 的完整分发路径。
const emitPdf = data => pdfHandlers.message({ source: pdfParent, origin: location.origin, data: {
    channel: "mineru-trace", documentId: "one", ...data
} });
emitPdf({ type: "init", entries: [entry], pageIndices: [2, 5] });
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
console.log("PDF source navigation handles horizontal zoom overflow");
