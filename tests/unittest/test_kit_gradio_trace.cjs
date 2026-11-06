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
