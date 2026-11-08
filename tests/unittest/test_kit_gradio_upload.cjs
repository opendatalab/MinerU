// 重放真实上传脚本，仅替换 DOM 和网络边界，覆盖进度、取消、错误及迟到回调。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const crypto = require("node:crypto").webcrypto;
const script = fs.readFileSync(require("node:path").join(__dirname, "../../mineru/resources/gradio_upload.js"), "utf8");
const handlers = {};
const documentHandlers = {};
const windowHandlers = {};
// 创建最小 DOM 节点，保留属性、显隐和事件语义供生产渲染函数使用。
function makeNode() {
    return {
        textContent: "", hidden: false, style: {}, attributes: {}, events: {}, classes: new Set(),
        // 记录真实属性，尤其是进度条的数值及无障碍名称。
        setAttribute(name, value) { this.attributes[name] = value; },
        // 未知传输总量时移除百分比属性，不能报告虚假的 0%。
        removeAttribute(name) { delete this.attributes[name]; },
        // SVG 使用 hidden 属性而非 HTMLElement 专用属性。
        toggleAttribute(name, value) { if (name === "hidden") this.hidden = value; },
        // 保存事件处理器，以便重放用户交互。
        addEventListener(name, handler) { this.events[name] = handler; },
        // 记录键盘焦点回到上传区域。
        focus() { this.focused = true; },
    };
}
const names = ["target", "file-row", "stem", "extension", "label", "size", "document-icon", "image-icon", "indicator",
    "percent", "finished", "fill", "meter", "remove", "status", "retry", "file-icon", "format-badge"];
const nodes = Object.fromEntries(names.map(name => [name, makeNode()]));
const icons = ["file", "document", "image", "sheet", "slides", "web", "ebook"].map(kind =>
    Object.assign(kind === "document" || kind === "image" ? nodes[`${kind}-icon`] : makeNode(), {dataset: {mineruUploadIcon: kind}}));
nodes["file-row"].classList = { toggle: (name, value) => value ? nodes["file-row"].classes.add(name) : nodes["file-row"].classes.delete(name) };
const input = Object.assign(makeNode(), {value: "", files: []});
const element = {
    querySelector: selector => selector === 'input[type="file"]' ? input : nodes[selector.replace(".mineru-upload-", "")],
    querySelectorAll: selector => selector === "[data-mineru-upload-icon]" ? icons : [],
    addEventListener: (name, handler) => { handlers[name] = handler; },
};
const requests = [];
// 模拟 XHR 的公开事件接口，允许主动注入中止后的迟到 progress/load。
class UploadRequest extends EventTarget {
    // 每个请求都有独立的上传事件源和中止状态。
    constructor() { super(); this.upload = new EventTarget(); this.aborted = false; requests.push(this); }
    // 保存目标地址以核对挂载前缀。
    open(method, url) { this.method = method; this.url = url; }
    // 仅捕获 multipart 数据，不执行真实网络传输。
    send(body) { this.file = body.get("files"); }
    // 中止请求按浏览器语义派发 abort。
    abort() { this.aborted = true; this.dispatchEvent(new Event("abort")); }
    // 以实际事件字段驱动生产进度计算。
    progress(loaded, total, lengthComputable = true) {
        this.upload.dispatchEvent(Object.assign(new Event("progress"), {loaded, total, lengthComputable}));
    }
    // 模拟服务端确认或 HTTP 错误，不跳过生产响应解析。
    respond(path, status = 200) {
        this.status = status;
        this.responseText = status === 200 ? JSON.stringify([path]) : path;
        this.dispatchEvent(new Event("load"));
    }
}
const values = [];
let clearClicks = 0;
const window = {
    addEventListener: (name, handler) => { windowHandlers[name] = handler; },
    __mineruI18n: {text: value => value},
    __mineruPdfPreview: {
        sourceId: "",
        // 同名文件仍以本次选择身份区分。
        local(file, id) { this.sourceId = id; },
        // 迟到文件不能绑定当前预览。
        bind(file, id) { return id === this.sourceId; },
        // 原生清除回调使用同一失效入口。
        clear() { this.sourceId = ""; window.__mineruUpload.invalidate(); },
    },
};
const document = {
    addEventListener: (name, handler) => { documentHandlers[name] = handler; },
    removeEventListener: name => { delete documentHandlers[name]; },
    getElementById: name => {
        assert.equal(name, "mineru-kit-clear");
        return {click: () => { clearClicks++; window.__mineruPdfPreview.clear(); }};
    },
};
const context = {upload_url: "https://example.test/mini/gradio_api/upload", file_url_prefix: "https://example.test/mini/gradio_api/file="};
let contextGate = null;
const server = {upload_context: () => contextGate || Promise.resolve(context)};
const props = {set value(value) { values.push(value); }};
vm.runInNewContext(`(${script})`, {window, document, crypto, setTimeout, clearTimeout, File, FormData, DOMException,
    XMLHttpRequest: UploadRequest})({element, props, server, accepted: [".pdf", ".png"], imageExtensions: ["png"]});
// 等待传输地址和文件桥接的异步边界，不依赖模拟上传速度。
const tick = () => new Promise(resolve => setTimeout(resolve, 10));
// 从原生文件选择入口提交文件。
const choose = async file => { input.files = [file]; handlers.change({target: input}); await tick(); };
// 点击实际叉号处理器，验证它会复用全局清除按钮。
const clear = () => nodes.remove.events.click({preventDefault() {}});
(async () => {
    assert.equal(nodes.target.hidden, false);
    assert.equal(nodes["file-row"].hidden, true);
    await choose(new File([new Uint8Array(10000)], "same.pdf"));
    const first = requests.at(-1);
    assert.equal(first.url, context.upload_url);
    assert.equal(first.withCredentials, true);
    assert.equal(first.file.name, "same.pdf");
    assert.equal(nodes.target.hidden, true);
    assert.equal(nodes.percent.textContent, "0%");
    first.progress(2500, 10000);
    assert.equal(nodes.percent.textContent, "25%");
    assert.equal(nodes.fill.style.width, "25%");
    assert.equal(nodes.size.textContent, "2.5 KB / 10.0 KB");
    first.progress(10000, 10000);
    assert.equal(nodes.status.textContent, "upload_confirming");
    assert.equal(nodes.finished.hidden, true);
    assert.equal(nodes.indicator.hidden, false);
    const secondFile = new File([new Uint8Array(20000)], "same.pdf");
    handlers.drop({preventDefault() {}, dataTransfer: {files: [secondFile]}});
    await tick();
    const second = requests.at(-1);
    assert.equal(first.aborted, true);
    first.progress(9000, 10000);
    first.respond("/old.pdf");
    await tick();
    assert.equal(nodes.percent.textContent, "0%");
    assert.equal(values.some(value => value.includes("/old.pdf")), false);
    second.progress(1, 0, false);
    assert.equal(nodes.percent.hidden, true);
    assert.equal(nodes.meter.attributes["aria-valuenow"], undefined);
    second.respond("/cache/new #1.pdf");
    await tick();
    const receipt = values.at(-1);
    const data = window.__mineruUpload.bridge(receipt);
    assert.equal(data.url, context.file_url_prefix + "/cache/new%20%231.pdf");
    assert.equal(data.size, 20000);
    assert.equal(nodes.finished.hidden, false);
    assert.equal(nodes.finished.textContent, "· upload_complete");
    assert.equal(nodes.indicator.hidden, true);
    assert.equal(nodes.percent.hidden, true);
    assert.equal(nodes.size.textContent, "20.0 KB");
    // 语言重新初始化只更新完成文案，不丢失文件大小或重新显示百分比。
    window.__mineruI18n.text = key => key === "upload_complete" ? "已上传" : key;
    window.__mineruUpload.localize();
    assert.equal(nodes.finished.textContent, "· 已上传");
    assert.equal(nodes.size.textContent, "20.0 KB");
    window.__mineruI18n.text = key => key;
    assert.equal(window.__mineruUpload.bridge(receipt).__type__, "update");
    window.__mineruUpload.prepared("prepare failed");
    assert.equal(nodes.retry.hidden, false);
    assert.equal(nodes.finished.hidden, true);
    const requestCount = requests.length;
    nodes.retry.events.click();
    await tick();
    assert.equal(requests.length, requestCount);
    assert.equal(JSON.parse(values.at(-1)).file.path, "/cache/new #1.pdf");
    assert.equal(nodes.finished.hidden, false);
    assert.equal(window.__mineruUpload.bridge(receipt).__type__, "update");

    await choose(new File(["file"], '<img src=x onerror=alert(1)>.pdf'));
    assert.equal(nodes.stem.textContent, '<img src=x onerror=alert(1)>');
    const rejected = requests.at(-1);
    rejected.respond("File too large", 413);
    await tick();
    assert.equal(nodes.status.textContent, "File too large");
    assert.equal(nodes.status.attributes.role, "alert");
    assert.equal(nodes.retry.hidden, false);
    assert.equal(nodes.finished.hidden, true);
    nodes.retry.events.click();
    assert.equal(nodes.finished.hidden, true);
    await tick();
    assert.notEqual(requests.at(-1), rejected);
    requests.at(-1).dispatchEvent(new Event("error"));
    await tick();
    assert.equal(nodes.status.textContent, "upload_failed");

    documentHandlers.paste({preventDefault() {}, clipboardData: {files: [], items: [
        {getAsFile: () => new File(["png"], "clipboard", {type: "image/png"})},
    ]}});
    await tick();
    const pasted = requests.at(-1);
    assert.ok(pasted.file.name.endsWith(".png"));
    assert.equal(nodes["image-icon"].hidden, false);
    const beforeClear = values.length;
    clear();
    assert.equal(clearClicks, 1);
    assert.equal(pasted.aborted, true);
    pasted.progress(100, 100);
    pasted.respond("/late.png");
    await tick();
    assert.equal(nodes.target.hidden, false);
    assert.equal(nodes["file-row"].hidden, true);
    assert.equal(nodes.finished.hidden, true);
    assert.equal(values.length, beforeClear);
    assert.equal(window.__mineruUpload.bridge(receipt).__type__, "update");

    let resolveContext;
    contextGate = new Promise(resolve => { resolveContext = resolve; });
    await choose(new File(["pending"], "pending.pdf"));
    const beforeContext = requests.length;
    clear();
    resolveContext(context);
    contextGate = null;
    await tick();
    assert.equal(requests.length, beforeContext);
    window.__mineruUpload.sync({path: "/example.pdf", orig_name: "example.pdf"});
    assert.equal(nodes.size.textContent, "upload_size_unknown");
    window.__mineruUpload.localize();
    assert.equal(nodes.stem.textContent, "example");
    await choose(new File([], "empty.pdf"));
    requests.at(-1).respond("/empty.pdf");
    await tick();
    assert.equal(nodes.size.textContent, "0 B");
    await choose(new File(["closing"], "closing.pdf"));
    const closing = requests.at(-1);
    windowHandlers.pagehide();
    assert.equal(closing.aborted, true);
    assert.equal(documentHandlers.paste, undefined);
    console.log("Upload progress, confirmation, replacement, clear/abort, retry, paste, names and late callbacks passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
