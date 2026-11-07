// 通过真实上传控件脚本重放选择、拖放、粘贴及迟到上传，不模拟框架内部组件状态。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const crypto = require("node:crypto").webcrypto;
const script = fs.readFileSync(require("node:path").join(__dirname, "../../mineru/resources/gradio_upload.js"), "utf8");
const handlers = {};
const documentHandlers = {};
const input = { value: "", files: [] };
const label = { textContent: "" };
const status = { textContent: "", setAttribute() {} };
const retry = { hidden: true, addEventListener(name, handler) { this[name] = handler; } };
const nodes = { 'input[type="file"]': input, '.mineru-upload-label': label, '.mineru-upload-status': status, '.mineru-upload-retry': retry };
const element = { querySelector: selector => nodes[selector], addEventListener: (name, handler) => handlers[name] = handler };
const uploaded = [];
const pending = [];
const values = [];
const window = {
    addEventListener() {},
    __mineruI18n: { text: value => value },
    __mineruPdfPreview: {
        sourceId: "",
        local(file, id) { this.sourceId = id; },
        bind(file, id) { return id === this.sourceId; }
    }
};
const document = { addEventListener: (name, handler) => documentHandlers[name] = handler, removeEventListener() {} };
const props = { set value(value) { values.push(value); } };
// 只替换传输边界，调用方仍执行生产异步闭包和身份校验。
const upload = file => { uploaded.push(file); return new Promise(resolve => pending.push(resolve)); };
vm.runInNewContext(`(${script})`, { window, document, crypto, setTimeout, File })({element, props, upload, accepted:[".pdf", ".png"], imageExtensions:["png"]});
// 等待控件显式安排的下一次提交，不依赖模拟时钟或固定上传耗时。
const tick = () => new Promise(resolve => setTimeout(resolve, 10));
(async () => {
    const first = new File(["first"], "same.pdf");
    const second = new File(["second"], "same.pdf");
    input.files = [first]; handlers.change({target:input});
    const firstId = window.__mineruPdfPreview.sourceId;
    handlers.drop({preventDefault(){},dataTransfer:{files:[second]}});
    const secondId = window.__mineruPdfPreview.sourceId;
    assert.notEqual(firstId, secondId);
    pending[0]({path:"/old.pdf",url:"/old.pdf"});
    await tick();
    assert.equal(values.some(value=>value.includes("/old.pdf")), false);
    pending[1]({path:"/new.pdf",url:"/new.pdf"});
    await tick();
    const receipt = values.at(-1);
    assert.equal(window.__mineruUpload.bridge(receipt).path,"/new.pdf");
    assert.equal(window.__mineruUpload.bridge(receipt).__type__,"update");
    window.__mineruUpload.prepared("prepare failed");
    assert.equal(retry.hidden,false);
    retry.click();
    await tick();
    assert.equal(uploaded.length,2);
    assert.equal(JSON.parse(values.at(-1)).file.path,"/new.pdf");
    assert.equal(window.__mineruUpload.bridge(receipt).__type__,"update");
    documentHandlers.paste({preventDefault(){},clipboardData:{files:[],items:[{getAsFile:()=>new File(["png"],"clipboard",{type:"image/png"})}]}});
    assert.ok(uploaded.at(-1).name.endsWith(".png"));
    const before = values.length;
    window.__mineruUpload.invalidate();
    pending[2]({path:"/late.png",url:"/late.png"});
    await tick();
    assert.equal(values.length,before);
    console.log("Upload identity, duplicate events, preparation retry, paste and clear passed");
})().catch(error=>{console.error(error);process.exitCode=1;});
