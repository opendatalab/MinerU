// 执行生产预览控制器，验证本地文件、服务器准备及结果更新共享唯一 iframe。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const crypto = require("node:crypto").webcrypto;
const script = fs.readFileSync(require("node:path").join(__dirname, "../../mineru/resources/gradio_pdf_preview.js"), "utf8");
const location = new URL("https://mineru.test/prefix/");
let disposed = 0;
let revoked = 0;
let blobs = 0;
const activations = [];
const events = {};
const host = { children: [], replaceChildren(...nodes) { this.children = nodes; } };
// 最小 DOM 替身只观察真实创建、释放和 URL，不模拟 Gradio 内部状态。
const document = {
    querySelector: selector => selector === ".mineru-pdf-host" ? host : null,
    createElement: tag => ({
        tag, attributes: {},
        setAttribute(name, value) { this.attributes[name] = value; },
        remove() { host.children = host.children.filter(node => node !== this); },
        contentWindow: { __mineruPdfDisposePreview() { disposed++; } },
    })
};
class LocalURL extends URL {
    // 使用浏览器同源 Blob URL，记录释放次数。
    static createObjectURL() { return `blob:${location.origin}/${++blobs}`; }
    static revokeObjectURL() { revoked++; }
}
const window = {
    addEventListener: (name, fn) => events[name] = fn,
    __mineruI18n: { text: key => key },
    __mineruTrace: { reset() {}, activate(...values) { activations.push(values); } },
    __mineruUpload: { isImage: name => name.endsWith(".png"), invalidate() {}, sync() {}, prepared() {} },
};
const invoke = vm.runInNewContext(`(${script})`, { window, document, location, crypto, URL: LocalURL, URLSearchParams });
// 使用真实文件路径而非编码后的 URL 判断文件身份。
const file = name => ({ path: `/cache/${name}`, url: `/prefix/gradio_api/file=/cache/${name}`, orig_name: name });
const first = file('中文 #?%&".pdf');
const viewer = file("assets/viewer.html");
invoke("setup", viewer);
const state = window.__mineruPdfPreview;
state.local({ name: first.orig_name }, "local-one");
const originalFrame = state.frame;
assert.ok(originalFrame && state.uploading);
assert.equal(new URLSearchParams(new URL(originalFrame.src).hash.slice(1)).get("file"), `blob:${location.origin}/1`);
invoke("reset", null);
assert.equal(state.frame, originalFrame);
const uploadingRevision = state.revision;
assert.equal(state.bind(first, "local-one"), true);
assert.ok(state.revision > uploadingRevision);
invoke("reset", first);
const ticket = JSON.parse(invoke("prepare", first));
assert.equal(invoke("prepared", JSON.stringify({ ...ticket, pdf: first })), first.path);
invoke("source", first, first, viewer);
for (const id of ["run-one", "run-two"]) {
    invoke("begin");
    invoke("result", file(`${id}/preview.pdf`), first, viewer, id);
    assert.equal(state.frame, originalFrame);
}
assert.deepEqual(activations.map(values => [...values]), [["run-one", "local-one"], ["run-two", "local-one"]]);
assert.equal(disposed, 0);
invoke("prepared", JSON.stringify({ ...ticket, error: "copy failed" }));
assert.equal(host.children[0], originalFrame);
invoke("clear");
assert.equal(disposed, 1);
assert.equal(revoked, 1);
assert.equal(invoke("prepared", JSON.stringify({ ...ticket, pdf: first })).value, undefined);
assert.equal(state.frame, null);

// 图片上传后就加载准备好的 PDF，转换结果不得再次创建查看器。
const picture = file("photo.png");
state.local({ name: "photo.png" }, "image-one");
assert.equal(state.frame, null);
state.bind(picture, "image-one");
invoke("reset", picture);
const imageTicket = JSON.parse(invoke("prepare", picture));
invoke("prepared", JSON.stringify({ ...imageTicket, pdf: file("sources/photo.pdf") }));
const imageFrame = state.frame;
assert.ok(imageFrame);
invoke("begin");
invoke("result", file("sources/photo.pdf"), picture, viewer, "image-run");
assert.equal(state.frame, imageFrame);
state.local({ name: "photo.png" }, "image-two");
assert.equal(state.bind(picture, "image-one"), false);
invoke("prepared", JSON.stringify({ ...imageTicket, pdf: file("old.pdf") }));
assert.equal(state.frame, null);

// 示例及 Windows 路径继续正确编码，并拒绝跨源文件。
invoke("clear");
invoke("reset", first);
invoke("source", first, first, viewer);
let params = new URLSearchParams(new URL(state.frame.src).hash.slice(1));
assert.equal(decodeURIComponent(new URL(params.get("file")).pathname), `/prefix/gradio_api/file=${first.path}`);
const win = { path: "C:\\cache\\中文 #1.pdf", url: "/prefix/gradio_api/file=C:/cache/中文 #1.pdf" };
invoke("reset", win);
invoke("source", win, win, viewer);
params = new URLSearchParams(new URL(state.frame.src).hash.slice(1));
assert.equal(decodeURIComponent(new URL(params.get("file")).pathname), "/prefix/gradio_api/file=C:/cache/中文 #1.pdf");
const external = { path: "/cache/external.pdf", url: "https://other.test/gradio_api/file=/cache/external.pdf" };
invoke("reset", external);
invoke("source", external, external, viewer);
assert.equal(state.frame, null);
assert.equal(host.children[0].attributes.role, "alert");
events.pagehide();
console.log("Persistent PDF, local upload identity, image preparation and URL checks passed");
