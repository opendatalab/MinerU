// 运行真实预览适配脚本，验证文件编码、事件乱序和清除失效，不模拟 Gradio 内部组件。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const script = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_pdf_preview.js"), "utf8");
const window = { location: new URL("https://mineru.test/prefix/"), __mineruI18n: { text: (key) => `中文 ${key}` } };
const invoke = vm.runInNewContext(`(${script})`, { window, URL, URLSearchParams });
// 使用原始文件路径构造典型 FileData，模拟框架回传的未编码 URL。
const file = (name) => ({ path: `/cache/${name}`, url: `/prefix/gradio_api/file=/cache/${name}`, orig_name: name });
// 提取真正传给 iframe 的地址，后续断言同时覆盖两层 URL 编码。
const iframe = (update) => new URL(update.value.match(/src="([^"]+)"/)[1].replaceAll("&amp;", "&"));
const first = file('中文 空格#?%&".pdf');
const second = file("second.pdf");
const viewer = file("assets/viewer.html");
assert.equal(invoke("reset", first).value, "");
const initial = invoke("source", first, first, viewer);
assert.equal(initial.visible, true);
const initialUrl = iframe(initial);
assert.equal(initialUrl.pathname, "/prefix/gradio_api/file=/cache/assets/viewer.html");
assert.ok(initialUrl.searchParams.has("document"));
const parameters = new URLSearchParams(initialUrl.hash.slice(1));
const pdfUrl = new URL(parameters.get("file"));
assert.equal(decodeURIComponent(pdfUrl.pathname), `/prefix/gradio_api/file=${first.path}`);
assert.equal(pdfUrl.search, "");
assert.equal(pdfUrl.hash, "");
assert.ok(initial.value.includes('sandbox="allow-scripts allow-same-origin"'));
assert.equal(JSON.parse(parameters.get("messages")).pdf_page, "中文 pdf_page");
// Gradio 的事件 FileData 可能缺少挂载前缀，静态入口的基址必须用于结果 PDF。
const relative = { ...first, url: first.url.replace("/prefix", "") };
const relativeParams = new URLSearchParams(iframe(invoke("source", relative, first, viewer)).hash.slice(1));
assert.equal(new URL(relativeParams.get("file")).pathname, pdfUrl.pathname);

invoke("begin");
const layout = file("run/layout.pdf");
assert.equal(invoke("result", layout, first, viewer, "run-1").visible, true);
assert.equal(invoke("source", first, first, viewer).value, undefined);
invoke("reset", second);
assert.equal(invoke("result", layout, first, viewer, "run-1").value, undefined);
assert.equal(invoke("source", first, second, viewer).value, undefined);
assert.equal(invoke("source", second, second, viewer).visible, true);
invoke("begin");
assert.equal(invoke("result", layout, second, viewer, "").value, undefined);
assert.equal(invoke("clear").value, "");
assert.equal(invoke("result", layout, second, viewer, "run-2").value, undefined);
assert.equal(invoke("source", second, second, viewer).value, undefined);
invoke("reset", first);
assert.notEqual(iframe(invoke("source", first, first, viewer)).search, initialUrl.search);

const image = file("photo.png");
invoke("reset", image);
assert.equal(invoke("source", null, image, viewer).value, "");
invoke("begin");
assert.equal(invoke("result", layout, image, viewer, "image-run").visible, true);
const win = { path: "C:\\cache\\中文 #1.pdf", url: "/prefix/gradio_api/file=C:/cache/中文 #1.pdf" };
invoke("reset", win);
const windowsUrl = new URL(new URLSearchParams(iframe(invoke("source", win, win, viewer)).hash.slice(1)).get("file"));
assert.equal(decodeURIComponent(windowsUrl.pathname), "/prefix/gradio_api/file=C:/cache/中文 #1.pdf");
const external = { path: first.path, url: `https://other.test/gradio_api/file=${first.path}` };
invoke("reset", external);
assert.ok(invoke("source", external, external, viewer).value.includes('role="alert"'));
console.log("PDF preview lifecycle and URL checks passed");
