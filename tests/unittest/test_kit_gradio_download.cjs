const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const downloads = [];
global.window = {};
global.document = {
    body: { appendChild() { /* 模拟临时下载链接挂载。 */ } },
    createElement() {
        // 记录真实脚本发起的下载，重复成功事件不得重复触发链接。
        return { click() { downloads.push([this.href, this.download]); }, remove() {} };
    },
};
const reduce = vm.runInThisContext(fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_download.js"), "utf8"));
const formats = [["zip", "ZIP"], ["html", "HTML"], ["docx", "DOCX"], ["latex", "LaTeX bundle"], ["epub", "EPUB"], ["pdf", "PDF"]];
// 调用实际产品脚本，保持 DOM 替身只负责记录下载行为。
const step = (action, format = "", ...args) => reduce(action, formats, format, formats.find(([name]) => name === format)?.[1], ...args);
const receipt = (request, error = "") => JSON.stringify({ request, error });
const file = { url: "/gradio_api/file=/cache/test.html", orig_name: "含 空格.html" };
step("activate", "", "run-a");
for (const [format, label] of formats) {
    for (let click = 0; click < 2; click++) {
        const [token, busy] = step("begin", format, "run-a");
        assert.equal(busy.interactive, false);
        assert.match(busy.value, /准备中/);
        assert.deepEqual(step("begin", format, "run-a"), [{ __type__: "update" }, { __type__: "update" }, { __type__: "update" }]);
        const count = downloads.length;
        const [ready, notice] = step("complete", format, file, receipt(token), "run-a");
        assert.equal(downloads.length, count + 1);
        assert.deepEqual(downloads.at(-1), [file.url, file.orig_name]);
        assert.equal(ready.value, label);
        assert.equal(ready.interactive, true);
        assert.equal(notice, "");
        step("complete", format, file, receipt(token), "run-a");
        assert.equal(downloads.length, count + 1);
    }
}
const [failed] = step("begin", "pdf", "run-a");
const [restored, error] = step("complete", "pdf", null, receipt(failed, "<bad> & failed"), "run-a");
assert.equal(restored.interactive, true);
assert.ok(error.includes("&lt;bad&gt; &amp; failed"));
assert.ok(!error.includes("<bad>"));
const [retry] = step("begin", "pdf", "run-a");
const count = downloads.length;
const reset = step("reset");
assert.equal(reset[0], "");
assert.deepEqual(reset.slice(1, 7), Array(6).fill(null));
assert.ok(reset.slice(-7, -1).every(update => update.interactive === false));
step("activate", "", "run-b");
const [current] = step("begin", "pdf", "run-b");
assert.deepEqual(step("busy", "pdf", retry, "run-b"), [{ __type__: "update" }, { __type__: "update" }]);
assert.equal(step("busy", "pdf", current, "run-b")[0].interactive, false);
assert.deepEqual(step("complete", "pdf", file, receipt(retry), "run-b"), [{ __type__: "update" }, { __type__: "update" }]);
assert.equal(downloads.length, count);
step("complete", "pdf", file, receipt(current), "run-b");
assert.equal(downloads.length, count + 1);
console.log("download lifecycle passed");
