const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const messages = JSON.parse(fs.readFileSync(0, "utf8"));
const resource = (name) => fs.readFileSync(path.join(__dirname, "../../mineru/resources/", name), "utf8");
const makeSpan = () => {
    const attributes = new Map([["data-mineru-i18n-key", "status_message"]]);
    return {
        textContent: "",
        isConnected: true,
        setAttribute(name, value) { attributes.set(name, value); },
        getAttribute(name) { return attributes.get(name) ?? null; },
        hasAttribute(name) { return attributes.has(name); },
        removeAttribute(name) { attributes.delete(name); },
    };
};
const makeStatus = (start, elapsed, span) => ({
    querySelector() { return span; },
    getAttribute(name) {
        return name === "data-mineru-processing-start" ? start : String(elapsed);
    },
});

let now = 0;
let currentStatus = null;
let tick = null;
let intervalMs = null;
let stopped = 0;
const document = {
    title: "",
    querySelector() { return currentStatus; },
    querySelectorAll() { return []; },
};
const context = vm.createContext({
    document,
    navigator: { languages: ["en-US"], language: "en-US" },
    performance: { now: () => now },
    setInterval(callback, ms) { tick = callback; intervalMs = ms; return 1; },
    clearInterval() { tick = null; stopped += 1; },
});
const i18n = vm.runInContext(resource("gradio_i18n.js"), context)(messages);
const timer = vm.runInContext(resource("gradio_status_timer.js"), context)(i18n);

const first = makeSpan();
currentStatus = makeStatus("100.000000000", 0, first);
timer.sync();
assert.equal(intervalMs, 10);
assert.equal(first.textContent, "Processing on server (0.00s)");
now = 40;
tick();
assert.equal(first.textContent, "Processing on server (0.04s)");
now = 50;
tick();
assert.equal(first.textContent, "Processing on server (0.05s)");

// 同一解析状态被 Gradio 重绘时，浏览器应沿用原起点。
const remounted = makeSpan();
first.isConnected = false;
currentStatus = makeStatus("100.000000000", 0.05, remounted);
now = 60;
tick();
assert.equal(remounted.textContent, "Processing on server (0.06s)");
assert.equal(first.hasAttribute("data-mineru-local-timer"), false);
assert.equal(stopped, 0);

context.navigator.languages = ["zh-CN"];
now = 125;
tick();
assert.equal(remounted.textContent, "服务端解析中（0.13 秒）");
// 原有本地化扫描不得覆盖浏览器实时数字。
const root = {
    querySelectorAll(selector) { return selector === "[data-mineru-i18n-key]" ? [remounted] : []; },
};
i18n.localize(root);
assert.equal(remounted.textContent, "服务端解析中（0.13 秒）");

currentStatus = null;
timer.sync();
assert.equal(tick, null);
assert.equal(stopped, 1);
assert.equal(remounted.hasAttribute("data-mineru-local-timer"), false);

const second = makeSpan();
currentStatus = makeStatus("200.000000000", 0.01, second);
timer.sync();
assert.equal(second.textContent, "服务端解析中（0.01 秒）");
now = 1050;
tick();
assert.equal(second.textContent, "服务端解析中（0.94 秒）");
currentStatus = null;
timer.sync();
assert.equal(stopped, 2);
