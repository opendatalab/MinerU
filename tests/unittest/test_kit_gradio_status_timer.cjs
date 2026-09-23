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
const makeStatus = (start, elapsed, span, queueKey = null) => ({
    querySelector() { return span; },
    getAttribute(name) {
        if (name === "data-mineru-processing-start") return start;
        if (name === "data-mineru-processing-elapsed") return String(elapsed);
        if (name === "data-mineru-queue-key") return queueKey;
        return null;
    },
});

let now = 0;
let currentStatus = null;
let currentPanel = null;
let tick = null;
let intervalMs = null;
let stopped = 0;
const document = {
    title: "",
    querySelector(selector) {
        return selector === ".mineru-status-panel .status-steps-panel" ? currentPanel : currentStatus;
    },
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

// 排队圆点在浏览器每秒推进，不依赖 Gradio 再次发送状态；重绘后沿用旧起点。
const queued = makeSpan();
now = 1100;
currentStatus = makeStatus(null, null, queued, "queued_on_server");
timer.sync();
assert.equal(intervalMs, 1000);
assert.equal(queued.textContent, "服务端排队中.");
now = 2100;
tick();
assert.equal(queued.textContent, "服务端排队中..");
now = 10100;
tick();
assert.equal(queued.textContent, `服务端排队中${".".repeat(10)}`);
now = 11100;
tick();
assert.equal(queued.textContent, "服务端排队中.");
i18n.localize({ querySelectorAll(selector) { return selector === "[data-mineru-i18n-key]" ? [queued] : []; } });
assert.equal(queued.textContent, "服务端排队中.");
const remountedQueue = makeSpan();
queued.isConnected = false;
currentStatus = makeStatus(null, null, remountedQueue, "queued_on_server");
now = 12100;
tick();
assert.equal(remountedQueue.textContent, "服务端排队中..");
assert.equal(stopped, 2);
assert.equal(queued.hasAttribute("data-mineru-local-animation"), false);

const localQueue = makeSpan();
currentStatus = makeStatus(null, null, localQueue, "queued_locally");
now = 12300;
timer.sync();
assert.equal(localQueue.textContent, "本地排队中.");
assert.equal(stopped, 3);

// 进入解析时停止排队动画并切换到 10 毫秒计时；离开后两种定时器都停止。
const afterQueue = makeSpan();
currentStatus = makeStatus("400.000000000", 0.04, afterQueue);
now = 12400;
timer.sync();
assert.equal(intervalMs, 10);
assert.equal(afterQueue.textContent, "服务端解析中（0.04 秒）");
assert.equal(localQueue.hasAttribute("data-mineru-local-animation"), false);
currentStatus = null;
timer.sync();
assert.equal(tick, null);

// 新任务点击时须立即切到准备阶段，并停止上一任务仍在运行的本地计时。
const preparingSpan = makeSpan();
const titleSpan = makeSpan();
const latestAttributes = new Map([
    ["data-mineru-processing-start", "300.000000000"],
    ["data-mineru-processing-elapsed", "0.010000"],
    ["data-mineru-queue-key", "queued_on_server"],
]);
const steps = Array.from({ length: 8 }, () => {
    const values = new Set(["is-done"]);
    return {
        values,
        classList: {
            toggle(name, enabled) { if (enabled) values.add(name); else values.delete(name); },
            remove(...names) { names.forEach((name) => values.delete(name)); },
        },
    };
});
currentPanel = {
    querySelectorAll() { return steps; },
    querySelector(selector) {
        if (selector === ".status-panel-title [data-mineru-i18n-key]") return titleSpan;
        if (selector === ".status-latest") {
            return {
                removeAttribute(name) { latestAttributes.delete(name); },
                querySelector() { return preparingSpan; },
            };
        }
        return null;
    },
};
currentStatus = makeStatus("300.000000000", 0.01, preparingSpan);
timer.sync();
assert.ok(tick);
timer.showPreparing();
assert.equal(tick, null);
assert.equal(titleSpan.textContent, "最新状态");
assert.equal(preparingSpan.textContent, "正在准备请求…");
assert.equal(preparingSpan.getAttribute("data-mineru-i18n-key"), "preparing_request");
assert.equal(latestAttributes.size, 0);
assert.ok(steps[0].values.has("is-active"));
assert.ok(steps.slice(1).every((step) => step.values.has("is-pending")));
