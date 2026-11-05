const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_advanced_options.js"), "utf8");
const listeners = new Map();
const timers = new Map();
let timerId = 0;
// 最小 DOM 夹具仅模拟自有节点，所有显隐、定时器和定位均执行生产脚本。
class Element {
    constructor(kind) { this.kind = kind; this.dataset = {}; this.style = {}; this.attrs = {}; this.inert = false; }
    matches(selector) { return this.kind === "trigger" && selector === "button"; }
    closest(selector) {
        if (this.kind === "trigger" && selector.includes(".mineru-advanced-open")) return this;
        if (["panel", "input"].includes(this.kind) && selector.includes(".mineru-advanced-popover")) return panel;
        return null;
    }
    getAttribute(name) { return this.attrs[name] ?? null; }
    setAttribute(name, value) { this.attrs[name] = value; }
    querySelector(selector) { return selector === "input" ? checkbox : null; }
    getBoundingClientRect() {
        return this.kind === "trigger" ? {left: 32, right: 332, top: 700, bottom: 732} : {height: 140};
    }
    focus() { document.activeElement = this; }
}
const trigger = new Element("trigger"), panel = new Element("panel"), checkbox = new Element("input");
checkbox.checked = true;
const document = {
    activeElement: null,
    querySelector: (selector) => selector === ".mineru-advanced-open" ? trigger : panel,
    addEventListener: (event, callback) => listeners.set(event, callback),
};
const window = { innerWidth: 390, innerHeight: 844, addEventListener() {} };
const context = vm.createContext({
    window, document, Element,
    setTimeout: (callback, delay) => { timers.set(++timerId, {callback, delay}); return timerId; },
    clearTimeout: (id) => timers.delete(id),
});
const controller = vm.runInContext(`(${source})`, context)();
// 推进虚拟悬停时间；已取消定时器不再执行。
function tick(delay) {
    for (const [id, timer] of [...timers]) if (timer.delay <= delay) {
        timers.delete(id);
        timer.callback();
    }
}
// 初始、非适用格式和清除均隐藏入口，且不触碰用户勾选。
controller.sync();
assert.equal(trigger.dataset.available, "false");
assert.equal(panel.inert, true);
controller.setFile({path: "UPPER.PDF"}, ["pdf", "png"]);
assert.equal(trigger.dataset.available, "true");
listeners.get("pointerover")({target: trigger, relatedTarget: null, pointerType: "mouse"});
tick(120);
assert.equal(panel.dataset.open, "true");
assert.equal(trigger.attrs["aria-expanded"], "true");
assert.equal(panel.inert, false);
// 窄屏浮窗必须落在视口内。
assert.ok(parseFloat(panel.style.left) >= 12);
assert.ok(parseFloat(panel.style.left) + parseFloat(panel.style.width) <= 378);
assert.ok(parseFloat(panel.style.top) + 140 <= 832);
// 鼠标经过按钮和浮窗之间的间隙，进入浮窗会取消关闭定时器。
listeners.get("pointerout")({target: trigger, relatedTarget: null});
listeners.get("pointerover")({target: panel, relatedTarget: null, pointerType: "mouse"});
tick(280);
assert.equal(panel.dataset.open, "true");
checkbox.checked = false;
controller.setFile({path: "report.csv"}, ["pdf", "png"]);
assert.equal(trigger.dataset.available, "false");
assert.equal(panel.dataset.open, "false");
assert.equal(checkbox.checked, false);
controller.setFile("photo.png", ["pdf", "png"]);
// 触摸不会误触发悬停；键盘点击展开并把焦点移入表单。
listeners.get("pointerover")({target: trigger, relatedTarget: null, pointerType: "touch"});
tick(120);
assert.equal(panel.dataset.open, "false");
listeners.get("click")({target: trigger, detail: 0});
assert.equal(panel.dataset.open, "true");
assert.equal(document.activeElement, checkbox);
listeners.get("keydown")({key: "Escape"});
assert.equal(panel.dataset.open, "false");
assert.equal(document.activeElement, trigger);
listeners.get("pointerover")({target: trigger, relatedTarget: null, pointerType: "mouse"});
controller.setFile(null, ["pdf", "png"]);
tick(280);
assert.equal(trigger.dataset.available, "false");
assert.equal(panel.dataset.open, "false");
assert.equal(checkbox.checked, false);
console.log("advanced options behavior passed");
