// 执行真实胶囊脚本，验证连续拖动不会提前提交、锁档不会污染偏好。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../../mineru/resources/gradio_tier.js"), "utf8");

// 最小 DOM 保留输入值、属性和事件；不替代生产状态转换算法。
function node() {
    return {
        children: [], dataset: {}, attributes: {}, events: {}, value: "0", disabled: false,
        style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} },
        // 创建按钮和刻度时保留它们的真实顺序。
        appendChild(child) { this.children.push(child); },
        // 记录无障碍标签和当前选中状态。
        setAttribute(name, value) { this.attributes[name] = value; },
        // 重放浏览器事件时调用生产回调。
        addEventListener(name, callback) { this.events[name] = callback; },
        // 指针捕获在此只记录请求，不模拟浏览器拖动算法。
        setPointerCapture(id) { this.capture = id; },
        // 点击目标就是当前按钮，无嵌套元素。
        closest() { return this; },
    };
}

// 每个控件独立持有 props 和提交记录，模拟 Gradio watch 的输出回执。
function setup(tiers, position = 0) {
    const nodes = Object.fromEntries(["capsule", "range", "choices", "dots", "title", "current", "options", "announcement"]
        .map(name => [name, node()]));
    const props = { value: position, selected_tier: tiers[position], locked: false };
    const submissions = [];
    const window = { __mineruI18n: { text: key => key } };
    let watcher;
    vm.runInNewContext(`(${script})`, { window, document: { createElement: node } })({
        element: { querySelector: selector => nodes[selector.replace(".mineru-tier-", "")] },
        props, tiers,
        trigger: event => { submissions.push([event, props.value]); },
        watch: (_keys, callback) => { watcher = callback; },
    });
    return { nodes, props, submissions, window,
        // 回执到来时同步现有实例，不能重建控件掩盖拖动竞态。
        update(patch) { Object.assign(props, patch); watcher(); },
        // 键盘默认行为必须被阻止，否则细步长会额外改变值。
        key(key) { let prevented = false; nodes.range.events.keydown({ key, preventDefault() { prevented = true; } }); assert(prevented); },
    };
}

const ui = setup(["flash", "basic", "standard", "advanced"], 1);
ui.nodes.range.events.pointerdown({button: 0, pointerId: 7});
ui.nodes.range.value = "1.67";
ui.nodes.range.events.input();
assert.equal(ui.nodes.current.textContent, "tier_compact_standard");
assert.equal(ui.props.value, 1);
assert.deepEqual(ui.submissions, []);
ui.nodes.range.events.pointerup();
ui.nodes.range.events.change();
assert.equal(ui.nodes.range.value, "2");
assert.deepEqual(ui.submissions, [["input", 2]]);
ui.key("End");
ui.key("ArrowRight");
assert.equal(ui.props.value, 3);
assert.equal(ui.submissions.length, 2);

// 文件切换中途锁到 Flash，即使迟到的 pointerup/change 也不能写回临时档位。
ui.nodes.range.events.pointerdown({button: 0, pointerId: 8});
ui.nodes.range.value = "0.4";
ui.update({ locked: true, selected_tier: "flash" });
ui.nodes.range.events.pointerup();
ui.nodes.range.events.change();
assert.equal(ui.nodes.options.hidden, true);
assert.equal(ui.nodes.current.textContent, "tier_compact_flash");
assert.equal(ui.props.value, 3);
assert.equal(ui.submissions.length, 2);
ui.update({ locked: false, selected_tier: "advanced" });
assert.equal(ui.nodes.range.value, "3");
assert.equal(ui.nodes.options.hidden, false);

// 页数回执不能打断进行中的拖动；取消指针则恢复已提交值。
ui.nodes.range.events.pointerdown({button: 0, pointerId: 9});
ui.nodes.range.value = "1.2";
ui.update({ value: 3 });
assert.equal(ui.nodes.range.value, "1.2");
ui.nodes.range.events.pointercancel();
assert.equal(ui.nodes.range.value, "3");
assert.equal(ui.submissions.length, 2);
ui.nodes.choices.events.click({ target: ui.nodes.choices.children[1] });
assert.equal(ui.props.value, 1);
assert.equal(ui.nodes.choices.children[1].attributes["aria-pressed"], "true");

// 缺档和单档只展示实际能力；原生文档的 Flash 不要求存在于公开 PDF 档位中。
for (const tiers of [["standard"], ["basic", "advanced"], ["advanced"]]) {
    const subset = setup(tiers);
    assert.deepEqual(subset.nodes.choices.children.map(button => button.textContent), tiers);
    assert.equal(subset.nodes.options.hidden, tiers.length === 1);
    assert.equal(subset.nodes.range.max, String(Math.max(1, tiers.length - 1)));
    subset.update({locked: true, selected_tier: "flash"});
    assert.equal(subset.nodes.current.textContent, "tier_compact_flash");
    subset.update({locked: false, selected_tier: tiers[0]});
    assert.equal(subset.nodes.current.textContent, `tier_compact_${tiers[0]}`);
}
