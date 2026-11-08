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
        // 创建刻度时保留它们的真实顺序。
        appendChild(child) { this.children.push(child); },
        // 记录无障碍标签和当前选中状态。
        setAttribute(name, value) { this.attributes[name] = value; },
        // 重放浏览器事件时调用生产回调。
        addEventListener(name, callback) { this.events[name] = callback; },
        // 指针捕获在此只记录请求，不模拟浏览器拖动算法。
        setPointerCapture(id) { this.capture = id; },
        // 单元测试只提供初始宽度，实际尺寸变化由真实浏览器覆盖。
        getBoundingClientRect() { return { width: 300 }; },
    };
}

// 每个控件独立持有 props 和提交记录，模拟 Gradio watch 的输出回执。
function setup(tiers, position = 0, defaultIndex = position) {
    const nodes = Object.fromEntries(["capsule", "range", "reset", "dots", "title", "current", "options", "announcement"]
        .map(name => [name, node()]));
    const props = { value: position, selected_tier: tiers[position], locked: false };
    const submissions = [];
    // 保留目标档位参数，验证动态按钮说明没有使用当前档位替代默认值。
    const window = { __mineruI18n: { text: (key, values = {}) => values.tier ? `${key}: ${values.tier}` : key } };
    let watcher;
    // 不模拟浏览器布局，只提供生产脚本所需的尺寸观察接口。
    class ResizeObserver {
        // 控件在本测试中不会改变布局，无需触发异步尺寸回调。
        observe() {}
    }
    vm.runInNewContext(`(${script})`, { window, document: { createElement: node }, ResizeObserver })({
        element: { querySelector: selector => nodes[selector.replace(".mineru-tier-", "")] },
        props, tiers, defaultIndex,
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
assert.equal(ui.nodes.reset.disabled, false);
assert.equal(ui.nodes.reset.attributes["aria-label"], "tier_reset: tier_compact_basic");

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
assert.equal(ui.nodes.reset.disabled, true);
assert.equal(ui.nodes.reset.attributes.title, "tier_locked");
ui.nodes.reset.events.click();
assert.equal(ui.props.value, 3);
assert.equal(ui.submissions.length, 2);
ui.update({ locked: false, selected_tier: "advanced" });
assert.equal(ui.nodes.range.value, "3");
assert.equal(ui.nodes.options.hidden, false);
assert.equal(ui.nodes.reset.disabled, false);

// 页数回执不能打断进行中的拖动；取消指针则恢复已提交值。
ui.nodes.range.events.pointerdown({button: 0, pointerId: 9});
ui.nodes.range.value = "1.2";
ui.update({ value: 3 });
assert.equal(ui.nodes.range.value, "1.2");
ui.nodes.range.events.pointercancel();
assert.equal(ui.nodes.range.value, "3");
assert.equal(ui.submissions.length, 2);
ui.nodes.reset.events.click();
assert.equal(ui.props.value, 1);
assert.equal(ui.nodes.reset.disabled, true);
assert.equal(ui.nodes.reset.attributes.title, "tier_at_default: tier_compact_basic");
assert.deepEqual(ui.submissions.at(-1), ["input", 1]);
ui.nodes.reset.events.click();
assert.equal(ui.submissions.length, 3);

// 缺档和单档只展示实际能力；原生文档的 Flash 不要求存在于公开 PDF 档位中。
for (const tiers of [["standard"], ["basic", "advanced"], ["advanced"]]) {
    const subset = setup(tiers);
    assert.equal(subset.nodes.dots.children.length, tiers.length);
    assert.equal(subset.nodes.options.hidden, tiers.length === 1);
    assert.equal(subset.nodes.range.max, String(Math.max(1, tiers.length - 1)));
    assert.equal(subset.nodes.reset.disabled, true);
    subset.update({locked: true, selected_tier: "flash"});
    assert.equal(subset.nodes.current.textContent, "tier_compact_flash");
    subset.update({locked: false, selected_tier: tiers[0]});
    assert.equal(subset.nodes.current.textContent, `tier_compact_${tiers[0]}`);
}

// 重置目标使用服务注入的默认位置，不能写死 Standard 或第一个可用档位。
for (const [tiers, defaultIndex, selectedIndex] of [
    [["flash", "basic", "standard", "advanced"], 2, 3],
    [["flash", "basic", "standard", "advanced"], 1, 3],
    [["basic", "standard", "advanced"], 1, 0],
    [["basic", "advanced"], 0, 1],
    [["flash", "advanced"], 0, 1],
]) {
    const resetUi = setup(tiers, selectedIndex, defaultIndex);
    resetUi.nodes.reset.events.click();
    assert.equal(resetUi.props.value, defaultIndex);
    assert.equal(resetUi.nodes.range.value, String(defaultIndex));
    assert.equal(resetUi.nodes.current.textContent, `tier_compact_${tiers[defaultIndex]}`);
    assert.deepEqual(resetUi.submissions, [["input", defaultIndex]]);
    // 无关回执和重复点击不能重复派发，也不能改变服务默认位置。
    resetUi.update({ value: defaultIndex });
    resetUi.nodes.reset.events.click();
    assert.equal(resetUi.submissions.length, 1);
}
