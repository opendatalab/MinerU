// 验证实际脚本处理乱序回执，不依赖 Gradio 私有实现。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const crypto = require('node:crypto');
const window = {};
const listeners = new Set();
const logs = [];
const document = {
    addEventListener(type, fn, capture) { assert.equal(type, 'load'); assert.equal(capture, true); listeners.add(fn); },
    removeEventListener(type, fn, capture) { assert.equal(type, 'load'); assert.equal(capture, true); listeners.delete(fn); },
};
const invoke = vm.runInNewContext(`(${fs.readFileSync(path.join(__dirname, '../../mineru/resources/gradio_conversion.js'), 'utf8')})`, {
    window, crypto, console: { info(...args) { logs.push(args); } }, document,
});
// 跨 VM 对象通过 JSON 比较，忽略执行上下文的原型差异。
const equal = (a, b) => assert.equal(JSON.stringify(a), JSON.stringify(b));
const skipped = (values) => values.every(value => JSON.stringify(value) === '{"__type__":"update"}');
const status = (id, sequence, terminal = false) => JSON.stringify({ run_id: id, sequence, terminal, html: `stage-${sequence}` });
const result = (id, sequence, final = 'Completed') => JSON.stringify({ run_id: id, sequence, outputs: Array(15).fill(final) });
for (let index = 0; index < 150; index++) {
    const [ticket, timer] = invoke('begin');
    const first = JSON.parse(ticket);
    assert.equal(timer.active, true);
    equal(invoke('status', status(first.run_id, 1)), ['stage-1', { __type__: 'update', active: true }]);
    assert.ok(skipped(invoke('status', status(first.run_id, 1))));
    assert.ok(skipped(invoke('status', status(first.run_id, 0))));
    const cancelled = invoke('cancel');
    assert.equal(cancelled[1], ticket);
    assert.equal(cancelled[2].active, false);
    assert.ok(skipped(invoke('result', result(first.run_id, 3))));
    const second = JSON.parse(invoke('begin')[0]);
    assert.ok(second.revision > first.revision && second.run_id !== first.run_id);
    assert.ok(skipped(invoke('status', status(first.run_id, 99, true))));
    assert.ok(skipped(invoke('result', result(first.run_id, 99))));
    assert.equal(invoke('status', status(second.run_id, 5, true))[1].active, false);
    assert.ok(skipped(invoke('status', status(second.run_id, 6))));
    // 终态快照先到时，同序号完整结果仍须被应用一次。
    const final = invoke('result', result(second.run_id, 5, index % 2 ? 'Failed' : 'Completed'));
    assert.equal(final.length, 16);
    assert.equal(final[15].active, false);
    assert.equal(listeners.size, 1);
    // 其他 iframe 的加载不能提前移除监听；真实预览加载只记录一次。
    for (const fn of listeners) fn({ target: { matches: () => false } });
    assert.equal(listeners.size, 1);
    for (const fn of listeners) fn({ target: { matches: () => true } });
    assert.equal(listeners.size, 0);
    assert.equal(JSON.parse(logs.at(-1)[1]).run_id, second.run_id);
    assert.ok(skipped(invoke('result', result(second.run_id, 5))));
    const third = JSON.parse(invoke('begin')[0]);
    assert.equal(invoke('result', result(third.run_id, 4))[15].active, false);
    assert.ok(skipped(invoke('status', status(third.run_id, 3))));
    assert.ok(skipped(invoke('status', status(third.run_id, 4, true))));
}
console.log('150 lifecycle iterations passed');
