const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
let sequence = 0;
const preview = vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../mineru/resources/gradio_ofd_preview.js'), 'utf8'), {
    window: {}, crypto: {randomUUID: () => String(++sequence)},
});
const first = JSON.parse(preview('begin', {path: '/first.ofd'})[0]);
const second = JSON.parse(preview('begin', {path: '/second.ofd'})[0]);
assert.equal(preview('apply', JSON.stringify({id: second.id, html: 'new'})), 'new');
assert.equal(preview('apply', JSON.stringify({id: first.id, html: 'old'})).__type__, 'update');
preview('clear');
assert.equal(preview('apply', JSON.stringify({id: second.id, html: 'stale'})).__type__, 'update');
const third = JSON.parse(preview('begin', {path: '/second.ofd'})[0]);
assert.notEqual(third.id, second.id);
assert.equal(preview('apply', JSON.stringify({id: third.id, html: 'reload'})), 'reload');
console.log('OFD preview: switch, clear, re-upload and stale receipts passed');
