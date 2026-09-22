const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
let sequence = 0;
const preview = vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../mineru/resources/gradio_source_preview.js'), 'utf8'), {
    window: {location: {href: 'https://demo.example.test/mineru/', origin: 'https://demo.example.test'}},
    URL,
    URLSearchParams,
    crypto: {randomUUID: () => String(++sequence)},
});
const first = JSON.parse(preview('begin', {path: '/first.html'})[0]);
const second = JSON.parse(preview('begin', {path: '/second.ofd'})[0]);
assert.equal(preview('apply', JSON.stringify({id: second.id, html: 'new'})), 'new');
assert.equal(preview('apply', JSON.stringify({id: first.id, html: 'old'})).__type__, 'update');
preview('clear');
assert.equal(preview('apply', JSON.stringify({id: second.id, html: 'stale'})).__type__, 'update');
const third = JSON.parse(preview('begin', {path: '/second.ofd'})[0]);
assert.notEqual(third.id, second.id);
assert.equal(preview('apply', JSON.stringify({id: third.id, html: 'reload'})), 'reload');

const epub = {path: '/tmp/中文 book.epub', url: '/gradio_api/file=/tmp/中文 book.epub'};
const viewer = {path: '/tmp/reader.html', url: '/gradio_api/file=/tmp/reader.html'};
const epubTicket = JSON.parse(preview('begin', epub)[0]);
const epubFrame = preview('apply', JSON.stringify({id: epubTicket.id, kind: 'epub'}), viewer);
assert.match(epubFrame, /class="mineru-epub-frame"/);
assert.match(epubFrame, /book=https%3A%2F%2Fdemo.example.test%2Fgradio_api%2Ffile%3D%2Ftmp%2F%25E4%25B8%25AD%25E6%2596%2587%2520book.epub/);
assert.match(epubFrame, /epub_spine/);
assert.doesNotMatch(epubFrame, /epub_page/);
preview('clear');
assert.equal(preview('apply', JSON.stringify({id: epubTicket.id, kind: 'epub'}), viewer).__type__, 'update');
console.log('Source preview: switch, clear, re-upload and stale receipts passed');
