const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
let sequence = 0;
const listeners = {};
const windowListeners = {};
const sourceFrames = [];
const browserWindow = {
    location: {href: 'https://demo.example.test/mineru/', origin: 'https://demo.example.test'},
    addEventListener(type, listener) {
        windowListeners[type] = listener;
    },
};
const browserDocument = {
    addEventListener(type, listener, capture) {
        assert.equal(capture, true);
        listeners[type] = listener;
    },
    querySelectorAll(selector) {
        assert.equal(selector, 'iframe.mineru-source-frame');
        return sourceFrames;
    },
};
const preview = vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../mineru/resources/gradio_source_preview.js'), 'utf8'), {
    window: browserWindow,
    document: browserDocument,
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

const htmlTicket = JSON.parse(preview('begin', {path: '/page.html'})[0]);
const sourceMarkup = '<div class="mineru-source-viewport"><div class="mineru-source-stage">' +
    '<iframe class="mineru-source-frame" title="HTML preview" srcdoc="&lt;p&gt;source&lt;/p&gt;"></iframe></div></div>';
const guardedMarkup = preview('apply', JSON.stringify({id: htmlTicket.id, html: sourceMarkup}));
assert.match(guardedMarkup, new RegExp(`data-mineru-source-preview-id="${htmlTicket.id}"`));
assert.equal(typeof listeners.load, 'function');

let restores = 0;
const sourceViewport = {
    classList: {contains: (name) => name === 'mineru-source-viewport'},
    clientWidth: 600,
    clientHeight: 800,
};
const sourceStage = {
    classList: {contains: (name) => name === 'mineru-source-stage'},
    parentElement: sourceViewport,
    style: {},
};
const sourceFrame = {
    tagName: 'IFRAME',
    classList: {contains: (name) => name === 'mineru-source-frame'},
    dataset: {mineruSourcePreviewId: String(htmlTicket.id), mineruSourceContentWidth: "1200"},
    parentElement: sourceStage,
    style: {},
    contentWindow: {},
    getAttribute: (name) => name === 'srcdoc' ? '<p>source</p>' : null,
    set srcdoc(value) {
        restores += 1;
        assert.equal(value, '<p>source</p>');
    },
};
sourceFrames.push(sourceFrame);
listeners.load({target: sourceFrame});
assert.equal(restores, 0);
assert.equal(sourceFrame.style.width, '100%');
assert.equal(sourceFrame.style.height, '100%');
assert.equal(sourceFrame.style.transform, '');
assert.equal(sourceStage.style.width, '1200px');
assert.equal(sourceStage.style.height, '1600px');
assert.equal(sourceStage.style.transform, 'scale(0.5)');
assert.equal(typeof windowListeners.message, 'function');
windowListeners.message({
    source: sourceFrame.contentWindow,
    data: {type: 'mineru-source-preview-size', width: 1200, height: 2400},
});
assert.equal(sourceFrame.style.width, '100%');
assert.equal(sourceFrame.style.height, '100%');
assert.equal(sourceFrame.style.transformOrigin, '');
assert.equal(sourceFrame.style.transform, '');
assert.equal(sourceStage.style.width, '1200px');
assert.equal(sourceStage.style.height, '1600px');
assert.equal(sourceStage.style.transformOrigin, '0 0');
assert.equal(sourceStage.style.transform, 'scale(0.5)');
listeners.load({target: sourceFrame});
assert.equal(restores, 1);
listeners.load({target: sourceFrame});
assert.equal(restores, 1);
listeners.load({target: sourceFrame});
assert.equal(restores, 2);

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
