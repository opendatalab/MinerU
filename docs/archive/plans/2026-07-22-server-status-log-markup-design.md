# Server Status Log Markup Safety Design

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

## Problem

`mineru server status` passes recent log strings directly to Rich `Panel`. Rich treats string renderables as markup. Log content such as `MinerU home [/Users/jinzhenj/.mineru]` is therefore parsed as a closing markup tag and raises `MarkupError`.

Logs are untrusted plain text and can contain arbitrary brackets, paths, traceback text, or text that resembles Rich tags.

## Decision

Wrap each recent-log panel body in `rich.text.Text` before constructing the panel. `Text` preserves the log literally and prevents Rich markup parsing while retaining the existing panel title and border styling.

The global Rich output helper remains unchanged because other CLI renderers may intentionally use Rich markup or structured renderables.

## Tests

Extend the server status log panel test with paths and markup-like text. Assert that each panel receives a `Text` object whose plain content exactly matches the source log, and render a panel through Rich to verify no `MarkupError` is raised.
