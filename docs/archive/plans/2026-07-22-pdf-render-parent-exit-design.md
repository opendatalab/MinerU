# PDF render worker parent-exit cleanup

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

PDF rendering uses a persistent spawn-based `ProcessPoolExecutor`. If its owning parse-server exits without running executor shutdown,
the worker can remain alive with `PPID=1`; the multiprocessing resource tracker then also remains alive because the worker still holds
the tracker pipe.

Each PDF render worker installs a daemon watcher through the executor initializer. The watcher waits on
`multiprocessing.parent_process()` and calls `os._exit(1)` when the parent process sentinel becomes ready. This uses Python's
cross-platform multiprocessing process sentinel and does not require PID polling or a separate MinerU lifecycle pipe.

Normal executor shutdown is unchanged. On abnormal parent exit, terminating the worker closes its resource-tracker descriptor, allowing
the standard-library resource tracker to observe EOF, clean registered resources, and exit. A worker permanently holding the GIL may
still prevent its watcher thread from running; that extreme case is outside this change.
