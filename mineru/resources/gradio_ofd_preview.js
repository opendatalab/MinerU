(action, ...args) => {
    // 浏览器会话保存最新请求标识，清除和重复上传同一文件也会使旧请求失效。
    const key = "__mineruOfdPreview";
    if (action === "begin" || action === "clear") {
        const file = action === "clear" ? null : args[0];
        const ticket = {id: (window[key] || 0) + 1, path: file?.path || ""};
        window[key] = ticket.id;
        return [JSON.stringify(ticket), ""];
    }
    const receipt = JSON.parse(args[0]);
    return receipt.id === window[key] ? receipt.html : {__type__: "update"};
}
