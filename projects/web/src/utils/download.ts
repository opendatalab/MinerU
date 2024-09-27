export async function downloadFileUseAScript(
  url: string,
  filename?: string
): Promise<void> {
  try {
    // 发起请求获取文件
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 获取文件内容的 Blob
    const blob = await response.blob();

    // 创建一个 Blob URL
    const blobUrl = window.URL.createObjectURL(blob);

    // 创建一个隐藏的<a>元素
    const link = document.createElement("a");
    link.style.display = "none";
    link.href = blobUrl;

    // 设置下载的文件名
    const contentDisposition = response.headers.get("Content-Disposition");
    const fileName =
      filename ||
      (contentDisposition
        ? contentDisposition.split("filename=")[1].replace(/['"]/g, "")
        : url.split("/").pop() || "download");

    link.download = fileName;

    // 将链接添加到文档中并触发点击
    document.body.appendChild(link);
    link.click();

    // 清理
    document.body.removeChild(link);
    window.URL.revokeObjectURL(blobUrl);
  } catch (error) {
    console.error("Download failed:", error);
  }
}
