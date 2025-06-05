"""演示如何使用 MinerU File转Markdown客户端的示例。"""

import os
import asyncio
from mcp.client import MCPClient


async def convert_file_url_example():
    """从 URL 转换 File 的示例。"""
    client = MCPClient("http://localhost:8000")

    # 转换单个 File URL
    result = await client.call(
        "convert_file_url", url="https://example.com/sample.pdf", enable_ocr=True
    )
    print(f"转换结果: {result}")

    # 转换多个 File URL
    urls = """
    https://example.com/doc1.pdf
    https://example.com/doc2.pdf
    """
    result = await client.call("convert_file_url", url=urls, enable_ocr=True)
    print(f"多个转换结果: {result}")


async def convert_file_file_example():
    """转换本地 File 文件的示例。"""
    client = MCPClient("http://localhost:8000")

    # 获取测试 File 的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    test_file_path = os.path.join(project_root, "test_files", "test.pdf")

    # 转换单个 File 文件
    result = await client.call(
        "convert_file_file", file_path=test_file_path, enable_ocr=True
    )
    print(f"文件转换结果: {result}")


async def get_api_status_example():
    """获取 API 状态的示例。"""
    client = MCPClient("http://localhost:8000")

    # 获取 API 状态
    status = await client.get_resource("status://api")
    print(f"API 状态: {status}")

    # 获取使用帮助
    help_text = await client.get_resource("help://usage")
    print(f"使用帮助: {help_text[:100]}...")  # 显示前 100 个字符


async def main():
    """运行所有示例。"""
    print("运行 File 到 Markdown 转换示例...")

    # 检查是否设置了 API_KEY
    if not os.environ.get("MINERU_API_KEY"):
        print("警告: MINERU_API_KEY 环境变量未设置。")
        print("使用以下命令设置: export MINERU_API_KEY=your_api_key")
        print("跳过需要 API 访问的示例...")

        # 仅获取 API 状态
        await get_api_status_example()
    else:
        # 运行所有示例
        await convert_file_url_example()
        await convert_file_file_example()
        await get_api_status_example()


if __name__ == "__main__":
    asyncio.run(main())
