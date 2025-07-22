"""MinerU File转Markdown服务的命令行界面。"""

import argparse
import sys

from . import config, server


def main():
    """命令行界面的入口点。"""
    parser = argparse.ArgumentParser(description="MinerU File转Markdown转换服务")

    parser.add_argument(
        "--output-dir", "-o", type=str, help="保存转换后文件的目录 (默认: ./downloads)"
    )

    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        default="stdio",
        help="协议类型 (默认: stdio,可选: sse,streamable-http)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8001,
        help="服务器端口 (默认: 8001, 仅在使用HTTP协议时有效)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器主机地址 (默认: 127.0.0.1, 仅在使用HTTP协议时有效)",
    )

    args = parser.parse_args()

    # 检查参数有效性
    if args.transport == "stdio" and (args.host != "127.0.0.1" or args.port != 8001):
        print("警告: 在STDIO模式下，--host和--port参数将被忽略", file=sys.stderr)

    # 验证API密钥 - 移动到这里，以便 --help 等参数可以无密钥运行
    if not config.MINERU_API_KEY:
        print(
            "错误: 启动服务需要 MINERU_API_KEY 环境变量。"
            "\\n请检查是否已设置该环境变量，例如："
            "\\n  export MINERU_API_KEY='your_actual_api_key'"
            "\\n或者，确保在项目根目录的 `.env` 文件中定义了该变量。"
            "\\n\\n您可以使用 --help 查看可用的命令行选项。",
            file=sys.stderr,  # 将错误消息输出到 stderr
        )
        sys.exit(1)

    # 如果提供了输出目录，则进行设置
    if args.output_dir:
        server.set_output_dir(args.output_dir)

    # 打印配置信息
    print("MinerU File转Markdown转换服务启动...")
    if args.transport in ["sse", "streamable-http"]:
        print(f"服务器地址: {args.host}:{args.port}")
    print("按 Ctrl+C 可以退出服务")

    server.run_server(mode=args.transport, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
