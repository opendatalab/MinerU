# MinerU web 


## 目录
- [前端本地开发](#前端本地开发)
- [技术栈](#技术栈)
## 前端本地开发

### 前置条件
- Node.js 18.x
- pnpm

### 安装步骤

1. 安装 Node.js 18
   - 访问 [Node.js 官网](https://nodejs.org/) 下载并安装 Node.js 18.x 版本

2. 安装 pnpm
   ```bash
   npm install -g pnpm
   ```
3. 克隆仓库
   ```
   1. git clone https://github.com/opendatalab/MinerU
   2. cd ./projects/web
   ```

4. 安装依赖
   ```
   pnpm install
   ```

5. 运行开发服务器
   ```
   pnpm run dev
   ```

6. ⚠️ 注意：此命令仅用于本地开发，不要用于部署！
打开浏览器访问 http://localhost:5173（或控制台输出的其他地址）
构建项目
要构建生产版本，请执行以下命令：

   ```
   pnpm run build
   ```
7. 请确保./projects/web_demo后端服务启动

8. 如果pnpm install执行error，可更换包管理器
   ```
   npm install -g yarn
   yarn
   yarn start
   ```

## 技术栈

- React
- Tailwind CSS
- typeScript
- zustand
- ahooks
