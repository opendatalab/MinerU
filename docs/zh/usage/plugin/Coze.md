# Coze 简介

Coze（中文版名称：扣子） 是字节跳动推出的零代码 AI 应用开发平台。无论用户是否有编程经验，都可以通过该平台快速创建各种类型的聊天机器人、智能体、AI 应用和插件，并将其部署在社交平台和即时聊天应用程序中。

目前，MinerU 插件已在 Coze 插件商店上线，通过其强大的文档解析能力，为用户搭建智能体与工作流提供文档解析能力，加快用户 AI 应用的开发。

![img](../../../assets/images/coze_0.png)

- 扣子官网地址：https://www.coze.cn/
- MinerU 扣子插件下载地址：https://www.coze.cn/store/plugin/7527957359730360354

# MinerU 在 Coze 中的使用方法

## **Coze：集成应用**

- 进入 https://www.coze.cn/home coze 开发平台

## 智能体

### 工作空间 -> 项目开发 -> 创建 -> 创建智能体 -> 创建 -> 输入项目名

![img](../../../assets/images/Coze_1.png)

![img](../../../assets/images/Coze_2.png)

### 插件配置 -> 添加 `插件` -> 搜索 `MinerU`

![img](../../../assets/images/Coze_3.png)

### 添加 `parse_file` 工具（在线版）

![img](../../../assets/images/Coze_4.png)

### 选择 `MinerU` 插件 -> 编辑参数 -> 填写 api key

![img](../../../assets/images/Coze_5.png)

![img](../../../assets/images/Coze_6.png)

> 记得关闭 url 和 token 显示

### 调试 `智能体`

![img](../../../assets/images/Coze_7.png)

## 工作流

> 用工作流的方式使用 minerU

### 工作流 -> 创建工作流

![img](../../../assets/images/Coze_8.png)

![img](../../../assets/images/Coze_9.png)

### 工作流插件配置 -> 添加 `插件` -> 搜索 `MinerU` -> 添加

![img](../../../assets/images/Coze_10.png)

![img](../../../assets/images/Coze_11.png)

###  选择`MinerU` 插件 -> 编辑参数 -> 填写 api key

![img](../../../assets/images/Coze_12.png)

###  选择开始节点 -> 配置 `input` 类型为文件类型 -> 连接到 `mineru` 节点

![img](../../../assets/images/Coze_13.png)

![img](../../../assets/images/Coze_14.png)

###  选择结束节点 -> 连接到 `mineru` 节点 -> 配置 `output` 输出为 `mineru` 节点的 `parse_file.text`

![img](../../../assets/images/Coze_15.png)

![img](../../../assets/images/Coze_16.png)

### 上传文件 -> 试运行

![img](../../../assets/images/Coze_17.png)

![img](../../../assets/images/Coze_18.png)

### 发布 -> 添加到当前智能体

![img](../../../assets/images/Coze_19.png)

![img](../../../assets/images/Coze_20.png)

### 移除 `mineru` 插件 -> 调试

![img](../../../assets/images/Coze_21.png)