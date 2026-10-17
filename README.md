# meme-EmoStance

本项目用于管理数据集、通过 EmoStance 训练好的模型、meme 猫素材，以及在树莓派上运行的播放软件。

| 文件夹 | 功能 |
| --- | --- |
| `communication/` | TCP JSON 通信、认证和客户端基础代码。 |
| `dataset/` | V1、V2 清洗后的完整对话和数据来源清单。 |
| `data_spilt/` | 两个版本的训练、验证、测试集、完整对话及逐轮标注。 |
| `meme_software/` | 存放在树莓派上运行的播放软件。 |
| `sucai_meme/meme_gif/` | 存放 meme 猫的 GIF 素材。 |
| `sucai_meme/meme_music/` | 存放 meme 猫的音乐。 |
| `model/` | 数据处理、EmoStance 训练代码、配置、测试和数据恢复工具。 |
| `model_canshu/` | 按版本归档的推理源码、配置、情绪空间和回复检索库。 |
| `model_information/` | 服务器模型推理、对话管理及树莓派通信接入。 |

`data_spilt` 沿用项目现有目录名称。服务器当前默认使用 V2.0.0，prefix 回复输入与训练的 `Dialogue/A/B` 格式一致，角色系统提示词关闭；每段最多 5 轮成功问答，并保留 10 秒空闲重置。使用说明见 [model_information/README.md](model_information/README.md)。

## 克隆后恢复数据

Git 保存核心功能代码、整理后的两版数据及模型归档的小型必要资产。四个超过普通 Git 单文件限制的 V2 JSONL 以无损 gzip 包保存在 `model/git_assets/`；两版归档的训练数据从对应 split 文件恢复，避免重复存储。在项目根目录运行：

```bash
python3 model/tools/restore_git_assets.py
```

脚本校验压缩包与恢复文件的 SHA256，恢复原来的路径和字节内容；已存在且正确的文件跳过，内容冲突时报错，不覆盖本地数据。文件映射、大小及校验值见 [git_upload_manifest.json](model/git_upload_manifest.json)。

## 未上传的大模型权重

以下四个文件超过 GitHub 普通 Git 的 100 MiB 单文件限制，保留在原服务器，不包含在本次 Git 提交中：

| 文件 | 大小 |
| --- | ---: |
| `model_canshu/v1.0.0/planner/best.pt` | 706.61 MiB |
| `model_canshu/v1.0.0/generator/best.pt` | 516.14 MiB |
| `model_canshu/V2.0.0/planner/best.pt` | 706.59 MiB |
| `model_canshu/V2.0.0/generator/best.pt` | 516.14 MiB |

新机器推理前须另行补齐相应版本的两份权重，并准备本地 Mistral 基座；原始归档清单保持不变，用于验证补齐后的文件。恢复数据脚本只恢复数据，不下载权重。

训练运行目录、优化器检查点、日志、缓存、原始数据压缩包和重复备份不上传。网络辅助代码 `zerotier_chat/prepare_wsl_network.py` 是一键启动的必要依赖；参考 `zerotier_chat/network.env.example` 配置本机网络，实际口令、SSH 身份和机器配置留在本地。
