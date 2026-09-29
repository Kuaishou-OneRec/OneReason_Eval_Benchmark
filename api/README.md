# LLM API 客户端

保留统一接口 `load_config`、`get_client`、`get_client_from_config`、`batch_generate` 以及各 provider 的客户端类。只在使用某个 provider 时加载其 SDK；导入 `api` 不要求安装所有 SDK。

## 安装

在项目根目录执行：

```bash
python3 -m pip install -e '.[judge]'
# 可选：Gemini / Claude
python3 -m pip install -e '.[gemini]'
python3 -m pip install -e '.[claude]'
```

OpenAI-compatible 和 DeepSeek 使用 openai SDK；Gemini 使用 google-genai 或 Vertex AI；Claude 使用 anthropic。具体模型和可用服务由用户或主办方指定。

## 配置文件

路径优先级：显式 `config_path` 参数 > `LLM_CONFIG_PATH` 环境变量 > 项目内 `api/config/llm_config.json`。仓库只提供 `llm_config.example.json`，不包含真实账号配置。

```bash
cp api/config/llm_config.example.json api/config/llm_config.json
export JUDGE_BASE_URL=https://judge.example.org/v1
export JUDGE_MODEL=your-judge-model
read -rs -p 'API key: ' JUDGE_API_KEY
export JUDGE_API_KEY
```

配置值 `${ENV_NAME}` 在加载时替换为对应环境变量，缺失时报错。不要把实际密钥写进示例文件或提交到 Git。Gemini 的 `credentials_path` 指向自行提供的认证文件；仓库不提供服务账号文件。

```python
from api import get_client_from_config
client = get_client_from_config("openai")
response = client.generate("请用一句话说明推荐系统的作用。")
```

直接使用客户端仍支持原接口：

```python
import os
from api import get_client
client = get_client(
    "openai",
    api_key=os.environ["JUDGE_API_KEY"],
    base_url=os.environ["JUDGE_BASE_URL"],
    model_name=os.environ["JUDGE_MODEL"],
)
```

## 比赛 caption 裁判

`ChallengeSid2CaptionEvaluator` 仍继承 `Sid2CaptionEvaluator`，调用原 `OpenAIClient` 并保留 fenced JSON 规范化。通过 `JUDGE_BASE_URL`、`JUDGE_API_KEY`、`JUDGE_MODEL` 配置。兼容旧的 `WQ_API_BASE_URL` / `WQ_API_KEY` 环境变量，但无任何内置服务地址或密钥。

地址可以是 API base URL，也可以包含 `/chat/completions` 后缀。缺失地址、密钥或裁判模型时明确报错。裁判收到预测和参考描述；正式复现需固定裁判及 BERTScore 模型版本。客户端 repr 仅显示配置键，不显示配置值。
