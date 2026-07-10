# AI Agent 短剧剧本工作室 · FastAPI 单体版

这是替代 Streamlit 的公开网页版本：

- Web 框架：FastAPI
- 页面渲染：Jinja2 HTML 模板 + CSS
- 状态存储：SQLite
- 后台执行：服务端线程
- 核心生成逻辑：复用 `studio.engine`、`studio.llm_service`、`studio.prompts`

## 本地运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

访问：

```text
http://127.0.0.1:8000
```

## Railway 部署

1. 将本仓库推送到 GitHub。
2. Railway → New Project → Deploy from GitHub Repo。
3. 选择本仓库。
4. Railway 会读取 `railway.json`，启动命令为：

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

5. 若需要持久保存 SQLite，请在 Railway 添加 Volume，并设置环境变量：

```text
SCRIPT_STUDIO_RUNTIME_DIR=/data
```

或使用 Railway 自动提供的 `RAILWAY_VOLUME_MOUNT_PATH`。

## Render 部署

仓库内已提供 `render.yaml`。在 Render 创建 Blueprint 或 Web Service 后，启动命令为：

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## 注意

- API Key 不写入 SQLite；服务重启后需要重新填写真实模型 Key。
- 浏览器刷新/断开不会中断后台线程。
- 平台重启会中断当前线程，但 SQLite 会保留已完成任务、日志和中断提示，可从页面继续执行。
- 高并发、多用户、长期商业化版本建议升级到 PostgreSQL + Redis/RQ。
