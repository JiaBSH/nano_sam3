import time
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.errors import FileDownloadError

# 替换为你实际的 sam3 模型在 ModelScope 上的 repo_id 
# (这里需要根据你之前执行 CLI 时的名字修改，比如 "damo/sam3" 或 "AI-ModelScope/sam3" 等)
model_id = "facebook/sam3" 

print("开始断点续传下载...")

for i in range(10): # 允许失败后自动重试 10 次
    try:
        # snapshot_download 默认会保存在 ~/.cache/modelscope/hub/ 下
        # 它会自动识别已经下载的 700M 并继续往下跑
        model_dir = snapshot_download(repo_id=model_id, cache_dir="./ms_cache")
        print(f"🎉 下载成功！模型保存在: {model_dir}")
        break
    except Exception as e:
        print(f"⚠️ 第 {i+1} 次下载被中断: {e}。等待 5 秒后自动重试...")
        time.sleep(5)