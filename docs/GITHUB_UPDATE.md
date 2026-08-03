# Windows 本地更新 GitHub

服务器无需连接 GitHub。

1. 使用 GitHub Desktop 克隆 `Clara0329/Ad-Evaluation`；
2. 解压本代码包；
3. PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\COPY_TO_GITHUB_WINDOWS.ps1 -RepositoryPath "C:\你的路径\Ad-Evaluation"
```

4. GitHub Desktop 检查 Changes；
5. Commit message 建议：`docs: restructure repository for project and paper`；
6. Push origin。

脚本不会删除目标仓库已有数据图片，只覆盖本包中列出的代码和文档文件。
