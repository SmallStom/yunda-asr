# 在项目中部署一个独立的 Python 3.11 便携环境，用于运行 DeepFilterNet
# 因为 DeepFilterNet 的 Rust 扩展 libdf 目前没有 Python 3.12 的 Windows wheel。

$ErrorActionPreference = "Stop"
$tools = "$PSScriptRoot\..\tools"
$pyDir = "$tools\py311"
New-Item -ItemType Directory -Force -Path $tools, $pyDir | Out-Null

# 1. 下载 Python 3.11 embeddable
$zip = "$env:TEMP\py311-embed.zip"
if (-not (Test-Path $zip)) {
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip -OutFile $zip
}
Expand-Archive -Path $zip -DestinationPath $pyDir -Force

# 2. 启用 site-packages 和 pip
$pth = Get-ChildItem $pyDir -Filter "python*._pth" | Select-Object -First 1
if ($pth) {
    $content = Get-Content $pth.FullName
    $content = $content -replace "^#?import site", "import site"
    # 确保 site-packages 路径存在
    if (-not ($content -match "Lib\\site-packages")) {
        $content += "`n.\Lib\site-packages"
    }
    $content | Set-Content $pth.FullName
}

# 3. 下载并安装 pip
$getPip = "$env:TEMP\get-pip.py"
if (-not (Test-Path $getPip)) {
    Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile $getPip
}
& "$pyDir\python.exe" $getPip --no-warn-script-location

# 4. 安装 DeepFilterNet 及其依赖
& "$pyDir\python.exe" -m pip install --upgrade pip -q
& "$pyDir\python.exe" -m pip install deepfilternet torch torchaudio soundfile numpy -q

Write-Host "Python 3.11 + DeepFilterNet 安装完成: $pyDir"
