# 解决旧版本 Python 容器中的 swe-rex 安装问题

## 问题描述

当使用 Python 版本较低的基础镜像时，可能会遇到以下错误：

```
RuntimeError: Container process terminated.
stderr:
/bin/sh: 1: swerex-remote: not found
no such option: --break-system-packages
ERROR: Could not find a version that satisfies the requirement swe-rex==1.2.0
ERROR: No matching distribution found for swe-rex==1.2.0
```

## 问题原因

1. **pip 版本过旧**：`--break-system-packages` 选项在 pip 23.0+ 才引入，旧版本 pip 不支持此选项
2. **swe-rex 版本问题**：无法找到 `swe-rex==1.2.0` 版本（可能是版本号错误或网络问题）
3. **swerex-remote 命令未找到**：因为 swe-rex 安装失败，导致命令不可用

## 解决方案

### 方案 1：使用更新的基础镜像（推荐）

在配置中使用 Python 3.11+ 的基础镜像，这些镜像通常自带较新版本的 pip：

```yaml
# 在 instances.yaml 或配置文件中
deployment:
  type: docker
  image: python:3.11  # 或 python:3.12
```

### 方案 2：通过环境配置升级 pip

如果必须使用旧版本 Python 镜像，可以通过 `post_startup_commands` 在容器启动后升级 pip，但**此方案可能不够**，因为错误发生在容器启动阶段（swerex 包内部安装 swe-rex 时）。

更好的方法是在实例配置中指定一个自定义镜像，该镜像已经预装了必要的依赖。

### 方案 3：创建自定义 Dockerfile（最佳方案）

创建一个自定义 Dockerfile，预安装 swe-rex 和升级 pip：

```dockerfile
FROM <your-base-image>

# 升级 pip（如果 Python 版本较旧）
RUN python3 -m pip install --upgrade pip || \
    python3 -m pip install --upgrade pip --user || \
    python3 -m pip install --upgrade pip --break-system-packages || true

# 安装 swe-rex（使用最新版本，而不是固定版本）
RUN python3 -m pip install swe-rex || \
    python3 -m pip install swe-rex --user || \
    python3 -m pip install swe-rex --break-system-packages || \
    python3 -m pip install 'swe-rex>=1.2.0' || true

# 验证安装
RUN swerex-remote --help || echo "swerex-remote not found, will be installed at runtime"
```

然后构建并推送镜像：

```bash
docker build -t your-username/your-image:tag -f Dockerfile .
docker push your-username/your-image:tag
```

### 方案 4：修改运行脚本添加预处理

在 `run_missing_pro.sh` 中添加对旧版本 Python 镜像的特殊处理，但这需要修改 swerex 包的内部逻辑，不太可行。

### 方案 5：使用 python_standalone_dir（如果适用）

某些情况下，可以使用独立的 Python 环境：

```bash
--instances.deployment.python_standalone_dir "/root/python3.11"
```

但这需要基础镜像中已经存在该目录。

## 推荐的修复步骤

1. **检查基础镜像的 Python 版本**：
   ```bash
   docker run --rm <your-image> python3 --version
   docker run --rm <your-image> python3 -m pip --version
   ```

2. **如果 pip 版本 < 23.0**，选择以下方案之一：
   - **方案 A**：切换到更新的基础镜像（如 `python:3.11`）
   - **方案 B**：创建自定义 Dockerfile 预安装依赖

3. **验证修复**：
   ```bash
   docker run --rm <your-image> swerex-remote --help
   ```

## 临时解决方案

如果无法修改镜像，可以尝试：

1. 在 `run_missing_pro.sh` 中添加环境变量，强制使用用户安装：
   ```bash
   export PIP_USER=true
   ```

2. 或者在配置文件中添加：
   ```yaml
   env:
     PIP_USER: "true"
   ```

但这可能无法完全解决问题，因为错误发生在 swerex 包内部的安装过程中。

## 长期解决方案

建议：
1. 统一使用 Python 3.11+ 的基础镜像
2. 或者在 SWE-bench Pro 的镜像生成过程中，确保所有镜像都预装了 swe-rex 或使用足够新的 pip 版本
3. 考虑向 swerex 项目提交 issue，请求支持旧版本 pip 的兼容性
