**下载miniconda**
```
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
**运行安装脚本
```
bash Miniconda3-latest-Linux-x86_64.sh
```
让配置生效
```
source ~/.bashrc
```
验证安装
```
conda --version
```

----
## Conda 环境管理
Conda是一个开源的软件包管理系统和环境管理系统，能够在同一系统上轻松管理多个项目的依赖关系。对于从事人工智能和数据科学的开发者来说，Conda是极其重要的工具，因为它可以简化复杂的环境配置过程。

#### 创建和管理虚拟环境
虚拟环境是一个隔离空间，允许用户为每个项目创建独立的Python环境，避免不同项目之间的依赖冲突。

##### 1.创建虚拟环境
```
conda create -n name(你想要的环境名称) python=3.10
```
这条命令创建了一个叫name的虚拟环境 并安装python为3.10的环境
##### 2.激活环境
```
conda activate name
```
激活后，所有的包操作（如安装、卸载）都将在这个虚拟环境中进行，而不会影响系统的的全局环境。
##### 3.停用虚拟环境
```
conda deactivate
```
停用当前环境并返回系统默认的环境

##### 4.列出所有虚拟环境
```
conda env list
```
这条命令列出了所有已经创建的Conda虚拟环境

##### 5.删除虚拟环境
```
conda remove -n myenv --all
```

### 安装和卸载包
```
conda install numpy
conda update numpy
conda remove numpy
```

### 换源加速包
默认情况下 Conda使用的源可能会导致下载的速度较慢，可以更换为国内的镜像源，可以显著提升包的下载速度。

#### 1.查看当前源
```
conda config --show channels
```
这条命令显示当前使用的源列表

#### 2.添加清华源
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

#### 3.移除旧源
```
conda config --remove channels https://repo.anaconda.com/pkgs/main/
conda config --remove channels https://repo.anaconda.com/pkgs/free/
```

#### 4.更新conda
```
conda update conda
```
以上命令可以更新conda工具本身，使得配置的源生效。


