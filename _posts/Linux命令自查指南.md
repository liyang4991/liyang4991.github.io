---
layout:     post
title:      Linux命令自查指南
subtitle:   自己常用的一些命令
date:       2019-07-24
author:     liyang
header-img: img/007.jpg
catalog: true
tags:
    - Linux
    - 终端
    - conda
---
自己整理的一些命令，有些简单常用就是记不住，有些不太常用但很有用，一并在此记录，以后也会更新。
没什么逻辑，比较乱，只当记录。

### 1. 链接CUDA：
	export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64

### 2. CUDA报错
        Q:  Couldn't open CUDA library libcupti.so.9.0. LD_LIBRARY_PATH: /usr/local/cuda-9.0/lib64
	解决方法： export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

### 3. Linux 查看文件大小
        显示文件大小：du -sh * | sort -n
	显示磁盘情况：df -h 
	统计当前目录下文件的个数（不包括目录）：
	$ ls -l | grep "^-" | wc -l
	统计当前目录下文件的个数（包括子目录）：
	$ ls -lR| grep "^-" | wc -l
	查看某目录下文件夹(目录)的个数（包括子目录）：
	$ ls -lR | grep "^d" | wc -l
	命令解析：
	ls -l
	长列表输出该目录下文件信息(注意这里的文件是指目录、链接、设备文件等)，每一行对应一个文件或目录，ls -lR是列出所有文件，包括子目录。
	grep "^-"
	过滤ls的输出信息，只保留一般文件，只保留目录是grep "^d"。
	wc -l
	统计输出信息的行数，统计结果就是输出信息的行数，一行信息对应一个文件，所以就是文件的个数。

### 4. Anaconda 更换源：
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
	conda config --set show_channel_urls yes
	
	恢复官方源：
	conda config --remove-key channels
	
### 5. conda 环境
	source activate py36   （py36是个环境名）
	
	退出环境
	
	source deactivate

### 6. 对于解压多个.tar.gz文件的，用下面命令：
	for tar in *.tar.gz;  do tar xvf $tar; done

### 7. 指定GPU
	export CUDA_VISIBLE_DEVICES=1

### 8. 设置文件权限
	例如，ssh用户是qinuxman,我不想它访问/root目录及下面的文件和目录，那么我就这样设置
	setfacl -R -m u:qinuxman:- /root
	-R 是递归的意思，-m就是设置和修改的意思，u就是user的意思，冒号后面是用户名又一个冒号后面是权限，-表示没有任何权限  /root是要设置的目录
	如果要给它读权限，把-替换成r就可以了，还有rwx需要什么就给什么
	设置好以后，ll看权限的时候，文件权限后面有个+号，表示这个文件设置了ACL规则，使用
	getfacl /root
	 命令查看/root目录的ACL规则详情。
	要删除ACL规则就
	setfacl -b /root
	
	setfacl -R -m u:usrname:- /root  ##-代表无权限，rwx为所有权限 /root 代表要设置的目录


### 9. pip使用清华镜像加速：
	pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple/

### 10. 终端配色代码：
    把下面的命令直接复制粘贴在 ./bashrc 文件中即可。

	# set color options for terminal
	export CLICOLOR=1
	PS1="\[\e[32;1m\][\[\e[33;1m\]\u\[\e[31;1m\]@\[\e[33;1m\]\h \[\e[36;1m\]\w\[\e[32;1m\]]\[\e[34;1m\]\$ \[\e[0m\]"
	unset LS_COLORS
	
	
	

### 11. pip 升级包
	pip install --upgrade pip
	
### 12. Vim查找
	在normal模式下按下 / 即可进入查找模式，输入要查找的字符串并按下回车。 
	Vim会跳转到第一个匹配。 按下 n 查找下一个，按下 N 查找上一个。 Vim查找支持正则表达式，例如 /vim$ 匹配行尾的 "vim


### 13. 在linux下执行sh文件时提示下面信息：
	-bash: ./xx.sh: Permission denied
	解决：
	chmod 777 xx.sh

### 14. 通用
	# -*- coding: utf8 -*-

### 15. 查看GPU信息
	watch --color gpustat
	watch -n 1 gpustat  -n 后加数字，表示刷新速度（秒）


### 16. 快速输出环境包
	pip freeze > requirements.txt # 输出本地包环境至文件
	pip install -r requirements.txt # 根据文件进行包安装

### 17. jupyter 清除输出：
	from IPython.display import clear_output
	clear_output(wait = True)

### 18. 检测设备信息：（gpu， cpu）
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	

### 19. 建立软链接，
	相当于把原目标文件夹复制一份到 目标文件夹
	ln -s 源文件 目标文件。
	有创建就有删除
	rm -rf symbolic_name 注意不是rm -rf symbolic_name/

	来自 <https://www.cnblogs.com/xiaochaohuashengmi/archive/2011/10/05/2199534.html> 

	
### 20. jupyter中添加conda虚拟环境
      首先安装ipykernel 
      在terminal下执行命令行：conda install ipykernel
      在虚拟环境下创建kernel文件 
      在terminal下执行命令行：conda install -n 环境名称 ipykernel 
      比如我的虚拟环境叫python27（后面举例都默认这个虚拟环境），那么我的就是：conda install -n python27 ipykernel
      激活conda环境 
      在terminal下执行命令行： 
      windows版本:source activate 环境名称 我的命令是：source activate python27 
      linux版本:source activate 环境名称我的命令是：activate python27
      将环境写入notebook的kernel中 
      python -m ipykernel install --user --name 环境名称 --display-name "在jupyter中显示的环境名称" 
      这里引号里面的名称自己可以随便起，用于在jupyter里面做标识，这里我仍然在jupyter里面叫python27，所以我的命令是：
      python -m ipykernel install --user --name python27 --display-name "python27"
      打开notebook服务器 
      在terminal下执行命令行jupyter notebook

      来自 <https://blog.csdn.net/u014665013/article/details/81084604> 

### 21. conda安装 opencv3
	conda install -c https://conda.binstar.org/menpo opencv3
