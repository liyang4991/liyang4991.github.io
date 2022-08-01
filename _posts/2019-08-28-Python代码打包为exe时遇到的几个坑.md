---
layout:     post
title:      Python代码打包为exe时遇到的几个坑
subtitle:   pyinstaller使用指南
date:       2019-08-28
author:     liyang
header-img: img/the-whale.png
catalog: true
tags:
    - Python
    - 编程
---

## Python代码打包为exe时遇到的几个坑

最近老师让我帮一个公司做一些数据处理的工作，我帮他们写了一个python程序。但是考虑到他们直接使用python代码的话，一个是操作不够友好，另一个是还要搭建环境。因此想着可以把python代码打包成exe，只要是windows系统电脑，可以直接运行。

查阅资料，比较好用的工具是pyinstaller。

### 安装pyinstaller

```cmd
pip install  pyinstaller
```

### pyinstaller相关参数

```bash
-F : 打包成单个可执行文件
-w : 打包之后运行程序,只有窗口不显示命令行
-c : 打包之后运行程序,显示命令行
```

**注意** :`-w`和`-c`不可以同时使用. 看解释就明白了..

### 打包命令

程序入口文件必须在项目根目录,

```cmd
# 打开CMD或者终端,切换到项目根目录
pyinstaller -F <程序入口文件名>.py
```

**注意** : 打包过程中可能会报出一堆,找不到`api-xxx-xxx-xx.dll`之类的警告,我打包之后运行程序没问题... 这个也自行测试.

### 打包完成标志

项目目录下存在`build`、`dist`和`<入口程序文件名>.spec`
其中打包好的可执行程序在`dist`目录中就能看见, 其他;两个文件没什么卵用,可以直接删除.

### 运行打包的程序

虽说生成了exe文件，双击运行就行，但是如果想看运行过程，尤其是如果出错，想看是什么错误，直接运行的话，控制台窗口会一闪而过，看不了错误信息。网上有人推荐用手机录像的方式查看报错，我差点就这样做了......

一个更优雅的做法是在命令行中运行：打开cmd， 输入exe程序的路径就好：

~~~python
(base) F:\RSdata\Toexe>dist\\PointToCity.exe
~~~

### 遇到的问题以及解决方案：

#### 1. 程序无法运行，弹窗显示以下错误信息：

~~~python
 failed to execute script *.exe
~~~

为了查看详细的错误，首先打包的时候不加 -w 参数， 然后报错在控制台窗口一闪而过，我差点就用手机录屏了，好看又看到在cmd运行的方法。发现错误是因为：

~~~python
ModuleNotFoundError: No module named 'fiona._shim' 
~~~

实际上，在我程序中我并没有显式导入fiona库。所以第一次尝试导入fiona，失败，然后只导入_shim, 依然失败，这个错误让我崩溃，我查一了一下午才终于找到解决方案：

```
from fiona import _shim, schema
```

![](https://i.loli.net/2019/08/28/f2iERLFnWUD4jxI.jpg)

参考自：https://github.com/pyinstaller/pyinstaller/issues/4277

但是让人绝望的是，程序不报上面的错误之后，又报了其他错误：

~~~python
StopIteration  Error
~~~

恰如之前回答问答那人所说。

#### 2. StopIteration  Error

又查了一下，好在这个错误也有人遇到过，解决方案是：注释掉

~~~python
import geopandas.datasets
~~~

![](https://i.loli.net/2019/08/29/DmMT7aiYHlXe91y.jpg)

https://stackoverflow.com/questions/51662773/cannot-import-geopandas-with-pyinstaller-executable-despite-running-fine-in-th

到这里程序终于完美运行了。

### 总结：

1. 实际上用pyinstaller打包exe挺简单，关键是在调用geopandas库中出现了很多问题。
2. 对于程序中需要调用的文件，只要写好路径即可，并不需做额外处理。
3. 打包后的程序很大，一个简单的输出hello，world程序有6M，我这个几十行的简单程序竟然有600M。避免太大的一个方法是只调用用到的函数，不要整个库都调用。但是我的代码中不知道该怎么修改了，就先这么大吧。