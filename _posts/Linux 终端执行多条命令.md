---
layout:     post
title:      Linux 终端执行多条命令
subtitle:   终端执行多条命令
date:       2022-08-01
author:     liyang
header-img: img/bg_007.jpg
catalog: true
tags:
    - Linux
    - 终端
---

# Linux 终端执行多条命令

（1）在每个命令之间用；（分号）隔开。

（2）在每个命令之间用&&隔开。

&&表示：若前一个命令执行成功，才会执行下一个。这样，可确保所有的命令执行完毕后，其执行过程都是成功的

这种执行方法，经常运用在自动安装的Script中。

例如：自动安装Apache及PHP的Script文件：

#！ /bin/bash

tar xvzf httpd-2.2.tar.gz &&

tar xvzf php-5.3.tar.gz &&

#设定Apache

echo "Configure apache……" &&

cd httpd-2.2 &&

 make &&

make install &&

……

echo "done"

echo 

这个Script文件，各指令都用&&串接，因此，若顺利执行完毕，表示中间的编译过程无误，并且，在执行后，安装确实是成功的。

另外，在script文件中，如果某一行太长写不完，可以行末，放置接续上行的符号"/"。

 

（3）在每个命令之间用||隔开。

||表示：若前一个命令执行成功，就不会执行下一条了。

 

（4）也可以把数个命令弄成一组，然后整组去执行它，方法有二：

1、（命令1；命令2；命令3；……）

（）会开启一个子Shell环境来执行此括号中的命令组。

以下是把一组命令放入后台中执行的范例：

（sort mydate -o test.txt;procdata test.txt） &&

2、{ 命令1;命令2；命令3；…… }   //注意，{的右边有一空格，}的左边也有一空格。

与上一种方法不同的是，此法是把这些命令组成在现行的Shell中执行，而非在子Shell中执行。

特别要注意的是，在”{“的右边 和”}“的左边，至少要间隔一个以上的空格，而且每个命令都要以；（分号）作为结尾。
