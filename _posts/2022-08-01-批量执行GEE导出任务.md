---
layout:     post
title:      GEE 任务批量导出
subtitle:   批量执行GEE导出任务 
date:       2022-08-01
author:     liyang
header-img: img/Rio2.png
catalog: true
tags:
    - GEE
---

# 批量执行GEE导出任务

/ ** 

 批量执行GEE导出任务 
 首先，您需要生成导出任务。并显示了运行按钮。 
 然后按F12进入控制台，然后将这些脚本粘贴到其中，然后按 
 输入。所有任务都将自动启动。 
 （支持Firefox和Chrome。其他浏览器我没有测试过。） 

  @Author： 

 *Dongdong Kong，2017年8月28日 

 中山大学 

 * / 
 function runTaskList() {
    var tasklist = document.getElementsByClassName('awaiting-user-config');
    for (var i = 0; i < tasklist.length; i++)
        tasklist[i].children[2].click();
}
function confirmAll() {
    var ok = document.getElementsByClassName('goog-buttonset-default goog-buttonset-action');
    for (var i = 0; i < ok.length; i++)
        ok[i].click();
}
runTaskList();
confirmAll();

