---
layout:     post
title:      ImageCollection (Error) Expected a homogeneous image collection, but an image with incompatible bands was encountered
subtitle:   GEE踩坑记录
date:       2023-06-09
author:     liyang
header-img: img\GoodnightCopenhagen.png
catalog: true
tags:
    - GEE
---

## ImageCollection (Error) Expected a homogeneous image collection, but an image with incompatible bands was encountered——一个困扰我两天的bug

在使用GEE中， 有一个需求：有一个ImageCollection，里面是2010年到2013年逐日的image，我想把逐日的image聚合成逐月的image，采用求和的方式。这其实是一个很简单的操作，大概用到的代码如下：

```javascript
// Google Earth Engine
//var monthly = daily.filterDate(start_month, end_month).sum().toFloat();

//具体代码如下
var ToMonth = function (n){ 
    var start_month = ee.Date(start).advance(n, 'month'); // Starting date
    var end_month = start_month.advance(1, 'month'); // Step by each iteration
    var month = start_month.get('month')
    var month_data = daily_data
                  .filterDate(start_month, end_month)
                  .reduce(ee.Reducer.max())
    return ee.Image(month_data)
                  .rename(new_name)
                  .set({
                  'date':start_month,
                  'system:time_start': start_month.millis(),
                  'year': start_month.get('year'),
                  'month': start_month.get('month')
              });
}

var month_sum = ee.List.sequence(0, 12 * 2 - 1).map(ToMonth);
print('month sum', month_sum)


```



开始的时候，上面的代码总是保存，我以为是图像的数据类型不一样，因此设置了强制转换，但是依然有上述的错误。最后不断尝试，才发现问题出在了daily_data 。daily_data里的每一个image为逐日的数据，我当时画蛇添足的为每个图像设置了不同了名字（名字前边计算并添加了day of year信息，估计应该是这个名字导致了图像not homogeneous，最后给他们换了同样的名字（其实就是去掉了 day of year信息），然后就跑通了。 

一个简单的操作，耽误了两天时间。唉！立贴纪念。