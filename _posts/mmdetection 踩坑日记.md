## mmdetection 踩坑日记

1. 运行验证代码： python tools/voc_eval.py result.pkl ./configs/cascade_rcnn_r50_fpn_1x.py

![1567694826604](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1567694826604.png)

解决方案： https://github.com/open-mmlab/mmdetection/issues/642

即添加下列代码到 label_names 初始化后：

```python
num_classes = len(label_names)
```

![](https://i.loli.net/2019/09/05/AFCKzZmjLlQdsg4.jpg)

问题解决。最后输出如下：

![1567694968254](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1567694968254.png)

