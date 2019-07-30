---
layout:     post
title:      Keras 回调函数中计算F1，Recall, Precission
subtitle:   解决fit_generator时
date:       2019-07-30
author:     liyang
header-img: img/Rio2.png
catalog: true
tags:
    - Kreas
    - F1
    - 多分类
---







#  Keras 回调函数中计算F1，Recall, Precission

最近在参加一个分类比赛，想在每次训练时输出precision，recall，和f1_score；但是keras中没有给出相关函数，查了一会，在知乎找到一个答案，实现方法应该是：添加一个callback，在on_epoch_end的时候通过sklearn的f1_score这些API去算。但是他的代码只能使用在用fit函数，validation_data不是一个生成器的情况下使用。但是我的代码中使用的都是生成器，程序一直报如下错：

~~~linux
TypeError: 'NoneType' object is not subscriptable
~~~

，仔细查看了源码，原来是Callback函数中没有初始化。于是在Github上找到了另外一个解决办法。不幸的是他的代码也有问题（主要是类别维度那有问题）。最后改了一下，终于跑通了。

~~~python
class Metrics(Callback):
    def __init__(self, val_data, batch_size = BATCH_SIZE):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        
    def on_train_begin(self, logs={}):
        print("on train begin", len(self.validation_data))
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        print("on epoch end :", len(self.validation_data))
        batches = len(self.validation_data) ## 等于所有数据 / batchsize
        total = batches * self.batch_size
        
        val_pred = np.zeros((total))
        val_true = np.zeros((total))
        
        for batch in range(batches):
            xVal, yVal = next(self.validation_data)
            # print(len(yVal))
            val_pred[batch * self.batch_size : (batch+1) * self.batch_size] = np.argmax(np.asarray(self.model.predict(xVal)), axis = 1).round()
            val_true[batch * self.batch_size : (batch+1) * self.batch_size] = np.argmax(yVal, axis = 1) 
        #val_pred = np.squeeze(val_pred)
        
        _val_f1 = f1_score(val_true, val_pred, average='macro')
        _val_recall = recall_score(val_true, val_pred)
        _val_precision = precision_score(val_true, val_pred)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:' ,_val_f1)
        return
~~~



## 参考

https://github.com/keras-team/keras/issues/10472