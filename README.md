# EGO-Algorithm
复现了论文“Efficient Global Optimization ofExpensive Black-Box Functions”中的EGO算法。
原论文地址"http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf"

## 复现结果
模型采用了Kriging模型，采样方法采用了LHS方法。初始采样点数设置为20，在Branin函数上的结果：最佳函数值与全局最小值的误差小于1%。得到的最佳函数值为0.398481，与全局最小值的实际误差为0.15%。
