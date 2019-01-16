# Coursera C程序设计进阶 第五周编程作业 编程题#2 二维数组右上左下遍历（附思路） 

### 描述

给定一个row行col列的整数数组array，要求从array[0][0]元素开始，按从右上到左下的对角线顺序遍历整个数组。

![示意图](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Gq_qOCD5EeWG5hLaeYFpVQ_888e644c307b3f277b12032c40c6f8d2_Screen-Shot-2015-07-02-at-1.24.44-PM.png?expiry=1537315200000&amp;hmac=gv8s_kkdaX4_xi64e3Nji1iozq0M9jW0E3khd4VDngc)

### 输入

输入的第一行上有两个整数，依次为row和col。

余下有row行，每行包含col个整数，构成一个二维整数数组。

（注：输入的row和col保证0 < row < 100, 0 < col < 100）

### 输出

按遍历顺序输出每个整数。每个整数占一行。

### 思路

首先是找规律，如果把上面的示意图旋转45°就可以发现这个遍历方式把二维数组转成了另一种形式的二维数组。

以一个三行四列的二维数组为例，我们写出转换完毕后的结果：[a[0][0]