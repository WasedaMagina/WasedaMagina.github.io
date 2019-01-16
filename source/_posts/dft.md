---
title: Use python to implement Discrete Fourier transform
date: 2018-01-30 01:58:05
tags: 课程作业
categories: Report
mathjax: true
---
In image processing, Discrete Fourier Transformation is a very useful method. However, its formula is so complicated that it is difficult to understand. As a result, I intend to deepen my understanding of it by implementing it. Since I am not familiar with c or c++, I use python to do this task. During the task, I realize that the efficiency of my method is not as good as I thought, so I use several methods to improve it. The goal of my work is implementing the Discrete Fourier Transformation(DFT) in the most efficient way in python.

In this report, I would like to show my work in the following aspects:
1. Introduction of Discrete Fourier Transformation.
2. Introduction of my experiment.
3. Implement the DFT with standard formula.
4. Implement the DFT with Euler's formula.
5. Implement the DFT with matrix.
6. The principle of Fast Fourier Transform(FFT).
7. Conclusion.

### 1. Introduction of Discrete Fourier Transformation
In 1822, Fourier, a French scientist, pointed out that any periodic function can be expressed as a composition of sine and cosine in different frequencies. The posterity found that in some conditions, even non-periodic function can be expressed by sine and cosine. Following this idea, Fourier Transformation(FT) is produced.

In image processing, the image data is discrete value. Based on the Fourier Transformation, Later researchers invent the Discrete Fourier Transformation(DFT) to hold the discrete value. Although the formula of FT and DFT is different, the principle of them is same. Therefore, we can use DFT to convert the image to a combine with sine and cosine. Even better, we could use the Inverse DFT to convert it back to image. As a result, DFT is very important in image processing.

### 2. Introduction to my experiment
#### 2.1 Purpose
I aim to use python to implement the DFT in the most efficient way. In this report, I implement the DFT in different ways and I would give the comparison of them.

#### 2.2 Programming environment
The version of python is 3.6, IDE is jupyter notebook.

In my program, I use three libraries.
1. Opencv. I use this library to read image from folder.
2. Numpy. I use this library to do matrix computing.
3. Matplotlib. I use this library to show my results in picture.

(If researches have the same version python with libraries that I mentioned before, they can copy my code to jupyter notebook and run it to check my work.)

#### 2.3 Data preparation

I prepare two different sizes of classic image Lena to test my program. One is the original image and its size is 512\*512. The other is shrank  image and its size is 50\*50. Because the original  image size is too large to do the test. Some of the algorithms will be too slow to get a result.

Before the experiment, I read images from my folder. Here is code and explanations. I use function imread to read image and plt to show my results. Variable lena saves the information of the original lena and lena50 saves the information of the small sizes lena image.


```python
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

lena = cv2.imread("C:\lena.jpg",0)   # read image
lena50 = cv2.imread("C:\lena50.jpg",0)
plt.subplot(121),plt.imshow(lena,'gray'),plt.title('lena_512*512')
plt.subplot(122),plt.imshow(lena50,'gray'),plt.title('lena_50*50')
plt.show()
```


![png](http://ozaeyj71y.bkt.clouddn.com/image/dft/output_3_0.png)


#### 2.4 The steps of evaluating the result of method.
I evaluate my result by following steps:
1. Get result from my function.
2. Get the standard answer from numpy's fft funtion.
3. Print out the result of each function.
4. Check the picture to see whether they are same or not.
5. Use allclose to check the result of my function whether correct or not.
6. For each function, use timeit function to calculate the cost of time.

### 3. Implement the DFT with standard formula.
#### 3.1 How to implement it.
The standard formula of the DFT is:
$$
F(u,v)= \sum\_{x=0}^{M-1}\sum\_{y=0}^{N-1}f(x,y)e^{-j2\pi(ux/M+vy/N)}
$$
$f(x,y)$ means the pixel value. $M$ and $N$ is the length and width of the image$f(x,y)$.

We can get several information from this formula:

1. The output of Fourier Transformation is a complex matrix.
2. Each pixel in the output is the sum of input pixel multiply a complicated formula.

Based on these information, I start coding. 
First, I use the shape function get the row and column information from input image.
Second, I build a complex matrix with same dimension of the input image.
Finally, I use four loops to implement the Fourier Transformation. Here is the code:


```python
def dft(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    output_img = np.zeros((rows,cols),complex)
    for m in range(0,rows):
        for n in range(0,cols):
            for x in range(0,rows):
                for y in range(0,cols):
                    output_img[m][n] += input_img[x][y] * np.exp(-1j*2*math.pi*(m*x/rows+n*y/cols))
    return output_img
```

#### 3.2 Result


```python
dft_lena50 = dft(lena50)
out_dft = np.log(np.abs(dft_lena50))
fft_lena50 = np.fft.fft2(lena50)
out_fft = np.log(np.abs(fft_lena50))

plt.subplot(121),plt.imshow(out_dft,'gray'),plt.title('dft_output')
plt.subplot(122),plt.imshow(out_fft,'gray'),plt.title('np.fft_output')
plt.show()
np.allclose(dft_lena50,fft_lena50)
```


![png](http://ozaeyj71y.bkt.clouddn.com/image/dft/output_8_0.png)

    True
```python
%timeit dft_lena50 = dft(lena50)
%timeit fft_lena50 = np.fft.fft2(lena50)
```

    26.3 s ± 574 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    96.1 µs ± 1.98 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


#### 3.3 Result analysis
We can see that the output image of dft is as same as the np.fft_output. And the result of np.allclose is true which means that each value in matrix dft_lena50 and fft_lena50 is equal. Therefore, I have already implemented the DFT. 

However,my method cost so much time. At the beginning, I use original image(size 512\*512) to do the test and I even can not get the result. After changing the size of it, I can get result. It is still too slow. By using my function, it costs 26.3 seconds in average. But by using the function in numpy, it costs only 96.1 us in average. As a result, I figure out two ways to improve my code. First, I think that the exponential calculation in the loop will consume a lot of computing resources, so I use Euler's formula to convert the exponent calculation into addition calculation.

### 4. Implement the DFT with Euler's formula.
#### 4.1 How to implement it.
I use Euler's formula to change the exponent calculation into addition calculation. Than I get a new formula:
$$
F(u,v)= \sum\_{x=0}^{M-1}\sum\_{y=0}^{N-1}f(x,y)\*(\cos(-2\pi(ux/M+vy/N))+j\sin(-2\pi(ux/M+vy/N)))
$$
Implement is very simple. I just change one row in dft then I get dft_ol function. 


```python
def dft_ol(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    output_img = np.zeros((rows,cols),complex)
    for m in range(0,rows):
        for n in range(0,cols):
            for x in range(0,rows):
                for y in range(0,cols):
                    w = -2*math.pi*(m*x/rows+n*y/cols)
                    output_img[m][n] += input_img[x][y] * (math.cos(w) + 1j*math.sin(w))
    return output_img
```

#### 4.2 Result
Then, I did the same evaluation of it. Here is the result:


```python
dftol_lena50 = dft_ol(lena50)
out_dftol = np.log(np.abs(dftol_lena50))
fft_lena50 = np.fft.fft2(lena50)
out_fft = np.log(np.abs(fft_lena50))

plt.subplot(121),plt.imshow(out_dftol,'gray'),plt.title('dftol_output')
plt.subplot(122),plt.imshow(out_fft,'gray'),plt.title('np.fft_output')
plt.show()
np.allclose(dftol_lena50,fft_lena50)
```


![png](http://ozaeyj71y.bkt.clouddn.com/image/dft/output_15_0.png)

    True
```python
%timeit dftol_lena50 = dft_ol(lena50)
%timeit fft_lena50 = np.fft.fft2(lena50)
```

    21.8 s ± 84 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    91.7 µs ± 855 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


#### 4.3 Result analysis
We could see that the result is correct. Although the average time is a little bit shorter than before(21.8s in average), it is still not acceptable.

I think the problem is I used four loops in my code. In python, the efficiency of loop is low so I wander that is there any possibility to replace the loops? I check the formula again. The input of it is a matrix and the output of it is also a matrix. If we apply numpy library to do matrix computing, efficiency of calculating is high. Therefore I tried to write the formula in matrix way.

### 5. Implement the DFT with matrix.
#### 5.1 How to implement it.
The image data is a two dimension matrix. Imply the DFT on two dimension data is a 2D-DFT problem.

The 2D-DFT can write in two 1D-DFT:
$$
F(u,v)=\sum\_{x=0}^{M-1}e^{-j2\pi ux/M}\sum\_{y=0}^{N-1}f(x,y)e^{-j2\pi vy/N}=\sum\_{x=0}^{M-1}F(x,v)e^{-j2\pi ux/M}
$$

$$
F(x,v) =\sum\_{y=0}^{N-1}f(x,y)e^{-j2\pi vy/N}
$$

and the 1D-DFT is easy to write in matrix way:
$$
F(x,v) = M\vec x
$$

$$
M = e^{-j2\pi \vec v \vec y/N}
$$

$\vec x $ means each row vectors of $f(x,y)$, $\vec v$ is a column vector $(0,1,2，\cdots，N-1)^T$,$\vec y$ is a row vector$(0,1,2，\cdots，N-1)$. 

For 2D-Fourier Transformation , we just need to do the 1D-DFT for each row of input and do 1D-DFT for each column of the output from 1D-DFT for rows.

I follow this new formula building the dft_matrix function.


```python
def dft_matrix(input_img):
    rows = input_img.shape[0]
    cols = input_img.shape[1]
    t = np.zeros((rows,cols),complex)
    output_img = np.zeros((rows,cols),complex)
    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows,1))
    y = n.reshape((cols,1))
    for row in range(0,rows):
        M1 = 1j*np.sin(-2*np.pi*y*n/cols) + np.cos(-2*np.pi*y*n/cols)
        t[row] = np.dot(M1, input_img[row])
    for col in range(0,cols):
        M2 = 1j*np.sin(-2*np.pi*x*m/cols) + np.cos(-2*np.pi*x*m/cols)
        output_img[:,col] = np.dot(M2, t[:,col])
    return output_img
```

#### 5.2 Result
Also, I use the same flow to evaluate it. Here is the result:


```python
dftma_lena50 = dft_matrix(lena50)
out_dftma = np.log(np.abs(dftma_lena50))
fft_lena50 = np.fft.fft2(lena50)
out_fftma = np.log(np.abs(fft_lena50))

plt.subplot(121),plt.imshow(out_dftma,'gray'),plt.title('dftma_output')
plt.subplot(122),plt.imshow(out_fftma,'gray'),plt.title('np.fft_output')
plt.show()
np.allclose(dftma_lena50,fft_lena50)
```


![png](http://ozaeyj71y.bkt.clouddn.com/image/dft/output_21_0.png)

    True
```python
%timeit dftma_lena50 = dft_matrix(lena50)
%timeit fft_lena50 = np.fft.fft2(lena50)
```

    10.9 ms ± 196 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    92 µs ± 523 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


The result is correct and the time is much shorter than other functions. It costs only 10.9ms in average. I think it is fast enough so I give the original lena as input and run it to see the result.


```python
dftma_lena = dft_matrix(lena)
out_dftma = np.log(np.abs(dftma_lena))
fft_lena = np.fft.fft2(lena)
out_fftma = np.log(np.abs(fft_lena))

plt.subplot(121),plt.imshow(out_dftma,'gray'),plt.title('dftma_output')
plt.subplot(122),plt.imshow(out_fftma,'gray'),plt.title('np.fft_output')
plt.show()
np.allclose(dftma_lena,fft_lena)
```


![png](http://ozaeyj71y.bkt.clouddn.com/image/dft/output_24_0.png)

    True
```python
%timeit dftma_lena = dft_matrix(lena)
%timeit fft_lena = np.fft.fft2(lena)
```

    16.3 s ± 448 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    11.3 ms ± 151 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


#### 5.3 Result analysis
The result is correct but the time of calculation increase rapidly as the size of image increases. Although np.fft cost more time than before, it is not increase so rapidly. I check this online, the method that numpy used is called Fast Fourier Transformation(FFT). Since this function is not in the lecture so I did not implement it in my report but I studied its principle. 

### 6. The principle of Fast Fourier Transform(FFT).

Since we can use two 1D-DFT to calculate the 2D-DFT, we only to improve the efficiency of 1D-DFT than we can improve the efficiency of 2D-DFT.

For a 1D-DFT:
$$
F(u)=\sum\_{x=0}^{M-1}f(x)W\_{M}^{ux}
$$
if M is divisible by 2, we can write it in two parts:
$$
M = 2K \\
F(u)=\sum\_{x=0}^{K-1}f(2x)W\_{K}^{ux} + \sum\_{x=0}^{M-1}f(2x+1)W\_{K}^{ux}W\_{2K}^{ux}
$$
But in this formation the length of $F(u)$ is only a half as before. We need to calculate the other part of it:
$$
F(u+K) = \sum\_{x=0}^{K-1}f(2x)W\_{K}^{ux} - \sum\_{x=0}^{M-1}f(2x+1)W\_{K}^{ux}W\_{2K}^{ux}
$$
This is called the symmetry of DFT. We only calculate half of DFT than we can got the result of whole DFT as soon as M is divisible by 2.

Even better we can set $M=2^n$ , than we can divide the DFT into several small DFT and use the symmetry of DFT to reduce the amount of computation. This is the principle of FFT.

Time complexity of FFT is $O(nlogn)$ , DFT is $O(n^2)$. Therefore, it is much faster than the DFT when the n is large.

### 7. Conclusion.

Here is the result of all functions:

| Function name              | Input image | Answer correct or not | Time in average |
| -------------------------- | ----------- | --------------------- | --------------- |
| DFT with lecture's formula | Lena50      | True                  | 26.3 seconds    |
| DFT with Euler's formula   | Lena50      | True                  | 21.8 seconds    |
| DFT with matrix            | Lena50      | True                  | 10.9 ms         |
| Numpy's fft function       | Lena50      |                       | 91.7 us         |
| DFT with matrix            | Lena        | True                  | 16.3 seconds    |
| Numpy's fft function       | Lena        |                       | 11.3 ms         |

As a result, I think the most efficient way to implement Discrete Fourier transform(DFT) in Python is use matrix to replace the loops. It is much faster than other method. However, the 

After this report, I feel I understand the Discrete Fourier Transform deeper than before. Here are the conclusion of this task:

1. The Fourier transform converts the image to a superposition of sine and cosine. It is not intuitive to imagine an image is a superposition of sine and cosine. However, if we treat all the pictures as an instant noodles, understanding the concept of Fourier Transformation will be much more easier. Images is an instant noodles, sine and cosine is the noodle in noodles. 
2. The reason why we use Fourier transform is someone like thick noodle and others like the thin noodle. Fourier Transformation is a method that decomposition the instant noodles into one and one different noodle. Then we can choose the most appropriate noodle for different people. In image processing, it means that in some conditions, we may be interested in high frequency items and sometimes we may need the low frequency items. We can use the Fourier Transformation to find the desire items.
3. In python, it is very important to learn how to use an appropriate library. In this task, I use the matrix to replace the loops in function. I use the numpy library to do the matrix computing. With this help, I reduce the time from 21.8 seconds to 10.9ms. If I use the numpy's FFT function directly, only cost 91.7us. Therefore, the efficiency of the functions in python library is very high. It is much more practical to find a corresponding library function when encounter a problem.
4. If we can't find the corresponding library, it would be better to use others programming language to implement it such as C or C++. Because, without the help of library, python is too slow. 

### 8. Reference

[1] Gonzalez, Rafael C, & Woods, Richard E. (1977). Digital image processing. **Prentice Hall International**,* *28*(4), 484 - 486.

[2] 快速傅里叶变换（FFT）算法【详解】