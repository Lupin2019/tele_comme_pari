---
typora-root-url: .
typora-copy-images-to: IMA204.assets
---

# IMA204 -- Active Contours

- [Assets](https://pan.baidu.com/s/1nIm7D0RtFk9SwTaHfgiZng)
  - TP
  - Cours



## 1. TH

- [Paper](https://link.springer.com/content/pdf/10.1007/BF00133570.pdf)

![图像分割—Snake模型](https://pic1.zhimg.com/70/v2-ba180385facbeb2d24c48ce3795ee0fd_1440w.image?source=172ae18b&biz_tag=Post)

### 1.1 [主动轮廓模型 CSDN](https://blog.csdn.net/sunlinju/article/details/52999872)

#### 1.1.1 基于边缘的主动轮廓模型

- **snake model energy:**
  $$
  \Large {
  J_{\mathrm{snake}} = J_{\mathrm{int}}  + J_{\mathrm{ext}} + J_{\mathrm{cons}} 
  }
  $$

  - $J_{\mathrm{int}}$:  continuity and smoothness of the contour curve.

  - $J_{\mathrm{ext}}$: image force
  - $J_{\mathrm{cons}} $: constraint term

- **internal energy:**
  $$
  \Large \begin{align*}
  J_{\mathrm{int}} &= J_{\mathrm{countin}}  + J_{\mathrm{smooth}} \\
  &= \alpha(s)|\frac{\mathrm{d}C}{\mathrm{d}s}|^2 + \beta(s)|\frac{\mathrm{d}^2C}{\mathrm{d}s^2}|^2
  
  \end{align*}
  $$

- **external energy :**
  $$
  \Large \begin{align*}
  J_{\mathrm{ext}} = \gamma(s)|\nabla I|^2 
  \end{align*}
  $$

- **snake :**
  $$
  \Large J_{snake}=\int_0^1 [ {\alpha(s)|\frac{dC}{ds}|^2+\beta(s)|\frac{d^2C}{ds^2}|^2}-\gamma(s)|\nabla I|^2]ds
  $$

  >通过求解上式的最小值，使轮廓曲线收敛在图像的最大梯度点，而图像的最大梯度一般在目标的边缘处取得，也即检测出了目标的边缘。 为了克服原始的Snake模型的不足，Cohen等人把==气球力==引入到Snake模型，使轮廓曲线可以越过非目标边界的局部极值点。但气球力依靠其系数的正负来决定轮廓曲线的收敛方向，而这一系数的正负是固定的，无法自适应轮廓曲线的初始位置变化，因此该模型的应用受到了局限。 而后，Caselles等人又在Snake模型的基础上提出了==测地线主轮廓模型==，通过选择合适的停止函数，使演化曲线停止在目标边界。测地线祝轮廓模型摆脱了初始轮廓曲线的位置对检测结果的影响，在很大程度上改进了Snake模型。 但是，无论是Sanke模型还是测地线主动轮廓模型，都是基于图像的梯度信息来判断目标的边缘，当图像的背景复杂或者是目标的边缘模糊时，无法准确的定位目标的边缘，造成检测结果不准确

#### 1.1.2 基于区域的主动轮廓模型

- **energy expression :**
  $$
  \Large \begin{align*}
  E(C) &= E_{in}(C)+E_{out}(C) \\
  &= \int_{inside(C)}|I(x,y)-c_0|^2 {\rm d}x{\rm d}y \\
  &+\int_{outside(C)}|I(x,y)-c_b|^2 {\rm d}x{\rm d}y
  \end{align*}
  $$

  - $c_0$: average grayscale of the internal region
  - $c_b$: average grayscale of the external region
  - $I(x, y)$: grayscale at $(x,y)$

- 

#### 1.1.3 水平集方法(Level Set)

- **Level set function :**
  $$
  u(x,y)=
  \begin{cases}
  d[(x,y),C]& \text{(x,y)在封闭曲线C的外部}\\
  -d[(x,y),C]& \text{(x,y)在封闭曲线C的内部}
  \end{cases}
  $$



---



### 1.2 [基于边缘的主动轮廓模型——从零到一用python实现snake](https://blog.csdn.net/weixin_40673873/article/details/106664590)



---



### 1.3 [CMU Active Contours(Slides)](https://www.cs.cmu.edu/~galeotti/methods_course/Segmentation2-Snakes.pdf)



### 1.4 [《Snakes: Active Contour Models》算法原理](https://blog.csdn.net/cfan927/article/details/108884457)

- 内部能量 $E_{int}$ , 一阶导作用是保持轮廓的==连续性==, 二阶导是控制==轮廓平滑程度==

- 图像能量 $E_{\mathrm{image}}$ :  $E_{\text{image}} = \omega_{\text{line}} E_{\text{line}} + \omega_{\text{edge}}E_{\text{edge}} +  \omega_{\text{term}}E_{\text{term}}$ , 引导轮廓到线, 边和末端

  - 控制$\omega_{line}$的正负号可以控制轮廓被吸引到较暗的线或是较亮的线，也就是使轮廓试图靠近轮廓的最暗或最亮处。snake轮廓在被吸引到零交点的同时，仍然受到它自己的平滑限制

  - $E_{\text{line}} = I(x, y)$
  - $E_{\text{line}} = - (G_{\sigma} * \nabla^2 I)^2$
  - $E_{\text{edge}} = - |\nabla I(x, y)|^2$
  - $E_{\text{term}} = \frac{\partial n_{\perp}}{\partial \theta} = \frac{\partial C}{\partial n} \frac{\partial^2 C}{\partial n_{\perp}^2} = \frac{(C_{x}^2 + C_{y}^2)^{3/2}}{C_{yy}C_{x}^2 - 2C_{xy}C_{x}C_{y} + C_{xx}C_{y}^2}$

  

- ![image-20231214210158968](/IMA204.assets/image-20231214210158968.png)



---

---



## 2. TP  :waxing_crescent_moon:

### 2.1 Previous

After undergoing the baptism of the previous TPs, I realized that it is not the best practice to take on a project and start working on it directly. A better approach is to review the entire notebook as a whole, gaining an understanding of the teacher's thought process in designing the project.

### 2.2 Some requests

1. Essential libraries include numpy, matplotlib, IPython, skimage, and ffmpeg.
2. Use the provided interactive tool (iview_IMA204.html) to visualize images and retrieve pixel coordinates with intensity values.
3. Complete and submit a Jupyter Notebook that accesses local images.
4. Include any new images in your submitted documents and ensure they are read from the `/images_misc/` local folder in your code.

### 2.3 Notebook Review

- lots of functions in `skimage`

- good format,  far much better than previous ones

  ```python
  s      = np.linspace(0, 2*np.pi, Nber_pts)
  Radius = R0  
  r      = r0 + Radius*np.sin(s)
  c      = c0 + Radius*np.cos(s) #col
  init   = np.array([r, c]).T
  ```

- nice use of parameter `as_gray` in `skimage.io.imread()`

- Some code segment can be optimized; there is too much repetition.

- We said "some routines won't work with *uint8*"

- ==TODO :==

  - [ ] Snake on a binary shape
  - [ ] Snake on a real image
  - [ ] A tool to visualise the deformations of the snake
  - [ ] Snake with Gradient Vector Flow
  - [ ] The active contour with fixed end points
  - [ ] Propose a motivated pipeline using the snake capabilities forme the acitve) contour function
  - [ ] Test the GEometric Level-Swet formulation using the Chan-Vese model
  - [ ] Geometric active contours with balloon force

### 2.4 Active Contours, here we go!

- `from IPython.display import HTML` <u>First time encountering this usage</u> :mag:

  ```python
  from IPython.core.display import HTML
  HTML('<a href="http://example.com">link</a>')
  
  # output:
  link
  ```

- `skimage.segmentation `

  - `chan_vese`: This algorithm was first proposed by Tony Chan and Luminita Vese, in a publication entitled “An Active Contour Model Without Edges” 

    ![Original Image, Chan-Vese segmentation - 200 iterations, Final Level Set, Evolution of energy over iterations](https://scikit-image.org/docs/stable/_images/sphx_glr_plot_chan_vese_001.png)

  - `morphological_chan_vese`

    ![Morphological ACWE segmentation, Morphological ACWE evolution, Morphological GAC segmentation, Morphological GAC evolution](https://scikit-image.org/docs/stable/_images/sphx_glr_plot_morphsnakes_001.png)

  - `checkerboard_level_set`

  - `morphological_geodesic_active_contour`

- `skimage.morphology`

  - `white_tophat`
  - `black_tophat`
  - `disk`

- ```python
  from IPython.core.interactiveshell import InteractiveShell
  InteractiveShell.ast_node_interactivity = "all"
  ```

  - Display full outputs in Jupyter Notebook, not only the last command's output. ==Why should we do this? Not sure for now==

  - InteractiveShell.ast_node_interactivity[¶](https://ipython.readthedocs.io/en/stable/config/options/terminal.html#configtrait-InteractiveShell.ast_node_interactivity)

    ‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’ specifying which nodes should be run interactively (displaying output from expressions).

    - Options:

      `'all'`, `'last'`, `'last_expr'`, `'none'`, `'last_expr_or_assign'`

    - Default:

      `'last_expr'`

- We offer some functions:

  ```python
  def edge_map(img,sigma):
  def edge_map2(img,sigma):
  def subtract_background(image, radius=5, light_bg=False):
  def define_initial_circle(R0,r0,c0,Nber_pts=400):
  def animate_cv(image, segs, interval=1000):
  def animate_snake(image, segs, interval=500):
  def store_evolution_in(lst):
  ```

- ```python
  img_test  = img_cell
  Sigma_val = 1
  edge_test = edge_map(img_test, sigma=Sigma_val) # sobel filter
  ```

  - sobel filter to get edges:

  $$
  G_x = 
  \begin{pmatrix}
  +1 & 0 & -1 \\
  +2 & 0 & -2 \\
  +1 & 0 & -1
  \end{pmatrix}
  
  * 
  \textbf{A}, \quad
  
  G_y = 
  \begin{pmatrix}
  +1 & +2 & +1 \\
  0 & 0 & 0 \\
  -1 & -2 & -1
  \end{pmatrix}
  
  * 
  \textbf{A}
  $$

- ```python
  edge_inv_test  = skimage.segmentation.inverse_gaussian_gradient(img_to_test, alpha=1.0, sigma=1.0) 
  ```

  - steepness: 陡度
  - gaussian -> gradient -> `return 1.0 / np.sqrt(1.0 + alpha * gradnorm)`

- ```python
  # Run all  OPTION_ENHANCE for display here       
  gamma_corrected       = skimage.exposure.adjust_gamma(img_to_test, 0.8)
  logarithmic_corrected = skimage.exposure.adjust_log(img_to_test, gain= 1,inv=True)
  img_open              = skimage.morphology.diameter_opening(img_to_test, 40, connectivity=2)
  img_adapteq           = skimage.exposure.equalize_adapthist(img_to_test, clip_limit=0.03)
  ```

  - | Function                                                     | Functionality                                             |
    | ------------------------------------------------------------ | --------------------------------------------------------- |
    | [`skimage.exposure.adjust_gamma`](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma) | Performs Gamma Correction on the input image.             |
    | [`skimage.exposure.adjust_log`](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_log) | Performs Logarithmic correction on the input image.       |
    | [`skimage.exposure.equalize_adapthist`](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) | Contrast Limited Adaptive Histogram Equalization (CLAHE). |
    | [`skimage.exposure.equalize_hist`](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist) | Return image after histogram equalization.                |
  
  - **限制对比度自适应直方图均衡化** CLAHE

- ```python
  Radius_val = 15
  img_test1  = subtract_background(img_to_test, radius=Radius_val, light_bg=False)
  img_test2  = subtract_background(img_to_test, radius=Radius_val, light_bg=True)
  ```

  - black_tophat
  - white_tophat
  - ==原图 minus topcat== 有什么明堂吗？

- Definition of Energy

  ![image-20231213014306695](/IMA204.assets/image-20231213014306695.png)

  ![image-20231213014327179](/IMA204.assets/image-20231213014327179.png)

  ![image-20231213014913112](/IMA204.assets/image-20231213014913112.png)

​	![image-20231213015003671](/IMA204.assets/image-20231213015003671.png)

​	![image-20231213024100436](/IMA204.assets/image-20231213024100436.png)





通过观察30次迭代后的结果可以发现， 当初始init的边缘距离目标分割图像边缘较近的时候， 轮廓可以逐渐被“吸引”过去。 



还有一个原因是， 左侧细胞图像的颜色梯度较小， 意思是颜色比较均匀， 干扰较少。
