```math
\newcommand{\dd}[1]{ \, \mathrm{d}{#1} \,}
```
custom functions and modules for A102 and A116
# modules
explain modules
## lablib102.gaussian2d.py

`gaussian_2d_iso` 函数的解析形式为:
```math
\begin{equation}
Z(x,y) = A\exp\left[-\frac{(x-x_0)^2 + (y-y_0)^2}{\sigma_r^2}\right] + C_\text{bg}
\end{equation}
```

其中 $\sigma_r^2 = \sigma_x^2 + \sigma^2_y$, 它们都是相应坐标值的二阶*中心*矩 $\mu_2^\mathrm{c} = \mathbb E[(x-\mu)^2]$ (see e.g. Berendsen 2011 p.30). r 是以质心 $x_0, y_0$ 作为原点的极坐标系 $r, \theta$ 的第一个坐标. 
```math
\begin{align}
\text{1st moment of x}\quad x_0 &= \frac{\int xZ(x,y) \dd{x}\!\dd{y}}{\int Z(x,y) \dd{x}\!\dd{y}}\\
\text{1st moment of y}\quad y_0 &= \frac{\int yZ(x,y) \dd{x}\!\dd{y}}{\int Z(x,y) \dd{x}\!\dd{y}}\\
\text{2nd moment of x}\quad \sigma_x^2 &\equiv  \frac{\int (x-x_0)^2Z(x,y) \dd{x}\!\dd{y}}{\int Z(x,y) \dd{x}\!\dd{y}}\\
\text{2nd moment of y}\quad \sigma_y^2 &\equiv  \frac{\int (y-y_0)^2Z(x,y) \dd{x}\!\dd{y}}{\int Z(x,y) \dd{x}\!\dd{y}}\\
\text{2nd moment of r} \quad \sigma_r^2 &\overset{\text{ISO}}{\equiv}  \frac{\int [(x-x_0)^2 + (y-y_0)^2]Z(x,y) \dd{x}\!\dd{y}}{\int Z(x,y) \dd{x}\!\dd{y}} = \sigma_x^2 + \sigma_y^2
\end{align}
```

其中 eq6 在 ISO 11145 3.3.2 中被称为 *second moment of the power distribution*, 写为 $\sigma^2(z)$,表示光斑是传播 z 方向的函数, 这里我们不考虑 z 方向传播, 省略 z 坐标. 从统计的角度, 上面五式将 $x, y, (x-x_0)^2, (y-y_0)^2, (x-x_0)^2 + (y-y_0)^2$ 这五个量视为随机变量, 求其期望.

最简单的二维高斯强度分布是两个相同的一维高斯的连乘 (Lawrence2019 p.83):
```math
\begin{align}
p(x,y) &= \frac{1}{\sigma\sqrt{2\pi}} \exp\left (\frac{-x^2}{2\sigma^2} \right) \frac{1}{\sigma\sqrt{2\pi}} \exp\left (\frac{-y^2}{2\sigma^2} \right)\\
&= \frac{1}{2\pi\sigma^2} \exp\left [\frac{-(x^2+y^2)}{2\sigma^2} \right]
\end{align}
```
其中 $\sigma = \sigma_x = \sigma_y$. 这种 x,y 无关联且边缘分布密度函数 $\varphi_1(x) = \int p(x,y) \dd{x}$, $\varphi_2(x) = \int p(x,y) \dd{y}$
形式相同 (都是同样的一维高斯) 的二维高斯 eq8 就是我们用于拟合的各向同性二维高斯. 在各向同性时, $2\sigma^2 \textcolor{magenta}{=} \sigma_x^2 + \sigma_y^2 \overset{\text{ISO}}{\equiv} \sigma_r^2$. 颜色等号是各向同性时恰好成立的等式, $\sigma_x^2 + \sigma_y^2$ 并不会在任何二维高斯公式中出现 (指数项: independent but different dispersions $x^2/\sigma_x^2 + y^2/\sigma_y^2$; correlated variables $x^2/\sigma_x^2 + y^2/\sigma_y^2-2\rho_{xy}/\sigma_x\sigma_y$), 但是 eq6 在 ISO 中被用来定义 D4sigma (ISO 定义式为 $d_\sigma = 2\sqrt 2 \sigma_r$, 其中 $d_\sigma$ 被 newport LBP2 manual 称为 $\text{D4}\sigma$)

再将 eq8 写为 eq1 时, 我们将统计量 $\sigma_x^2 + \sigma_y^2 \equiv \sigma_r^2$ 视为了一个拟合参数, 这个拟合参数在各向同性高斯情况下, 对应 $\text{D4}\sigma^2/8$. 