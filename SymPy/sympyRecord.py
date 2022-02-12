# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:19:31 2022

@author: DELL
"""
#%% 所有模块https://docs.sympy.org/latest/modules/index.html
from sympy import exp,ln
from sympy import Function,symbols
from sympy.abc import x,y,z
# 或者：x, y,z = symbols("x y z")
f = Function('f')
# 或者：f = symbols('f', cls=Function)
#%% 1.代数方程求解 https://docs.sympy.org/latest/modules/solvers/solvers.html#algebraic-equations
from sympy import solve
# 函数solve()，可以传入flag（如set=True,dict=True）使返回格式为集合或字典

# 当x，y都为变量时，例：求解 x**2-xy+7=0; x+y-6=0.
expr = [x**2 - x*y + 7, x + y -6]
result1 = solve(expr, dict=True)

# 当x为变量，y为常数时，例：求解x**2-xy+7=0
result2 = solve(expr[0],x)  # x为变量，结果中用y来表示x

# 当x，y为常量，z为变量时，例：求解x**y+ln(z)+e**x=0
result3 = solve(x**y+ln(z)+exp(x), x, z)

#%% 微分方程计算
from sympy import Derivative,dsolve

# 简单微分方程，例：y''+9y=0，初值为f(0)=5,f'(6)=7
result1 = dsolve(Derivative(f(x),(x,2)) + 9*f(x), f(x), ics={f(0):5, f(x).diff().subs(x,6):7})
# 其中Derivative(f(x),(x,2))也可以写为f(x).diff(x,2)或者Derivative(f(x),x,x)

from sympy import erfi
 


