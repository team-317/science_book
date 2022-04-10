'''
Author: your name
Date: 2022-03-04 08:28:21
LastEditTime: 2022-03-08 08:36:43
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \EssentialCplusd:\ProgramCoding\ScienceBook\Draft\draft.py
'''


def hello(a: int) -> int:
    print(a)
    return a


b = hello({'a': 9})
hello.name = 'hello'

print(hello.name)
