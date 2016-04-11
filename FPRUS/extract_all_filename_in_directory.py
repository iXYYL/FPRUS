# -*- coding: utf-8 -*-

# 提取指定文件夹下的所有文件名，并加上相对于脚本所在位置路径，将其存放在指定文件夹下的文本文件all_filename中。
# 其中，all_filename文件中每行存放一个文件名

import os
import sys

print 'Usage: python extract_all_filename_in_directory.py dir_name all_filename'

dir_name = ""
all_filename = ""
if len(sys.argv) < 2:
    dir_name = '.'
    all_filename = 'all_filename.txt'
elif not os.path.isdir(sys.argv[1]):
    dir_name = '.'
elif len(sys.argv) < 3:
    dir_name = sys.argv[1]
    all_filename = 'all_filename.txt'
else:
    dir_name = sys.argv[1]
    all_filename = sys.argv[2]

if dir_name[-1] != '/':
    dir_name = dir_name + '/'

if not len(all_filename):
    all_filename = 'all_filename.txt'

fileList = os.listdir(dir_name)
fout = open(dir_name + all_filename, 'w')
print dir_name + all_filename
for item in fileList:
    if os.path.isfile(dir_name + item) and item != all_filename:
        fout.write(dir_name + item)
        fout.write('\n')

fout.close()

