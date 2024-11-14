# coding=utf-8
import numpy as np 
import os
#导出task3_result.txt
def store_txt(label,result_txt):
    with open(result_txt,"w") as w:
        for index,result in enumerate(label,1):#label后面的1,代表下标从1开始
            w.write(str(index)+" "+str(result)+"\n")#按网站要求格式写入，imageID（int）+" "（此处为空格非\t）+label（-1\1,int）+"\n"
def main():
    label = np.random.binomial(1, 0.5, 1000)#二项分布以0.5概率生成1000长度的1,0标签,长度保持和测试集一致
    label[label==0] = -1 #标签映射到1,-1
    store_txt(label,'./task3_result.txt')#结果存储到txt文件,默认在demo.py文件路径下
if __name__ == "__main__":
    main()           