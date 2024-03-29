import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import math
import time  # 引入time模块
import sys as sys
from copy import deepcopy

# 判断是否存在路径，存在返回1，不存在返回0
def exist_way(a,b,c,d):  # 地图理解为左shang角为原点(x,y)访问map是y行x列map【y】【x】
    x1=a
    y1=b
    x2=c
    y2=d
    nx1=int(math.floor(x1))
    nx2 = int(math.floor(x2))
    ny1 = int(math.floor(y1))
    ny2 = int(math.floor(y2))
    k=0
    if (x1==x2):
        k=0
    else:
        if ((y2-y1)/(x2-x1)<99999 and (y2-y1)/(x2-x1)>-99999):
            k=(y2-y1)/(x2-x1)
        else:
            if ((y2-y1)/(x2-x1)<-99999):
                k=-99999
            else:
                k=99999
    b=y1-k*x1
    if (b>99999):b=99999
    if (b < -99999): b = -99999
    # print(k,"    ",b)
    if ( x1 < 0 )or ( x2 < 0 )or ( y1 < 0 )or ( y2 < 0 )or\
            ( x1 >= len(map) )or ( x2 >= len(map) )or ( y1 >= len(map[0]) )or ( y2 >= len(map[0])):
        # print("Out of range")
        return 0
    if map[nx1][ny1]==0:
        # print("Start point is block",nx1," ",ny1)
        return 0
    if map[nx2][ny2]==0:
        # print("End point is block")
        return 0
    if x1<x2:
        x_min=x1
        x_max=x2
    else:
        x_min=x2
        x_max=x1
    if y1<y2:
        y_min=y1
        y_max=y2
    else:
        y_min=y2
        y_max=y1
    dis=0
    x_min=int(math.ceil(x_min))
    x_max = int(math.ceil(x_max))
    y_min = int(math.ceil(y_min))
    y_max = int(math.ceil(y_max))

    for i in range(x_min,x_max):
        a=int(math.floor(k*i+b))
        if a >= 0 and a < len(map[0]):
            if(map[i][a]==0):
                dis=1
            if (i-1>=0):
                if(map[i-1][a] == 0):
                    dis = 1
    for i in range(y_min,y_max):
        if (k!=0):
            a= int(math.floor((i-b)/k))
            if a>=0 and a<len(map):
                if (map[a][i] == 0):
                    dis = 1
                if (i - 1 >= 0):
                    if (map[a][i-1] == 0):
                        dis = 1

    if dis==0 :return 1
    else: return 0

# 计算方位角函数，网上刨下来的代码，输入两个点的xy信息返回角度的弧度制表示
def azimuthAngle( x1,  y1,  x2,  y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle)

class PSO(object):
    def __init__(self, population_size, step, iteration,k1,k2,iiii):
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 =0.2
        self.population_size = population_size  # 粒子群数量
        self.dim = 2  # 搜索空间的维度,每条路径中路径点的个数
        self.points = step  # 路径由几个点组成
        self.iteration = iteration  # 迭代次数
        self.x_bound = [0, 20]  # 解空间范围
        self.y_bound = [0, 20]  # 解空间范围
        self.mu=0  # 控制启发式初始化正态分布的变量，不要问我为什么没说作用，因为我忘记正态分布两个变量的名字了
        self.sigma=1  # 控制启发式初始化正态分布的变量
        self.k1=k1   # 适应度中路径长度的惩罚系数
        self.k2=k2   # 适应度中平滑度的惩罚系数
        self.k4=0.5  # 适应度中惩罚场的惩罚系数
        self.x=np.zeros((self.population_size, self.points, self.dim))# 初始化粒子群位置
        self.initx()#调用启发式初始化的程序
        self.blocks = []
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 0:
                    pd = -1
                    for k1 in range(3):
                        for k2 in range(3):
                            ppx = i + k1 - 1
                            ppy = j + k2 - 1
                            if (ppx >= 0 and ppx < len(map) and ppy >= 0 and ppy < len(map[0])):
                                if map[ppx][ppy] == 1:
                                    pd = 1
                    if pd == 1:
                        self.blocks.append([i, j])
        # print(len(self.blocks))
        self.v = np.random.rand(self.population_size, self.points,self.dim)  # 初始化粒子群速度
        for i in range(self.population_size):   #因为起点终点不能变，所有粒子的期待你终点的随机速度都是0
            self.v[i][0][0]=0
            self.v[i][0][1] = 0
            self.v[i][self.points-1][0] = 0
            self.v[i][self.points-1][1] = 0
        fitness = self.calculate_fitness(self.x)#调用计算适应度的函数，获得适应度的矩阵
        self.p = self.x  # 个体的最佳路径          我们的适应度是越小越好
        self.pg =deepcopy( self.x[np.argmin(fitness)])  # 全局最佳路径
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度
        self.per_correct = 5  # 单步最大纠正次数
        self.correct=step*self.per_correct  # 最大单路径纠正次数
        self.maxdis=0  # 粒子间最大距离
        self.blocks=[]
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j]==0:
                    self.blocks.append([i,j])

        plt.clf()
        for j in range(len(self.blocks)):
            xx = [self.blocks[j][0], self.blocks[j][0], self.blocks[j][0] + 1, self.blocks[j][0] + 1, self.blocks[j][0]]
            yy = [self.blocks[j][1], self.blocks[j][1] + 1, self.blocks[j][1] + 1,self.blocks[j][1], self.blocks[j][1]]
            plt.plot(xx, yy)
        my_x_ticks = np.arange(self.x_bound[0], self.x_bound[1], 1)
        my_y_ticks = np.arange(self.y_bound[0], self.y_bound[1], 1)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        # for j in range(len(map)):
        #     for k in range(len(map[j])):
        #         if (map[j][k] == 0):
        #             xx = [j, j, j + 1, j + 1, j]
        #             yy = [k, k + 1, k + 1, k, k]
        #             plt.plot(xx, yy)
        #


        for i in range(self.population_size): plt.plot(self.x[i, :, 0], self.x[i, :, 1])
        plt.title('This is init ')
        plt.xlim(self.x_bound[0], self.x_bound[1])
        plt.ylim(self.y_bound[0], self.y_bound[1])
        plt.pause(0.01)

    def get_w(self,i,it):  # 惯性权重相似度更新
        w_max=1.0
        w_min=0.1
        dis=0
        for j in range(self.points):
            dis=dis+(self.x[i][j][0]-self.pg[j][0])*(self.x[i][j][0]-self.pg[j][0])+(self.x[i][j][1]-self.pg[j][1])*(self.x[i][j][1]-self.pg[j][1])
        r=dis/self.maxdis
        self.w= w_min + math.sqrt((self.iteration-it)/self.iteration)*((w_max-w_min)*(1-r))

    def update_max_dis(self):
        maxdis=-1
        for i in range(self.population_size):
            for j in range(self.population_size):
                dis = 0
                if (i!=j):
                    for k in range(self.points):
                        dis=dis+(self.x[i][k][0]-self.x[j][k][0])*(self.x[i][k][0]-self.x[j][k][0])+(self.x[i][k][1]-self.x[j][k][1])*(self.x[i][k][1]-self.x[j][k][1])
                    if (dis>maxdis):
                        maxdis=dis
        self.maxdis=maxdis


    #获得一段路径，正态分布生成一个区间在【-pi，pi】的角度，弧度制，以及一段距离【0，2】（可能不精确）
    def get_a_small_route(self):
        theta = np.random.normal(self.mu, math.pi/3, 1)    #正太分布生成一个角度，弧度制
        while (theta[0] < -math.pi or theta[0] > math.pi):  #如果这个值不在[-pi,pi]的范围
            theta = np.random.normal(self.mu, self.sigma, 1)    #重新随机一个值
        tdis = np.random.normal(self.mu,1/3, 1)     #随机一个长度
        while (tdis[0] < -1 or tdis[0] > 1):    #如果太长或太短
            tdis = np.random.normal(self.mu, self.sigma, 1) #重新随机
        dis = 40/self.points + tdis[0] / 3   #新的移动距离是根据路径中的点数目而定
        return  theta[0],dis

    #计算适应度，通过距离（系数k1）加上平滑度(系数k2)吧，适应度越小越好
    def calculate_fitness(self, x):
        #路径系数
        k1=self.k1
        #平滑度系数
        k2=self.k2
        k4=self.k4
        a=0#临时变量
        fit = np.zeros(len(x))  #适应度最终结果存放在这
        d = np.zeros(self.points)   #用来存放临时计算的距离，d[0]中存放的是第零个点到第一个点的距离
        for i in range(len(x)):#枚举每条路径/粒子
            fitx=0  #从0开始计算
            for j in range(self.points-1):#枚举除了终点外所有点
                d[j]=(np.sqrt(np.square(x[i][j+1][0]-x[i][j][0])+np.square(x[i][j+1][1]-x[i][j][1])))#计算距离
                fitx=fitx+k1*d[j]#加上距离

            for j in range(1,self.points-1):#枚举除了起点终点外所有点
                a=math.sqrt((x[i][j-1][0]-x[i][j+1][0])*(x[i][j-1][0]-x[i][j+1][0])+(x[i][j-1][1]-x[i][j+1][1])*(x[i][j-1][1]-x[i][j+1][1]))
                #计算这个点的对边长度
                if (d[j]!=0) and ( d[j-1]!=0):
                    theta=(a*a-d[j]*d[j]-d[j-1]*d[j-1])/(-2*d[j]*d[j-1])#余弦定理，计算出这个点的余弦值
                    if (theta<-1):
                        theta=-1
                    if theta>1:
                        theta=1
                    u=math.acos(theta)
                    degree=180-abs(math.degrees(u))
                    # print(degree )
                    #
                    # if degree>35:
                    #     # fitx = fitx + k2 * theta  # 日常加上平滑度
                    #     p = degree*degree/100+22.75
                    #     fitx = fitx + k2 * p  # 日常加上平滑度
                    # else:
                    #     p = degree
                    #     fitx = fitx + k2 * p  # 日常加上平滑度
                    #     # fitx=fitx+k2*theta#日常加上平滑度

                    if theta>-0.81915204:
                        # fitx = fitx + k2 * theta  # 日常加上平滑度
                        p = (theta+1)*4
                        fitx = fitx + k2 * p  # 日常加上平滑度
                    else:
                        p = theta+1
                        fitx = fitx + k2 * p  # 日常加上平滑度
                        # fitx=fitx+k2*theta#日常加上平滑度
            for j in range(self.points-1):#枚举除了终点外所有点
                if(exist_way(self.x[i][j][0],self.x[i][j][1],self.x[i][j+1][0],self.x[i][j+1][1])==0):
                    fitx = fitx +punish

            for j in range(self.points - 1):  #self.x[i][self.points-1] 这个是终点
                if (exist_way(self.x[i][j][0], self.x[i][j][1], self.x[i][j + 1][0], self.x[i][j + 1][1]) == 1):
                    fitx = fitx+k4*self.punish_field(self.x[i][j][0],self.x[i][j][1],self.x[i][j+1][0],self.x[i][j+1][1])
            fit[i]=fitx

        return fit


    def punish_field11(self,x1,y1,x2,y2):
        # print(x1,y1,x2,y2)
        if x1 < x2:
            x_min = x1
            x_max = x2
        else:
            x_min = x2
            x_max = x1
        if y1 < y2:
            y_min = y1
            y_max = y2
        else:
            y_min = y2
            y_max = y1
        x_min=x_min-safe_distance-1
        x_max=x_max+safe_distance+1
        y_min = y_min - safe_distance-1
        y_max = y_max + safe_distance+1
        x_min=math.ceil(x_min)
        x_max=math.floor(x_max)
        y_min = math.ceil(y_min)
        y_max = math.floor(y_max)
        if (x_min<0):x_min=0
        if (x_max>len(map)):x_max=len(map)
        if (y_min < 0): y_min = 0
        if (y_max > len(map[0])): y_max = len(map[0])
        p_min=self.points*100
        for i in range(len(self.blocks)):
            if (self.blocks[i][0] <= x_max and self.blocks[i][0] >= x_min and self.blocks[i][1] <= y_max and
                    self.blocks[i][1] >= y_min):
                bx=self.blocks[i][0]
                by=self.blocks[i][1]
                p_min=self.shortest_distance(x1,y1,x2,y2,bx,by)
                # print(bx , ',', by, '    ', p_min)
                a=self.shortest_distance(x1,y1,x2,y2,bx+1,by)
                # print(bx+1,',',by,'    ',a)
                if p_min>a:p_min=a
                a = self.shortest_distance(x1, y1, x2, y2, bx + 1, by+1)
                # print(bx + 1, ',', by+1, '    ', a)
                if p_min > a: p_min = a

                a = self.shortest_distance(x1, y1, x2, y2, bx , by+1)
                # print(bx, ',', by+1, '    ', a)
                if p_min > a: p_min = a



        return p_min

    def punish_field(self,x1,y1,x2,y2):
        # print(x1,y1,x2,y2)
        if x1 < x2:
            x_min = x1
            x_max = x2
        else:
            x_min = x2
            x_max = x1
        if y1 < y2:
            y_min = y1
            y_max = y2
        else:
            y_min = y2
            y_max = y1
        x_min=x_min-safe_distance-1
        x_max=x_max+safe_distance+1
        y_min = y_min - safe_distance-1
        y_max = y_max + safe_distance+1
        x_min=math.ceil(x_min)
        x_max=math.floor(x_max)
        y_min = math.ceil(y_min)
        y_max = math.floor(y_max)
        if (x_min<0):x_min=0
        if (x_max>len(map)):x_max=len(map)
        if (y_min < 0): y_min = 0
        if (y_max > len(map[0])): y_max = len(map[0])

        punish_number=0
        for i in range(len(self.blocks)):
            if (self.blocks[i][0] <= x_max and self.blocks[i][0] >= x_min and self.blocks[i][1] <= y_max and
                    self.blocks[i][1] >= y_min):
                bx=self.blocks[i][0]
                by=self.blocks[i][1]
                p_min=self.shortest_distance(x1,y1,x2,y2,bx,by)
                # print(bx , ',', by, '    ', p_min)
                a=self.shortest_distance(x1,y1,x2,y2,bx+1,by)
                # print(bx+1,',',by,'    ',a)
                if p_min>a:p_min=a
                a = self.shortest_distance(x1, y1, x2, y2, bx + 1, by+1)
                # print(bx + 1, ',', by+1, '    ', a)
                if p_min > a: p_min = a

                a = self.shortest_distance(x1, y1, x2, y2, bx , by+1)
                # print(bx, ',', by+1, '    ', a)
                if p_min > a: p_min = a
                # print(bx,'  ',by,'    ',p_min)
                if (p_min<safe_distance):
                    punish_number=punish_number+1/(p_min*p_min)
                    # print('[',self.blocks[i][0],',',self.blocks[i][1],']is close to [',x1,',',y1,'],[',x2,',',y2,']  ',1/(p_min*p_min))


        return punish_number

    def shortest_distance(self,x1,y1,x2,y2,bx,by):
        k = 0
        # print('==============')
        # print('from ',x1,' ',y1,' to ',x2,' ',y2,' point is  ',bx,' ',by)
        if x1 == x2:
            k = 0
        else:
            k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        # print('k and b ',k,'  ',b)
        if x1 < x2:
            x_min = x1
            x_max = x2
        else:
            x_min = x2
            x_max = x1
        # print('(',x_min,',',x_max,')')
        if k==0:
            if (bx<=x_max and bx>=x_min):
                return by-b
            else:
                a1=((bx-x1)*(bx-x1)+(by-y1)*(by-y1))
                a2= ((bx - x2) * (bx - x2) + (by - y2) * (by - y2))
                a_min=a1
                if a_min>a2:a_min=a2
                return math.sqrt(a_min)
        else:
            k1=-1/k
            b1=by - k1 * bx
            fx=(b1-b)/(k-k1)
            fy=k1*fx+b1
            if (fx <= x_max and fx >= x_min):
                # print('in   ',fx,' ',fy,'  '  ,math.sqrt((bx-fx)*(bx-fx)+(by-fy)*(by-fy)))
                return math.sqrt((bx-fx)*(bx-fx)+(by-fy)*(by-fy))
            else:
                # print('out')
                a1=(bx-x1)*(bx-x1)+(by-y1)*(by-y1)
                a2 = (bx - x2) * (bx - x2) + (by - y2) * (by - y2)
                if a1>a2:
                    return math.sqrt(a2)
                else:
                    return math.sqrt(a1)
    #迭代程序
    def evolve(self, name):
        # f1 = open("E:/创新实验/code/GA-PSO/pictures/"+name+".txt","w")
        fig = plt.figure()
        acnumber=0
        for step in range(self.iteration):
            self.update_max_dis()
            #速度计算，这里还没有实现线性递减的惯性权重啊，或者是相似度的惯性权重，最简单的固定值
            for i in range(self.population_size):
                self.get_w(i,step)  # 粒子惯性权重相似度更新
                ac=10086
                while (ac> 0):
                    ra1 = np.random.rand(self.points, self.dim)
                    ra2 = np.random.rand(self.points, self.dim)
                    ra1[0][0] = 0
                    ra1[0][1] = 0
                    ra1[self.points - 1][0] = 0
                    ra1[self.points - 1][1] = 0
                    ra2[0][0] = 0
                    ra2[0][1] = 0
                    ra2[self.points - 1][0] = 0
                    ra2[self.points - 1][1] = 0
                    nv= self.w * self.v[i] + self.c1 * ra1 * (self.p[i] - self.x[i]) + self.c2 * ra2 * (self.pg - self.x[i])
                    nx=nv+self.x[i]
                    if allow_punish==1:
                        self.v[i] = nv
                        self.x[i] = nx
                        for j in range(self.points):
                            if (self.x[i][j][0]<0):
                                self.x[i][j][0]=0
                            if (self.x[i][j][0]>len(map[0])):
                                self.x[i][j][0]=len(map[0])
                            if (self.x[i][j][1] < 0):
                                self.x[i][j][1] = 0
                            if (self.x[i][j][1] > len(map)):
                                self.x[i][j][1] = len(map)
                            if (self.x[i][j][0] < 0):
                                self.x[i][j][0] = 0
                            if (self.x[i][j][0] > len(map[0])):
                                self.x[i][j][0] = len(map[0])
                            if (self.x[i][j][1] < 0):
                                self.x[i][j][1] = 0
                            if (self.x[i][j][1] > len(map)):
                                self.x[i][j][1] = len(map)
                        break;
                    else:
                        #=================================================画图分界线===============
                        for j in range(len(map)):
                            for k in range(len(map[j])):
                                if (map[j][k] == 0):
                                    xx = [j, j, j + 1, j + 1, j]
                                    yy = [k, k + 1, k + 1, k, k]
                                    plt.plot(xx, yy)
                        plt.plot(nx[:, 0],nx[:, 1],'m-d')
                        plt.plot(nv[:, 0], nv[:, 1], 'b.')
                        plt.plot(self.x[i, :, 0], self.x[i, :, 1], 'b.')
                        plt.plot(self.pg[:, 0], self.pg[:, 1],'k--')
                        plt.plot(self.p[i, :, 0], self.p[i, :, 1],'k--')
                        plt.title('This is ' + str(step) + 'iteration ' + str(i)+ ' partical ')
                        plt.xlim(self.x_bound[0], self.x_bound[1])
                        plt.ylim(self.y_bound[0], self.y_bound[1])
                        my_x_ticks = np.arange(self.x_bound[0], self.x_bound[1], 1)
                        my_y_ticks = np.arange(self.y_bound[0], self.y_bound[1], 1)
                        plt.xticks(my_x_ticks)
                        plt.yticks(my_y_ticks)
                        plt.pause(0.1)
                        # =================================================画图分界线===============

                        corr=0
                        for j in range(1,self.points):  # 对每一小段路径进行判断是不是可行
                            per_count=0
                            while (exist_way(nx[j][0], nx[j][1], nx[j - 1][0], nx[j - 1][1]) == 0):  # 等于0是不可行，那就不用了
                                if (corr>self.correct):
                                    break
                                corr = corr + 1
                                per_count = per_count + 1
                                if(per_count>3* self.per_correct ):
                                    break
                                ra1 = np.random.rand(self.dim)
                                ra2 = np.random.rand(self.dim)
                                nv[j] = self.w * self.v[i][j] + self.c1 * ra1 * (self.p[i][j] - self.x[i][j]) + self.c2 * ra2 * (
                                            self.pg[j] - self.x[i][j])
                                nx[j] = nv[j] + self.x[i][j]
                                # =================================================画图分界线===============
                                for j1 in range(len(map)):
                                    for k in range(len(map[j])):
                                        if (map[j1][k] == 0):
                                            xx = [j1, j1, j1 + 1, j1 + 1, j1]
                                            yy = [k, k + 1, k + 1, k, k]
                                            plt.plot(xx, yy)
                                plt.plot(nx[:, 0], nx[:, 1], 'm-d')
                                plt.plot(nv[:, 0], nv[:, 1], 'b.')
                                plt.plot(self.x[i, :, 0], self.x[i, :, 1], 'b.')
                                plt.plot(self.pg[:, 0], self.pg[:, 1], 'k--')
                                plt.plot(self.p[i, :, 0], self.p[i, :, 1], 'k--')
                                plt.title('This is ' + str(step) + 'iteration ' + str(i) + ' partical '+str(corr))
                                plt.xlim(self.x_bound[0], self.x_bound[1])
                                plt.ylim(self.y_bound[0], self.y_bound[1])
                                my_x_ticks = np.arange(self.x_bound[0], self.x_bound[1], 1)
                                my_y_ticks = np.arange(self.y_bound[0], self.y_bound[1], 1)
                                plt.xticks(my_x_ticks)
                                plt.yticks(my_y_ticks)
                                plt.pause(0.1)
                                # =================================================画图分界线===============
                            if (per_count > 3* self.per_correct):
                                break

                        ac=0
                        for j in range(self.points - 1):  # 对每一小段路径进行判断是不是可行
                            if (exist_way(nx[j][0], nx[j][1], nx[j + 1][0], nx[j + 1][1]) == 0):  # 等于0是不可行，那就不用了
                                ac = ac+1  # 不可行
                        print('step ', step, ' partical  ', i, ' error times ', ac)
                        if (ac==0):
                            self.v[i]=nv

            #画图
            plt.clf()
            for j in range(len(map)):
                for k in range(len(map[j])):
                    if (map[j][k] == 0):
                        xx = [j, j, j + 1, j + 1, j]
                        yy = [k, k + 1, k + 1, k, k]
                        plt.plot(xx, yy)
            for i in range(self.population_size): plt.plot(self.x[i,:,0],self.x[i,:,1])
            plt.title('This is '+str(step)+' iteration  ')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.y_bound[0], self.y_bound[1])
            my_x_ticks = np.arange(self.x_bound[0], self.x_bound[1], 1)
            my_y_ticks = np.arange(self.y_bound[0], self.y_bound[1], 1)
            plt.xticks(my_x_ticks)
            plt.yticks(my_y_ticks)
            plt.pause(0.01)

            #画图结束
            #重新计算适应度
            fitness = self.calculate_fitness(self.x)

            if name == "ga" or name == "ga_smooth":
                # --------- 遗传算法 -----------#
                tmp_x = deepcopy(self.x)
                # 交叉互换
                for i in range(int(len(self.x) * genetic_percent)):
                    p1 = np.random.randint(0, self.population_size)  # 随机选取两个粒子
                    p2 = np.random.randint(0, self.population_size)
                    rand_point = np.random.randint(1, self.points-2)
                    tmp = tmp_x[p1][rand_point]
                    tmp_x[p1][rand_point] = tmp_x[p2][rand_point]
                    tmp_x[p2][rand_point] = tmp
                tmp_fitness = self.calculate_fitness(tmp_x)
                for i in range(len(tmp_fitness)):
                    if tmp_fitness[i] < fitness[i]:
                        self.x[i] = tmp_x[i]

                # 变异
                tmp_x = deepcopy(self.x)
                # 变异
                for i in range(int(len(self.x))):
                    rand_point = np.random.randint(1, self.points-1) # 变化路径中的一个点
                    tmp_x[i][rand_point][0] = (tmp_x[i][rand_point-1][0] + tmp_x[i][rand_point+1][0])/2
                    tmp_x[i][rand_point][1] = (tmp_x[i][rand_point-1][1] + tmp_x[i][rand_point+1][1])/2
                tmp_fitness = self.calculate_fitness(tmp_x)
                for i in range(len(tmp_fitness)):
                    if tmp_fitness[i] < fitness[i]:
                        self.x[i] = tmp_x[i]

                tmp_x = deepcopy(self.x)
                for i in range(int(len(self.x))):
                    x_rand = -1 + 2*np.random.rand()
                    y_rand = -1 + 2*np.random.rand()
                    rand_point = np.random.randint(1, self.points-1) # 变化路径中的一个点
                    tmp_x[i][rand_point][0] += x_rand
                    tmp_x[i][rand_point][1] += y_rand
                tmp_fitness = self.calculate_fitness(tmp_x)
                for i in range(len(tmp_fitness)):
                    if tmp_fitness[i] < fitness[i]:
                        self.x[i] = tmp_x[i]

                # 替换适应度最大的粒子
                tmp_x = deepcopy(self.x)
                if step < self.iteration / 2:
                    i = 0 # 适应度值最大的粒子
                    for t in range(len(fitness)):
                        if fitness[t] > fitness[i]:
                            i = t
                    j = 0  # 从0开始推算路径
                    # point-1的位置存的是终点
                    while (j <= self.points - 3):
                        o_theta = azimuthAngle(self.x[i][j][0], self.x[i][j][1], end_point[0],
                                               end_point[1])  # 求出弧度制的当前点到终点的方向
                        t_theta, dis = self.get_a_small_route()  # 获得一个弧度制角度与就离
                        # 以x[i][j]第i条路径第j个点为当前点，如果到新的落点不存在一条路径就重新生成一个落点
                        while (exist_way(self.x[i][j][0], self.x[i][j][1],
                                         self.x[i][j][0] + dis * math.cos(t_theta + o_theta),
                                         self.x[i][j][1] + dis * math.sin(t_theta + o_theta)) == 0):
                            t_theta, dis = self.get_a_small_route()  # 每个落点由新的角度距离决定
                        # while结束后新的落点必定是可以到达的
                        self.x[i][j + 1][0] = self.x[i][j][0] + dis * math.cos(t_theta + o_theta)  # 存进下一个点x
                        self.x[i][j + 1][1] = self.x[i][j][1] + dis * math.sin(t_theta + o_theta)  # 存进下一个点y
                        j = j + 1  # j永远代表着当前的落点位置
                        # 为什么上面while退出条件是-3？因为j=point-3的时候必定会触发下面这句j=point-2(因为j++)
                        if (j == self.points - 2):  # 如果当前点是终点前一个点就需要判断了，因为终点是固定的
                            if (exist_way(end_point[0], end_point[1], self.x[i][j][0], self.x[i][j][1]) == 0):
                                j = 0
                                # 如果当前点到终点不存在一条直线，那么j=0一切重来
                            else:  # 如果存在路径，将重点放进路径末尾
                                #  print(end_point[0],'  ', end_point[1],'  ', self.x[i][j][0],'  ' ,self.x[i][j][1],'  ',exist_way(end_point[0], end_point[1], self.x[i][j][0] ,self.x[i][j][1] ))
                                self.x[i][j + 1][0] = end_point[0]
                                self.x[i][j + 1][1] = end_point[1]
                                break
                # --------- 遗传算法 -----------#

            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            for i in range(self.population_size):
                if fitness[i]<self.individual_best_fitness[i]:
                    self.individual_best_fitness[i]=fitness[i]
                    self.p[i]=self.x[i]
            if np.min(fitness) < self.global_best_fitness:#寻找最小适应度的是不是更新了全局
                self.pg = deepcopy(self.x[np.argmin(fitness)])
                self.global_best_fitness = deepcopy(np.min(fitness))

            print(step, "  当前平均适应度",np.mean(fitness)," 最优适应度  ",self.global_best_fitness)
            # f1.write(str(self.global_best_fitness)+"\n")
        # f1.close()
    #初始化生成粒子x的部分
    def initx(self):
        for i in range(self.population_size):#枚举所有x
            j=0#从0开始推算路径
            #当point=30时，point-1的位置存的是终点
            while(j<=self.points-3):

                o_theta=azimuthAngle(self.x[i][j][0],self.x[i][j][1],end_point[0],end_point[1])#求出弧度制的当前点到终点的方向
                t_theta,dis=self.get_a_small_route()#获得一个弧度制角度与就离
                #以x[i][j]第i条路径第j个点为当前点，如果到新的落点不存在一条路径就重新生成一个落点
                while(exist_way(self.x[i][j][0],self.x[i][j][1],
                                self.x[i][j][0]+dis*math.cos(t_theta+o_theta),self.x[i][j][1]+dis*math.sin(t_theta+o_theta))==0):
                    t_theta, dis = self.get_a_small_route()#每个落点由新的角度距离决定
                #while结束后新的落点必定是可以到达的
                self.x[i][j+1][0] =  self.x[i][j][0] + dis * math.cos(t_theta + o_theta)#存进下一个点x
                self.x[i][j + 1][1] = self.x[i][j][1] + dis * math.sin(t_theta + o_theta)#存进下一个点y
                j=j+1#j永远代表着当前的落点位置
                # 为什么上面while退出条件是-3？因为j=point-3的时候必定会触发下面这句j=point-2(因为j++)
                if (j==self.points-2):#如果当前点是终点前一个点就需要判断了，因为终点是固定的
                    if(exist_way(end_point[0], end_point[1], self.x[i][j][0] ,self.x[i][j][1] ) == 0):
                        j=0
                        #如果当前点到终点不存在一条直线，那么j=0一切重来
                    else:#如果存在路径，将重点放进路径末尾
                     #  print(end_point[0],'  ', end_point[1],'  ', self.x[i][j][0],'  ' ,self.x[i][j][1],'  ',exist_way(end_point[0], end_point[1], self.x[i][j][0] ,self.x[i][j][1] ))
                        self.x[i][j + 1][0] = end_point[0]
                        self.x[i][j + 1][1] = end_point[1]
                        break
        return 0

    def judge(self,x):
        dis=0
        d = np.zeros(self.points)  # 用来存放临时计算的距离，d[0]中存放的是第零个点到第一个点的距离
        for j in range(self.points - 1):  # 枚举除了终点外所有点
            d[j] =(np.sqrt(np.square(x[j + 1][0] - x[j][0]) + np.square(x[j + 1][1] - x[j][1])))
            dis= dis+  d[j] # 计算距离
        print("距离  ",dis)
        danger_point=0
        for j in range(self.points - 1):  # self.x[i][self.points-1] 这个是终点
            a1 = float(x[j][0])
            a2 = float(x[j][1])
            a3 = float(x[j+1][0])
            a4 = float(x[j+1][1])
            if self.punish_field(a1, a2,a3, a4)!=0:
                danger_point=danger_point+1
        print("危险路径比例  ",danger_point*100/(self.points-1),"%",danger_point)
        max_angle=0
        total_angle=0
        for j in range(1, self.points - 1):  # 枚举除了起点终点外所有点
            a = math.sqrt((x[j - 1][0] - x[j + 1][0]) * (x[j - 1][0] - x[j + 1][0]) + (
                        x[j - 1][1] - x[j + 1][1]) * (x[j - 1][1] - x[j + 1][1]))
            # 计算这个点的对边长度
            if (d[j] != 0) and (d[j - 1] != 0):
                theta = (a * a - d[j] * d[j] - d[j - 1] * d[j - 1]) / (-2 * d[j] * d[j - 1])  # 余弦定理，计算出这个点的余弦值
                if (theta < -1):
                    theta = -1
                if theta > 1:
                    theta = 1
                u = math.acos(theta)
                degree = 180-abs(math.degrees(u))
                if (degree>max_angle):
                    max_angle=degree
                total_angle=total_angle+degree
        print("平均转角度数  ",total_angle/(self.points-2))
        print("最大转角度数  ",  max_angle)
        total_dis=0
        count=0
        for j in range(self.points - 1):  # self.x[i][self.points-1] 这个是终点
            a1 = float(x[j][0])
            a2 = float(x[j][1])
            a3 = float(x[j + 1][0])
            a4 = float(x[j + 1][1])
            d=self.punish_field11(a1, a2, a3, a4)
            if (d<safe_distance):
                count=count+1
                total_dis=total_dis+d
            avg1=0
            if count>0:
                avg1=total_dis/count
        print("平均危险距离为  ",avg1,"  ",total_dis,"  ",count)
start_time=time.time()
#地图信息，0为不能走，1为可以走，后面加上势场之后就将1换成别的数值
map1=[
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
[1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
[1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,],
[0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,],
[0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,],
[1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,],
[1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,],
[1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
[1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,],
[1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,],
[1,1,1,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,],
[1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,],
[1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,],
[1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,],
[1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,],
[1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1,],
[1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,1,1,0,0,1,],
[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
]

#起点
start_point=[0,0]
#终点
end_point=[19,19]
# 程序里使用到的地图全部定向到这个map上
map=map1
#粒子的数量
number_of_particle=70
#每条路径有多少个点
step_per_route=35
allow_punish=1      #允许惩罚开关
punish=1000  #惩罚值
#迭代次数
iteration=100
safe_distance=0.5
# 遗传算法比例
genetic_percent = 0.3
plt.clf()
for j in range(len(map)):
    for k in range(len(map[j])):
        if (map[j][k] == 0):
            xx = [j, j, j + 1, j + 1, j]
            yy = [k, k + 1, k + 1, k, k]
            plt.plot(xx, yy)
x1=[19,16.78980877158566 ]
y1=[19, 8.39007763126324]
#print(exist_way(x1[0],y1[0],x1[1],y1[1]))
plt.plot(x1,y1)
pso = PSO(number_of_particle, step_per_route,iteration,1,1,0)#初始化
pso.evolve("ga_smooth")#开始迭代

#print(pso.global_best_fitness)
plt.show()
end_time=time.time()
print("using time = ",end_time-start_time)
print("avg time = ",(end_time-start_time)/number_of_particle)
print("最终最优适应度：",pso.global_best_fitness)
pso.judge(pso.pg)