import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import math
import time  # 引入time模块
import sys as sys


# 判断是否存在路径，存在返回1，不存在返回0
def exist_way(x1,y1,x2,y2):  # 地图理解为左下角为原点

    nx1=int(math.floor(x1))
    nx2 = int(math.floor(x2))
    ny1 = int(math.floor(y1))
    ny2 = int(math.floor(y2))
    k=(y2-y1)/(x2-x1)
    b=y1-k*x1
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
    x_min=0
    x_max = 0
    y_min = 0
    y_max = 0
    if nx1<nx2:
        x_min=nx1
        x_max=nx2
    else:
        x_min=nx2
        x_max=nx1
    if ny1<ny2:
        y_min=ny1
        y_max=ny2
    else:
        y_min=ny2
        y_max=ny1
    # print(x_min,"  ",x_max)
    # print(y_min, "  ", y_max)
    dis=0

    for i in range(x_min,x_max+1):

        a=int(math.floor(k*i+b))
        # print("1  ",i, "   ", a)
        if a >= 0 and a < len(map[0]):
            if(map[i][a]==0):
                dis=1

    for i in range(y_min,y_max+1):

        a= int(math.floor((i-b)/k))
        # print("2  ", a, "   ", i)
        if a>=0 and a<len(map):
            if (map[a][i] == 0):
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
    def __init__(self, population_size, step, iteration):
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = 2  # 搜索空间的维度,每条路径中路径点的个数
        self.points = step  # 路径由几个点组成
        self.iteration = iteration  # 迭代次数
        self.x_bound = [0, 20]  # 解空间范围
        self.y_bound = [0, 20]  # 解空间范围
        self.mu=0  # 控制启发式初始化正态分布的变量，不要问我为什么没说作用，因为我忘记正态分布两个变量的名字了
        self.sigma=1  # 控制启发式初始化正态分布的变量
        self.k1=1   # 适应度中路径长度的惩罚系数
        self.k2=1   # 适应度中平滑度的惩罚系数
        self.x=np.zeros((self.population_size, self.points, self.dim))# 初始化粒子群位置
        self.initx()#调用启发式初始化的程序
        self.v = np.random.rand(self.population_size, self.points,self.dim)  # 初始化粒子群速度
        for i in range(self.population_size):   #因为起点终点不能变，所有粒子的期待你终点的随机速度都是0
            self.v[i][0][0]=0
            self.v[i][0][1] = 0
            self.v[i][self.points-1][0] = 0
            self.v[i][self.points-1][1] = 0
        fitness = self.calculate_fitness(self.x)#调用计算适应度的函数，获得适应度的矩阵
        self.p = self.x  # 个体的最佳路径          我们的适应度是越小越好
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳路径
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    #获得一段路径，正态分布生成一个区间在【-pi，pi】的角度，弧度制，以及一段距离【0，2】（可能不精确）
    def get_a_small_route(self):
        theta = np.random.normal(self.mu, self.sigma, 1)    #正太分布生成一个角度，弧度制
        while (theta[0] < -math.pi or theta[0] > math.pi):  #如果这个值不在[-pi,pi]的范围
            theta = np.random.normal(self.mu, self.sigma, 1)    #重新随机一个值
        tdis = np.random.normal(self.mu, self.sigma, 1)     #随机一个长度
        while (tdis[0] < -1 or tdis[0] > 1):    #如果太长或太短
            tdis = np.random.normal(self.mu, self.sigma, 1) #重新随机
        dis = 1 + tdis[0] / 3   #新的移动距离是1加上三分之一随机，即距离的范围是【0.67777，1.333333】
        return  theta[0],dis

    #计算适应度，通过距离（系数k1）加上平滑度(系数k2)吧，适应度越小越好
    def calculate_fitness(self, x):
        #路径系数
        k1=self.k1
        #平滑度系数
        k2=self.k2
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
                theta=(a*a-d[j]*d[j]-d[j-1]*d[j-1])/(-2*d[j]*d[j-1])#余弦定理，计算出这个点的余弦值
                fitx=fitx+k2*theta#日常加上平滑度
            fit[i]=fitx

        return fit

    #迭代程序
    def evolve(self):
        fig = plt.figure()
        acnumber=0
        for step in range(self.iteration):
            #获得两个随机值r
            r1 = np.random.rand(self.population_size, self.points,self.dim)
            r2 = np.random.rand(self.population_size, self.points,self.dim)
            #不管怎样，起点终点永远不会变，所以起点终点的随机值是0，不接受起点终点的移动
            for i in range(self.population_size):
                r1[i][0][0]=0
                r1[i][0][1] = 0
                r1[i][self.points-1][0] = 0
                r1[i][self.points-1][1] = 0
                r2[i][0][0] = 0
                r2[i][0][1] = 0
                r2[i][self.points - 1][0] = 0
                r2[i][self.points - 1][1] = 0
            #速度计算，这里还没有实现线性递减的惯性权重啊，或者是相似度的惯性权重，最简单的固定值
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            #位置更新
            k = self.v + self.x#临时变量存下新的位置
            for i in range(self.population_size):#对每个粒子的新的位置进行判断
                ac=1#默认新的位置是可以用的
                for j in range(self.points-1):#对每一小段路径进行判断是不是可行
                    if(exist_way(k[i][j][0],k[i][j][1],k[i][j+1][0],k[i][j+1][1])==0):#等于0是不可行，那就不用了
                        ac=0#不可行
                        break#退出for循环

                if ac==1:#如果可行
                    self.x[i]=k[i]#x的新位置更新
                    acnumber=acnumber+1#一个可行解的计数器
            #画图
            plt.clf()
            #你看看这里你会不会改让画图能画出路径
            plt.plot(self.pg[:, 0],self.pg[:, 1])
            # for i in range(number_of_particle):
            #     aaa = []
            #     bbb = []
            #     for j in range(self.points):
            #         aaa.append(self.x[i][j][0])
            #         bbb.append(self.x[i][j][1])
            #     plt.plot(aaa,bbb)#这句话有问题，他这里是二维的x，但是我改了后变成三维的x了，我不知道这里要怎么改！！！！！
            plt.title('This is '+str(step)+'iteration')
            plt.xlim(self.x_bound[0], self.x_bound[1])
            plt.ylim(self.y_bound[0], self.y_bound[1])
            plt.pause(0.01)
            #画图结束
            #重新计算适应度
            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体，只有适应度变高了的粒子的id才会返回进update——id
            update_id = np.greater(self.individual_best_fitness, fitness)
            #更新粒子的历史最优
            self.p[update_id] = self.x[update_id]#存下更小适应度的路径
            self.individual_best_fitness[update_id] = fitness[update_id]#存下更小适应度的适应度……
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:#寻找最小适应度的是不是更新了全局
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            # print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))
            print("当前平均适应度",np.mean(fitness)," 最优适应度  ",self.global_best_fitness )
        print("新的路径可行的有 ",acnumber,"条")
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
                        self.x[i][j+1][0]=end_point[0]
                        self.x[i][j+1][1]  =end_point[1]
        return 0



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
number_of_particle=100
#每条路径有多少个点
step_per_route=35
#迭代次数
iteration=100
pso = PSO(number_of_particle, step_per_route,iteration)#初始化
pso.evolve()#开始迭代
plt.show()
end_time=time.time()
print("using time = ",end_time-start_time)
print("avg time = ",(end_time-start_time)/number_of_particle)
print(pso.global_best_fitness)