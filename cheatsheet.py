import datetime
print(datetime.datetime.now())
# 速度について
# 普通のinput()よりも高速。sys.stdin.readline()
import sys
input = sys.stdin.readline
N = int(input())
A = [int(input()) for _ in range(N)]

n,m=map(int,input().split())
a=list(map(int,input().split()))
ab=[list(map(int,input().split())) for _ in range(n)]

from random import randint,shuffle


"""速度早いテンプレ

from numba import njit
import numpy as np
@njit
def main():
  pass
if __name__=='__main__':
    main()



# 再帰回数の上限を引き上げる
# 文字列をinput()した場合、末尾に改行が入るので注意
# 入力データ数が 10^6 の場合だと、0.3~0.4 sec の差がでる
import sys
sys.setrecursionlimit(10**7)
import sys
input = sys.stdin.readline
# 文字列をinput()した場合、末尾に改行が入るので注意
def main():
  pass
if __name__=='__main__':
  #import datetime
  #print(datetime.datetime.now())
  main()
  #print(datetime.datetime.now())

"""
from heapq import heappop,heappush,heapify

# 二次元配列のkeyに使う。lambdaの代わり
import sys
input = sys.stdin.readline
# sort, itemgetter
from operator import itemgetter
A.sort(key=itemgetter(1))

# べき乗
# 速い
pow(m,n,mod)
# 遅い
m**n%mod
# リストのfor文
# 速い
for a in A:
    pass
# 遅い
for i in range(len(A)):
    pass
# リストの追加
# 内包表記が一番早く、appendが一番遅い
# append()
A = []
for i in range(N):
    A.append(i)
# A[i] = i, 代入
A = [None] * N
for i in range(N):
    A[i] = i
# [i for i in range(N)], 内包表記
A = [i for i in range(N)]

# リストの初期化
# 速い
[None] * N
# 遅い
[None for _ in range(N)]
# 速い
[[None] * N for _ in range(N)]
# 遅い
[[None for _ in range(N)] for _ in range(N)]
# NG
[[None] * N] * N

def main():
    """"ここに今までのコード"""
if __name__ == '__main__':
    main()
# forはwhileより速い
# ④enumerate
# for e in a: という書き方が速くなるとありましたが、これの補足としてindexと要素を知りたいときはenumerateを使うといいです。
# 正規表現による置換
import re
text = "子供用Bagや女性用バックなど様々なバッグを取り揃えています"
text_mod = re.sub('バック|バッグ',"Bag",text)
print (text_mod)

#約分
def make_divisors(n):  
  divisors = []
  for i in range(1, int(n**0.5)+1):
    if n % i == 0:
      divisors.append(i)
      if i != n // i:
        divisors.append(n//i)
  # divisors.sort()
  return divisors

# numpyを使った約分
def calc_div(N):
  sq = int(N**.5 + 10)
  x = np.arange(1, sq)
  x = x[N % x == 0]
  x = np.concatenate((x, N // x))
  return np.unique(x)



#nの素数判定
def is_prime(n):
    if n == 1:
        return False
    for i in range(2,int(n**0.5)+1):
        if n % i == 0:
            return False
    return True
# アルファベット(a〜z)→数値(1〜26)小文字限定
a2n = lambda c: ord(c) - ord('a') + 1
# 数値(1〜26)→アルファベット(a〜z)
n2a = lambda c: chr(c+64).lower()
#二分探索
import bisect



# Binary Indexed Tree:BIT木。累積和を高速で更新する。平衡二分木の代わりにもなる。順序を保ったままO(logN)で挿入と取り出しが可能
# https://qiita.com/Salmonize/items/638da118cd621d2628d1?utm_campaign=popular_items&utm_medium=feed&utm_source=popular_items

# Binary Indexed Tree (Fenwick Tree)
# 1-indexed
class BIT:
  def __init__(self, n):
    self.n = n
    self.data = [0]*(n+1)
    self.el = [0]*(n+1)
  # sum(ary[:i])
  def sum(self, i):
    s = 0
    while i > 0:
      s += self.data[i]
      i -= i & -i
    return s
  # ary[i]+=x
  def add(self, i, x):
    # assert i > 0
    self.el[i] += x
    while i <= self.n:
      self.data[i] += x
      i += i & -i
  # sum(ary[i:j])
  def get(self, i, j=None):
    if j is None:
      return self.el[i]
    return self.sum(j) - self.sum(i)

# 区間加算可能なBIT。内部的に1-indexed BITを使う
class BIT_Range():
  def __init__(self,n):
    self.n=n
    self.bit0=BIT(n+1)
    self.bit1=BIT(n+1)
  # for i in range(l,r):ary[i]+=x
  def add(self,l,r,x):
    l+=1
    self.bit0.add(l,-x*(l-1))
    self.bit0.add(r+1,x*r)
    self.bit1.add(l,x)
    self.bit1.add(r+1,-x)
  # sum(ary[:i])
  def sum(self,i):
    if i==0:return 0
    #i-=1
    return self.bit0.sum(i)+self.bit1.sum(i)*i
  # ary[i]
  def get(self,i):
    return self.sum(i+1)-self.sum(i)
  # sum(ary[i:j])
  def get_range(self,i,j):
    return self.sum(j)-self.sum(i)


# 0-indexed binary indexed tree
class BIT:
  def __init__(self, n):
    self.n = n
    self.data = [0]*(n+1)
    self.el = [0]*(n+1)
  # sum of [0,i) sum(a[:i])
  def sum(self, i):
    if i==0:return 0
    s = 0
    while i > 0:
      s += self.data[i]
      i -= i & -i
    return s
  def add(self, i, x):
    i+=1
    self.el[i] += x
    while i <= self.n:
      self.data[i] += x
      i += i & -i
  # sum of [l,r)   sum(a[l:r])
  def sumlr(self, i, j):
    return self.sum(j) - self.sum(i)
  # a[i]
  def get(self,i):
    i+=1
    return self.el[i]


# UnionFind
class UnionFind:
  def __init__(self,n):
    self.n=n
    self.par=[-1]*n # par[i]:i根ならグループiの要素数に-1をかけたもの。i根じゃないならiの親
    self.rank=[0]*n
  
  # iの根を返す
  def find(self,i):
    if self.par[i]<0:return i
    ii=i
    while self.par[i]>=0:
      i=self.par[i]
    while i!=self.par[ii]:
      ii,self.par[ii]=self.par[ii],i
    return i

  # iとjをunionし、根頂点を返す
  def union(self,i,j):
    i,j=self.find(i),self.find(j)
    if i==j:return i
    elif self.rank[i]>self.rank[j]: # par[i]:グループiの要素数で判断してもいい
      self.par[i]+=self.par[j]
      self.par[j]=i
    else:
      self.par[j]+=self.par[i]
      self.par[i]=j
      # 深さ(rank)が同じものを併合した場合1を足す
      if self.rank[i]==self.rank[j]:
        self.rank[j]+=1
    return self.find(i)

  # iとjが同じグループに属するか判断
  def same(self,i,j):
    return self.find(i)==self.find(j)

  # ノードiが属する木のサイズを返す
  def size(self,i):
    return -self.par[self.find(i)]


# 角度を求める
# ラジアンから度に変換するのがmath.degrees()で、度からラジアンに変換するのがmath.radians()。
import math
print(math.degrees(math.atan2(0,1)))
print(math.degrees(math.atan2(1,1)))
print(math.degrees(math.atan2(1,0)))

# https://img.atcoder.jp/abc061/editorial.pdf
# 上のD問題
# BellmanFord
# ベルマンフォード法
# edges:エッジ、有向エッジ[a,b,c]a->bのエッジでコストc
# num_v:頂点の数
# source:始点
def BellmanFord(edges,num_v,source):
  #グラフの初期化
  inf=float("inf")
  dist=[inf for i in range(num_v)]
  dist[source]=0  
  #辺の緩和をnum_v-1回繰り返す。num_v回目に辺の緩和があればそれは閉路。-1を返す。
  for i in range(num_v-1):
    for edge in edges:
      if dist[edge[0]] != inf and dist[edge[1]] > dist[edge[0]] + edge[2]:
        dist[edge[1]] = dist[edge[0]] + edge[2]
        if i==num_v-1: return -1
  #閉路に含まれる頂点を探す。
  negative=[False]*n
  for i in range(num_v):
    for edge in edges:
      if negative[edge[0]]:negative[edge[1]]=True
      if dist[edge[0]] != inf and dist[edge[1]] > dist[edge[0]] + edge[2]:
        negative[edge[1]] = True
  return dist[n-1],negative[n-1]


# ワーシャルフロイド法
# 全頂点間の最短距離を計算
# O(N^3)

def warshall_floyd(d):
  #d[i][j]: iからjへの最短距離
  for k in range(n):
    for i in range(n):
      for j in range(n):
        d[i][j] = min(d[i][j],d[i][k] + d[k][j])
  return d
inf=float('inf')
d=[[inf]*n for _ in range(n)]
for i in range(n):
  d[i][i]=0
for a,b,l in abl:
  d[a][b]=min(d[a][b],l)
d=warshall_floyd(d)

# ワーシャルフロイド法  n:頂点数
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
d=[[0]*n for _ in range(n)]
for a,b,c in abc:
    a,b=a-1,b-1
    d[a][b]=c
    d[b][a]=c
d=np.array(floyd_warshall(csgraph=d, directed=False, return_predecessors=False))
print(d)

# ワーシャルフロイド法で作った全頂点最短距離テーブルdについて、xy間に距離zの辺を追加した時の更新
for x,y,z in xyz:
  x,y=x-1,y-1
  # np.resize(d[:,x],(n,n)) 全頂点からxへの最短距離のベクトルを行列にリサイズ
  # np.resize(d[y,:],(n,n)).T 全頂点からyへの最短距離のベクトルを行列にリサイズ
  # 上二つの距離にzを足し、任意の頂点間i,jについてi->x->y->jの距離を計算し、既存の最短距離より短いなら更新する
  d=np.minimum(d,np.resize(d[:,x],(n,n))+z+np.resize(d[y,:],(n,n)).T,out=d)
  d=np.minimum(d,d.T)
  print(int(d.sum()//2))


# ベクトルの成す角
import numpy as np
from numpy import linalg as LA
def tangent_angle(u: np.ndarray, v: np.ndarray):
  i=np.inner(u,v)
  n=LA.norm(u)*LA.norm(v)
  c=i/n
  return np.rad2deg(np.arccos(np.clip(c,-1.0,1.0)))
a=np.array([3,4])
b=np.array([-4,3])
print(tangent_angle(a, b))
# 90.0
# 3点の成す角
# 角度 p0-p1-p2
x0,y0=0,0
x1,y1=0,1
x2,y2=1,0
import math
(math.atan2(y2-y1,x2-x1)-math.atan2(y0-y1,x0-x1))/math.pi*180

# 凸包。AGC021-Bで使用
def Monotone_Chain(n,xy):
  xy=[[x,y,i]for i,(x,y) in enumerate(xy)]
  xy.sort(key=lambda x:x[0]*10**7 + x[1])
  s,t=xy[0],xy[-1]
  ary0=[s]
  ary1=[s]
  if t[0]-s[0]==0:
    ary=[s,t]
    return ary
  base=(t[1]-s[1])/(t[0]-s[0])
  # 凸包
  for x,y,i in xy[1:]:
    bd=base*(x-s[0])+s[1]
    #print(x,y,bd,base)
    if y>bd or i==t[2]:
      while len(ary0)>1:
        x0,y0,i0=ary0[-1]
        x1,y1,i1=ary0[-2]
        if x-x0!=0 and x-x1!=0:
          a0=(y-y0)/(x-x0)
          a1=(y-y1)/(x-x1)
          if a0<a1:break
        ary0.pop()
      ary0.append([x,y,i])
    if y<bd or i==t[2]:
      while len(ary1)>1:
        x0,y0,i0=ary1[-1]
        x1,y1,i1=ary1[-2]
        if x-x0!=0 and x-x1!=0:
          a0=(y-y0)/(x-x0)
          a1=(y-y1)/(x-x1)
          if a0>a1:break
        else:
          if x-x1!=0:
            break
        ary1.pop()
      ary1.append([x,y,i])
  ret=[0]*n
  ary1.reverse()
  ary=ary0+ary1[1:-1]
  return ary


# 最大公約数
import math
math.gcd(x,y)
import fractions
fractions.gcd(x,y)
# 最大公約数 再帰定義
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)
def gcd(a, b):
    while b:
        a, b = b, a%b
    return a
# n個の数字の最大公約数
import functools
def gcdn(nums):
    return functools.reduce(gcd, nums)

# 最小公倍数
def lcm(x, y):
    return (x * y) // math.gcd(x, y)

# n個の数字の最小公倍数
import functools
def lcmn(nums):
        return functools.reduce(lcm, nums)

# 場合の数 nCr
import math
def ncr(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

# 場合の数をmodで割った余り nCr % mod
def ncr_mod(n,r,mod):
  a,b=1,1
  for i in range(r):
    a*=n-i
    a%=mod
    b*=i+1
    b%=mod
  return (a*pow(b,mod-2,mod))%mod

# nまでのcmbを計算O(N^2)
memo={}
def cmb(m,r):
  r=min(r,m-r)
  if (m,r) in memo:return memo[(m,r)]
  if r==1:
    ret=i
  elif r==0:
    ret=1
  else:
    ret=memo[(m,r-1)]*(m-r+1)/r
  memo[(m,r)]=int(ret)
  return ret
for i in range(1,n+1):
  for j in range(1,i+1):
    cmb(i,j)


# コンビネーション。あらかじめO(N)の計算をすることでのちの計算が早くなる
def cmb(n,r,mod):
  if (r<0 or r>n):
    return 0
  r=min(r,n-r)
  return (g1[n]*g2[r]*g2[n-r])%mod
g1=[1,1] # g1[i]=i! % mod　:階乗
g2=[1,1] # g2[i]=(i!)^(-1) % mod　:階乗の逆元
inverse=[0,1]
for i in range(2,n+1):
  g1.append((g1[-1]*i)%mod)
  inverse.append((-inverse[mod%i]*(mod//i))%mod)
  g2.append((g2[-1]*inverse[-1])%mod)


# 巡回セールスマン。n<=17で通る
# https://inarizuuuushi.hatenablog.com/entry/2017/01/31/083835
def tsp(d):
  n = len(d)
  # DP[A] = {v: value}
  DP = dict()
  
  for A in range(1, 1 << n):
    if A & 1 << 0 == 0:# 集合Aが0を含まない
      continue
    if A not in DP:
      DP[A] = dict()

    # main
    for v in range(n):
      if A & 1 << v == 0:
        if A == 1 << 0:
          DP[A][v] = d[0][v] if d[0][v] > 0 else float('inf')
        else:
          DP[A][v] = min([DP[A ^ 1 << u][u] + d[u][v] for u in range(n) 
                          if u != 0 and A & 1 << u != 0 and d[u][v] > 0]
                          + [float('inf')])
  # 最後だけ例外処理
  V = 1 << n
  DP[V] = dict()
  DP[V][0] = min([DP[A ^ 1 << u][u] + d[u][0] for u in range(n) 
                  if u != 0 and A & 1 << u != 0 and d[u][0] > 0]
                  + [float('inf')]) 
  return DP[V][0]

if __name__ == '__main__':
  n=int(input())
  xyz=[list(map(int,input().split())) for _ in range(n)]
  dist=[[0]*n for _ in range(n)]
  for i in range(n):
    for j in range(n):
      a,b,c=xyz[i]
      p,q,r=xyz[j]
      dist[i][j]=abs(a-p)+abs(b-q)+max(0,r-c)
  res = tsp(dist)
  print(res)

# 分割数
def main0(n,m):
  mod=998244353
  dp = [[0]*(n+1) for i in range(m+1)]
  dp[0][0] = 1
  for i in range(1,m+1):
      for j in range(n+1):
          if j-i >= 0:
              dp[i][j] = (dp[i-1][j] + dp[i][j-i]) % mod
          else:
              dp[i][j] = dp[i-1][j]
  return dp[n][m]
  #return (dp[m][n]-dp[m-1][n])%mod



# 4bit全探索 ABC119-C
m=n*2
ans=pow(10,9)
for i in range(2**m):
  g=[[]for _ in range(4)]
  for j in range(n):
    g[(i//pow(4,j))%4].append(j)
  tmp=func(g)
  if tmp>=0:
    ans=min(ans,tmp)


# 指定した整数以下の素数を列挙。エラトステネスの篩
def eratosthenes(n):
  l0=list(range(2,n+1))
  l1=[1]*(n+1)
  l1[0]=0
  l1[1]=0
  for li in l0:
    if li>int(n**0.5):
      break
    if l1[li]==1:
      k=2
      while k*li<=n:
        l1[k*li]=0
        k+=1
  ret=[i for i,li in enumerate(l1) if li==1]
  return ret

# 素因数分解
def factorization(n):
  arr = []
  temp = n
  for i in range(2, int(-(-n**0.5//1))+1):
    if temp%i==0:
      cnt=0
      while temp%i==0:
        cnt+=1
        temp //= i
      arr.append([i, cnt])
  if temp!=1:
    arr.append([temp, 1])
  if arr==[]:
    arr.append([n, 1])
  return arr
#factorization(24) 
## [[2, 3], [3, 1]] 
##  24 = 2^3 * 3^1

"""
高速素因数分解
問題： A以下の数がN個与えられる。全て素因数分解せよ。
 前計算としてエラトステネスの篩を行い、「その数をふるい落とした素数」を配列 Dに記録します。例えば D[4]=D[6]=2,D[35]=5です。
 xが素数のときは D[x]=xとしておきます。この配列はエラトステネスの篩と同様O(AloglogA)で構築できます。
 D[x]はxを割り切る最小の素数なので、この配列Dを利用すると素因数分解を行うときに「試し割り」をする必要がなくなり(D[x]で割ればよい)、
 1つの数の素因数分解が素因数の個数である O(logA) でできるようになります。
"""
# セグメントツリー

class SegmentTree():
  def __init__(self,size,f=lambda x,y:x+y,default=0):
    self.size=pow(2,(size-1).bit_length())
    self.f=f
    self.default=default
    self.data=[default]*(self.size*2)
  # list[i]をxに更新
  def update(self,i,x):
    i+=self.size
    self.data[i]=x
    while i:
      i>>=1
      self.data[i]=self.f(self.data[i*2],self.data[i*2+1])
  # はじめの構築をO(N)で行う
  def init_build(self,ary):
    n=len(ary)
    for i in range(n):
      self.data[i+self.size]=ary[i]
    for i in reversed(range(self.size)):
      self.data[i]=self.f(self.data[i*2],self.data[i*2+1])
  # 区間[l,r)へのクエリ
  def query(self,l,r):
    l,r=l+self.size,r+self.size
    lret,rret=self.default,self.default
    while l<r:
      if l&1:
        lret=self.f(self.data[l],lret)
        l+=1
      if r&1:
        r-=1
        rret=self.f(self.data[r],rret)
      l>>=1
      r>>=1
    return self.f(lret,rret)
  def get(self,i):
    return self.data[self.size+i]
  # fがsumの時に使える
  def add(self,i,x):
    self.update(i,self.get(i)+x)
  # f=min
  # RMinQ(Range Minimum Query)
  # x以下の要素を持つ最右idxを返す。存在しない場合-1を返す。
  def min_right(self,x):
    if self.data[1]>x:return -1
    now=1
    while now<self.size:
      if self.data[2*now+1]<=x:
        now=2*now+1
      else:
        now=2*now
    return now-self.size
  # f=min
  # RMinQ(Range Minimum Query)
  # x以下の要素を持つ最左idxを返す。存在しない場合-1を返す。
  def min_left(self,x):
    if self.data[1]>x:return -1
    now=1
    while now<self.size:
      if self.data[2*now]<=x:
        now=2*now
      else:
        now=2*now+1
    return now-self.size
  # f=max
  # RMaxQ(Range Maximum Query)
  # x以上の要素を持つ最右idxを返す。存在しない場合-1を返す。
  def max_right(self,x):
    if self.data[1]<x:return -1
    now=1
    while now<self.size:
      if self.data[2*now+1]>=x:
        now=2*now+1
      else:
        now=2*now
    return now-self.size
  # f=max
  # RMaxQ(Range Maximum Query)
  # x以上の要素を持つ最左idxを返す。存在しない場合-1を返す。
  def max_left(self,x):
    if self.data[1]<x:return -1
    now=1
    while now<self.size:
      if self.data[2*now]>=x:
        now=2*now
      else:
        now=2*now+1
    return now-self.size


# n: 頂点数
# ki: 木
# Euler Tour の構築
S=[] # Euler Tour
F=[0]*n # F[v]:vにはじめて訪れるステップ
depth=[0]*n # 0を根としたときの深さ
def dfs(v,pare,d):
    F[v]=len(S)
    depth[v]=d
    S.append(v)
    for w in ki[v]:
        if w==pare:continue
        dfs(w,v,d+1)
        S.append(v)
dfs(0,-1,0)
print('S',S)
print('F',F)
print('depth',depth)

# Sをセグメント木に乗せる
# u,vのLCAを求める:S[F[u]:F[v]+1]のなかでdepthが最小の頂点を探せば良い
# F[u]:uに初めてたどるつくステップ
# S[F[u]:F[v]+1]:はじめてuにたどり着いてつぎにvにたどるつくまでに訪れる頂点
# 存在しない範囲は深さが他よりも大きくなるようにする
INF = (n, None)
# LCAを計算するクエリの前計算
M = 2*n
M0 = 2**(M-1).bit_length() # M以上で最小の2のべき乗
data = [INF]*(2*M0)
for i, v in enumerate(S):
  data[M0-1+i] = (depth[v], i)
for i in range(M0-2, -1, -1):
  data[i] = min(data[2*i+1], data[2*i+2])
print('data',data)
# LCAの計算 (generatorで最小値を求める)
def _query(a, b):
  yield INF
  a += M0; b += M0
  while a < b:
    if b & 1:
      b -= 1
      yield data[b-1]
    if a & 1:
      yield data[a-1]
      a += 1
    a >>= 1; b >>= 1
# LCAの計算 (外から呼び出す関数)
def query(u, v):
  fu = F[u]; fv = F[v]
  if fu > fv:
    fu, fv = fv, fu
  return S[min(_query(fu, fv+1))[1]]



# 平衡二分木の代わりにできる。
# 順序を保ったまま要素の挿入削除最小値の取り出しをO(logN)でできる。
# ABC170-Eで使用
import heapq
class pqheap:
  def __init__(self,key=None):
    self.p = list()
    self.q = list()

  def insert(self,x):
    heapq.heappush(self.p, x)
    return

  def erase(self,x):
    heapq.heappush(self.q, x)
    return

  def minimum(self):
    while self.q and self.p[0] == self.q[0]:
      heapq.heappop(self.p)
      heapq.heappop(self.q)
    return self.p[0] if len(self.p)>0 else None

# MP。文字列 S が与えられたときに、各 i について「文字列S[0,i-1]の接頭辞と接尾辞が最大何文字一致しているか」を記録した配列を O(|S|)で構築するアルゴリズムです
# https://snuke.hatenablog.com/entry/2014/12/01/235807
s='aabaabcaa'
a=[0]*(len(s)+1)
a[0]=-1
j=-1
for i in range(len(s)):
  while j>=0 and s[i]!=s[j]:
    j=a[j]
  j+=1
  a[i+1]=j
print(a)

# Manacher
# 文字列が与えられた時、各 i について「文字 i を中心とする最長の回文の半径」を記録した配列 R を O(|S|) で構築するアルゴリズムです。半径というのは、(全長+1)/2です。
# 普通のManacherをやると奇数長の回文しか検出できませんが、「a$b$a$a$b」みたいにダミー文字を間に挟むと偶数長のものも検出できるようにできます。
#s='abaaababa'
s='aaaaa'
i,j=0,0
r=[0]*len(s)
while i<len(s):
  while i-j>=0 and i+j < len(s) and s[i-j]==s[i+j]:
    j+=1
  r[i]=j
  k=1
  while i-k>=0 and k+r[i-k]<j:
    r[i+k]=r[i-k]
    k+=1
  i+=k
  j-=k
print(r)


# Z-algorithm
# 文字列が与えられた時、各 i について「S と S[i:|S|-1] の最長共通接頭辞の長さ」を記録した配列 A を O(|S|) で構築するアルゴリズムです。
s='aaabaaaab'
a=[0]*len(s)
a[0]=len(s)
i,j=1,0
while i<len(s):
  while i+j<len(s) and s[j]==s[i+j]:
    j+=1
  a[i]=j
  if j==0:
    i+=1
    continue
  k=1
  while i+k<len(s) and k+a[k]<j:
    a[i+k]=a[k]
    k+=1
  i+=k
  j-=k
print(a)


# トポロジカルソートのパターン数
# n:頂点、m:ヒント数
# ヒントxy:x->yの順序
n,m=map(int,input().split())
xy=[list(map(int,input().split())) for _ in range(m)]
g=[[] for _ in range(n)]
for x,y in xy:
  x,y=x-1,y-1
  g[y].append(x)
dp=[-1]*(2**n)
dp[0]=1
def func(s):
  if dp[s]>0:
    return dp[s]
  else:
    ret=0
    p=[]
    for i in range(n):
      if (s>>i)&1:
        p.append(i)
    if len(p)==1:
      dp[s]=1
      return 1
    else:
      for pi in p:
        if not set(g[pi]).intersection(set(p)):
          ret+=func(s-2**pi)
      dp[s]=ret
      return ret
for i in range(2**n):
  func(i)
print(dp[2**n-1])

# トライ木
class Trie:
  class Node:
    def __init__(self,c):
      self.c=c
      self.next_c_id={}
      #self.common=0
      self.end=-1

  def __init__(self):
    self.trie=[self.Node('')]

  def add(self,id,s):
    i=0
    now=0
    ary=[0]
    while i<len(s) and s[i] in self.trie[now].next_c_id:
      now=self.trie[now].next_c_id[s[i]]
      ary.append(now)
      #self.trie[now].common+=1
      i+=1
    while i<len(s):
      self.trie[now].next_c_id[s[i]]=len(self.trie)
      now=len(self.trie)
      ary.append(now)
      self.trie.append(self.Node(s[i]))
      #self.trie[now].common+=1
      i+=1
    self.trie[now].end=id
    return ary

  def search(self,s):
    i=0
    now=0
    ary=[0]
    while i<len(s) and s[i] in self.trie[now].next_c_id:
      now=self.trie[now].next_c_id[s[i]]
      ary.append(now)
      i+=1
    if i<len(s):return []
    return ary

"""
ローリングハッシュでは、文字列A(m文字)から、
互いに素な基数bとmodの除数hを用いて以下の式でハッシュ値を求めます。
hash(A) =  A_0*b^(m-1) + A_1*b^(m-2) + ... + A_(m-1)*b^0) mod h
"""
# ローリングハッシュ
a2n=lambda x:ord(x)-ord('a')+1
b=10**9+7
h=pow(2,61)-1

# 最大流
# https://tjkendev.github.io/procon-library/python/max_flow/dinic.html
import sys
sys.setrecursionlimit(10**7)
from collections import deque
class MaxFlow:
  # n:頂点数
  def __init__(self,n):
    self.n=n
    self.g=[[] for i in range(n)]
  
  # 辺を追加する
  # fr:辺の始点
  # to:辺の終点
  # cap:辺のキャパシティ
  def add_edge(self,fr,to,cap):
    forward=[to,cap,None]
    forward[2]=backward=[fr,0,forward]
    # forward[2]=backward,backward[2]=forwardとなるように再帰的にforwardとbackwardを定義
    self.g[fr].append(forward)
    self.g[to].append(backward)
    #print('add_edge',fr,to)

  # sから各頂点への距離を返す。フローを流すごとにcapが減り、最終的に通れる辺が減り、tまで辿り着けなくなる。それまでフローを流す
  def bfs(self,s,t):
    self.level=level=[None]*self.n
    deq=deque([s])
    level[s]=0
    g=self.g
    while deq:
      v=deq.popleft()
      nlv=level[v]+1
      for nv,cap, _ in g[v]:
        if cap and level[nv] is None:
          level[nv]=nlv
          deq.append(nv)
    return level[t] is not None
  # v->tにfを流す。再帰的に呼び出す。v=tとなるまで続ける。
  # sから遠ざかるようなパスを見つけ、フローを流す
  def dfs(self,v,t,f):
    if v==t:
      return f
    level=self.level
    # self.it[v]:頂点vから伸びる辺。一本の辺について着目するとfordwardとbackwardが交互に呼び出される。
    for e in self.it[v]:
      nv,cap,rev=e
      if cap and level[v]<level[nv]:
        d=self.dfs(nv,t,min(f,cap))
        if d:
          e[1]-=d
          rev[1]+=d
          return d
    return 0

  # sからtへの最大フローを返す。
  # 以下の処理をフローを流しきるまで繰り返す。
  # BFSでsourceから各頂点までの距離(level)を計算
  # DFSでsourceからの距離が遠くなるようなパスを見つけ、フローを流す
  def flow(self,s,t):
    flow=0
    INF=10**9+7
    g=self.g
    # sから各頂点への距離を計算。sからtへ辿り着けない場合、終了。距離はself.levelに保存される。
    while self.bfs(s,t):
      # グラフの各要素をイテレータに変換ものをself.itに入れる。????
      *self.it,=map(iter,self.g)
      f=INF
      while f:
        # s->tにフローを流す。流せた量をflowに加算。流せる量があるまで続ける。
        f=self.dfs(s,t,INF)
        flow+=f
    return flow

# 最小費用流
from heapq import heappush, heappop
class MinCostFlow:
  INF = 10**18

  def __init__(self, N):
    self.N = N
    self.G = [[] for i in range(N)]

  def add_edge(self, fr, to, cap, cost):
    forward = [to, cap, cost, None]
    backward = forward[3] = [fr, 0, -cost, forward]
    self.G[fr].append(forward)
    self.G[to].append(backward)

  def flow(self, s, t, f):
    N = self.N; G = self.G
    INF = MinCostFlow.INF

    res = 0
    H = [0]*N
    prv_v = [0]*N
    prv_e = [None]*N

    d0 = [INF]*N
    dist = [INF]*N

    while f:
      dist[:] = d0
      dist[s] = 0
      que = [(0, s)]

      while que:
        c, v = heappop(que)
        if dist[v] < c:
          continue
        r0 = dist[v] + H[v]
        for e in G[v]:
          w, cap, cost, _ = e
          if cap > 0 and r0 + cost - H[w] < dist[w]:
            dist[w] = r = r0 + cost - H[w]
            prv_v[w] = v; prv_e[w] = e
            heappush(que, (r, w))
      if dist[t] == INF:
        return None

      for i in range(N):
        H[i] += dist[i]

      d = f; v = t
      while v != s:
        d = min(d, prv_e[v][1])
        v = prv_v[v]
      f -= d
      res += d * H[t]
      v = t
      while v != s:
        e = prv_e[v]
        e[1] -= d
        e[3][1] += d
        v = prv_v[v]
    return res

# 木の入力例
n=10
ab=[]
for i in range(n-1):
  #ab.append([i//2+1,i+2])
  #ab.append([i+1,i+2])
  ab.append([1,i+2])


# ACL Begnner Contest E問題で使用
class LazySegmentTree():
  def __init__(self, n, op, e, mapping, composition, id):
    self.n = n
    self.op = op
    self.e = e
    self.mapping = mapping
    self.composition = composition
    self.id = id
    self.log = (n - 1).bit_length()
    self.size = 1 << self.log
    self.data = [e] * (2 * self.size)
    self.lazy = [id] * (self.size)

  def update(self, k):
    #print(self.data[2*k],self.data[2*k+1])
    self.data[k] = self.op(self.data[2 * k], self.data[2 * k + 1])

  def all_apply(self, k, f):
    self.data[k] = self.mapping(f, self.data[k])
    if k < self.size:
      self.lazy[k] = self.composition(f, self.lazy[k])

  def push(self, k): # 親の遅延配列の値を子に反映させる
    self.all_apply(2 * k, self.lazy[k])
    self.all_apply(2 * k + 1, self.lazy[k])
    self.lazy[k] = self.id

  def build(self, arr):
    #assert len(arr) == self.n
    for i, a in enumerate(arr):
      self.data[self.size + i] = a
    for i in range(self.size-1,0,-1):
      self.update(i)

  def set(self, p, x):
    #assert 0 <= p < self.n
    p += self.size
    #事前に関係のある遅延配列を全て反映させてしまう
    for i in range(self.log, 0, -1):
      self.push(p >> i)
    self.data[p] = x #値を更新する
    #関係のある区間の値も更新する
    for i in range(1, self.log + 1):
      self.update(p >> i)

  def get(self, p):
    #assert 0 <= p < self.n
    p += self.size
    #関係のある遅延配列を全て反映させる
    for i in range(1, self.log + 1):
      self.push(p >> i)
    return self.data[p]

  def prod(self, l, r):
    #assert 0 <= l <= r <= self.n
    if l == r: return self.e
    l += self.size
    r += self.size
    for i in range(self.log, 0, -1):
      if ((l >> i) << i) != l: self.push(l >> i)
      if ((r >> i) << i) != r: self.push(r >> i)
    sml = smr = self.e
    while l < r:
      if l & 1:
        sml = self.op(sml, self.data[l])
        l += 1
      if r & 1:
        r -= 1
        smr = self.op(self.data[r], smr)
      l >>= 1
      r >>= 1
    return self.op(sml, smr)

  def all_prod(self):
    return self.data[1]

  def apply(self, p, f):
    #assert 0 <= p < self.n
    p += self.size
    for i in range(self.log, 0, -1):
      self.push(p >> i)
    self.data[p] = self.mapping(f, self.data[p])
    for i in range(1, self.log + 1):
      self.update(p >> i)

  def range_apply(self, l, r, f):
    #assert 0 <= l <= r <= self.n
    if l == r: return
    l += self.size
    r += self.size
    for i in range(self.log, 0, -1):
      if ((l >> i) << i) != l: self.push(l >> i)
      if ((r >> i) << i) != r: self.push((r - 1) >> i)
    l2 = l
    r2 = r
    while l < r:
      if l & 1:
        self.all_apply(l, f)
        l += 1
      if r & 1:
        r -= 1
        self.all_apply(r, f)
      l >>= 1
      r >>= 1
    l = l2
    r = r2
    for i in range(1, self.log + 1):
      if ((l >> i) << i) != l: self.update(l >> i)
      if ((r >> i) << i) != r: self.update((r - 1) >> i)

  def max_right(self, l, g):
    #assert 0 <= l <= self.n
    #assert g(self.e)
    if l == self.n: return self.n
    l += self.size
    for i in range(self.log, 0, -1):
      self.push(l >> i)
    sm = self.e
    while True:
      while l % 2 == 0: l >>= 1
      if not g(self.op(sm, self.data[l])):
        while l < self.size:
          self.push(l)
          l = 2 * l
          if g(self.op(sm, self.data[l])):
            sm = self.op(sm, self.data[l])
            l += 1
        return l - self.size
      sm = self.op(sm, self.data[l])
      l += 1
      if (l & -l) == l: return self.n

  def min_left(self, r, g):
    #assert 0 <= r <= self.n
    #assert g(self.e)
    if r == 0: return 0
    r += self.size
    for i in range(self.log, 0, -1):
      self.push((r - 1) >> i)
    sm = self.e
    while True:
      r -= 1
      while r > 1 and r % 2: r >>= 1
      if not g(self.op(self.data[r], sm)):
        while r < self.size:
          self.push(r)
          r = 2 * r + 1
          if g(self.op(self.data[r], sm)):
            sm = self.op(self.data[r], sm)
            r -= 1
        return r + 1 - self.size
      sm = self.op(self.data[r], sm)
      if (r & -r) == r: return 0

# ACL Begnner Contest E問題での使用例
import sys
input = sys.stdin.readline

INF = pow(10,18)
mod = 998244353
po32 = pow(2,32)

n,q=map(int,input().split())
lrd=[list(map(int,input().split())) for _ in range(q)]

def op(x, y):
  xv, xw = x
  yv, yw = y
  return (xv+yv)%mod,(xw+yw)%mod
def mapping(p, x): #pが更新後の値, xが更新する前の値
  if p==INF:return x
  xv,xw=x
  return (xw*p)%mod,xw
def composition(p, q):
  if p!=INF:return p
  return q
e=0,0
id=INF
ary=[(pow(10,n-i-1,mod),pow(10,n-i-1,mod)) for i in range(n)]
lst=LazySegmentTree(n,op,e,mapping,composition,id)
lst.build(ary)
ans = []
for l,r,d in lrd:
  lst.range_apply(l - 1, r, d)
  v = lst.all_prod()
  ans.append(v[0]%mod)
print(*ans, sep='\n')


# 平方分割
class squr_block:
  def __init__(self,n,f=lambda x,y:x+y,default=0):
    from math import sqrt,ceil
    self.f=f
    self.default=default
    self.block_size=ceil(sqrt(n))
    self.n=self.block_size**2
    self.data=[self.default]*self.n
    self.block_data=[self.default]*self.block_size

  def init_array(self,ary):
    idx=0
    tmp=self.default
    for i,x in enumerate(ary):
      if i and i%self.block_size==0:
        self.block_data[idx]=tmp
        idx+=1
        tmp=self.default
      self.data[i]=x
      tmp+=x
    if idx<self.block_size:self.block_data[idx]=tmp

  def update(self,i,x):
    self.data[i]=x
    t=self.block_size
    blockidx=i//t
    block_value=self.data[blockidx*t]
    for i in range(t-1):
      block_value=self.f(block_value,self.data[blockidx*t+i+1])
    self.block_data[blockidx]=block_value

  def get(self,i):
    return self.data[i]
  # [l,r)へのクエリ
  def query(self,l,r):
    ret=self.default
    t=self.block_size
    for i in range((l+t-1)//t,r//t):
      ret=self.f(ret,self.block_data[i])
    for i in range(l,min(l+(t-l)%t,r)):
      ret=self.f(ret,self.data[i])
    if min(l+(t-l)%t,r)==r:return ret
    for i in range(max(l,r-r%t),r):
      ret=self.f(ret,self.data[i])
    return ret

for i in range(10):
  # iで立っている一番小さいビット
  print(i,-i&i)

# sの部分集合を全列挙
s=19
sub = s
while sub:
  sub = (sub-1)&s

# 強連結成分分解(SCC): グラフGに対するSCCを行う
# 入力: <N>: 頂点サイズ, <G>: 順方向の有向グラフ, <RG>: 逆方向の有向グラフ
# 出力: (<ラベル数>, <各頂点のラベル番号>)
def scc(N, G, RG):
    order = []
    used = [0]*N
    group = [None]*N
    def dfs(s):
        used[s] = 1
        for t in G[s]:
            if not used[t]:
                dfs(t)
        order.append(s)
    def rdfs(s, col):
        group[s] = col
        used[s] = 1
        for t in RG[s]:
            if not used[t]:
                rdfs(t, col)
    for i in range(N):
        if not used[i]:
            dfs(i)
    used = [0]*N
    label = 0
    for s in reversed(order):
        if not used[s]:
            rdfs(s, label)
            label += 1
    return label, group

# 縮約後のグラフを構築
def construct(N, G, label, group):
    G0 = [set() for i in range(label)]
    GP = [[] for i in range(label)]
    for v in range(N):
        lbs = group[v]
        for w in G[v]:
            lbt = group[w]
            if lbs == lbt:
                continue
            G0[lbs].add(lbt)
        GP[lbs].append(v)
    return G0, GP

# 高速ゼータ変換
# https://ikatakos.com/pot/programming_algorithm/dynamic_programming/subset_convolution
# ARC100-Eで使用
# n:要素の数
# dp:要素の各部分集合に対する値。
def upd(now,x):
  if now[0]<x:
    now=[x,now[0]]
  elif now[1]<x:
    now[1]=x
  return now

def update(now,next):
  ary=now+next
  ary.sort(reverse=True)
  return ary[:2]

def main(n,a):
  dp=[[a[i],0] for i in range(2**n)]
  now=a[0]
  for j in range(n):
    bit=1<<j
    for i in range(1<<n):
      # i&bit==1とし、i^bitをiが部分集合として含む集合としてdp更新もできる。
      if i&bit==0:
        # i|bit:iを部分集合として含む数字
        dp[i|bit]=update(dp[i],dp[i|bit])
  now=sum(dp[0])
  for k in range(1,2**n):
    now=max(now,sum(dp[k]))
    print(now)
n=int(input())
a=list(map(int,input().split()))
main(n,a)


"""
n=3 # 集合の要素数
dp=[3,1,4,5,1,9,2,6] # 2^n の長さの配列
for j in range(n):
  bit=1<<j
  for i in range(1<<n):
    if i&bit==0:
      dp[i]+=dp[i|bit]
print(dp)
# => [31, 21, 17, 11, 18, 15, 8, 6]
# dp[i]:iをbit的に含むもののidxの合計
"""

# 高速ゼータ変換。ARC106-Eでの使用例
from collections import Counter,deque
def main(n,k,a):
  ary=[(1<<n)-1]*(2*n*k+1) 
  # ary[i]:i日目に働かない従業員のbitset(=int)
  # O(2*k*n^2)
  for i in range(n):
    day=1
    tmp=1<<i
    while day<=2*n*k:
      ary[day]-=tmp
      if day%a[i]==0:
        day+=a[i]+1
      else:
        day+=1
  def func(d):# d日で可能か
    # dp[s]:sに含まれる従業員がxの日数
    # 初期値は「sに含まれる従業員がxで他の従業員はoの日数」
    dp=[0]*(1<<n)
    ca=Counter(ary[1:d+1])
    for key,val in ca.items():
      # val:keyに含まれる従業員のみ働いていない日数
      dp[key]=val
    miary=[1]*(1<<n)
    # 高速ゼータ変換
    for j in range(n):
      bit=1<<j
      for i in range(1<<n):
        if i&bit==0:
          dp[i]+=dp[i|bit]
    for s in range(1,1<<n):
      l=0
      sub=s
      while sub:
        l+=1
        sub^=sub&(-sub)
      # k*l:必要なメダルの数。dary[s]:sに含まれる従業員すべてがxの日数
      if d-dp[s]>=k*l:continue
      else:return False
    return True
  l,r=1,2*n*k
  while r-l>1:
    d=(l+r)//2
    if func(d):
      l,r=l,d
    else:
      l,r=d,r
  return l if func(l) else r

if __name__=='__main__':
  n,k=map(int,input().split())
  a=list(map(int,input().split()))
  print(main(n,k,a))


def dot(mtr0,mtr1):
  a,b=len(mtr0),len(mtr1[0])
  t=len(mtr0[0])
  if t!=len(mtr1):return None
  ret=[[0]*b for _ in range(a)]
  for i in range(a):
    for j in range(b):
      tmp=0
      for k in range(t):
        tmp+=mtr0[i][k]*mtr1[k][j]
      ret[i][j]=tmp
  return ret


# ふたつの円の交点を返す。
import math
def intersection(x1,y1,r1,x2,y2,r2):
  xr=x2-x1
  yr=y2-y1
  if abs(-r2**2+(r1**2+xr**2+yr**2))<abs(-r1**2+(r2**2+xr**2+yr**2)):
    x1,y1,r1,x2,y2,r2=x2,y2,r2,x1,y1,r1
    xr*=-1
    yr*=-1
  R2=xr**2+yr**2
  R=math.sqrt(R2)
  aux=-r2**2+(r1**2+R2)
  s=r1+r2+R
  D=s*(s-2*r1)*(s-2*r2)*(s-2*R)
  if D<0.: 
    return []
  D=max(0.,D)
  xm=(xr*aux)/(2*R2)+x1
  ym=(yr*aux)/(2*R2)+y1
  dx=(yr*math.sqrt(D))/(2*R2)
  dy=-(xr*math.sqrt(D))/(2*R2)
  return (xm+dx,ym+dy),(xm-dx,ym-dy)
