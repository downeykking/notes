随机数：

```
为CPU中设置种子，生成随机数
torch.manual_seed(number)
为特定GPU设置种子，生成随机数
torch.cuda.manual_seed(number)
为所有GPU设置种子，生成随机数
torch.cuda.manual_seed_all()
```

[dataloader](https://blog.csdn.net/zw__chen/article/details/82806900)

[tensor，numpy的转化](https://blog.csdn.net/weixin_41490373/article/details/114407305?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.no_search_link)

[一个pipeline](https://www.analyticsvidhya.com/blog/2021/07/perform-logistic-regression-with-pytorch-seamlessly/)

[pytorch教程](https://github.com/yunjey/pytorch-tutorial)

[_, predicted = torch.max(outputs.data, dim=1))的理解](https://blog.csdn.net/weixin_48249563/article/details/111387501)

[squeeze和unsqueeze](https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch?rq=1)

[nn.embedding]()

[torch.nn.Embedding()中的padding_idx参数解读](https://blog.csdn.net/weixin_40426830/article/details/108870956?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)

```
一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。

实例一：创建查询矩阵进行查询
embedding = nn.Embedding(5, 3)  # 定义一个具有5个单词，维度为3的查询矩阵
print(embedding.weight)  # 展示该矩阵的具体内容
test = torch.LongTensor([[0, 2, 0, 1],
                         [1, 3, 4, 4]])  # 该test矩阵用于被embed，其size为[2, 4]
# 其中的第一行为[0, 2, 0, 1]，表示获取查询矩阵中ID为0, 2, 0, 1的查询向量
# 输入[batch_size, seq_len]
# 输出[batch_size, seq_len, dim]
# 可以在之后的test输出中与embed的输出进行比较
test = embedding(test)
print(test.size())  # 输出embed后test的size，为[2, 4, 3]，增加
# 的3，是因为查询向量的维度为3
print(test)  # 输出embed后的test的内容


——————————————————————————————————————
输出：
Parameter containing:
tensor([[-1.8056,  0.1836, -1.4376],
        [ 0.8409,  0.1034, -1.3735],
        [-1.3317,  0.8350, -1.7235],
        [ 1.5251, -0.2363, -0.6729],
        [ 0.4148, -0.0923,  0.2459]], requires_grad=True)
torch.Size([2, 4, 3])
tensor([[[-1.8056,  0.1836, -1.4376],
         [-1.3317,  0.8350, -1.7235],
         [-1.8056,  0.1836, -1.4376],
         [ 0.8409,  0.1034, -1.3735]],

        [[ 0.8409,  0.1034, -1.3735],
         [ 1.5251, -0.2363, -0.6729],
         [ 0.4148, -0.0923,  0.2459],
         [ 0.4148, -0.0923,  0.2459]]], grad_fn=<EmbeddingBackward>)
可以看出创建了一个具有5个ID(可以理解为拥有5个词的词典)的查询矩阵，每个查询向量的维度是3维，然后用一个自己需要Embedding的矩阵与之计算，其中的内容就是需要匹配的ID号，注意！如果需要Embedding的矩阵中的查询向量不为1，2这种整数，而是1.1这种浮点数，就不能与查询向量成功匹配，会报错，且如果矩阵中的值大于了查询矩阵的范围，比如这里是5，也会报错。

实例二：寻找查询矩阵中特定ID(词)的查询向量(词向量)

# 访问某个ID，即第N个词的查询向量(词向量)
print(embedding(torch.LongTensor([3])))  # 这里表示查询第3个词的词向量

————————————————————————————————————————
输出：
tensor([[-1.6016, -0.8350, -0.7878]], grad_fn=<EmbeddingBackward>)
```

[整数转one-hot](https://blog.csdn.net/hello_program_world/article/details/111867464?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.no_search_link&spm=1001.2101.3001.4242.1)

```
# input.scatter_(dim, index, src)
# 将src中数据根据index中的索引按照dim的方向填进input中
# pytorch的标记默认从0开始
tensor = torch.tensor([0, 2, 1, 3])
N = tensor.size(0)
num_classes = 4
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
```

[torchtext的一个不错tutorial](https://blog.csdn.net/u012436149/article/details/79310176)

torchtext Field执行流程：

torchtext对于Field，先执行preprocess，preprocess中先执行tokenize，之后执行自定义的preprocessing（参数x就是一个example，在文本中对应的就是一段字符串）。

之后执行process（参数batch，是指的多个x），process中执行pad，numericalize，postprocessing，最后返回一个tensor

[sampler，dataset，dataloader](https://www.cnblogs.com/marsggbo/p/11308889.html)



**pytorch中随机数相关**

torch.rand()
torch.rand(sizes, out=None)
产生一个服从均匀分布的张量，张量内的数据包含从区间[0,1)的随机数。参数size是一个整数序列，用于定义张量大小。

torch.randn()
torch.randn(sizes, out=None)
产生一个服从标准整正态分布的张量，张量内数据均值为0，方差为1，即为高斯白噪声。sizes作用同上。

torch.normal()
torch.normal(means, std, out=None)
产生一个服从离散正态分布的张量随机数，可以指定均值和标准差。其中，标准差std是一个张量包含每个输出元素相关的正态分布标准差。

torch.randperm()
torch.randperm(n, out=None, requires_grad=True)
返回从0到n-1的整数的随机排列数

torch.randint()
torch.randint(low=0, high, size, out=None, requires_grad=False)
返回一个张量，该张量填充了在[low,high)均匀生成的随机整数。
张量的形状由可变的参数大小定义。



[二分类 ，分类损失函数的区别](https://www.cnblogs.com/zhangxianrong/p/14773075.html)

[softmax和log_softmax的区别、CrossEntropyLoss() 与 NLLLoss()](https://blog.csdn.net/hao5335156/article/details/80607732?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=2)



torch.repeat()

```
https://blog.csdn.net/flyingluohaipeng/article/details/125039368
x = torch.tensor([1,2,3])  # size = (3,)
从右到左看
x1 = x.repeat(4) 此时dim=0 repeat四次，变成 tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])  # size = (12,)

x2 = x.repeat(4,1) dim=1 repeat一次, 增加维度变成 size=(1, 1) dim=0 repeat四次
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
repeat() 里面的参数<=2时，从右到左看分别dim=1，dim=0的复制数    
对于含有的参数大于2时，如(b,m,n)，即表示先在dim=0上面进行重复n次，再在dim=1上面重复m次，最后在通道上面重复b次
eg.
a = torch.randn(3, 2)
c = a.repeat(5, 6, 7, 8)

print(c.size())
torch.Size([5, 6, 21, 16])
```



torch.repeat_interleave() 

```
https://blog.csdn.net/flyingluohaipeng/article/details/125039411
区分torch.repeat两个函数方法最大的区别就是repeat_interleave是一个元素一个元素地重复，而repeat是一组元素一组元素地重复.
```



torch.expand()

```
https://blog.csdn.net/jacke121/article/details/124069005
```



[torch.where(condition, x, y):](https://blog.csdn.net/tszupup/article/details/108130366)

```
1. condition：判断条件
2. x：若满足条件，则取x中元素
3. y：若不满足条件，则取y中元素
4. 对于tensor.where(condition, y)，满足条件取本身，不满足条件取y
```



[torch.mm() torch.sparse.mm() torch.bmm() torch.mul() torch.matmul()的区别](https://blog.csdn.net/qq_33952811/article/details/120710801)



[稀疏矩阵之scipy中的coo_matrix函数](https://www.cnblogs.com/datasnail/p/11021835.html)

```
>>> # Constructing a matrix using ijv format
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])

经常的用法大概是这样的：coo_matrix((data, (i, j)), [shape=(M, N)])

- 这里有三个参数：

  data[:] 就是原始矩阵中的数据，例如上面的4,5,7,9；i[:] 就是行的指示符号；例如上面row的第0个元素是0，就代表data中第一个数据在第0行；j[:] 就是列的指示符号；例如上面col的第0个元素是0，就代表data中第一个数据在第0列；综合上面三点，对data中第一个数据4来说，就是原始矩阵中有4这个元素，它在第0行，第0列，即A[i[k], j[k]] = data[k]。以此类推，data中第2个数据5，在第3行，第3列。最后，有个shape参数是告诉coo_matrix原始矩阵的形状，除了上述描述的有数据的行列，其他地方都按照shape的形式补0。
```



torch.scatter

```
A.scatter_(dim, index, B) # 基本用法, tensorB中元素被扩散到tensorA中的index位置
```

![img](https://pic3.zhimg.com/80/v2-efcb5b75c1835d3dcf375ed31e6ef50a_1440w.jpg)

```
如上图，src为x，x中的元素被扩散到torch.zeros(3, 5)中，index为(2,5)的矩阵，共有十个元素。所以有10个数字被扩散到torch.zeros()中，位置为index指定的位置。下面将举例如何找index的位置。

1.判断dim。上图给定dim为0，则index对应的torch.zeros中的维度为0维发生变化。
对应上图：
首先为index第一行：
index=0时，此时对应的torch.zeros()中[0][0]元素，使用x[0][0]填充。
index=1时，此时对应的torch.zeros()中[1][1]元素，使用x[0][1]填充。
index=2时，此时对应的torch.zeros()中[2][2]元素，使用x[0][2]填充。
index=0时，此时对应的torch.zeros()中[0][3]元素，使用x[0][3]填充。
index=0时，此时对应的torch.zeros()中[0][4]元素，使用x[0][4]填充。
之后为index第二行：
index=2时，此时对应的torch.zeros()中[2][0]元素，使用x[1][0]填充。
index=0时，此时对应的torch.zeros()中[0][1]元素，使用x[1][1]填充。
index=0时，此时对应的torch.zeros()中[0][2]元素，使用x[1][2]填充。
index=1时，此时对应的torch.zeros()中[1][3]元素，使用x[1][3]填充。
index=2时，此时对应的torch.zeros()中[2][4]元素，使用x[1][4]填充。


src = torch.arange(1, 11).reshape((2, 5))
print(src)

index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
print(torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src))

tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
        
tensor([[ 5,  2,  3,  0,  0],
        [ 8,  9, 10,  0,  0],
        [ 0,  0,  0,  0,  0]])
2.判断dim。上图给定dim为1，则index对应的torch.zeros中的维度为1维发生变化。
对应上图：
首先为index第一行：
index=0时，此时对应的torch.zeros()中[0][0]元素，使用x[0][0]填充。
index=1时，此时对应的torch.zeros()中[0][1]元素，使用x[0][1]填充。
index=2时，此时对应的torch.zeros()中[0][2]元素，使用x[0][2]填充。
index=0时，此时对应的torch.zeros()中[0][0]元素，使用x[0][3]填充。
index=0时，此时对应的torch.zeros()中[0][0]元素，使用x[0][4]填充。
之后为index第二行：
index=2时，此时对应的torch.zeros()中[1][0]元素，使用x[1][0]填充。
index=0时，此时对应的torch.zeros()中[1][1]元素，使用x[1][1]填充。
index=0时，此时对应的torch.zeros()中[1][2]元素，使用x[1][2]填充。
index=1时，此时对应的torch.zeros()中[1][0]元素，使用x[1][3]填充。
index=2时，此时对应的torch.zeros()中[1][0]元素，使用x[1][4]填充。

所以，index为几，就把对应位置的元素放入目标tensor的第几行（dim=0时，列不变），大概就是这么个意思。以此类推，dim=1时，把对应位置的元素放入目标tensor的第几列（dim=1时，行不变）

此外，示例中还涉及到reduce参数：
不填就是None，直接覆盖
填multiply就是（src元素*target元素）
填add就是（src元素+target元素）

self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

五个约束：
（1）张量self，张量index和张量src的维度数量必须相同（即三者的.dim()必须相等，注意不是维度大小）；

（2）对于每一个维度d，有index.size(d)<=src.size(d)；

（3）对于每一个维度d，如果d!=dim，有index.size(d)<=self.size(d)；

同时，张量index中的数值大小也有2个约束：

（4）张量index中的每一个值大小必须在[0, self.size(dim)-1]之间；

（5）张量index沿dim维的那一行中所有值都必须是唯一的（弱约束，违反不会报错，但是会造成没有意义的操作）。

对于scatter_add，会将原本覆盖的值进行累加
src = torch.arange(1, 11).reshape((2, 5))
print(src)

index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
print(torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src))

tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
        
tensor([[10,  2,  3,  0,  0],
        [15,  9, 16,  0,  0],
        [ 0,  0,  0,  0,  0]])
        
首先为index第一行：
index=0时，此时对应的torch.zeros()中[0][0]元素，使用x[0][0]填充。
index=1时，此时对应的torch.zeros()中[0][1]元素，使用x[0][1]填充。
index=2时，此时对应的torch.zeros()中[0][2]元素，使用x[0][2]填充。
index=0时，此时对应的torch.zeros()中[0][0]元素，重复，x[0][0]+x[0][3]
index=0时，此时对应的torch.zeros()中[0][0]元素，重复，x[0][0]+x[0][3]+x[0][4]
之后为index第二行：
index=2时，此时对应的torch.zeros()中[1][0]元素，使用x[1][0]填充。
index=0时，此时对应的torch.zeros()中[1][1]元素，使用x[1][1]填充。
index=0时，此时对应的torch.zeros()中[1][2]元素，使用x[1][2]填充。
index=1时，此时对应的torch.zeros()中[1][0]元素，重复，x[1][0]+x[1][3]
index=2时，此时对应的torch.zeros()中[1][0]元素，重复，x[1][0]+x[1][3]+x[1][4]


参考：https://zhuanlan.zhihu.com/p/158993858


用在one hot和pyg中度的生成
```



scipy.sparse csr_matrix

```
import numpy as np
import scipy.sparse as sp
data = np.array([1, 2, 3, 4, 5, 6])      #所有非零数值
indices = np.array([0, 2, 2, 0, 1, 2])   #所有值的列索引
indptr = np.array([0, 2, 3, 6])          #最终matrix中每行有的非零数据 data[i：i+1]
mtx = sp.csr_matrix((data,indices,indptr),shape=(3,3))
print(mtx.toarray())

# [[1 0 2]
#  [0 0 3]
#  [4 5 6]]

https://blog.csdn.net/jeffery0207/article/details/100064602
```



RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.

```
这种情况是对于一个含有梯度的叶子节点进行了in-place操作，如 evc[evc > 0] = evc2[evc > 0]

正确的做法应该是先 evc = evc.clone()
```



torch.flatten()

```
torch.flatten(t, start_dim=0, end_dim=-1)

假设类型为 torch.tensor 的张量 t 的形状如下所示：(2,4,3,5,6)，则 torch.flatten(t, 1, 3).shape 的结果为 (2, 60, 6)。将索引为 start_dim 和 end_dim 之间（包括该位置）的数量相乘，其余位置不变。
当 默认 start_dim=0，end_dim=-1， torch.flatten(t) 返回的只有一维的数据。

```



torch.chunk()

```
torch.chunk(tensor, chunks, dim=0)
在给定维度(轴)上将输入张量进行分块，分成chunks块,返回的是元组

a = torch.randn(3, 2)
b, c = torch.chunk(a, chunks=2, dim=1)

print(a.size())
print(b.size(), c.size())
torch.Size([3, 2])
torch.Size([3, 1]) torch.Size([3, 1])
```



torch.spilt()

```
torch.split(tensor, split_size_or_sections, dim=0)
在给定维度(轴)上将输入张量进行分块，每一块的数目由split_size_or_sections给定（因此可以是个list）,返回的是元组,可以不均匀划分

a = torch.randn(3, 2)
e, f = torch.split(a, 2, dim=0)

print(e.size(), f.size())
torch.Size([2, 2]) torch.Size([1, 2])
```



torch.mask_select()

```
# torch.masked_select(input, mask, *, out=None) → Tensor
# 返回input中mask为True的元素，组成1维tensor
# input 和mask的维度不一定要相同,但是需要可以扩展成同样维度

https://blog.csdn.net/m0_46483236/article/details/115263417
```



torch.mask_fill()

```
torch.Tensor.masked_fill(mask, value)
输入的mask需要与当前的基础Tensor的形状一致。
将mask中为True的元素对应的基础Tensor的元素设置为值value。

https://stubbornhuang.blog.csdn.net/article/details/126341289
```



nonzero(),index_select()

```
https://blog.csdn.net/kz_java/article/details/123769974
```

