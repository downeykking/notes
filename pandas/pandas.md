#### 默认df为DataFrame

1. 获取dataFrame的列索引，即属性名，行索引是index

   col = df.columns # 获取到的col是<class 'pandas.core.indexes.base.Index'>

   col = df.columns.values # 获取到的是np.array

   df.index获取索引

2. 区分 df.loc[:, 'SalePrice'].values 和 df.loc[:, ['SalePrice']].values

   前者shape为(m,) 后者为(m,1)

3. 区分 shape为(x,)和shape为(x,1)，尽管它们都是一维的形式，但是具体机理不同：

   - (x,)意思是一维数组，数组中有2个元素

   - (x,1)意思是一个2维数组，每行有1个元素

     (5,3)(3,) 最后会降维为(5,)

     (5,3)(3,1) 才是正确保持维度 (5,1)

     (5,)+(5,1) 会变成(5,5)  

     [参见numpy广播](https://www.cnblogs.com/jiaxin359/p/9021726.html)

     [参见numpy广播](https://www.cnblogs.com/yangmang/p/7125458.html)
     
     2个规则弄懂numpy的broadcast广播机制 - 司南牧(李韬)的文章 - 知乎 https://zhuanlan.zhihu.com/p/60365398
   
4. 查看columns的数据类型 df.dtypes

5. 区分iloc和loc

   使用行号切片时使用iloc 使用index切片时使用loc

   eg:  	

   df.iloc[3:5,0:2]    不包括右边界

   df.loc['20130102':'20130104',['A','B']]    包括右边界
   
6. df.describe() 参数的设置可以选择统计数据的类型 [官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)
	
7. df根据条件选择行列时直接设置condition [参考1](https://www.it1352.com/904249.html)  [参考2](http://www.voidcn.com/article/p-tgrfizhp-bxx.html)

   ```python
   con = (df['Math']>=90) & (df['Math']<=100)
   print(df[con])
   或者
   mask = (df['Math']>=90) & (df['Math']<=100)
   df.loc[mask, 'flag'] = 'Y'
   ```

9. df按某一列进行排序 `df.sort_values(by=column)`  或 `df.sort_values(by=['AAA_cat','BBB_cat'],ascending = [0,1],inplace = True )`

10. 把不符合cond的转换为other

    DataFrame.where(***cond***, *other=nan***,** *inplace=False***,** *axis=None***,** *level=None***,** *errors='raise'***,** *try_cast=False***)**
	Series.where*(***cond**, *other=nan***,** *inplace=False***,** *axis=None***,** *level=None***,** *errors='raise'***,** *try_cast=False***)**
    
    区分np.where()
    
11. 导出为csv数值型数据变成科学计数法

    解决办法:数字改成字符串格式，字符串后加横向制表符’\t’再写入csv

    ```
    df['id'].apply(lambda x : str(x)+"\t") 
    df.to_csv(csv_path)
    ```

11. `DataFrame`中`axis`的概念：在`DataFrame`对象的大多数方法中，都会有`axis`这个参数，它控制了你指定的操作是沿着0轴还是1轴进行。`axis=0`代表操作对`列columns`进行，`axis=1`代表操作对行`row`进行。

12. map(), apply()和applymap()

    对series：map() 是一个[Series](https://so.csdn.net/so/search?q=Series&spm=1001.2101.3001.7020)的函数，DataFrame结构中没有map()。map()将一个自定义函数应用于Series结构中的每个元素(elements)。apply()可以解决传入`map`的函数只能接收一个参数的问题。

    ```
    boolean=[True,False]
    gender=["男","女"]
    color=["white","black","yellow"]
    data=pd.DataFrame({
        "height":np.random.randint(150,190,100),
        "weight":np.random.randint(40,90,100),
        "smoker":[boolean[x] for x in np.random.randint(0,2,100)],
        "gender":[gender[x] for x in np.random.randint(0,2,100)],
        "age":np.random.randint(15,90,100),
        "color":[color[x] for x in np.random.randint(0,len(color),100) ]
    }
    )
    
    #①使用字典进行映射
    data["gender"] = data["gender"].map({"男":1, "女":0})
    
    def apply_age(x,bias):
        return x+bias
    
    #以元组的方式传入额外的参数
    data["age"] = data["age"].apply(apply_age,args=(-3,))
    ```

    对dataframe：

    - 当沿着`轴0（axis=0）`进行操作时，会将各列(`columns`)默认以`Series`的形式作为参数，传入到你指定的操作函数中，操作后合并并返回相应的结果。最终生成行数据。
    - 当`apply`设置了`axis=1`对行进行操作时，会默认将每一行数据以`Series`的形式（Series的索引为列名）传入指定函数，返回相应的结果，生成列数据。

    applymap()将函数做用于DataFrame中的所有元素(elements)

    ```
    df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                       'key2' : ['one', 'two', 'one', 'two', 'one'],
                       'data1' : np.arange(5),
                       'data2' : np.arange(5,10)})
    df
      key1 key2  data1  data2
    0    a  one      0      5
    1    a  two      1      6
    2    b  one      2      7
    3    b  two      3      8
    4    a  one      4      9
    
    
    #axis =1 ,apply function to each row. 
    #axis =0,apply function to each column,default 0
    df['total'] = df[['data1','data2']].apply(lambda x : x.sum(),axis=1 ) 
    df
      key1 key2  data1  data2  total
    0    a  one      0      5      5
    1    a  two      1      6      7
    2    b  one      2      7      9
    3    b  two      3      8     11
    4    a  one      4      9     13
    
    df.loc['total'] = df[['data1','data2']].apply(lambda x : x.sum(),axis=0 )
    #此处由于是行数据，需要用索引的方式df.loc['total']增加行
    df
          key1 key2  data1  data2  total
    0        a  one    0.0    5.0    5.0
    1        a  two    1.0    6.0    7.0
    2        b  one    2.0    7.0    9.0
    3        b  two    3.0    8.0   11.0
    4        a  one    4.0    9.0   13.0
    total  NaN  NaN   10.0   35.0    NaN
    
    ```

13.  groupby

    `groupby`的过程就是将原有的`DataFrame`按照`groupby`的字段，划分为若干个`分组DataFrame`，被分为多少个组就有多少个`分组DataFrame`。**所以说，在`groupby`之后的一系列操作（如`agg`、`apply`等），均是基于`子DataFrame`的操作**，并且是按列操作。
    
    agg聚合操作：在新生成的df中按列进行操作。
    
    transform操作：`transform`和`agg`所不一样的地方，对`agg`而言，会计算得到`A`，`B`，`C`公司对应的均值并直接返回，但对`transform`而言，则会**对每一条数据求得相应的结果，同一组内的样本会有相同的值**，组内求完均值后会**按照原索引的顺序**返回结果。
    
    apply操作：将12中的apply传series改为传df。
    
14. merge

    示例数据及语句：

    ```
    df_1=pd.DataFrame({
        "userid":['a', 'b', 'c', 'd'],
        "age":[10,20,30,40]
    }
    )
    
    df_2 = pd.DataFrame({
        "userid":['a', 'c',],
        "payment":[2000, 3000]
    }
    )
    
    df_1.merge(df_2,how='inner',on='userid')
    df_1.merge(df_2,how='left',on='userid')
    df_2.merge(df_1,how='right',on='userid')
    df_1.merge(df_2,how='outer',on='userid')
    ```

    `merge`用于拼接两张表的，那么拼接时自然就需要将用户信息**一一对应**地进行拼接，所以进行拼接的两张表需要有一个共同的识别用户的**键（key）**。总结来说，整个`merge`的过程就是将信息**一一对应匹配**的过程，下面介绍`merge`的四种类型，分别为`'inner'`、`'left'`、`'right'`和`'outer'`。

    1. inner内连接

       它在拼接的过程中会取**两张表的键（key）的交集**进行拼接。

    2. `'left'`和`'right'`的`merge`方式其实是类似的，分别被称为**左连接**和**右连接**。这两种方法是可以互相转换的，所以在这里放在一起介绍。

       - `'left'`

       `merge`时，以**左边表格的键为基准**进行配对，如果左边表格中的键在右边不存在，则用缺失值`NaN`填充。

       - `'right'`

       `merge`时，以**右边表格的键为基准**进行配对，如果右边表格中的键在左边不存在，则用缺失值`NaN`填充。

    3. `'outer'`是**外连接**，在拼接的过程中它会取**两张表的键（key）的并集**进行拼接。

15. 常见函数：

    ```
    df = pd.DataFrame({'company' : [np.nan, 'A', 'A', 'C', 'A', 'C', 'A', 'B', 'B'],
                       'salary' : [43, 8, 28, 42, 33, 20, 48, 25, 39],
                       'age' : [21, 41, 26, 28, 26, 18, 43, 23, 18]})
    ```

    1. head()

       作用对象：`Series`和`DataFrame`

       主要用途：返回`DataFrame`的前N行。当数据量较大时，使用`.head()`可以快速对数据有个大致了解。

    2. info()

       作用对象：`DataFrame`

       主要用途：打印所用数据的一些基本信息，包括索引和列的数据类型和占用的内存大小。
       
    3. describe()

       作用对象：`Series`和`DataFrame`

       主要用途：生成描述性统计汇总，包括数据的计数和百分位数，有助于了解大致的数据分布

    4. value_counts()

       作用对象：`Series`

       主要用途：统计分类变量中每个类的数量，比如`company`中各个公司都有多少人

       主要参数：

       - normalize （*boolean, default False*）
         返回各类的占比
       - sort （*boolean, default True*）
         是否对统计结果进行排序
       - ascending （*boolean, default False*）
         是否升序排列

    5. isna()

       作用对象：`Series`和`DataFrame`

       主要用途：判断数据是否为缺失值，是的话返回`True`，否的话返回`False`
       
    6. any()

       作用对象：`Series`和`DataFrame`

       主要用途：any()一个序列中满足一个True，则返回True；大多数情况下数据量较大，不可能直接`isna()`后一个一个看是否是缺失值。`any()`和`isna()`结合使用可以判断某一列是否有缺失值。

    7. dropna()

       作用对象：`Series`和`DataFrame`

       主要用途：删掉含有缺失值的数据

    8. fillna()

       作用对象：`Series`和`DataFrame`

       主要用途：填充缺失数据

       主要参数：

       - value （*scalar, dict, Series, or DataFrame*）
         用于填充缺失值的值
       - method （*{‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None*）
         缺失值的填充方式，常用的是`bfill`后面的值进行填充，`ffill`用前面的值进行填充
       - inplace （*boolean, default False*）
         是否作用于原对象

    9. sort_index()

       作用对象：`Series`和`DataFrame`

       主要用途：对数据按照索引进行排序

       主要参数：

       - ascending （*boolean, default True*）
         是否升序排列
       - inplace （*boolean, default False*）
         是否作用于原对象

    10. sort_values()

        作用对象：`Series`和`DataFrame`

        主要用途：对`DataFrame`而言，按照某列进行排序（用`by`参数控制），对`Series`按数据列进行排序。

        主要参数：

        - by （*str or list of str*）
          作用于`DataFrame`时需要指定排序的列
        - ascending （*boolean, default False*）
          是否升序排列

    11. as_type()

        作用对象：`Series`和`DataFrame`

        主要用途：修改字段的数据类型，数据量大的情况下可用于**减小数据占用的内存**，多用于`Series`。

    12. rename()

        作用对象：`Series`,`DataFrame`(大多数情况下)

        主要用途：多用于修改`DataFrame`的列名

        主要参数：

        - columns （*dict-like or function*）
          指定要修改的列名以及新的列名，一般以字典形式传入
        - inplace （*boolean, default False*）
          是否作用于原对象

    13. set_index()

        作用对象：`DataFrame`

        主要用途：将`DataFrame`中的某一（多）个字段设置为索引

    14. reset_index()

        作用对象：`Series`,`DataFrame`

        主要用途：重置索引，默认重置后的索引为`0~len(df)-1`

        主要参数：

        - drop （*boolean, default False*）
          是否丢弃原索引，具体看下方演示
        - inplace （*boolean, default False*）
          是否作用于原对象

    15. drop_duplicates()

        作用对象：`Series`,`DataFrame`

        主要用途：去掉重复值，作用和`SQL`中的`distinct`类似

    16. drop()

        作用对象：`Series`,`DataFrame`

        主要用途：常用于删掉`DataFrame`中的某些字段

        主要参数：

        - columns （*single label or list-like*）
          指定要删掉的字段

    17. isin()

        作用对象：`Series`,`DataFrame`

        主要用途：常用于构建布尔索引，对`DataFrame`的数据进行条件筛选

    18. pd.cut()

        主要用途：将连续变量离散化，比如将人的年龄划分为各个区间

        主要参数：

        - x （*array-like*）
          需要进行离散化的一维数据
        - bins （*int, sequence of scalars, or IntervalIndex*）
          设置需要分成的区间，可以指定**区间数量**，也可以指定**间断点**
        - labels （*array or bool, optional*）
          设置区间的标签

    19. pd.qcut()

        主要用途：将连续变量离散化，区别于`pd.cut()`用具体数值划分，`pd.qcut()`使用**分位数**进行区间划分

        主要参数：

        - x （*array-like*）
          需要进行离散化的一维数据
        - q（*integer or array of quantiles*）
          设置需要分成的区间，可以指定**区间格式**，也可以指定**间断点**
        - labels （*array or boolean, default None*）
          设置区间的标签

    20. where()

        作用对象：`Series`,`DataFrame`

        主要用途：将不符合条件的值替换掉成指定值，相当于执行了一个`if-else`

        主要参数：

        - cond （*boolean Series/DataFrame, array-like, or callable*）
          用于筛选的条件
        - other（*scalar, Series/DataFrame, or callable*）
          对不符合`cond`条件的值（结果为为`False`），用`other`的值进行替代

    21. pd.concat()

        主要用途：将多个`Series`或`DataFrame`拼起来（横拼或者竖拼都可以）

        主要参数：

        - objs （*a sequence or mapping of Series or DataFrame objects*）
          用于拼接的`Series`或`DataFrame`，一般都放在一个列表中传入
        - axis （*0/’index’, 1/’columns’*）
          控制数据是横向拼接还是纵向拼接，默认为纵向拼接。
        - ignore_index （*bool, default False*）
          是否忽略原`Seires`或`DataFrame`内部的索引，如果为`True`则对拼接而成的数据生成新索引（0~n-1）

    22. pivot_table()

        作用对象：`DataFrame`

        主要用途：对`DataFrame`进行数据透视，相当于Excel中的数据透视表

        主要参数：

        - values （*column to aggregate, optional*）
          用于聚合运算的字段（数据透视的目标变量）
        - index （*column, Grouper, array, or list of the previous*）
          类比于数据透视表中的**行标签**
        - columns （*column, Grouper, array, or list of the previous*）
          类比于数据透视表中的**列标签**
        - aggfunc （ *function, list of functions, dict, default numpy.mean*）
          对values进行什么聚合运算

16. [时间处理](https://zhuanlan.zhihu.com/p/106675563)

    Pandas时序处理中最常见的两种数据类型为`datetime`和`timedelta`。一个`datetime`可以如下图所示：
    
    <img src="./imgs/v2-34978b4b0b063a429f38dfb1c72f0b73_720w.jpg" alt="v2-34978b4b0b063a429f38dfb1c72f0b73_720w" style="zoom:50%;" />

​		`resample`翻译过来是重采样的意思，官方文档中是这么描述`resample`的：

​		**`resample()`** is a **time-based** groupby

​		根据采样是**从低频到高频**还是**从高频到低频**可以分为**升采样**和**降采样**两种方式。[具体参考此文](https://zhuanlan.zhihu.com/p/106675563)

17.  [效率优化](https://zhuanlan.zhihu.com/p/97012199)
18. [`style`用法，它主要是用来美化`DataFrame`和`Series`的输出，能够更加直观地显示数据结果。](https://zhuanlan.zhihu.com/p/126223075)







