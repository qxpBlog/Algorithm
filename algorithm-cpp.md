# C++常用STL

![image-20230924202139079](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20230924202139079.png)

常用方法：

## （1）vector

```c++
// 变长数组，倍增的思想 
size() // 返回元素个数 
empty() // 返回是否为空 
clear() // 清空 
front()/back() // 返回第一个/最后一个数据
push_back()/pop_back() // 尾部插入一个数据/删除一个数据
[] // 支持比较运算，按字典序 
```

## （2）stack

```c++
size() 
empty() 
push() // 向栈顶插入一个元素 
top() // 返回栈顶元素 
pop() // 弹出栈顶元素
```

## （3）string

```c++
size()/length() // 返回字符串长度 
empty() 
clear() 
erase(iterator p) // 删除字符串中迭代器p所指向的字符
substr(起始下标，(子串长度)) 返回子串 
c_str()  // 返回字符串所在字符数组的起始地址
find (str, pos)	// 在当前字符串的pos索引位置（默认为0）开始，查找子串str，返回找到的位置索引，-1表示查找不到子串
```

## （4）map

```c++
// (1)map：
insert() // 插入的数是一个pair 
erase() // 输入的参数是pair或者迭代器 
find(key) // 在容器中搜索键值等于 k 的元素，如果找到，则返回一个指向该元素的迭代器，否则返回一个指向end的迭代器。 
[] // 注意multimap不支持此操作。 时间复杂度是 O(logn) 
lower_bound()/upper_bound()

// (2)unordered_map:
for(auto& [k,v]:map) // unordered_map遍历,其中如果只使用key或者value，那么另一个可以设为_
    

map和unordered_map区别:
1.map(内部用红黑树实现，具有自动排序（按键从小到大）功能)：
优点：内部用红黑树实现，内部元素具有有序性，查询删除等操作复杂度为O(logN)
缺点：占用空间，红黑树里每个节点需要保存父子节点和红黑性质等信息，空间占用较大。
2.unordered_map(内部用哈希表实现，内部元素无序杂乱)：
优点：内部用哈希表实现，查找速度非常快（适用于大量的查询操作）。
缺点：建立哈希表比较耗时。
```

## （5）queue

```c++
size() empty() 
push() // 向队尾插入一个元素 
front() // 返回队头元素 
back() // 返回队尾元素 
pop() // 弹出队头元素
```

## （6）deque

```c++
//双端队列
front()/back() 
push_back()/pop_back() 
push_front()/pop_front() 
begin()/end() size() empty() clear() 
[]
```

## （7）pair

```c++
// 只存储两个元素
pair<,> p = {element1, element2} // 赋值
pair<,> p = make_pair(element1, element2)
```

## （8）set

```c++
// set里面的元素不重复 且有序（基于平衡二叉树（红黑树），动态维护有序序列）
insert()  // 插入一个数 
find()  // 查找一个数 
count()  // 返回某一个数的个数，即1，若没有找到则返回0
s.rbegin() // 返回逆序迭代器，指向容器元素最后一个位置
clear()
erase() 
    (1) 输入是一个数x，删除所有x O(k + logn) 
    (2) 输入一个迭代器，删除这个迭代器 
lower_bound()/upper_bound() 
    lower_bound(x)  // 返回大于等于x的最小的数的迭代器 
    upper_bound(x)  // 返回大于x的最小的数的迭代器
```

## （9）priority_queue

```c++
// 优先队列具有队列的所有特性，包括基本操作，只是在这基础上添加了内部的一个排序，它本质是一个堆实现的

top() // 访问队头元素
pop()  // 弹出队头元素
push()  // 插入元素到队尾 (并排序)
emplace()  // 原地构造一个元素并插入队列
empty()  // 队列是否为空
size()  // 返回队列内元素个数
swap()  // 交换内容

// 设置优先级
priority_queue<int> pq; // 默认大根堆, 即每次取出的元素是队列中的最大值
priority_queue<int, vector<int>, greater<int> > q; // 小根堆, 每次取出的元素是队列中的最小值
```

## （10）tuple

```c++
// 元组tuole是pair的扩展，tuple可以声明二元组，也可以声明三元组。
// 可以封装不同类型任意数量的对象。

// 基础用法
tuple<int, int, string> t1;  // 声明一个空的tuple三元组
t1 = make_tuple(1, 1, "hahaha");  // 赋值
tuple<int, int, int, int> t2(1, 2, 3, 4);  // 创建的同时初始化
int first = get<0>(t);  // 获取tuple对象t的第一个元素
```



# C++常用库函数

```c++
(1) move(value) // 功能是把左值强制转化为右值，让右值引用可以指向左值。
    
(2) find(InputIterator first, InputIterator last, const T& val) // 如果找到元素，则返回一个指向该元素第一次出现的迭代器，否则返回last。
    
(3) abs(x) // 返回x的绝对值
    
(4) atoi(string s.c_str()) 或者 stoi(string s) // 字符串转整数
    
(5) to_string(int x) // 整数转字符串
    
(6) __builtin_popcount(x) // 返回x的二进制中1的个数
    
(7) pow(x, y) // 返回x的y次幂
    
(8) unique(it.begin(), it.end()) // “去除”容器或者数组中相邻元素的重复出现的元素,返回值是去重之后的尾地址。
    
(9) lower_bound(it.begin(), it.end(), val) // 有序的情况下，lower_bound返回第一个大于等于val值的位置。（通过二分查找）
    
(10) copy(iterator source_first, iterator source_end, iterator target_start); // 容器拷贝函数

(11) count(it.begin(), it.end(), val) // 查找val个数

(12) reverse(it.begin(), it.end()) // 反转数组
    
(13) isdigit(string)   // 判断字符是否是数字
    
(14) __lg(num)  // 返回数字num以2为底的对数，用于求num可以分解为2进制的最高位
```

# C++常用类

```
(1) stringstream ss
	ss << s; 
	while(ss >> ts){ }  // ss会将s按照空格进行分割，之后将分割后的每个单词放到ts字符串中进行处理
```

# C++常用常量

```
(1) 1LL   // 1的long long数据类型，1LL会在运算时把后面的临时数据扩容成long long类型，再在赋值给左边时转回对应类型。
```

# 算法基础

## （1）排序

### 快速排序

基本思想是：通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

复杂度：时间复杂度O(nlogn)

代码框架：

```c++
void quick_sort(vector<int> &q, int l, int r){
	// 递归终止的情况
	if(l >= r) return;
	// 选区分界线，这里选取数组中间的那个数
	int i = l - 1, j = r + 1, x = q[l + r >> 1];
	while(i < j){
		do i++; while(q[i] < x);
		do j--; while(q[j] > x);
		if(i < j) swap(q[i], q[j]);
	}
	quick_sort(q, l, i - 1);
	quick_sort(q, j + 1, r);
}
```

#### 题单

[215. 数组中的第K个最大元素 - 力扣（LeetCode）](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

### 归并排序

基本思想：先将待排序的数组不断拆分，直到拆分到区间里只剩下一个元素的时候。不能再拆分的时候。这个时候我们再想办法合并两个有序的数组，得到长度更长的有序数组。当前合并好的有序数组为下一轮得到更长的有序数组做好了准备。一层一层的合并回去，直到整个数组有序。

复杂度：时间复杂度O(nlogn)

代码框架：

```c++
void merge_sort(vector<int> &arr, int l, int r) { 
    // 递归终止情况
    if(l >= r) return;
    // 第一步：分成子问题
    int mid = l + r >> 1;
    
    // 第二步：递归处理子问题
    merge_sort(arr, l, mid);
    merge_sort(arr, mid + 1, r);
    
    // 第三步：合并子问题
    int k = 0, i = l, j = mid + 1;
    vector<int>  temp(r - l + 1);
    while(i <= mid && j <= r){
    	if(arr[i] <= arr[j]) {
    		temp[k++] = arr[i++];
    	} else{
    		temp[k++] = arr[j++];
    	}
    }
    while(i <= mid) temp[k++] = arr[i++];
    while(j <= r) temp[k++] = arr[j++];
    
    // 第四步：复制回原数组
    for(int i = l, j = 0; i <= r; i++, j++){ 
    	arr[i] = temp[j];
    }
}
```

#### 题单

[LCR 170. 交易逆序对的总数 - 力扣（LeetCode）](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/description/)

[315. 计算右侧小于当前元素的个数 - 力扣（LeetCode）](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/description/)

## （2）二分查找

定义:给定一个 n 个元素有序的整型数组 nums 和一个目标值 target  ，返回 nums 中与 target相关的位置。取自该题[34. 在排序数组中查找元素的第一个和最后一个位置 - 力扣（LeetCode）](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

代码框架:

```c++
// lower_bound 返回最小的满足 nums[i] >= target 的 i
// 如果数组为空，或者所有数都 < target，则返回 nums.size()
// 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]
// 返回大于等于target的第一个位置
// 其他几种情况可以用>=的结果进行转换：
// >target : >= (target + 1)
// <target : (>=target) - 1
// <=target : (>target) - 1
    // 闭区间写法
    int lower_bound(vector<int> &nums, int target) {
        int left = 0, right = (int) nums.size() - 1; // 闭区间 [left, right]
        while (left <= right) { // 区间不为空
            // 循环不变量：
            // nums[left-1] < target
            // nums[right+1] >= target
            int mid = left + (right - left) / 2;
            if (nums[mid] < target)
                left = mid + 1; // 范围缩小到 [mid+1, right]
            else
                right = mid - 1; // 范围缩小到 [left, mid-1]
        }
        return left; // 或者 right+1
    }

    // 左闭右开区间写法
    int lower_bound2(vector<int> &nums, int target) {
        int left = 0, right = nums.size(); // 左闭右开区间 [left, right)
        while (left < right) { // 区间不为空
            // 循环不变量：
            // nums[left-1] < target
            // nums[right] >= target
            int mid = left + (right - left) / 2;
            if (nums[mid] < target)
                left = mid + 1; // 范围缩小到 [mid+1, right)
            else
                right = mid; // 范围缩小到 [left, mid)
        }
        return left; // 或者 right
    }
    
	// 常用！！！！！
    // 开区间写法
    int lower_bound3(vector<int> &nums, int target) {
        int left = -1, right = nums.size(); // 开区间 (left, right)
        while (left + 1 < right) { // 区间不为空
            // 循环不变量：
            // nums[left] < target
            // nums[right] >= target
            int mid = left + (right - left) / 2;
            if (nums[mid] < target)
                left = mid; // 范围缩小到 (mid, right)
            else
                right = mid; // 范围缩小到 (left, mid)
        }
        return right; // 或者 left+1
    }

```

#### 题单

[704. 二分查找 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-search/)

[34. 在排序数组中查找元素的第一个和最后一个位置 - 力扣（LeetCode）](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

## （3）回溯

代码框架:

```c++
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素) {
        处理节点;
        backtracking(路径，选择列表); 
        回溯，撤销处理结果
    }
}
```

## （4）前缀和

用途：前缀和是一种预处理，可以用于快速计算某个区间的总和。

### 一维前缀和

说明：假设有一维数组a和前缀和数组preSum，那么它们之间的关系如下：

| preSum[i] = preSum[i - 1] + a[i] （当i = 0时，preSum[i] = a[i]） |
| :----------------------------------------------------------: |

代码框架:

```c++
// 预处理之后，求[l, r]的区间和为preSum[r] - preSum[l - 1];
for(int i = 0; i < n; i++){
	if(i == 0){
		preSum[i] = a[i];
	} else{
		preSum[i] = preSum[i - 1] + a[i];
	}
}
```



### 二维前缀和

说明：假设有二维数组a和前缀和数组preSum，那么它们之间的关系如下：

| preSum[ i] [ j] = preSum[i - 1] [ j] + preSum[ i] [ j - 1]  - preSum[i - 1] [ j - 1] + a[ i] [ j] （当i,j= 0时，preSum[ i] [ j] = a[ i] [ j]） |
| :----------------------------------------------------------: |

代码框架：

```c++
// a(i, j)的前缀和是x <= i && y <= j的全部元素之和，即（i, j）左上角的元素和
// 计算以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
// s = preSum[x2][y2] - preSum[x1 - 1][y2] - preSum[x2][y1 - 1] + preSum[x1 - 1][y1 - 1]
for(int i = 0; i < m; i++){
	for(int j = 0; j < n; j++){
		if(i == 0 && j == 0){
			preSum[i][j] = a[i][j];
		} else{
			preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + a[i][j];
		}
	}
}
```

### 题单

[1310. 子数组异或查询 - 力扣（LeetCode）](https://leetcode.cn/problems/xor-queries-of-a-subarray/description/)

[2615. 等值距离和 - 力扣（LeetCode）](https://leetcode.cn/problems/sum-of-distances/)

[2602. 使数组元素全部相等的最少操作次数 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-operations-to-make-all-array-elements-equal/description/)

## （5）差分

用途：用于**快速修改数组中某一段区间的值**。

思想：差分是前缀和的逆运算，对于一个数组a，其差分数组b的每一项都是a [ i ]和前一项a [ i − 1 ]的差。**注意：差分数组和原数组必须分开存放！！！！**

### 一维差分

说明：一维差分是指给定一个长度为n的序列a，要求支持操作pro(l,r,c)表示对a[l]~a[r]区间上的每一个值都加上或减去常数c，并求修改后的序列a。

作用：让一个序列中某个区间内的所有值均加上或减去一个常数。可以将对a数组任意区间的同一操作优化到O(1)。

代码框架：

```c++
// 假设数组a的差分数组为b，即b[i] = a[i] - a[i - 1], 当i = 0时，b[i] = a[i]
// 在区间[l, r]加上一个常数c，操作如下
void insert(int l, int r, int c){
    b[l] += c;
	b[r + 1] -= c;	
}
```

### 二维差分

说明：二维差分是指对于一个n*m的矩阵a，要求支持操作pro(x1,y1,x2,y2,a)，表示对于以(x1,y1)为左上角，(x2,y2)为右下角的矩形区域，每个元素都加上常数a。求修改后的矩阵a。

作用：与一维差分一样二维差分可以把对矩阵的同一操作优化到O(1)。

代码框架：

```c++
// 给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c
void insert(int x1, int y1, int x2, int y2, int c){ // 区间操作
	b[x1, y1] += c;
    b[x2 + 1, y1] -= c;
    b[x1, y2 + 1] -= c;
	b[x2 + 1, y2 + 1] += c;
}
// 构建差分数组
void getDif(){
	for(int i = 1; i <= m; i++){
		for(int j = 1; j <= n; j++){
			insert(i, j, i, j, a[i][j]);
		}
	}	
}
// 求原数组
void getStart(){
    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            b[i][j] = b[i-1][j] + b[i][j-1] - b[i-1][j-1] + b[i][j]; // 二维前缀和
        }
	}
}
```

### 题单

[1109. 航班预订统计 - 力扣（LeetCode）](https://leetcode.cn/problems/corporate-flight-bookings/)

[798. 得分最高的最小轮调 - 力扣（LeetCode）](https://leetcode.cn/problems/smallest-rotation-with-highest-score/)

## （6）位运算

说明：C++ 提供了按位与（&）、按位或（| ）、按位异或（^）、取反（~）、左移（<<）、右移（>>）这 6 种位运算符。  这些运算符只能用于整型操作数，即只能用于带符号或无符号的 char、short、int 与 long 类型。

做题思路：

​	（1）能不能拆位

​	（2）分析XOR、AND、OR的性质

​	（3）Trie字典树

### 位运算常用式子和性质

```
// XOR异或，本质上是模2加法
(1) (a&b)^(a&c) = a&(b^c) // 结合律
(2) (a ^ b ^ c) ^ (a) = b ^ c // 异或性质
(3) a ^ 0 = a  /  a ^ a = 0  // 异或性质

// AND与，与的数越多该数就越小
(1) x & -x  // 返回x的最后一位1

// OR或，或的数越多数越大

// 移位
(1) n >> k & 1  // 求n的第k位数字
(2) a & (1<<i) // 获取a的第i为数字
(3) a & ~(1<<i) // 将a的第i为清零
(4) a | (1<<i)  // 将a的第i为置1

// 其他
(1) c(x|y)+c(x&y)=c(x)+c(y) // 记 c(x)为x的二进制表示中1的个数

```

### 位运算常用函数

```

```

### 参考链接

[分享｜从集合论到位运算，常见位运算技巧分类总结！ - 力扣（LeetCode）](https://leetcode.cn/circle/discuss/CaOJ45/)

### 题单

[1835. 所有数对按位与结果的异或和 - 力扣（LeetCode）](https://leetcode.cn/problems/find-xor-sum-of-all-pairs-bitwise-and/)

[2354. 优质数对的数目 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-excellent-pairs/)

[2897. 对数组执行操作使平方和最大 - 力扣（LeetCode）](https://leetcode.cn/problems/apply-operations-on-array-to-maximize-sum-of-squares/)

[137. 只出现一次的数字 II - 力扣（LeetCode）](https://leetcode.cn/problems/single-number-ii/?envType=daily-question&envId=2023-10-15)

## （7）双指针

用途:

(1) 对于一个序列，用两个指针维护一段区间 

(2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作 

代码框架：

```c++
for (int i = 0, j = 0; i < n; i ++ ) { 
	while (j < i && check(i, j)) j++ ; 
	// 具体问题的逻辑 
}
```

### 题单

[11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/)

[167. 两数之和 II - 输入有序数组 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

[75. 颜色分类 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-colors/)

## （8）滑动窗口

概念：滑动窗口是一种基于双指针的一种思想，两个指针指向的元素之间形成一个窗口。

说明：在滑动窗口无效或者即将无效的情况下，更新维护的变量，并且收缩滑动窗口的左边界，两种情况如下：

​			滑动窗口的长度是**固定**的，使用**if条件**来更新 

​			滑动窗口的长度是**可变**的，使用**while条件**来更新

应用场景：

1. 一般给出的数据结构是数组或者字符串
2. 求取某个子串或者子序列最长最短等最值问题或者求某个目标值时
3. 该问题本身可以通过暴力求解

代码框架：

```c++
int slidwindow(vector<int>& nums,int k) { // k为窗口大小
	// （1）定义维护变量：
	int n = nums.size();
	unordered_map<char,int> m;	// 在需要统计字符或者数字出现的次数的时候，使用哈希表
	int sum = 0, res = 0; // 在需要记录整数数组中的子序列和或者其他求和时，使用sum记录每一次滑动窗口的子和，再利用res得到最大的或者最小的结果	 
	int len = 0,start = 0;	// 得到字符串的字串，len记录字串长度，start标识字串开始位置
	//（2）确定滑动窗口的边界，开始滑动：
	int left = 0, right = 0;
	while (right < n) {  
		//（3）更新答案：每进行一次滑动时，必须要更新的变量：如(1)的哈希表，sum,res,len与start等等
		if (满足条件) {   //满足某一种条件：例如滑动窗口的长度：right-left+1 与要求窗口大小k相等时，则更新res
			// 更新如res = max(res,sum)  
		}
		//（4）更新滑动窗口左端点left
		//1.滑动窗口的长度可变，使用while来更新窗口
		while (窗口合法) {  // 主要是用来求最小长度问题
			// 窗口left收缩
		}
		//2.滑动窗口的长度固定，使用if来更新窗口	
		if (right - left >= k-1) { //窗口大小固定为k，超过则需要left收缩
			left++; // 窗口left收缩
		}
		right++; //此处可以改为for循环
	}
	return res;
}
```

#### 定长滑动窗口题单

[1052. 爱生气的书店老板 - 力扣（LeetCode）](https://leetcode.cn/problems/grumpy-bookstore-owner/)

[2841. 几乎唯一子数组的最大和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-sum-of-almost-unique-subarray/description/)

[2461. 长度为 K 子数组中的最大和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-sum-of-distinct-subarrays-with-length-k/description/)

[2156. 查找给定哈希值的子串 - 力扣（LeetCode）](https://leetcode.cn/problems/find-substring-with-given-hash-value/description/)

#### 不定长滑动窗口题单

（1）求最长或者最大

[3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

[1004. 最大连续1的个数 III - 力扣（LeetCode）](https://leetcode.cn/problems/max-consecutive-ones-iii/)

[2024. 考试的最大困扰度 - 力扣（LeetCode）](https://leetcode.cn/problems/maximize-the-confusion-of-an-exam/description/)

（2）求最短或者最小

[209. 长度最小的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-size-subarray-sum/)

（3）求子数组个数

[713. 乘积小于 K 的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/subarray-product-less-than-k/)

（4）多指针滑动窗口

[930. 和相同的二元子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-subarrays-with-sum/)

## （9）双指针和滑动窗口的区别

一般来说，把长度不固定的叫做双指针，长度固定的叫做滑动窗口

## （10）离散化

概述：离散化的本质是建立了一段数列到自然数之间的映射关系（value -> index)，通过建立新索引，来缩小目标区间，使得可以进行一系列连续数组可以进行的操作比如二分，前缀和等

算法过程：离散化首先需要排序去重：

（1）排序：sort(alls.begin(),alls.end())

（2）去重：alls.earse(unique(alls.begin(),alls.end()),alls.end());

代码框架：

```c++
vector<int> start; // 原序列
vector<int> t(start); // 存储所有待离散化的值，初始化为原序列
vector<int> l; // 原序列离散化后的值
sort(t.begin(), t.end()); // 将所有值排序 
alls.erase(unique(t.begin(), t.end()), t.end()); // 去掉重复元素 

// 二分求出对应的离散化的值 
for(int i = 0; i < start.size(); i++){
    l[i] = lower_bound(t.begin(), t.end(), start[i]) - t.begin() + 1; // 二分查找,映射为1,2,...n
}
```

### 题单

[1331. 数组序号转换 - 力扣（LeetCode）](https://leetcode.cn/problems/rank-transform-of-an-array/description/)



## （11）区间合并

用途：将所有存在交集的区间合并

算法步骤：

（1）把要合并的区间按区间左端点从小到大排序
（2）用st和ed指针从前往后维护区间
（3）比较ed 和后一个区间的左端点，分情况更新ed和first

代码框架：

```c++
typedef pair<int, int> PII; //用来存储区间的左右两端first左端点，second右端点

void merge(vector<PII> &segs) {
    vector<PII> res;
    sort(segs.begin(), segs.end());   //把所有区间排序，sort默认是按照pair的first升序排序，如果first相同，则按照second进行升序排序
    int st = -2e9, ed = -2e9;   //左右端点初始值
    for(auto &seg : segs) {  //扫描所有区间
        if(ed < seg.first) {  //枚举的区间在我们维护区间左端，无交集
            if(st != -2e9) res.push_back({st, ed});  //将要维护的区间存入res中
            st = seg.first, ed = seg.second;
        } else{
        	ed = max(ed, seg.second);//有交点，将右端点更新为较长的一端
        }
    }
    if(st != -2e9) {
    	res.push_back({st, ed});  //判断是防止输入没有任何区间，如果有区间，将最后一个区间存入res中去
    }
    segs = res;  //将区间更新为res
}
```

### 题单

[56. 合并区间 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-intervals/)



# 数据结构

## （1）单调栈

定义:单调栈是栈数据结构的一种变形，在满足栈先进后出（FILO）的条件下，还要满足栈内元素遵循单调性。比如从栈底到栈顶元素保持递增的单调递增栈。

单调栈的维护（以单调递增栈为例）：当插入一个新元素时，为了维护栈内的单调性，我们将该元素与栈顶元素进行比较，若不满足单调性，就将栈顶元素弹出，不断重复，直到栈空或者满足单调性为止，最后再将该元素塞入栈顶。

代码框架：

```c++
stack<int> st；
st.push(a[0]);
for(int i=1;i<=n;i++){
    while(!st.empty() && st.top() >= a[i]) {
    	st.pop();
    }
    st.push(a[i]);
}
```

#### 题单：

[496. 下一个更大元素 I - 力扣（LeetCode）](https://leetcode.cn/problems/next-greater-element-i/description/)

[100048. 美丽塔 II - 力扣（LeetCode）](https://leetcode.cn/problems/beautiful-towers-ii/description/)

[42. 接雨水 - 力扣（LeetCode）](https://leetcode.cn/problems/trapping-rain-water/)

[456. 132 模式 - 力扣（LeetCode）](https://leetcode.cn/problems/132-pattern/description/)\

## （2）单调队列

定义：单调队列是一个限制只能 **队尾插入**，但是可以 **两端删除** 的 **双端队列** 。**单调队列** 存储的元素值，是从 **队首** 到 **队尾** 呈单调性的（要么单调递增，要么单调递减）。

作用：求区间最大值的问题->维护一个 **单调递减** 的队列。求区间最小值的问题->维护一个 **单调递增** 的队列。

​	同时**单调队列**是一种主要用于解决**滑动窗口**类问题的数据结构，即，在长度为 n 的序列中，求每个长度为 m的区间的区间最值。它的时间复杂度是 O(n) ，在这个问题中比O(n logn)的[ST表](https://zhuanlan.zhihu.com/p/105439034)和[线段树](https://zhuanlan.zhihu.com/p/106118909)要优。

代码框架：

```c++
//  求区间最大值
// m是滑动窗口大小
vector<int> a(n); 
deque<int> dq; // 存储的是编号
for (int i = 0; i < n; ++i) {
    if (!dq.empty() && i - dq.front() >= m) // 队列记录的区间大小超过滑动窗口大小，队首出列
        dq.pop_front();
    while (!dq.empty() && V[dq.back()] < a[i]) // 新进元素大于队尾元素，队尾元素出队（求区间最小值把这里改成>即可）
        dq.pop_back();
    dq.push_back(i); // 新元素入队
    if (i >= m - 1) // 当前窗口的最大值
        cout << a[dq.front()] << " ";
}
```

#### 题单

[LCR 184. 设计自助结算系统 - 力扣（LeetCode）](https://leetcode.cn/problems/dui-lie-de-zui-da-zhi-lcof/description/)（单调队列模板题）

[239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/)

[1438. 绝对差不超过限制的最长连续子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

## （3）并查集

定义：并查集是一种树型的数据结构，用于处理一些不相交集合的合并及查询问题（即所谓的并、查）。并查集的思想是用一个数组表示了整片森林（parent），树的根节点唯一标识了一个集合，我们只要找到了某个元素的的树根，就能确定它在哪个集合里。

作用：并查集的主要作用是**求连通分支数**

支持操作：

- **合并**（Union）：把两个不相交的集合合并为一个集合。
- **查询**（Find）：查询两个元素是否在同一个集合中。

代码框架：

```c++
class UnionFind {
public:
    //par数组用来存储根节点，par[x]=y表示x的根节点为y
    vector<int> par;
    UnionFind(int n){
        par = vector<int>(n);
        init();
    }
    //初始化
    void init(){
        for (int i = 0; i < par.size(); i++) {
            par[i]=i;
        }
    }
    //查找x所在集合的根
    int find(int x){
        // return par[x] == x ? x : find(par[x]); //无路径压缩
        // 路径压缩：查询过程中经过的每个元素都属于该集合，我们可以将其直接连到根节点以加快后续查询
        return par[x] == x ? x : par[x] = find(par[x]);
    }
    //合并x与y所在集合
    void unite(int x,int y){
        par[find(x)] = find(y);
    }
    bool isSame(int x, int y){
        return find(x) == find(y);
    }
};
```

#### 题单

[1971. 寻找图中是否存在路径 - 力扣（LeetCode）](https://leetcode.cn/problems/find-if-path-exists-in-graph/)

[684. 冗余连接 - 力扣（LeetCode）](https://leetcode.cn/problems/redundant-connection/)

[685. 冗余连接 II - 力扣（LeetCode）](https://leetcode.cn/problems/redundant-connection-ii/description/)

## （4）堆（优先队列）

基本思想：**堆**（Heap）是一类数据结构，它们拥有树状结构，且能够保证父节点比子节点大（或小）。当根节点保存堆中最大值时，称为**大根堆**；反之，则称为**小根堆**。

用途：**二叉堆**（Binary Heap）是最简单、常用的堆，是一棵符合堆的性质的**完全二叉树**。它可以实现 O(logn)地插入或删除某个值，并且 O(1) 地查询最大（或最小）值。

代码框架：

```c++
// 以大根堆为例

// h[N]存储堆中的值, h[1]是堆顶，x的左儿子是2x, 右儿子是2x + 1 
// ph[k]存储第k个插入的点在堆中的位置 
// hp[k]存储堆中下标是k的点是第几个插入的 
int h[N], ph[N], hp[N], size; 

// 交换两个点，及其映射关系 
void heap_swap(int a, int b) { 
	swap(ph[hp[a]],ph[hp[b]]); 
	swap(hp[a], hp[b]); 
	swap(h[a], h[b]); 
}

// 下滤 ，根节点破坏了堆的堆序性，重新构建成堆
void down(int u) { 
	int t = u; 
	// 选出左右最大子结点
	if (u * 2 <= size && h[u * 2] > h[t]) t = u * 2; 
	if (u * 2 + 1 <= size && h[u * 2 + 1] >	 h[t]) t = u * 2 + 1;
    
	if (u != t) { // 继续下滤，直到要插入的节点u大于其左右子结点
		heap_swap(u, t); 
		down(t); 
	} 
}

// 上滤，最后一个元素破坏了堆序性，重新构建成堆
// 主要用于插入数据
void up(int u) { 
	while (u / 2 && h[u] > h[u / 2]) { // 当前结点是否大于其父结点
		heap_swap(u, u / 2); 
		u >>= 1; 
	} 
}


// 建堆方法：
// （1）自顶向下：1.插入数据到堆的尾部；2.上滤操作。复杂度为O(nlogn)
// （2）自底向上：从最后一个父结点开始进行下滤操作，直至根节点下滤完成。复杂度为O(n)
// O(n)建堆，大根堆
for (int i = size / 2; i; i -- ) down(i);
```

#### 应用

##### 优先队列

主要操作：

（1）弹出最小元素

（2）插入队列

```c++
// 优先队列主要是使用小根堆

// （1）弹出最小元素，取出根节点，并将根节点与尾部结点交换，然后下滤
int pop(){
	int ans = h[1];
	swap(h[1], h[size--]); // 交换节点
	down(1); // 下滤,建堆
}

// （2）插入队列, 尾插然后上滤
void insert(int x){
	h[++size] = x;
	up(size);
}
```

##### 堆排序

算法思想：将根节点与尾结点交换，同时size--，之后将根节点进行下滤操作，重新建堆。重复上述操作，直至size为零

```c++
// 以大根堆为例
void heap_sort(){
	// 建堆
	for (int i = size / 2; i; i -- ) down(i);
	while(size > 0){
		swap(h[1], h[size--]);
		down(1);
	}
}
```

#### 题单

[373. 查找和最小的 K 对数字 - 力扣（LeetCode）](https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/description/)

[1439. 有序矩阵中的第 k 个最小数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/find-the-kth-smallest-sum-of-a-matrix-with-sorted-rows/description/)

[LCR 060. 前 K 个高频元素 - 力扣（LeetCode）](https://leetcode.cn/problems/g5c51o/description/)

[23. 合并 K 个升序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-k-sorted-lists/)

## （5）线段树

基础知识：线段树，是一种二叉搜索树。它将一段区间划分为若干单位区间，每一个节点都储存着一个区间。它功能强大，支持区间求和，区间最大值，区间修改，单点修改等操作。

基本思想：和分治思想很相像，线段树的每一个节点都储存着一段区间[L…R]的信息，其中叶子节点L=R。它的大致思想是：将一段大区间平均地划分成2个小区间，每一个小区间都再平均分成2个更小区间……以此类推，直到每一个区间的L等于R（这样这个区间仅包含一个节点的信息，无法被划分）。通过对这些区间进行修改、查询，来实现对大区间的修改、查询。

复杂度：一个包含n个区间的线段树，空间复杂度O(n)，查询的时间复杂度则为O(log(n+k))，其中k是符合条件的区间数量。

代码框架：

```c++
#define 100 maxd
vector<int> a(maxd); // 原数组
vector<int> tree(4 * maxd); // 树存储区间值
vector<int> mark(4 * maxd); // 懒标记/延迟标记
// （1）初始化建树
void build(int p,int l,int r){
    if(l==r){ // 到达叶子节点
        tree[p]=a[l];
        return;
    }
    int mid = (r+l) / 2;   // 防止爆范围
    build(p*2,l,mid);     // 左子树
    build(p*2+1,mid+1,r);  // 右子树
    tree[p]=tree[p*2]+tree[p*2+1];  // 该节点的值等于左右子节点之和
}
// （2）单点更新：将x位置的值加上num
void update(int p,int l,int r,int x,int num){
    if(x>r||x<l) return;  // x不在区间[l, r]中
    if(l==r && l==x){ //找到x位置
        tree[p] += num; //灵活变
        return;
    }
    int mid = (r+l)/2;
    update(p*2,l,mid,x,num);
    update(p*2+1,mid+1,r,x,num);
    tree[p]=tree[p*2]+tree[p*2+1];
}
// （3）区间修改：在区间[l,r]都加上d，p为当前节点，[cl,cr]为当前区间
void update(int l, int r, int d, int p = 1, int cl, int cr) {
    if (cl > r || cr < l) {   // 区间无交集
        return; // 剪枝
    } else if (cl >= l && cr <= r) {  // 当前节点对应的区间[cl,cr]包含在目标区间[l,r]中
        tree[p] += (cr - cl + 1) * d; // 更新当前区间的值
        if (cr > cl) {  // 如果不是叶子节点
            mark[p] += d; // 给当前区间打上标记
        }
    }  else {   // 与目标区间有交集，但不包含于其中
        int mid = (cl + cr) / 2;
        push_down(p, cr - cl + 1); // 懒标记向下传递
        update(l, r, d, p * 2, cl, mid); // 递归地往下寻找
        update(l, r, d, p * 2 + 1, mid + 1, cr);
        tree[p] = tree[p * 2] + tree[p * 2 + 1]; // 根据子节点更新当前节点的值
    }
}
void push_down(int p, int len){ // 懒标记向下传递
    mark[p * 2] += mark[p]; // 标记向下传递
    mark[p * 2 + 1] += mark[p];
    tree[p * 2] += mark[p] * (len - len / 2); // 往下更新一层
    tree[p * 2 + 1] += mark[p] * (len / 2);
    mark[p] = 0; // 清除标记
}
// （4）查询区间[x,y]和
int query(int p,int l, int r ,int x, int y){
    if(x<=l && r<=y) return tree[p]; // 区间[l, r]被包含于目的区间[x, y]
    if(x>r || y<l) return 0;  // 区间[x, y]与区间[l, r]无交集
    int mid = (r+l)/2;
    push_down(p, cl - cr + 1); // 懒标记向下传递
    return query(p*2,l,mid,x,y) + query(p*2+1,mid+1,r,x,y);
}
```

#### 参考链接

[看了也不会系列（一）------ 线段树 - AcWing](https://www.acwing.com/blog/content/1684/)

[算法学习笔记(14): 线段树 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/106118909)

#### 题单

[2213. 由单个字符重复的最长子字符串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-of-one-repeating-character/description/)

[2569. 更新数组后处理求和查询 - 力扣（LeetCode）](https://leetcode.cn/problems/handling-sum-queries-after-update/description/)

## （6）字典树（Trie树）

基础知识：Trie 树也叫前缀树，是一种多叉树的结构，每个节点保存一个字符，一条路径表示一个字符串。

用途：Trie 树最基本的作用是在树上查找字符串。其他用途如下所示：

- 1、维护字符串集合（即**字典**）。
- 2、向字符串集合中插入字符串（即**建树**）。
- 3、查询字符串集合中是否有某个字符串（即**查询**）。
- 4、统计字符串在集合中出现的个数（即**统计**）。
- 5、将字符串集合按字典序排序（即**字典序排序**）。
- 6、求集合内两个字符串的LCP（即**求最长公共前缀**）。

性质：

1. 根节点不包含字符，除根节点外每一个节点都只包含一个字符。
2. 从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串。
3. 每个节点的所有子节点包含的字符都不相同。

代码框架：

```c++
// 0号点既是根节点，又是空节点 
// son为Trie树，存储树中每个节点的子节点 
// cnt[]存储以每个节点结尾的单词数量，cnt[x] 表示：以 编号为 x 为结尾的字符串的个数
// idx为各个节点的编号，根节点的编号为0
// 注意：用数组存储Trie开销较大，可以使用链表的方式来优化空间开销，例如一下结构体：
/*
struct Node{
	Node *son[26]{}; // 结构体指针数组，初始化为nullptr
	int score = 0; // 作用与cnt数组一致 
}
*/

int N = 1e5 + 7;
int son[N][26], cnt[N], idx; 

// （1）建树：插入一个字符串 
void insert(string s){
    int p = 0; //指向根节点
    for(int i = 0; i < s.size(); i++){
        // 将当前字符转换成数字（a->0, b->1,...）
        int u = s[i] - 'a';
        if(!son[p][u]){  // 如果当前节点p的子结点不包含当前字符s[i]，为当前字符创建新的节点，保存该字符
            // 新节点编号为 idx + 1
            son[p][u] = ++idx;
        }
        p = son[p][u]; // 移动到当前符合当前字符的子结点
    }
    // 这个时候，p 等于字符串s的尾字符所对应的 idx
    // cnt[p]保存的是字符串 s 出现的次数
    // 故 cnt[p]++
    cnt[p] ++;
}

// （2）查询：查询字符串出现的次数 
int query(string s){
    int p = 0;  // 指向根节点
    for(int i = 0; i < s.size(); i++){
        // 将当前字符转换成数字（a->0, b->1,...）
        int u = s[i] - 'a';
        if(!son[p][u]){ // 如果走不通了，即树中没有保存当前字符，则说明树中不存在该字符串
            return 0;
        }
        // 指向下一个节点
        p = son[p][u];
    }
    // 循环结束的时候，p 等于字符串 s 的尾字符所对应的 idx
    // cnt[p] 就是字符串 s 出现的次数
    return cnt[p];
}
```

#### 参考链接

[AcWing 835. Trie树图文详解 - AcWing](https://www.acwing.com/solution/content/27771/)

[算法学习笔记(43): 字典树 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/173981140)

#### 题单

[208. 实现 Trie (前缀树) - 力扣（LeetCode）](https://leetcode.cn/problems/implement-trie-prefix-tree/description/)（模板题）

[745. 前缀和后缀搜索 - 力扣（LeetCode）](https://leetcode.cn/problems/prefix-and-suffix-search/description/)

[2416. 字符串的前缀分数和 - 力扣（LeetCode）](https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/)



# 字符串

## （1）字符串哈希

定义：字符串哈希实质上就是**把每个不同的字符串转成不同的整数**。

算法思想：将字符串看成P进制数，P的经验值是131或13331，取这两个值的冲突概率低 。字符串的哈希值为
$$
h[s] = ∑^n_{i=1}s[i] * p^{n - i}(mod \space M)
$$
小技巧：取模的数用2^64，这样直接用unsigned long long存储，溢出的结果就是取模的结果

代码框架：

```c++
typedef unsigned long long ULL; 
const int P = 131;
ULL h[N], p[N]; // h[k]存储字符串前k个字母的哈希值, p[k]存储 P^k mod 2^64 
// 初始化 ,h[n]为字符串str的哈希值
p[0] = 1; 
h[0] = 0;
int n = str.size();
for (int i = 1; i <= n; i ++ ) { 
	h[i] = h[i - 1] * P + str[i]; // 相当于前缀和
	p[i] = p[i - 1] * P; 
}
// 计算子串 str[l ~ r] 的哈希值 ,即求区间和
ULL get(int l, int r) { 
	return h[r] - h[l - 1] * p[r - l + 1]; 
}
// 判断两个子串是否相等
bool is_same(int l1, int l2, int r1, int r2){
    return get(l1, r1) == get(l2, r2);
}
```

#### 题单

[P3370 【模板】字符串哈希 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P3370)

[187. 重复的DNA序列 - 力扣（LeetCode）](https://leetcode.cn/problems/repeated-dna-sequences/description/)

[1044. 最长重复子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-duplicate-substring/)

## （2）KMP算法

算法思想：求出模式串的next数组，用于找出当发生不匹配时，模式串的指针回退的位置，此时主串位置不变。 next 数组定义为：next[i] 表示 p[0] ~ p[i] 这一个子串，使得 **前k个字符**（真前缀）恰等于**后k个字符** （真后缀）的最大的k. 特别地，k不能取i+1（因为这个子串一共才 i+1 个字符，自己肯定与自己相等，就没有意义了）。

用途：字符串匹配,时间复杂度为O(n+m) 

代码框架：

```c++
// s[]是主串，p[]是模式串，n是s的长度，m是p的长度 
const int N = 100010, M = 1000010; 
int n, m; 
int ne[N]; 
char s[M], p[N];
// 求模式串的next数组： 
ne[0] = 0;
for (int i = 1, j = 0; i < m; i++) { 
	while(j && p[i] != p[j]) j = ne[j - 1]; 
	if(p[i] == p[j]) j++; 
	ne[i] = j; 
}
// 匹配 
for (int i = 0, j = 0; i < n; i++) { 
	while(j && s[i] != p[j]) j = ne[j - 1]; 
	if(s[i] == p[j]) j++;  // 匹配
	if(j == m) {   // 匹配成功
        j = ne[j - 1]; // 继续匹配
		// return i - m + 1; // 返回匹配成功的位置
	}
}
```

#### 题单

[P3375 【模板】KMP - 洛谷 ](https://www.luogu.com.cn/problem/P3375)

[28. 找出字符串中第一个匹配项的下标 - 力扣（LeetCode）](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

#### 参考链接

[ 如何更好地理解和掌握 KMP 算法? - 知乎 (zhihu.com)](https://www.zhihu.com/question/21923021/answer/37475572)

## （3）回文子串

### （1）中心拓展

算法思想：枚举每一个可能的回文中心，然后用两个指针分别向左右两边拓展，当两个指针指向的元素相同的时候就拓展，否则停止拓展。

算法复杂度：时间复杂度O(n ^ 2), 空间复杂度O(1)。

代码框架：

```c++
int countSubstrings(string s) { // 求回文子串个数
	int n = s.size(), ans = 0;
	for (int i = 0; i < 2 * n - 1; ++i) {
    	int l = i / 2, r = l + i % 2;
        while (l >= 0 && r < n && s[l] == s[r]) {
        	--l;
             ++r;
             ++ans;
        }
     }
     return ans;
}
```

### （2）Manacher（马拉车）算法

算法思想：

​	1.通过插入其他符号统一奇偶长度的回文处理（每个回文串的中心都可以落在字符上，即都是奇回文串，若中心字符是其他符号那么复原到原字符串就是偶回文串；若中心字符就是原字符串的字符，那么该回文子串复原到原字符串就是奇回文串）。

​	2.维护一个数组max_extend[], 其中m_e[i] 代表以位置i为中心的最长回文子串到中心的距离，即[i - m_e[i] + 1, i + m_e[i] - 1]为最长回文子串。

​	3.维护一个能覆盖到最远（右）的回文中心 j

​		case1:最远覆盖未覆盖到当前位置，无对称信息可用，暴力中心扩展

​		case2: a 对称位置i^回文较短，即i^的最长回文子串左端点位于最远覆盖之内，直接使用m_e[i] = m_e[i^]  = m_e[j * 2 - i]；

​					b 对称位置i^回文较长，即i^的最长回文子串左端点位于最远覆盖之外，直接使用m_e[i] = m_e[j] + j - i；

​					c 对称位置i^刚好位于最远覆盖边界，令m_e[i] = m_e[j] + j - i后，继续中心暴力扩展。

用途：给定一个字符串S，求改字符串的**最长回文子串**和回文子串数目。时间复杂度和空间复杂度均为O(n)。

代码框架：

```c++
string func(string s){
	int n = s.size();
	string t = "$*"; // 向原字符串字符之间添加*，首部添加$尾部添加#,构造新的字符串；s中位置i的字符在t中位置变为(i + 1) * 2
	for(char c:s){
		t += c;
    	t += '*';
	}
	t += '#';
	int m = t.size();
	vector<int> m_e(m, 0);
	m_e[1] = 1;
	int j = 1; // 维护最远覆盖中心下标
	for(int i = 2; i <= 2 * n; i++){  // 从s的第一个字符遍历到最后一个字符
        // j * 2 - i即i关于j的对称位置i^
		int cur_max_extend = j + m_e[j] > i ? min(m_e[j * 2 - i], j + m_e[j] - i) : 1;
		while(t[i - cur_max_extend] == t[i + cur_max_extend]){
			cur_max_extend++;
		}
		if(i + cur_max_extend > j + m_e[j]){  // 更新最远覆盖中心下标
			j = i;
		}
		m_e[i] = cur_max_extend;
	}
	int mx = 0, p = -1;
	for(int i = 2; i <= 2 * n; i++){
		if(m_e[i] - 1 > mx){
			mx = m_e[i] - 1;  // 原字符串中回文串长度为m_e[i] - 1
			p = (i - m_e[i]) / 2; // 原字符串对应第一个字符的位置
            // 若求回文子串个数则：cnt += m_e[i] / 2;
		}
	}
	return s.substr(p, mx);
}
```

#### 参考链接

[[Algorithm\][018] 最长回文子串 Manacher [OTTFF]_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1AX4y1F79W/?vd_source=354d995c8d91b9fafe3f6274db4fbc10)

[【算法ABC × Manim】探究字符串的对称奥秘，小学三年级都能听懂的马拉车算法_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ft4y117a4/?vd_source=354d995c8d91b9fafe3f6274db4fbc10)

[(6 封私信 / 32 条消息) 有什么浅显易懂的Manacher Algorithm讲解？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/37289584)

[Manacher 算法详解：O(n) 复杂度求最长回文子串_manacher算法求最长回文子串-CSDN博客](https://blog.csdn.net/synapse7/article/details/18908413)

[算法学习笔记(83): Manacher算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/404216115)

### 题单

[5. 最长回文子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-palindromic-substring/)

[647. 回文子串 - 力扣（LeetCode）](https://leetcode.cn/problems/palindromic-substrings/description/)

[2472. 不重叠回文子字符串的最大数目 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/)

# 数学

## （1）基础知识

### 模运算

基础概念：给定一个正整数p，任意一个整数n，一定存在等式 n = kp + r ；其中k、r是整数，且 0 ≤ r < p，称呼k为n除以p的商，r为n除以p的余数

相关性质：

```
// 常用性质
（1） (a + b) % p = (a % p + b % p) % p 
（2） (a * b) % p = (a % p * b % p) % p 
```

参考链接：

[模（Mod）运算的运算法则、公式（不包括同余关系） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/60533528)

[模运算及其性质_模的基本性质-CSDN博客](https://blog.csdn.net/openstack_/article/details/103951102)



## （2）数论

### 质数

基础概念：质数又称素数。一个大于1的自然数，除了1和它自身外，不能被其他自然数整除的数叫做质数；否则称为合数（规定1既不是质数也不是合数）。

#### 1. 试除法判定质数

代码框架：

```c++
bool is_prime(int x){
	if(x < 2) return false;
	for(int i = 2; i <= x / i; i++){
		if(x % i == 0){
			return false;
		}
	}
	return true;
}
```

参考链接：

[试除法判定质数-CSDN博客](https://blog.csdn.net/m0_56015524/article/details/119780265)

#### 2. 试除法分解质因数

基础概念：质因数（素因数或质因子）在数论里是指能整除给定正整数的**质数**。根据算术基本定理，**任何正整数皆有独一无二的质因子分解式**。

代码框架：

```c++
vector<int> divide(int x){  // x为正整数
	vector<int> res;
	for(int i = 2; i <= x / i; i++){
		if(x % i == 0){  // 如果 i 能够整除 x，说明 i 为 x 的一个质因子。
			while(x % i == 0) x /= i;
			res.push_back(i);
		}
	}
	if(x > 1){ // 说明再经过操作之后 x 留下了一个素数
		res.push_back(x);
	}
	return res;
}
```

参考链接：

[分解质因数 - OI Wiki (oi-wiki.org)](https://oi-wiki.org/math/number-theory/pollard-rho/)

#### 3. 埃氏筛法求质数

基础知识：埃氏筛法又称埃拉托斯特尼筛法。主要是用来求小于等于n有多少个质数。时间复杂度O(n * log(log n))

算法思想：对于任意一个大于1的正整数n，那么它的x倍就是合数（x > 1）。利用这个结论，我们可以避免很多次不必要的检测。从小到大考虑每个数，然后同时把当前这个数的所有（比自己大的）倍数记为合数，那么运行结束的时候没有被标记的数就是素数了。

代码框架：

```c++
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉
void get_primes(int n) {
    for (int i = 2; i <= n; i ++ ) {
        if(st[i]) continue;
        primes[cnt++] = i;
        for (int j = i + i; j <= n; j += i) {
        	st[j] = true;
        }  
    }
}
```

#### 4. 线性筛法求质数

基础知识：线性筛法也叫欧拉筛法。时间复杂度为O(n)。

算法思想：**让每一个合数被其最小质因数筛到**。

代码框架：

```c++
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉
void get_primes(int n){
    for (int i = 2; i <= n; i ++ ){
        if (!st[i]) primes[cnt++] = i;
        for (int j = 0; primes[j] <= n / i; j++){
            st[primes[j] * i] = true; // 以primes[j]为最小质因子筛
            if (i % primes[j] == 0) {
                break;
            }
        }
    }
}
```

参考链接：

[欧拉筛法（线性筛）的学习理解-CSDN博客](https://blog.csdn.net/qq_39763472/article/details/82428602)

#### 题单

（1）求质数

[P3383 【模板】线性筛素数 - 洛谷 ](https://www.luogu.com.cn/problem/P3383)

（2）质因数分解

[2507. 使用质因数之和替换后可以取到的最小值 - 力扣（LeetCode）](https://leetcode.cn/problems/smallest-value-after-replacing-with-sum-of-prime-factors/)

[2584. 分割数组使乘积互质 - 力扣（LeetCode）](https://leetcode.cn/problems/split-the-array-to-make-coprime-products/)

[793. 阶乘函数后 K 个零 - 力扣（LeetCode）](https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/)



### 约数

基础知识：约数，又称[因数](https://baike.baidu.com/item/因数/9539111?fromModule=lemma_inlink)。[整数](https://baike.baidu.com/item/整数/1293937?fromModule=lemma_inlink)a除以整数b(b≠0) 除得的[商](https://baike.baidu.com/item/商/3820976?fromModule=lemma_inlink)正好是整数而没有[余数](https://baike.baidu.com/item/余数/6180737?fromModule=lemma_inlink)，我们就说a能被b整除，或b能整除a。a称为b的[倍数](https://baike.baidu.com/item/倍数/7827981?fromModule=lemma_inlink)，b称为a的约数。

#### 1. 试除法求所有约数

算法思想：从 **1 ~ n** 依次[枚举](https://so.csdn.net/so/search?q=枚举&spm=1001.2101.3001.7020)试除每一个数，判断该数是否可以整除 **n**，如果可以整除 **n**，说明该数是 **n** 的约数。试除法还可以优化一下，即我们可以发现约数都是成对出现的，如果 **d** 可以整除 **n**，那么 **n / d** 也可以整除 **n**。

代码框架：

```c++
vector<int> get_divisors(int x) {
    vector<int> res;
    for (int i = 1; i <= x / i; i ++){  // 只需要枚举到√n即可
        if (x % i == 0) {
            res.push_back(i);
            if (i != x / i) res.push_back(x / i);  // i如果为√n,那么只需要添加一次就行了
        }
    }
    sort(res.begin(), res.end());
    return res;
}
```

参考链接：

[试除法求数的约数-CSDN博客](https://blog.csdn.net/qq_41575507/article/details/115590408)



### 最大公约数

#### 1. 欧几里得算法

基础知识：欧几里得算法又称**辗转相除法**，是指用于计算两个非负整数a，b的最大公约数。计算公式**gcd(a,b) = gcd(b,a mod b)**。

算法思想：两个整数的最大公约数等于其中较小的数和两数相除余数的最大公约数。

代码框架：

```c++
int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
}
```

### 快速幂

代码框架：

```c++
// 求 m^k mod p，时间复杂度 O(logk)。
// 二进制取幂
int qmi(int m, int k, int p) {
    int res = 1 % p, t = m;
    while (k) {
        if(k&1) res = res * t % p; // 当前二进制末位是否为1
        t = t * t % p;
        k >>= 1;
    }
    return res;
}
```

参考资料

[算法学习笔记(4)：快速幂 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/95902286)

### 扩展欧几里得算法

基础知识：裴蜀定理，又称贝祖定理，如果a、b是正整数，那么一定存在整数x、y使得ax+by=gcd(a,b)。换句话说，如果ax+by=m有解，那么m一定是gcd(a,b)的若干倍。

主要用途：

​	1、求解不定方程

​	2、求解模的逆元

​	3、求解线性同余方程

代码框架：

```c++
// 求x, y，使得ax + by = gcd(a, b)
// 返回gcd(a, b)的值
int exgcd(int a, int b, int &x, int &y) {  
    if (!b) {
        x = 1; 
        y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}
```

参考链接：

[扩展欧几里得算法详解-CSDN博客](https://blog.csdn.net/destiny1507/article/details/81750874)

[扩展欧几里得算法求系数x，y_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1bp4y1S7bW/?spm_id_from=333.337.search-card.all.click&vd_source=354d995c8d91b9fafe3f6274db4fbc10)

#### 题单

[P1082 [NOIP2012 提高组\] 同余方程 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1082)



### 逆元

基础概念：如果一个线性同余方程 ax ≡ 1(mod b)  ，则 x 称为 a 的逆元，记作 a^(-1)或者inv(a)。

**求解逆元的三种方法：**

#### 1. 扩展欧几里得算法

#### 2. 费马小定理+快速幂

基础知识：费马小定理：若 **p 是质数**，且gcd(a, p) = 1, 则有a^(p - 1) ≡ 1(mod p)。因此inv(a)  = a^(p - 2) 

代码框架：

```c++
qmi(a, p - 2, p); // 求a的p-2次方幂模p，具体快速幂方法见上
```

#### 3. 线性求逆元

代码框架：

```
// 求1到n每个数在模数p下的逆元
inv[1] = 1;
for (int i = 2; i <= n; ++i) {
  inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
```

#### 参考链接

[【乘法逆元】模运算下除以一个整数，就相当于乘以这个整数的乘法逆元_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1kv4y1P7WV/?spm_id_from=333.337.search-card.all.click&vd_source=354d995c8d91b9fafe3f6274db4fbc10)

[算法学习笔记(9)：逆元 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/100587745)

[乘法逆元 - OI Wiki (oi-wiki.org)](https://oi-wiki.org/math/number-theory/inverse/)

#### 题单

## （3）组合数学

### 求组合数

#### 1. 递推法

算法思想：从a个数中选出b个数，那么对于第a个数，如果选，那么剩余的数要从a-1中选择b-1；如果不选那么要从a-1个数中选择b个数，结果为两种情况的总和。

适用范围：n，m大小在10^3左右。

代码框架：

```c++
// （1）非递归形式
// c[n][m]表示从n个元素中取m个的方案数
for(int i= 0; i <= N; i++){
	for(int j = 0; j <= i; j++){
        if(!j) c[i][j] = 1;
        else c[i][j] = c[i - 1][j] + c[i - 1][j - 1];
    }
}
// 递归形式
int c(int n, int m){
    if(m == 0 || n == m) return 1;
    return c(n - 1, m) + c(n - 1, m - 1);
}
```

#### 2. 预处理+逆元

#### 3. Lucas定理

#### 4. 分解质因数

####  参考链接

[【精选】【数论】求组合数的四种方法_复杂组合数-CSDN博客](https://blog.csdn.net/qq_58207591/article/details/121106898)

#### 题单



### 容斥原理

基础知识：先不考虑重叠的情况，把包含于某内容中的所有对象的数目先计算出来，然后再把计数时重复计算的数目[排斥](https://baike.baidu.com/item/排斥/10717656?fromModule=lemma_inlink)出去，使得计算的结果既无遗漏又无重复，这种计数的方法称为容斥原理。

公式：
$$
|A∪B∪C| = |A| + |B| + |C| - |A∩B| - |A∩C| - |B∩C| - |A∩B∩C|
$$


## （4）线性代数







## （5）博弈论

### 公平组合游戏

定义：

- 游戏有两个人参与，二者轮流做出决策，双方均知道游戏的完整信息；
- 任意一个游戏者在某一确定状态可以作出的决策集合只与当前的状态有关，而与游戏者无关；
- 游戏中的同一个状态不可能多次抵达，游戏以玩家无法行动为结束，且游戏一定会在有限步后以非平局结束。

定理：定义必胜状态为先手必胜的状态，必败状态为先手必败的状态。                                                                                                           	定理 1：没有后继状态的状态是必败状态
	定理 2：一个状态是必胜状态当且仅当存在至少一个必败状态为它的后继状态
	定理 3：一个状态是必败状态当且仅当它的所有后继状态均为必胜状态
	对于定理 1，如果游戏进行不下去了，那么这个玩家就输掉了游戏
	对于定理 2，如果该状态至少有一个后继状态为必败状态，那么玩家可以通过操作到该必败状态；此时对手的状态为必败状态——对手必定是失败的，而相反地，自己就获得了胜利
	对于定理 3，如果不存在一个后继状态为必败状态，那么无论如何，玩家只能操作到必胜状态；此时对手的状态为必胜状态——对手必定是胜利的，自己就输掉了游戏

类型：

# 动态规划(DP)

## （1）背包问题

![image-20230918153814927](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20230918153814927.png)

### 01背包

定义：有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大。

二维dp数组代码框架：

```c++
//（1）dp数组下标含义:dp[i][j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。
//（2）递推公式：dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
//（3）初始化：
vector<vector<int>> dp(n, vector<int>(w + 1, 0));
for (int j = weight[0]; j <= bagweight; j++) {
    dp[0][j] = value[0];
}
//（4）遍历数组：
for(int i = 1; i < weight.size(); i++) { // 遍历物品
        for(int j = 0; j <= bagweight; j++) { // 遍历背包容量
            if (j < weight[i]) {
            	dp[i][j] = dp[i - 1][j];
            } else {
            	dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
            }
        }
}
```

一维dp数组代码框架:

```c++
//（1）dp数组下标含义:dp[j]表示容量为j的背包，所背的物品价值可以最大为dp[j]。
//（2）递推公式：dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
//（3）初始化:全部初始化为0即可
//（4）遍历顺序：
for(int i = 0; i < weight.size(); i++){
	for(int j = w; j >= weight[i]; j--){
		dp[j] = max(dp[j],dp[j - weight[i]] + value[i]);
	}
}
```

#### 题单

[416. 分割等和子集 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-equal-subset-sum/description/)

[279. 完全平方数 - 力扣（LeetCode）](https://leetcode.cn/problems/perfect-squares/solutions/822940/wan-quan-ping-fang-shu-by-leetcode-solut-t99c/)

[1049. 最后一块石头的重量 II - 力扣（LeetCode）](https://leetcode.cn/problems/last-stone-weight-ii/discussion/)

[494. Target Sum - 力扣（LeetCode）](https://leetcode.cn/problems/target-sum/)

[474. 一和零 - 力扣（LeetCode）](https://leetcode.cn/problems/ones-and-zeroes/description/)

### 完全背包

定义:有N件物品和一个最多能背重量为W的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品都有无限个（也就是可以放入背包多次）**，求解将哪些物品装入背包里物品价值总和最大。

代码框架:

```c++
for(int i = 0; i < n; i++){
	for(int j = w[i]; j <= w; j++){
		dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
	}
}
```

#### 题单

[518. 零钱兑换 II - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change-ii/)

[322. 零钱兑换 - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change/)

[377. Combination Sum IV - 力扣（LeetCode）](https://leetcode.cn/problems/combination-sum-iv/)

### 多重背包

定义:有N种物品和一个容量为W 的背包。第i种物品最多有Si件可用，每件耗费的空间是Wi ，价值是Vi 。求解将哪些物品装入背包可使这些物品的耗费的空间 总和不超过背包容量，且价值总和最大。

代码框架：

```c++
//朴素算法
for(int i=1;i<=n;++i){
    for(int j=W;j>=w[i];--j){ 
        for (int k = 0; k <= s[i] && k * w[i] <= j; k++) {
            dp[j] = Math.max(dp[j], dp[j - k*w[i]] + k*v[i]);
        }
    }
}
//二进制优化
int g
for(int i=0;i<n;++i){
    int a = w[i]; //体积
    int b = v[i]; //价值
    int c = s[i]; //数量
    for(int j=1;j<=c;j<<=1){
        w[g]=j*a;
        v[g++]=j*b;
        c-=j;
    }
    if(c>0){
        w[g]=c*a;
        v[g++]=c*b;
    }
}
for(int i=1;i<=g;++i){
    for(int j=m;j>=w[i];--j){ //与01背包相似
        dp[j]=Math.max(dp[j],dp[j-w[i]]+v[i]);
    }
}
```

## （2）树形DP

定义：树形 DP，即在树上进行的 DP。由于树固有的递归性质，树形 DP 一般都是递归进行的。

#### 题单

[337. 打家劫舍 III - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber-iii/description/)

[P1352 没有上司的舞会 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1352)

## （3）线性DP

### 最长上升子序列（LIS）

定义:给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

代码框架：

```c++
int n = nums.size();
vector<int> dp(n);
int res = 0; //长度
for(int i = 0; i < n; i++){
	//dp[i]中第i个数字一定要选择
	dp[i] = 1;
	for(int j = 0; j < i; j++){
        if(nums[j] < nums[i]){
            dp[i] = max(dp[i], dp[j] + 1);
        }
	}
    if(dp[i] > res){
    	res = dp[i];
    }
}
```

#### 题单

[300. 最长递增子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-increasing-subsequence/description/)

### 最长公共子序列（LCS）

定义：给定两个字符串 s1 和 s2，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

代码框架:

```c++
int lenx = s1.size();
int leny = s2.size();
for(int i=1;i<=lenx;++i){
    for(int j=1;j<=leny;++j){
        if(s1[i-1] == s2[j-1]) {
        	dp[i][j] = dp[i-1][j-1]+1;
        }
        else {
        	dp[i][j] = max(dp[i-1][j],dp[i][j-1]); //当当前对应位置不等时，就看是不要串a的，还是不要串b中的
        }
    }
}
```

#### 题单

[1143. 最长公共子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-common-subsequence/)

[1035. 不相交的线 - 力扣（LeetCode）](https://leetcode.cn/problems/uncrossed-lines/description/)

[72. 编辑距离 - 力扣（LeetCode）](https://leetcode.cn/problems/edit-distance/description/)

[583. 两个字符串的删除操作 - 力扣（LeetCode）](https://leetcode.cn/problems/delete-operation-for-two-strings/description/)

## （4）区间DP

1.定义：在一段区间上进行动态规划，一般做法是由长度较小的区间往长度较大的区间进行递推，最终得到整个区间的答案，而边界就是长度为1以及2的区间。常见转移方程如下：

dp(i,j) = min{dp(i,k-1) + dp(k,j)} + w(i,j)   (i < k <= j)。

其中`dp(i,j)`表示在区间`[i,j]`上的最优值，`w(i,j)`表示在转移时需要额外付出的代价，min也可以是max。

2.性质：

**合并**：即将两个或多个部分进行整合，当然也可以反过来；

**特征**：能将问题分解为能两两合并的形式；

**求解**：对整个问题设最优值，枚举合并点，将问题分解为左右两个部分，最后合并两个部分的最优值得到原问题的最优值。

#### 题单

[516. 最长回文子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-palindromic-subsequence/description/)

[1039. 多边形三角剖分的最低得分 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/)

## （5）计数DP

概念：计数类DP就是常说的统计可行解数目的问题，区别于求解最优解，此类问题需要统计所有满足条件的可行解，而求解最优值的DP问题往往只需要统计子问题时满足不漏的条件即可，但是计数类DP需要满足不重不漏的条件，是约束更高的。

算法应用：一个正整数 n 可以表示成若干个正整数之和，我们将这样的一种表示称为正整数 n 的一种划分。现在给定一个正整数 n，请你求出 n共有多少种不同的划分方法。

代码框架：

```c++
// 状态转移方程 ：f[j]=(f[j-i]+f[j])
// 完全背包的写法 
const int M=1e9+7; 
int f[1010],n; 
int main() { 
	cin >> n; 
	f[0]=1; 
	for (int i = 1; i <= n; i ++ ) {
		for (int j = i; j <= n; j ++ ){ 
			f[j]=(f[j-i]+f[j])%M; 
		} 
	}
	cout<<f[n]<<endl; 
	return 0; 
}
```

#### 题单

[62. 不同路径 - 力扣（LeetCode）](https://leetcode.cn/problems/unique-paths/description/)

## （6）数位DP

基础概念：数位是指把一个数字按照个、十、百、千等等一位一位地拆开，关注它每一位上的数字。如果拆的是十进制数，那么每一位数字都是 0~9，其他进制可类比十进制。

用途：用来解决一类特定问题，这种问题比较好辨认，一般具有这几个特征：

1. 要求统计满足一定条件的数的数量（即最终目的为计数）；
2. 这些条件经过转化后可以使用「数位」的思想去理解和判断；
3. 输入会提供一个数字区间（有时也只提供上界）来作为统计的限制；
4. 上界很大（比如10^18)，暴力枚举验证会超时。

​	例如给定一个闭区间[ l , r ]，让你求这个区间中满足某种条件的数的总数。

算法思想:从高到低枚举每一位，再考虑每一位都可以填哪些数字，最后利用通用答案数组统计答案。

技巧：

​	1. 把一个区间内的答案拆成两部分相减，即ans[l，r] = ans[0，r] - ans[0，l]。

代码框架:

```c++
// 记忆化搜索
// 如果一个正整数每一个数位都是 互不相同 的，我们称它是 特殊整数 。返回1-n之间的特殊整数
int countSpecialNumbers(int n) {
    auto s = to_string(n);
    int m = s.length(), memo[m][1 << 10];
    memset(memo, -1, sizeof(memo)); // -1 表示没有计算过
    // mask表示前面选过的数字集合，换句话说，第 i位要选的数字不能在 mask中，比如说你选择填入数字5，那么mask第5位不能为1,这里	   // 是题目要求，不同的题目要求整数所具备的性质不一样，所需参数也不一样
    // mask为可选参数，根据题目要求不同可以删改
    function<int(int, int, bool, bool)> f = [&](int i, int mask, bool lim, bool is_num) -> int {
        if (i == m)
            return is_num; // is_num 为 true 表示得到了一个合法数字,将0排除
        if (!lim && is_num && memo[i][mask] != -1) // 之前已经计算过
            return memo[i][mask];
        int res = 0;
        if (!is_num) // 可以跳过当前数位
            res = f(i + 1, mask, false, false);
        int up = lim ? s[i] - '0' : 9; // 如果前面填的数字都和 n 的一样，那么这一位至多填数字 s[i]（否则就超过 n 啦）
        for (int d = 1 - is_num; d <= up; ++d) // 枚举要填入的数字 d，如果is_num为true那么从0开始枚举，否则从1开始枚举	
            if ((mask >> d & 1) == 0) // d 不在 mask 中
                res += f(i + 1, mask | (1 << d), lim && d == up, true);
        if (!lim && is_num)
            memo[i][mask] = res;
        return res;
    };
    return f(0, 0, true, false);
}
参考链接：
https://leetcode.cn/problems/count-special-integers/solutions/1746956/shu-wei-dp-mo-ban-by-endlesscheng-xtgx/
```

#### 题单

[902. 最大为 N 的数字组合 - 力扣（LeetCode）](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)

[1012. 至少有 1 位重复的数字 - 力扣（LeetCode）](https://leetcode.cn/problems/numbers-with-repeated-digits/)

[233. 数字 1 的个数 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-digit-one/)

[2376. 统计特殊整数 - 力扣（LeetCode）](https://leetcode.cn/problems/count-special-integers/description/)

[600. 不含连续1的非负整数 - 力扣（LeetCode）](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/)

[1397. 找到所有好字符串 - 力扣（LeetCode）](https://leetcode.cn/problems/find-all-good-strings/)

#### 参考链接

[数位 DP - OI Wiki (oi-wiki.org)](https://oi-wiki.org/dp/number/)

[[Algorithm\][016] 数位 DP [OTTFF]_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1MT4y1376C/?vd_source=354d995c8d91b9fafe3f6274db4fbc10)

[数位 DP 通用模板_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1rS4y1s721/?vd_source=354d995c8d91b9fafe3f6274db4fbc10)



## （7）状态压缩DP

基础概念：**状态压缩**就是使用某种方法，简明扼要地以最小代价来表示某种状态，通常是用一串01数字（二进制数）来表示各个点的状态。这就要求使用状态压缩的对象的点的状态必须只有两种，0 或 1；当然如果有三种状态用**三进制**来表示也未尝不可。

算法思想：其精髓就是将所有物品的状态（一般是选或不选，用01表示，当然也有特殊情况）压缩成一个整数，进行状态的转移并节约空间。状压 DP 本质上就是在集合与集合之间转移。

应用场景：常用于处理包含排列的问题等。

1. 数据范围
   范围在20左右时正常的状压~~当然凡事有二般情况~~
   很多时候会有一些NP问题会用状压求解。
2. 是否可以压缩
   一般的状态压缩都是选择或者不选择，放或者不放，遇见这种东西一般时状压。八皇后问题也是一个状压的代表。
3. 常见题目模型
   比如TSP，覆盖问题之类的。经常会有这种模型的题出现就可以使用状压。

代码框架：

```c++
/* 
	以旅行商问题（TSP）为例：说是有一个商人想要旅行各地并进行贸易。各地之间有若干条单向的通道相连，商人从一个地方出发，想要用最短的路程把所有地区环游一遍，请问环游需要的最短路程是多少？在这题当中，我们假设商人从0位置出发，最后依然回到位置0。
*/
int n; // 节点从0到n-1
int dp[n][1<<n]; // dp[i][s]表示当前在i节点，已经访问过节点集合为s，所有经过路径的最小权值和
memset(dp, INT_MAX, sizeof(dp));
for(int s = 0; s < (1 << n); s++){  // 枚举所有状态
    for(int i = 0; i < n; i++){
		if(s&(1<<i)){ // 节点i在集合s中
            for(int j = 0; j < n; j++){
                if(!(s&(1<<j)) && g[i][j]){ // 如果节点j不在集合s中，并且节点i到j有通路
                    dp[i][s|(1<<j)] = min(dp[i][s|(1<<j)], dp[i][s] + g[i][j]);
                }
            }
        }
    }
}
```

#### 参考链接

[浅谈状压DP - yijan's blog - 洛谷博客 (luogu.com.cn)](https://www.luogu.com.cn/blog/yijan/zhuang-ya-dp)

[状态压缩DP --算法竞赛专题解析（15)](https://blog.csdn.net/weixin_43914593/article/details/106432695)

[[Algorithm\][014] 状态压缩DP [OTTFF]_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wE41147Bw/?spm_id_from=333.999.0.0&vd_source=354d995c8d91b9fafe3f6274db4fbc10)

#### 题单

[U204941 蒙德里安的梦想 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/U204941)

[U122241 最短Hamilton路径 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/U122241)

[动态规划课程状压dp例题_竞赛题单_ACM/NOI/CSP基础提高训练专区_牛客竞赛OJ (nowcoder.com)](https://ac.nowcoder.com/acm/problem/collection/808)

[动态规划课程状压dp习题_竞赛题单_ACM/NOI/CSP基础提高训练专区_牛客竞赛OJ (nowcoder.com)](https://ac.nowcoder.com/acm/problem/collection/810)

[691. 贴纸拼词 - 力扣（LeetCode）](https://leetcode.cn/problems/stickers-to-spell-word/)

# 图论

## （1）图的存储

### 邻接矩阵

```
g[a][b] 存储边a->b的距离
```

### 邻接表

```
// 又叫做链式向前星存储（头插法）
// 首先 idx 是用来对边进行编号的，然后对存图用到的几个数组作简单解释：
// he 数组：存储是某个节点所对应的边的集合（链表）的头结点；
// e  数组：用于访问某一条边指向的节点；
// ne 数组：由于是以链表的形式进行存边，该数组就是用于找到下一条边；
// w  数组：用于记录某条边的权重为多少。
int[] he = new int[N], e = new int[M], ne = new int[M], w = new int[M];
int idx = 0; // 为每一条边进行编号
// 初始化
memset(he, -1, sizeof he);
// 添加一条a到b的边，权重为c
void add(int a, int b, int c) {
    e[idx] = b;
    ne[idx] = he[a];
    he[a] = idx;
    w[idx] = c;
    idx++;
}
// 遍历a的出边
for(int i = he[a]; i != -1; i = ne[i]){
	int j = e[i], v = w[i]; // j为a的出边， v为权重
}
```

### 类

```
class Edge {
    // 代表从 a 到 b 有一条权重为 c 的边
    int a, b, c;
};
vector<Edge> g;
```

## （2）图的遍历

### DFS（深度优先搜索）

代码框架：

```c++
void dfs(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本节点所连接的其他节点) {
        处理节点;
        dfs(图，选择的节点); 
        回溯，撤销处理结果
    }
}
void main_function(参数){
    for(遍历所有节点){
		if(节点未遍历){
            dfs(该节点)
        }
    }
}
```

### BFS（广度优先搜索）

代码框架：

```c++
int dir[4][2] = {0, 1, 1, 0, -1, 0, 0, -1}; // 表示四个方向
void bfs(vector<vector<char>>& grid, vector<vector<bool>>& visited, int x, int y) {
	int m = grid.size(),n = grid[0].size();
    queue<pair<int, int>> que; // 定义队列
    que.push({x, y}); // 起始节点加入队列
    visited[x][y] = true; // 只要加入队列，立刻标记为访问过的节点
    while(!que.empty()) { // 开始遍历队列里的元素
        auto cur = que.front(); // 从队列取元素
        que.pop(); 
        int x = cur.first;
        int y = cur.second; // 当前节点坐标
        for (int i = 0; i < 4; i++) { // 开始想当前节点的四个方向左右上下去遍历
            int tx = x + dir[i][0];
            int ty = y + dir[i][1]; // 获取周边四个方向的坐标
            if (tx >= 0 && tx < m && ty >= 0 && ty < n && !visited[tx][ty]) { // 如果节点没被访问过
                que.push({tx, ty});  // 队列添加该节点为下一轮要遍历的节点
                visited[tx][ty] = true; // 只要加入队列立刻标记，避免重复访问
            }
        }
    }
}
```

## （3）最短路径

### Floyd算法

原理：Floyd本质上是一个**动态规划**的思想，每一次循环更新**经过前k个节点，i到j的最短路径**。

用途：Floyd算法是求解**多源最短路**时通常选用的算法，经过一次算法即可求出任意两点之间的最短距离，并且**可以处理有负权边**的情况（但**无法处理负权环**）。时间复杂度为O(n3)。

代码框架：

```c++
#define N 100
const int INF = 0x3f3f3f3f;
int d[N][N];
// 代码初始化,共有n个顶点
for(int i = 0; i < n; i++){
	for(int j = 0; j < n; j++){
		d[i][j] = i == j ? 0 : INF;
	}
}
// 将每条边的值加入到dis中去，这里不再赘叙
// Floyd算法
for(int k = 0; k < n; k++){
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
		}
	}
}
```

### Dijkstra算法

原理：将结点分成两个集合：已确定最短路长度的点集（记为S集合）的和未确定最短路长度的点集（记为T集合）。一开始所有的点都属于T集合。

初始化dis(S) = 0，其他点的S均为INT_MAX。

然后重复这些操作：

1. 从T集合中，选取一个最短路长度最小的结点，移到S集合中。
2. 对那些刚刚被加入S集合的结点p的所有出边执行松弛操作。松弛操作即更新dis(T)的值，具体公式为：dis(T) = min(dis(T), dis(p) + w(p)(T))。

直到T集合为空，算法结束。

用途：基于**贪心**思想的一种求解 **非负权图** 上**单源最短路径**的算法。暴力的话O(n * n)。

代码框架：

```c++
// 假设共有n个节点
#define N 100
vector<vector<int>> w; // 储存每条边的权重
int dis[N]; // 储存开始节点到其他节点的最短距离
bool s[N]; // 储存已找到最短路径的节点
int dijkstra(int x, int des){ // 假设x节点为开始节点,des目的节点
	// 初始化dis
	memset(dis, 0x3f, sizeof(dis));
	dis[x] = 0;
	for(int i = 0; i < n; i++){
		int p = -1; // 本次循环加入到S集合的节点
		for(int j = 0; j < n; j++){ // 在集合T中寻找距离最小的节点
			if(!s[j] && (p == -1 || dis[p] > dis[j])){
				t = j;
			}
		}
		//用p更新其他节点到开始节点x的最短距离
		for(int j = 0; j < n; j++){
			dis[j] = minx(dis[j], dis[p] + w[p][j]);
		}
		s[p] = true;
	}
	return dis[des];
}
```

### Bellman-Ford算法

原理：逐基于动态规划，遍的对图中每一个边去迭代计算起始点到其余各点的最短路径（松弛操作），执行n - 1遍，最终得到起始点到其余各点的最短路径。

用途：Bellman–Ford 算法是一种基于**松弛**（relax）操作的**单源最短路算法**，可以处理负权边与负权回路。**对于一个不包含负权环的V个点的图，任意两点之间的最短路径至多包含V-1条边。**如果存在负权环，每次在负权环上走一圈都会使环上的每一个点的距离减少，因此不存在最短路径。时间复杂度为O(nm)，其中n为节点个数，m为边数。

​	可以解决**边数限制**的最短路径问题，SPFA无法代替。

代码框架：

```c++
// 假设共有n个节点，m条边
struct Edge { // 边u表示出点，v表示入点，w表示边的权重 
	int u, v, w; 
}edges[m];
int dis[100]; // 储存开始节点到其他节点的最短距离

int Bellman_Ford(int start, int des){ // 开始节点为start，结束节点为des
	memset(dis, 0x3f, sizeof(dis));
	dis[start] = 0;
	for(int i = 0; i < n; i++){ // 迭代n 次
		for(int j = 0; j < m; j++){
            if(i == n - 1 && dis[edges[i].v] > dis[edges[i].u] + edges[i].w){// 判断是否出现负权回路
                // 最短距离发生更新，说明存在负权回路，返回-1
                return -1;
			}
			dis[edges[j].v] = min(dis[edges[j].v], dis[edges[j].u] + edges[j].w);
		}
	}
	return dis[des] > 0x3f3f3f3f / 2 ? -1 : dis[des];
}
```

### SPFA算法

原理：初始时将起点加入队列。每次从队列中取出一个元素，并对所有与它相邻的点进行修改，若某个相邻的点修改成功，则将其入队。直到队列为空时算法结束。算法的流程为：

1. 将除源点之外的所有的点当前距离初始化为无穷，并标记为未入队。源点的当前距离为0，将源点入队。
2. 取出队首u，遍历u的所有出边，检查是否能更新所连接的点v的当前距离。如果v的当前距离被更新并且v不在队中，则将v入队。重复该操作直到队列为空。
3. 检查是否存在负权环的方法为：记录一个点的入队次数，如果超过n-1次说明存在负权环，因为最短路径上除自身外至多n-1个点，故一个点不可能被更新超过n-1次。

用途：SPFA是队列优化的Bellman-Ford算法，因此用途与Bellman-Ford算法用途相同，但是时间复杂度更低。平均复杂度O(m)，最坏复杂度O(n * m)。

代码框架：

```c++
// 假设共有n个节点，m条边
#define N 100
struct Edge { // v表示出边，w表示边的权重 
	int v, w; 
};
vector<Edge> e[n]; // 与各个节点相连的边
int dis[N]; // 储存开始节点到其他节点的最短距离
bool s[N]; // 判断节点是否在队列中
int cnt[N]; // 记录边数
int spfa(int start, int des){ // 开始节点为start，结束节点为des
	memset(dis, 0x3f, sizeof(dis));
	dis[start] = 0;
	queue<int> q;
	q.push(start);
	s[start] = true;
	while(!q.empty()){
		int u = q.front();
		q.pop();
		s[u] = false;
		for(auto &ed : e[u]){ // 遍历节点p能直接到达的节点,松弛操作
			int v = ed.v, w = ed.w;
			if(dis[v] > dis[u] + w){
				dis[v] = dis[u] + w;
				cnt[v] = cnt[u] + 1;
				if(cnt[v] >= n){ // 出现负权回路
					return -1;
				}
				if(!s[v]){
					q.push(v);
					s[v] = true;
				}
			}
		}
	}
	return dis[des] > 0x3f3f3f3f / 2 ? -1 : dis[des];
}
```

### 总结

（1）单源最短路：给定V中的一个顶点，称为源。要计算从源到其他所有各顶点的最短路径长度。这里的长度就是指路上各边权之和。这个问题通常称为单源最短路径 问题。

​	所有边权都是正数：

​		朴素Dijkstra算法 O(n^2) 适合稠密图，贪心思想

​		堆优化版的Dijkstra算法 O(mlogn)适合稀疏图，贪心思想

​	存在负权边：

​		Bellman-ford O(nm)，动态规划思想

​		SPFA 一般：O(m)，最坏O(nm)

（2）多源汇最短路：任意两点最短路径被称为多源最短路径，即给定任意两个点，一个出发点，一个到达点，求这两个点的之间的最短路径，就是任意两点最短路径问题：Floyd算法 O(n^3)

### 题单

[743. 网络延迟时间 - 力扣（LeetCode）](https://leetcode.cn/problems/network-delay-time/)

[787. K 站中转内最便宜的航班 - 力扣（LeetCode）](https://leetcode.cn/problems/cheapest-flights-within-k-stops/)

[1928. 规定时间内到达终点的最小花费 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/)

## （4）最小生成树

问题描述：在连通网的所有生成树中，所有边的代价和最小的生成树，称为最小生成树。

### Kruskal算法

原理：基本思想是从小到大加入边，是个贪心算法。我们将图中的每个边按照权重大小进行排序，每次从边集中取出权重最小且两个顶点都不在同一个集合的边加入生成树中。**注意：如果这两个顶点都在同一集合内，说明已经通过其他边相连，因此如果将这个边添加到生成树中，那么就会形成环。**这样反复做，直到选出n-1条边。时间复杂度为O(m*logm)

算法过程：此算法可以称为**“加边法”**，初始最小生成树边数为0，每迭代一次就选择一条满足条件的最小代价边，加入到最小生成树的边集合里。 具体过程如下所示：
	步骤1：先对图中所有的边按照权值进行排序
	步骤2：如果当前这条边的两个顶点不在一个集合里面，那么就用并查集的Union函数把他们合并在一个集合里面(也就是把他们放在最小生成树里面)，如果在一个并查集里面，我们就舍弃这条边，不需要这条边。
	步骤3：一直执行步骤2，知道当边数等于n-1（n为节点个数），那就说明这n个顶点就连合并在一个集合里面了；如果边数不等于顶点数目减去1，那么说明这些边就不连通，即无法构成最小生成树。

代码框架：

```c++
int n, m; // n是点数，m是边数 
int p[n + 1]; // 并查集的父节点数组 
struct Edge{ // 存储边  
	int a, b, w; 
	bool operator< (const Edge &W)const { 
		return w < W.w; 
	} 
}edges[m]; 

int find(int x){ // 并查集核心操作 
     return p[x] == x ? x : p[x] = find(p[x]);
}
void init(){ // 初始化并查集 
	for(int i = 1; i <= n; i++){
		p[i] = i;
	}
}
int kruskal() {
	sort(edges, edges + m); 
	init();
	int res = 0, cnt = 0; 
	for (int i = 0; i < m; i++) { // 从m条边选择n-1条边
		int a = edges[i].a, b = edges[i].b, w = edges[i].w; 
		a = find(a), b = find(b); 
		if (a != b)  { // 如果两个连通块不连通，则将这两个连通块合并
			p[a] = b; 
			res += w; 
			cnt++; 
		} 
	}
	if (cnt < n - 1) return INF; 
	return res; 
}
```

### Prim算法

原理：基本思想是从一个结点开始，不断加点。因此该算法可以称为“加点法”，每次迭代选择代价最小的边对应的点，加入到最小生成树中。算法从某一个顶点s开始，逐渐长大覆盖整个连通网的所有顶点。时间复杂度为O(n * n + m)。

算法过程：

1. 用两个集合A{}，B{}分别表示找到的点集，和未找到的点集；
2. 我们以A中的点为起点a，在B中找一个点为终点b，这两个点构成的边（a，b）的权值是其余边中最小的
3. 重复上述步骤#2，直至B中的点集为空，A中的点集为满

代码框架：

```c++
int n; // 节点个数
vector<vector<int>> g(n, vector<int>(n)); // 邻接矩阵，存储所有边
vector<int> dis(n); // 存储其他节点到当前最小生成树的距离
vector<bool> v(n); // 存储每个节点是否加入到最小生成树中

// 如果图不连通，则返回INF(值是0x3f3f3f3f), 否则返回最小生成树的树边权重之和
int prim(){
	const int inf = 0x3f3f3f3f;
	memset(dis, 0x3f, sizeof dis);
	int res = 0;
	for(int i = 0; i < n; i++){
		int p = -1;
		for(int j = 0; j < n; j++){
			if(!v[j] && (p == -1 || dis[j] < dis[p])){
				p = j;
			}
		}
		if(i && dis[p] == inf){ // dis[p] = inf说明找到的节点与最小生成树不连通，但是当i = 0说明是第一个节点，不考虑连通
			return inf;
		}
		if(i){
			res += dis[p];
		}
		v[p] = true;
		for(int j = 0; j < n; j++){
			dis[j] = min(dis[j], g[p][j]); // 与Dijkstra算法的区别
		}
	}
    return res;
}

```

### 题单

[1584. 连接所有点的最小费用 - 力扣（LeetCode）](https://leetcode.cn/problems/min-cost-to-connect-all-points/)

## （5）拓扑排序

概念：一个**有向图**，如果图中有入度为 0 的点，就把这个点删掉，同时也删掉这个点所连的边。一直进行上面的处理，如果所有点都能被删掉，则这个图可以进行拓扑排序。**拓扑排序**是对**DAG**（有向无环图）上的节点进行排序，使得对于每一条有向边u->v，u 都在v之前出现。简单地说，是在不破坏节点**先后顺序**的前提下，把**DAG**拉成一条**链**。

算法过程：构造拓扑序列步骤

1. 从图中选择一个入度为零的点。
2. 输出该顶点，从图中删除此顶点及其所有的出边。

重复上面两步，直到所有顶点都输出，拓扑排序完成，或者图中不存在入度为零的点，此时说明图是有环图，拓扑排序无法完成，陷入死锁。

代码框架：

```
int n;
vector<int> g[MAXN]; // 储存节点出边
int in[MAXN];  // 存储每个结点的入度
bool toposort() {
  vector<int> l; // 排序结果
  queue<int> q;
  for (int i = 0; i < n; i++){ // 入度为0的节点入队
  	if (in[i] == 0) {
  		q.push(i);
  	}
  }
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    l.push_back(u);
    for (auto v : G[u]) { // 删除与节点u直接相连的边
      if (--in[v] == 0) { // 出现新入度为零的节点入队
        q.push(v);
      }
    }
  }
  return l.size() == n;
}
```

### 题单

[207. 课程表 - 力扣（LeetCode）](https://leetcode.cn/problems/course-schedule/)

[210. 课程表 II - 力扣（LeetCode）](https://leetcode.cn/problems/course-schedule-ii/)

## （6）二分图

基础知识：二分图是一类特殊的图，它可以被划分为**两个部分**，每个部分**内部的顶点互不相连**。**注意:**一个图是二分图，当且仅当他不含奇数边数的圈

![image-20231010164049576](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20231010164049576.png)

### 染色法判别二分图

用途：判断一个图是否为二分图

算法思想：一条边连接的两个顶点颜色不相同，用两种颜色对所有边进行染色。染色结束后，若所有相邻顶点的颜色不同，那么就是二分图。具体算法过程如下：

- 我们规定1或2代表一个点属于两个集合。
- 首先我们任选一个点染色成1，把和它相连的所有点染色成2。
- 然后再把所有和染色成2的点相邻的点染色成1
- 在每一次染色点时首先要判断一下这个点是否被染色过，如果被染色过并且和上一个点颜色相同，则代表染色失败，该图不是二分图。

代码框架：

```c++
int n;   // n表示点数 
int h[N], e[M], ne[M], idx; // 邻接表存储图 
int color[N]; // 表示每个点的颜色，-1表示未染色，0表示白色，1表示黑色 

// 参数：u表示当前节点，c表示当前点的颜色 
bool dfs(int u, int c) { 
	color[u] = c; 
	for (int i = h[u]; i != -1; i = ne[i]) { 
		int j = e[i]; 
		if (color[j] == -1) { 
			if (!dfs(j, !c)) {  // 将j染色成与u不一样的颜色
                return false; 
             }
		} else if (color[j] == c) {  // u和j两个点的颜色一致
            return false;
        }
	}
	return true; 
}
bool check() { 
	memset(color, -1, sizeof color); 
	bool flag = true; 
	for (int i = 1; i <= n; i ++ ) 
		if (color[i] == -1) {  // 未染色
			if (!dfs(i, 0)) { 
				flag = false; 
				break; 
			} 
        }
	}
	return flag; 
}
```

#### 题单

[785. 判断二分图 - 力扣（LeetCode）](https://leetcode.cn/problems/is-graph-bipartite/description/)

### 匈牙利算法

相关概念：

​	（1）匹配：在图论中，一个「匹配」是一个边的集合，其中任意两条边都没有公共顶点。

​	（2）最大匹配：一个图所有匹配中，所含匹配边数最多的匹配，称为这个图的最大匹配。

​	（3）完美匹配：如果一个图的某个匹配中，所有的顶点都是匹配点，那么它就是一个完美匹配。

​	（4）交替路：从一个未匹配点出发，依次经过非匹配边、匹配边、非匹配边…形成的路径叫交替路。

​	（5）增广路：从一个未匹配点出发，走交替路，如果途径另一个未匹配点（出发的点不算），则这条交替 路称为增广路（agumenting path）。

用途：求二分图的**最大匹配数**和**最小点覆盖数**。

​	最小点覆盖数：我们想找到**最少**的一些**点**，使二分图所有的边都**至少有一个端点**在这些点之中。倒过来说就是，删除包含这些点的边，可以删掉所有边。

​	**柯尼希定理（Konig）**：一个二分图中的最大匹配数**等于**这个图中的最小点覆盖数。

算法思想：

​	1. 匈牙利算法寻找最大匹配，就是通过不断**寻找原有匹配M的增广路径**，因为找到一条M匹配的增广路径，就意味着一个更大的匹配M', 其恰好比M 多一条边。

2. 对于图来说，最大匹配不是唯一的，但是最大匹配的大小是唯一的。

代码框架：

```c++
// n1表示第一个集合中的点数，n2表示第二个集合中的点数
int n1, n2; 
// 邻接表存储所有边，匈牙利算法中只会用到从第一个集合 指向第二个集合的边，所以这里只用存一个方向的边
int h[N], e[M], ne[M], idx; 
int match[N];  // 存储第二个集合中的每个点当前匹配的第一个集合中的点是哪个 
bool st[N]; // 表示第二个集合中的每个点是否已经被遍历过 

bool find(int x) { 
	// 遍历自己喜欢的女孩 
	for (int i = h[x]; i != -1; i = ne[i]) { 
        int j = e[i]; 
        if (!st[j]) {   // 如果在这一轮模拟匹配中,这个女孩尚未被预定
            st[j] = true;  // 那x就预定这个女孩了 
            // 如果女孩j没有男朋友，或者她原来的男朋友能够预定其它喜欢的女孩。配对成功 
            if (match[j] == 0 || find(match[j])) { 
                match[j] = x; 
                return true; 
            } 
        } 
	}
	//自己中意的全部都被预定了。配对失败。 
	return false; 
}
// 求最大匹配数，依次枚举第一个集合中的每个点能否匹配第二个集合中的点 
int res = 0; 
for (int i = 1; i <= n1; i++) { 
	memset(st, false, sizeof st); 
	if (find(i)) res++; 
}
```

#### 题单

[P3386 【模板】二分图最大匹配 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P3386)

[1349. 参加考试的最大学生数 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-students-taking-exam/)

[LCP 04. 覆盖 - 力扣（LeetCode）](https://leetcode.cn/problems/broken-board-dominoes/)





## （7）基环树

# 其他

## （1）记忆化搜索

概念：记忆化搜索是动态规划的一种实现方式。在递归函数中, 在函数返回前，**记录函数的返回结果**。在下一次以同样参数访问函数时直接返回记录下的结果。也就是对递归树进行剪枝，遇到已经计算过的节点就不再继续往下计算，直接返回储存在hash table中的值。

本质：**递归搜索+保存计算结果=记忆化搜索**

算法思想：我们在使用记忆化搜索解决问题的时候，其基本步骤如下：

​	（1）写出问题的动态规划「状态」和「状态转移方程」。

​	（2）定义一个缓存（数组或哈希表），用于保存子问题的解。

​	（3）定义一个递归函数，用于解决问题。在递归函数中，首先检查缓存中是否已经存在需要计算的结果，如果存在则直接返回结果，否则进行计算，并将结果存储到缓存中，再返回结果。

​	（4）在主函数中，调用递归函数并返回结果。

代码框架：

```c++
unordered_map<int,int> record;
int memo[m][n]; // 用来保存计算结果
int dfs(int i) {
    // （1）判断i的合法性
    // （2）判断该状态是否已经被计算过了，若已经计算过，那么则直接返回record[i]
    // （3）之前未计算过该状态，现在计算
    // （4）保存该计算结果至record，并返回。
}
```
