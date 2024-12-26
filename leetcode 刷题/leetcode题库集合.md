

### 1两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数组下标。

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

输入：nums = [3,2,4], target = 6
输出：[1,2]

输入：nums = [3,3], target = 6
输出：[0,1]

```cpp
/*
方法一：暴力枚举，枚举数组中的每一个数 x，寻找数组中是否存在 target - x

时间复杂度：O(N^2)，其中 N是数组中的元素数量。最坏情况下数组中任意两个数都要被匹配一次。

空间复杂度：O(1)。
*/
class Solution
{
public:
    vector<int> twoSum(vector<int> &nums, int target)
    {
        int n = nums.size();
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                if (nums[i] + nums[j] == target)
                {
                    return {i, j};
                }
            }
        }
        return {};
    }
};

/*
方法二：哈希表
时间复杂度：O(N)，其中 N 是数组中的元素数量。对于每一个元素 x，我们可以 O(1) 地寻找 target - x。

空间复杂度：O(N)，其中 NNN 是数组中的元素数量。主要为哈希表的开销

*/
class Solution
{
public:
    vector<int> twoSum(vector<int> &nums, int target)
    {
        unordered_map<int, int> hashtable;

        //
        for (int i = 0; i < nums.size(); ++i)
        {
            auto it = hashtable.find(target - nums[i]);
            if (it != hashtable.end())
            {
                return {it->second, i};
            }
            hashtable[nums[i]] = i;
        }
        return {};
    }
};

/*
方法三：双指针
*/

```



### 2两数相加

两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字

将两个数相加，并以相同形式返回一个表示和的链表

你可以假设除了数字 0 之外，这两个数都不会以 0 开头

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

输入：l1 = [0], l2 = [0]
输出：[0]

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]

分析：
时间复杂度：O(max⁡(m,n))，其中 m,n为两个链表的长度。我们要遍历两个链表的全部位置，而处理每个位置只需要 O(1) 的时间。
空间复杂度：O(max⁡(m,n))。答案链表的长度最多为较长链表的长度 +1

```cpp
//
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *head = nullptr, *tail = nullptr;
        int carry = 0;
        while (l1 || l2) {
            int n1 = l1 ? l1->val: 0;
            int n2 = l2 ? l2->val: 0;
            int sum = n1 + n2 + carry;
            if (!head) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail->next = new ListNode(sum % 10);
                tail = tail->next;
            }
            carry = sum / 10;
            if (l1) {
                l1 = l1->next;
            }
            if (l2) {
                l2 = l2->next;
            }
        }
        if (carry > 0) {
            tail->next = new ListNode(carry);
        }
        return head;
    }
};
```

### 3无重复字符的最长子串

给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。


输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

输入: s = ""
输出: 0

```cpp
//
class Solution
{
public:
    int lengthOfLongestSubstring(string s)
    {
        // 哈希集合，记录每个字符是否出现过
        unordered_set<char> occ;

        
        int n = s.size();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        // 枚举左指针的位置，初始值隐性地表示为 -1
        for (int i = 0; i < n; ++i)
        {
            if (i != 0)
            {
                // 左指针向右移动一格，移除一个字符
                occ.erase(s[i - 1]);
            }
            while (rk + 1 < n && !occ.count(s[rk + 1]))
            {
                // 不断地移动右指针
                occ.insert(s[rk + 1]);
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1);
        }
        return ans;
    }
};

```

### 4寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5

输入：nums1 = [0,0], nums2 = [0,0]
输出：0.00000

输入：nums1 = [], nums2 = [1]
输出：1.00000

输入：nums1 = [2], nums2 = []
输出：2.00000

```cpp
//
class Solution {
public:
    int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int m = nums1.size();
        int n = nums2.size();
        int index1 = 0, index2 = 0;

        while (true) {
            // 边界情况
            if (index1 == m) {
                return nums2[index2 + k - 1];
            }
            if (index2 == n) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return min(nums1[index1], nums2[index2]);
            }

            // 正常情况
            int newIndex1 = min(index1 + k / 2 - 1, m - 1);
            int newIndex2 = min(index2 + k / 2 - 1, n - 1);
            int pivot1 = nums1[newIndex1];
            int pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            }
            else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }

    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totalLength = nums1.size() + nums2.size();
        if (totalLength % 2 == 1) {
            return getKthElement(nums1, nums2, (totalLength + 1) / 2);
        }
        else {
            return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
        }
    }
};
```



### 5最长回文子串

给你一个字符串 s，找到 s 中最长的回文子串。

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

输入：s = "cbbd"
输出："bb"

输入：s = "a"
输出："a"

输入：s = "ac"
输出："a"

```cpp
//方法一：动态规划
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        string ans;
        for (int l = 0; l < n; ++l) {
            for (int i = 0; i + l < n; ++i) {
                int j = i + l;
                if (l == 0) {
                    dp[i][j] = 1;
                } else if (l == 1) {
                    dp[i][j] = (s[i] == s[j]);
                } else {
                    dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
                }
                if (dp[i][j] && l + 1 > ans.size()) {
                    ans = s.substr(i, l + 1);
                }
            }
        }
        return ans;
    }
};

//方法二：中心扩展算法
class Solution {
public:
    pair<int, int> expandAroundCenter(const string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left;
            ++right;
        }
        return {left + 1, right - 1};
    }

    string longestPalindrome(string s) {
        int start = 0, end = 0;
        for (int i = 0; i < s.size(); ++i) {
            auto [left1, right1] = expandAroundCenter(s, i, i);
            auto [left2, right2] = expandAroundCenter(s, i, i + 1);
            if (right1 - left1 > end - start) {
                start = left1;
                end = right1;
            }
            if (right2 - left2 > end - start) {
                start = left2;
                end = right2;
            }
        }
        return s.substr(start, end - start + 1);
    }
};

//方法三：Manacher算法
class Solution {
public:
    int expand(const string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left;
            ++right;
        }
        return (right - left - 2) / 2;
    }

    string longestPalindrome(string s) {
        int start = 0, end = -1;
        string t = "#";
        for (char c: s) {
            t += c;
            t += '#';
        }
        t += '#';
        s = t;

        vector<int> arm_len;
        int right = -1, j = -1;
        for (int i = 0; i < s.size(); ++i) {
            int cur_arm_len;
            if (right >= i) {
                int i_sym = j * 2 - i;
                int min_arm_len = min(arm_len[i_sym], right - i);
                cur_arm_len = expand(s, i - min_arm_len, i + min_arm_len);
            } else {
                cur_arm_len = expand(s, i, i);
            }
            arm_len.push_back(cur_arm_len);
            if (i + cur_arm_len > right) {
                j = i;
                right = i + cur_arm_len;
            }
            if (cur_arm_len * 2 + 1 > end - start) {
                start = i - cur_arm_len;
                end = i + cur_arm_len;
            }
        }

        string ans;
        for (int i = start; i <= end; ++i) {
            if (s[i] != '#') {
                ans += s[i];
            }
        }
        return ans;
    }
};

```

### 8字符串转换整数

请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。


函数 myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。

注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符

输入：s = "42"
输出：42
解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
第 1 步："42"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："42"（读入 "42"）
           ^
解析得到整数 42 。
由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。

```cpp
//自动机
class Automaton
{
    string state = "start";
    unordered_map<string, vector<string>> table = {
        {"start", {"start", "signed", "in_number", "end"}},
        {"signed", {"end", "end", "in_number", "end"}},
        {"in_number", {"end", "end", "in_number", "end"}},
        {"end", {"end", "end", "end", "end"}}};

    int get_col(char c)
    {
        if (isspace(c))
            return 0;
        if (c == '+' or c == '-')
            return 1;
        if (isdigit(c))
            return 2;
        return 3;
    }

public:
    int sign = 1;
    long long ans = 0;

    void get(char c)
    {
        state = table[state][get_col(c)];
        if (state == "in_number")
        {
            ans = ans * 10 + c - '0';
            ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
        }
        else if (state == "signed")
            sign = c == '+' ? 1 : -1;
    }
};

class Solution
{
public:
    int myAtoi(string str)
    {
        Automaton automaton;
        for (char c : str)
            automaton.get(c);
        return automaton.sign * automaton.ans;
    }
};

```

### 10RegularExpressionMatching

给定一个字符串和一个正则表达式（regular expression, regex），求该字符串是否可以被匹配

输入输出样例
	输入是一个待匹配字符串和一个用字符串表示的正则表达式，输出是一个布尔值，表示是否
可以匹配成功。

Input: s = "aab", p = "c*a*b"
Output: true
在个样例中，我们可以重复 c 零次，重复 a 两次

```cpp
//
bool isMatch(string s, string p)
{
	int m = s.size(), n = p.size();
	vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
	dp[0][0] = true;
	for (int i = 1; i < n + 1; ++i)
	{
		if (p[i - 1] == '*')
		{
			dp[0][i] = dp[0][i - 2];
		}
	}
	for (int i = 1; i < m + 1; ++i)
	{
		for (int j = 1; j < n + 1; ++j)
		{
			if (p[j - 1] == '.')
			{
				dp[i][j] = dp[i - 1][j - 1];
			}
			else if (p[j - 1] != '*')
			{
				dp[i][j] = dp[i - 1][j - 1] && p[j - 1] == s[i - 1];
			}
			else if (p[j - 2] != s[i - 1] && p[j - 2] != '.')
			{
				dp[i][j] = dp[i][j - 2];
			}
			else
			{
				dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
			}
		}
	}
	return dp[m][n];
}
```

### 15三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组

答案中不可以包含重复的三元组

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]

输入：nums = []
输出：[]

输入：nums = [0]
输出：[]

```cpp
//方法一：排序 + 双指针
class Solution
{
public:
    vector<vector<int>> threeSum(vector<int> &nums)
    {
        int n = nums.size();
        sort(nums.begin(), nums.end()); //
        vector<vector<int>> ans;
        // 枚举 a
        for (int first = 0; first < n; ++first)
        {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1])
            {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second)
            {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1])
                {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target)
                {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third)
                {
                    break;
                }
                if (nums[second] + nums[third] == target)
                {
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};

```



### 19删除链表中倒数第N个结点

删除链表的倒数第 n 个结点，并且返回链表的头结点

输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

输入：head = [1], n = 1
输出：[]

输入：head = [1,2], n = 1
输出：[1]

```cpp
//方法一
class Solution
{
public:
    int getLength(ListNode *head)
    {
        int length = 0;
        while (head)
        {
            ++length;
            head = head->next;
        }
        return length;
    }

    ListNode *removeNthFromEnd(ListNode *head, int n)
    {
        ListNode *dummy = new ListNode(0, head);
        int length = getLength(head);
        ListNode *cur = dummy;
        for (int i = 1; i < length - n + 1; ++i)
        {
            cur = cur->next;
        }
        cur->next = cur->next->next;
        ListNode *ans = dummy->next;
        delete dummy;
        return ans;
    }
};

//使用栈
class Solution
{
public:
    ListNode *removeNthFromEnd(ListNode *head, int n)
    {
        ListNode *dummy = new ListNode(0, head);
        stack<ListNode *> stk;
        ListNode *cur = dummy;
        while (cur)
        {
            stk.push(cur);
            cur = cur->next;
        }
        for (int i = 0; i < n; ++i)
        {
            stk.pop();
        }
        ListNode *prev = stk.top();
        prev->next = prev->next->next;
        ListNode *ans = dummy->next;
        delete dummy;
        return ans;
    }
};

//使用双指针
class Solution
{
public:
    ListNode *removeNthFromEnd(ListNode *head, int n)
    {
        ListNode *dummy = new ListNode(0, head);
        ListNode *first = head;
        ListNode *second = dummy;
        for (int i = 0; i < n; ++i)
        {
            first = first->next;
        }
        while (first)
        {
            first = first->next;
            second = second->next;
        }
        second->next = second->next->next;
        ListNode *ans = dummy->next;
        delete dummy;
        return ans;
    }
};

```

### 20有效的括号

给定一个只由左右原括号、花括号和方括号组成的字符串，求这个字符串是否合法。合法的
定义是每一个类型的左括号都有一个右括号一一对应，且括号内的字符串也满足此要求

输入输出样例
	输入是一个字符串，输出是一个布尔值，表示字符串是否合法

Input: "{[]}()"
Output: true

```cpp
//
bool isValid(string s)
{
	stack<char> parsed;
	for (int i = 0; i < s.length(); ++i){
		if (s[i] == '{ ' || s[i] == '[' || s[i] == '(')
		{
			parsed.push(s[i]);
		}
		else
		{
			if (parsed.empty())
			{
				return false;
			}
			char c = parsed.top();
			if ((s[i] == '}' && c == '{ ') ||
				(s[i] == ']' && c == '[') ||
				(s[i] == ')' && c == '('))
			{
				parsed.pop();
			}
			else
			{
				return false;
			}
		}
	}
	return parsed.empty();
}

```

### 21MergeTwoSortedLists

给定两个增序的链表，试将其合并成一个增序的链表

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

输入：l1 = [], l2 = []
输出：[]

输入：l1 = [], l2 = [0]
输出：[0]

```cpp
//递归写法  递增
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2)
{
	//
	if (!l2)
	{
		return l1;
	}
	if (!l1)
	{
		return l2;
	}

	//对比两个链表的结点值
	if (l1->val > l2->val)
	{
		l2->next = mergeTwoLists(l1, l2->next); //l2值较小，故该节点返回l2，并比较l2->next和l1
		return l2;
	}
	else
	{
		l1->next = mergeTwoLists(l1->next, l2);
		return l1;
	}
}

//迭代写法
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2)
{
	ListNode *dummy = new ListNode(0), *node = dummy; // node：主操作结点
	while (l1 && l2)
	{
		if (l1->val <= l2->val)
		{
			node->next = l1;
			l1 = l1->next;
		}
		else
		{
			node->next = l2;
			l2 = l2->next;
		}
		node = node->next;
	}
	
	//传入的l1或l2其中一个为空时，直接返回
	node->next = l1 ? l1 : l2;
	return dummy->next; //返回dummy结点的下一个，因为该链表头节点为0
}
```



### 23合并K个排序链表

给定 k 个增序的链表，试将它们合并成一条增序链表

输入输出样例
	输入是一个一维数组，每个位置存储链表的头节点；输出是一条链表

Input:
[1->4->5,
1->3->4,
2->6]
Output: 1->1->2->3->4->4->5->6

```cpp
//
struct Comp
{
    bool operator()(ListNode *l1, ListNode *l2)
    {
        return l1->val > l2->val;
    }
};


ListNode *mergeKLists(vector<ListNode *> &lists)
{
    if (lists.empty())
        return nullptr;

    //
    priority_queue<ListNode *, vector<ListNode *>, Comp> q;
    for (ListNode *list : lists)
    {
        if (list)
        {
            q.push(list);
        }
    }
    ListNode *dummy = new ListNode(0), *cur = dummy;

    //
    while (!q.empty())
    {
        cur->next = q.top();
        q.pop();
        cur = cur->next;
        if (cur->next)
        {
            q.push(cur->next);
        }
    }
    return dummy->next;
}

```

### 24SwapNodesInPairs

给定一个链表，交换每个相邻的一对节点

Input: 1->2->3->4
Output: 2->1->4->3

```CPP
//
ListNode *swapPairs(ListNode *head)
{
    ListNode *p = head, *s;

    //
    if (p && p->next)
    {
        s = p->next;
        p->next = s->next; //
        s->next = p;
        head = s;//

        //
        while (p->next && p->next->next)
        {
            s = p->next->next;
            p->next->next = s->next;
            s->next = p->next;
            p->next = s;
            p = s->next;
        }
    }
    return head;
}

//递归解法
class Solution
{
    public:
    ListNode *swapPairs(ListNode *head)
    {
        if (head == nullptr || head->next == nullptr)
        {
            return head;
        }
        ListNode *newHead = head->next;
        head->next = swapPairs(newHead->next);
        newHead->next = head;
        return newHead;
    }
};

//官方迭代
class Solution
{
    public:
    ListNode *swapPairs(ListNode *head)
    {
        ListNode *dummyHead = new ListNode(0);
        dummyHead->next = head;
        ListNode *temp = dummyHead;
        while (temp->next != nullptr && temp->next->next != nullptr)
        {
            ListNode *node1 = temp->next;
            ListNode *node2 = temp->next->next;
            temp->next = node2;
            node1->next = node2->next;
            node2->next = node1;
            temp = node1;
        }
        return dummyHead->next;
    }
};

```

### 25K个一组反转链表

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
k 是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序

输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]

输入：head = [1,2,3,4,5], k = 1
输出：[1,2,3,4,5]

输入：head = [1], k = 1
输出：[1]

```CPP
//
class Solution {
public:
    // 翻转一个子链表，并且返回新的头与尾
    pair<ListNode*, ListNode*> myReverse(ListNode* head, ListNode* tail) {
        ListNode* prev = tail->next;
        ListNode* p = head;
        while (prev != tail) {
            ListNode* nex = p->next;
            p->next = prev;
            prev = p;
            p = nex;
        }
        return {tail, head};
    }

    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* hair = new ListNode(0);
        hair->next = head;
        ListNode* pre = hair;

        while (head) {
            ListNode* tail = pre;
            // 查看剩余部分长度是否大于等于 k
            for (int i = 0; i < k; ++i) {
                tail = tail->next;
                if (!tail) {
                    return hair->next;
                }
            }
            ListNode* nex = tail->next;
            // 这里是 C++17 的写法，也可以写成
            // pair<ListNode*, ListNode*> result = myReverse(head, tail);
            // head = result.first;
            // tail = result.second;
            tie(head, tail) = myReverse(head, tail);
            // 把子链表重新接回原链表
            pre->next = head;
            tail->next = nex;
            pre = tail;
            head = tail->next;
        }

        return hair->next;
    }
};
/*
时间复杂度：O(n)
空间复杂度：O(1)
*/
```

### 28ImplementStrStr

判断一个字符串是不是另一个字符串的子字符串，并返回其位置

输入输出样例
	输入一个母字符串和一个子字符串，输出一个整数，表示子字符串在母字符串的位置，若不
存在则返回-1

Input: haystack = "hello", needle = "ll"
Output: 2

```CPP
// 主函数
int strStr(string haystack, string needle) {
	int k = -1, n = haystack.length(), p = needle.length();
	if (p == 0) 
		return 0;
	vector<int> next(p, -1); // -1表示不存在相同的最大前缀和后缀
	calNext(needle, next); // 计算next数组
	for (int i = 0; i < n; ++i) {
		while (k > -1 && needle[k + 1] != haystack[i]) {
			k = next[k]; // 有部分匹配， 往前回溯
		}
		if (needle[k + 1] == haystack[i]) {
			++k;
		}
		if (k == p - 1) {
			return i - p + 1; // 说明k移动到needle的最末端， 返回相应的位置
		}
	}
	return -1;
}
// 辅函数 - 计算next数组
void calNext(const string &needle, vector<int> &next) {
	for (int j = 1, p = -1; j < needle.length(); ++j) {
		while (p > -1 && needle[p + 1] != needle[j]) {
			p = next[p]; // 如果下一位不同， 往前回溯
		}
		if (needle[p + 1] == needle[j]) {
			++p; // 如果下一位相同， 更新相同的最大前缀和最大后缀长
		}
		next[j] = p;
	}
}

//409
//3
//772
//5 

```

### 33搜索旋转排序数组

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 


示例 1：

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例 2：

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例 3：

输入：nums = [1], target = 0
输出：-1

```CPP
//二分查找
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = (int)nums.size();
        if (!n) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
};
```

### 34在排序区间中查找给定元素的第一个和最后一个位置

给定一个增序的整数数组和一个值，查找该值第一次和最后一次出现的位置

输入输出样例：
	输入是一个数组和一个值，输出为该值第一次出现的位置和最后一次出现的位置（从 0 开
始）；如果不存在该值，则两个返回值都设为-1

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

```cpp
// 主函数
vector<int> searchRange(vector<int> &nums, int target)
{
	if (nums.empty())
		return vector<int>{-1, -1};
	int lower = lower_bound(nums, target);
	int upper = upper_bound(nums, target) - 1; // 这里需要减1位
	if (lower == nums.size() || nums[lower] != target)
	{
		return vector<int>{-1, -1};
	}
	return vector<int>{lower, upper};
}
// 辅函数
int lower_bound(vector<int> &nums, int target)
{
	int l = 0, r = nums.size(), mid;
	while (l < r)
	{
		mid = (l + r) / 2;
		if (nums[mid] >= target)
		{
			r = mid;
		}
		else
		{
			l = mid + 1;
		}
	}
	return l;
}
// 辅函数
int upper_bound(vector<int> &nums, int target)
{
	int l = 0, r = nums.size(), mid;
	while (l < r)
	{
		mid = (l + r) / 2;
		if (nums[mid] > target)
		{
			r = mid;
		}
		else
		{
			l = mid + 1;
		}
	}
	return l;
}
```

### 42接雨水

给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

输入：height = [4,2,0,3,2,5]
输出：9

```cpp
//动态规划
class Solution
{
public:
    int trap(vector<int> &height)
    {
        int n = height.size();
        if (n == 0)
        {
            return 0;
        }
        vector<int> leftMax(n);
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i)
        {
            leftMax[i] = max(leftMax[i - 1], height[i]);
        }

        vector<int> rightMax(n);
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i)
        {
            rightMax[i] = max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i)
        {
            ans += min(leftMax[i], rightMax[i]) - height[i];
        }
        return ans;
    }
};

//单调栈
class Solution
{
public:
    int trap(vector<int> &height)
    {
        int ans = 0;
        stack<int> stk;
        int n = height.size();
        for (int i = 0; i < n; ++i)
        {
            while (!stk.empty() && height[i] > height[stk.top()])
            {
                int top = stk.top();
                stk.pop();
                if (stk.empty())
                {
                    break;
                }
                int left = stk.top();
                int currWidth = i - left - 1;
                int currHeight = min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stk.push(i);
        }
        return ans;
    }
};

//双指针
class Solution
{
public:
    int trap(vector<int> &height)
    {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        while (left < right)
        {
            leftMax = max(leftMax, height[left]);
            rightMax = max(rightMax, height[right]);
            if (height[left] < height[right])
            {
                ans += leftMax - height[left];
                ++left;
            }
            else
            {
                ans += rightMax - height[right];
                --right;
            }
        }
        return ans;
    }
};

```

### 43字符串相乘

给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

输入: num1 = "2", num2 = "3"
输出: "6"

输入: num1 = "123", num2 = "456"
输出: "56088"

```cpp
//方法一：加法
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") {
            return "0";
        }
        string ans = "0";
        int m = num1.size(), n = num2.size();
        for (int i = n - 1; i >= 0; i--) {
            string curr;
            int add = 0;
            for (int j = n - 1; j > i; j--) {
                curr.push_back(0);
            }
            int y = num2.at(i) - '0';
            for (int j = m - 1; j >= 0; j--) {
                int x = num1.at(j) - '0';
                int product = x * y + add;
                curr.push_back(product % 10);
                add = product / 10;
            }
            while (add != 0) {
                curr.push_back(add % 10);
                add /= 10;
            }
            reverse(curr.begin(), curr.end());
            for (auto &c : curr) {
                c += '0';
            }
            ans = addStrings(ans, curr);
        }
        return ans;
    }

    string addStrings(string &num1, string &num2) {
        int i = num1.size() - 1, j = num2.size() - 1, add = 0;
        string ans;
        while (i >= 0 || j >= 0 || add != 0) {
            int x = i >= 0 ? num1.at(i) - '0' : 0;
            int y = j >= 0 ? num2.at(j) - '0' : 0;
            int result = x + y + add;
            ans.push_back(result % 10);
            add = result / 10;
            i--;
            j--;
        }
        reverse(ans.begin(), ans.end());
        for (auto &c: ans) {
            c += '0';
        }
        return ans;
    }
};


//方法二：乘法
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") {
            return "0";
        }
        int m = num1.size(), n = num2.size();
        auto ansArr = vector<int>(m + n);
        for (int i = m - 1; i >= 0; i--) {
            int x = num1.at(i) - '0';
            for (int j = n - 1; j >= 0; j--) {
                int y = num2.at(j) - '0';
                ansArr[i + j + 1] += x * y;
            }
        }
        for (int i = m + n - 1; i > 0; i--) {
            ansArr[i - 1] += ansArr[i] / 10;
            ansArr[i] %= 10;
        }
        int index = ansArr[0] == 0 ? 1 : 0;
        string ans;
        while (index < m + n) {
            ans.push_back(ansArr[index]);
            index++;
        }
        for (auto &c: ans) {
            c += '0';
        }
        return ans;
    }
};

```

### 46求数组排列

给定一个无重复数字的整数数组，求其所有的排列方式。

输入是一个一维整数数组，输出是一个二维数组，表示输入数组的所有排列方式

Input: [1,2,3]
Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

```cpp
// 主函数
vector<vector<int>> permute(vector<int> &nums)
{
	vector<vector<int>> ans;
	backtracking(nums, 0, ans);
	return ans;
}
// 辅函数
void backtracking(vector<int> &nums, int level, vector<vector<int>> &ans)
{
	//
	if (level == nums.size() - 1)
	{
		ans.push_back(nums);
		return;
	}
	for (int i = level; i < nums.size(); i++)
	{
		swap(nums[i], nums[level]);			// 修改当前节点状态
		backtracking(nums, level + 1, ans); // 递归子节点，
		swap(nums[i], nums[level]);			// 回改当前节点状态
	}
}
```

### 48RotateImage

给定一个 n × n 的矩阵，求它顺时针旋转 90 度的结果，且必须在原矩阵上修改（in-place）。
怎样能够尽量不创建额外储存空间呢

输入输出样例
	输入和输出都是一个二维整数矩阵

Input:
[[1,2,3],
[4,5,6],
[7,8,9]]
Output:
[[7,4,1],
[8,5,2],
[9,6,3]]

```cpp
//
void rotate(vector<vector<int>> &matrix)
{
	int temp = 0, n = matrix.size() - 1;

	//
	for (int i = 0; i <= n / 2; ++i)
	{
		for (int j = i; j < n - i; ++j)
		{
			temp = matrix[j][n - i];
			matrix[j][n - i] = matrix[i][j];
			matrix[i][j] = matrix[n - j][i];
			matrix[n - j][i] = matrix[n - i][n - j];
			matrix[n - i][n - j] = temp;
		}
	}
}
```

### 53最大子序和

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

输入：nums = [1]
输出：1

输入：nums = [0]
输出：0

输入：nums = [-1]
输出：-1

```cpp
//方法一：动态规划
class Solution
{
public:
    int maxSubArray(vector<int> &nums)
    {
        int pre = 0, maxAns = nums[0];
        for (const auto &x : nums)
        {
            pre = max(pre + x, x); //
            maxAns = max(maxAns, pre); //
        }
        return maxAns;
    }
};

//方法二：分治
class Solution
{
public:
    /*
    lSum 表示 [l,r] 内以 l 为左端点的最大子段和
    rSum 表示 [l,r] 内以 r 为右端点的最大子段和
    mSum 表示 [l,r] 内的最大子段和
    iSum 表示 [l,r] 的区间和
    */
    struct Status
    {
        int lSum, rSum, mSum, iSum;
    };

    Status pushUp(Status l, Status r)
    {
        int iSum = l.iSum + r.iSum;
        int lSum = max(l.lSum, l.iSum + r.lSum);
        int rSum = max(r.rSum, r.iSum + l.rSum);
        int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
        return (Status){lSum, rSum, mSum, iSum};
    };


    //表示查询a序列[l, r]区间内的最大字段和
    Status get(vector<int> &a, int l, int r) 
    {
        if (l == r)
        {
            return (Status){a[l], a[l], a[l], a[l]};
        }
        int m = (l + r) >> 1;
        Status lSub = get(a, l, m);
        Status rSub = get(a, m + 1, r);
        return pushUp(lSub, rSub);
    }

    int maxSubArray(vector<int> &nums)
    {
        return get(nums, 0, nums.size() - 1).mSum;
    }
};

```

### 54.螺旋矩阵

给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

```cpp
//模拟
class Solution {
private:
    static constexpr int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return {};
        }
        
        int rows = matrix.size(), columns = matrix[0].size();
        vector<vector<bool>> visited(rows, vector<bool>(columns));
        int total = rows * columns;
        vector<int> order(total);

        int row = 0, column = 0;
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
            order[i] = matrix[row][column];
            visited[row][column] = true;
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return order;
    }
};

//按层模拟
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return {};
        }

        int rows = matrix.size(), columns = matrix[0].size();
        vector<int> order;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            for (int column = left; column <= right; column++) {
                order.push_back(matrix[top][column]);
            }
            for (int row = top + 1; row <= bottom; row++) {
                order.push_back(matrix[row][right]);
            }
            if (left < right && top < bottom) {
                for (int column = right - 1; column > left; column--) {
                    order.push_back(matrix[bottom][column]);
                }
                for (int row = bottom; row > top; row--) {
                    order.push_back(matrix[row][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return order;
    }
};
```

### 69求开方

给定一个非负整数，求它的开方，向下取整

输入输出样例
	输入一个整数，输出一个整数
	Input: 8
	Output: 2

分析：在1和该数之间逐个遍历，取二分查找

```cpp
//
int mySqrt(int a)
{
	if (a == 0)
		return a;
	int l = 1, r = a, mid, sqrt;
	while (l <= r)
	{
		mid = l + (r - l) / 2;
		sqrt = a / mid;
		if (sqrt == mid) //
		{
			return mid;
		}
		else if (mid > sqrt)
		{
			r = mid - 1;
		}
		else
		{
			l = mid + 1;
		}
	}
	return r;
}


//方法二：牛顿迭代法
int mySqrt(int a)
{
	long x = a;
	while (x * x > a)
	{
		x = (x + a / x) / 2;
	}
	return x;
}
```

### 70上台阶

给定 n 节台阶，每次可以走一步或走两步，求一共有多少种方式可以走完这些台阶

输入是一个数字，表示台阶数量；输出是爬台阶的总方式

Input: 3
Output: 3

分析：
	定义一个数组 dp， dp[i] 表示走到第 i 阶的方法数。因为
	我们每次可以走一步或者两步，所以第 i 阶可以从第 i-1 或 i-2 阶到达

dp[i] = dp[i-1] + dp[i-2]

```cpp
//方法一：递归
int climbStairs(int n)
{
	if (n <= 2)
		return n;
	vector<int> dp(n + 1, 1); //n+1个元素
	for (int i = 2; i <= n; ++i)
	{
		dp[i] = dp[i - 1] + dp[i - 2];
	}
	return dp[n];
}
//方法二：空间复杂度O(1)
int climbStairs(int n)
{
	if (n <= 2)
		return n;
	int pre2 = 1, pre1 = 2, cur;
	for (int i = 2; i < n; ++i)
	{
		cur = pre1 + pre2;
		pre2 = pre1;
		pre1 = cur;
	}
	return cur;
}
```

### 72字符串编辑

给定两个字符串，已知你可以删除、替换和插入任意字符串的任意字符，求最少编辑几步可
以将两个字符串变成相同

输入输出样例：
	输入是两个字符串，输出是一个整数，表示最少的步骤

Input: word1 = "horse", word2 = "ros"
Output: 3

在这个样例中，一种最优编辑方法是（1） horse -> rorse （2） rorse -> rose（3） rose -> ros

```cpp
//
int minDistance(string word1, string word2)
{
	int m = word1.length(), n = word2.length();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 0; i <= m; ++i)
	{
		for (int j = 0; j <= n; ++j)
		{
			if (i == 0)
			{
				dp[i][j] = j;
			}
			else if (j == 0)
			{
				dp[i][j] = i;
			}
			else
			{
				dp[i][j] = min(
					dp[i - 1][j - 1] + ((word1[i - 1] == word2[j - 1]) ? 0 : 1),
					min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
			}
		}
	}
	return dp[m][n];
}

```



### 75颜色分类

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]

输入：nums = [2,0,1]
输出：[0,1,2]

输入：nums = [0]
输出：[0]

输入：nums = [1]
输出：[1]

```cpp

//方法一：双指针
class Solution
{
public:
	void sortColors(vector<int> &nums)
	{
		int n = nums.size();
		int p0 = 0, p1 = 0;
		for (int i = 0; i < n; ++i)
		{
			if (nums[i] == 1)
			{
				swap(nums[i], nums[p1]);
				++p1;
			}
			else if (nums[i] == 0)
			{
				swap(nums[i], nums[p0]);
				if (p0 < p1)
				{
					swap(nums[i], nums[p1]);
				}
				++p0;
				++p1;
			}
		}
	}
};

```

### 76最小覆盖子串

给定两个字符串 S 和 T，求 S 中包含 T 所有字符的最短连续子字符串的长度，同时要求时间复杂度不得超过 O(n)

示例：输入是两个字符串 S 和 T，输出是一个 S 字符串的子串

- Input: S = "ADOBECODEBANC", T = "ABC"
Output : "BANC"

```cpp
//
string minWindow(string S, string T)
{
	vector<int> chars(128, 0);	   //表示目前每个字符缺少的数量
	vector<bool> flag(128, false); //表示每个字符是否在 T 中存在
	// 先统计T中的字符情况
	for (int i = 0; i < T.size(); ++i)
	{
		flag[T[i]] = true;
		++chars[T[i]]; //
	}
	// 移动滑动窗口， 不断更改统计数据
	int cnt = 0, l = 0, min_l = 0, min_size = S.size() + 1;
	for (int r = 0; r < S.size(); ++r)
	{
		if (flag[S[r]])
		{
			if (--chars[S[r]] >= 0)
			{
				++cnt;
			}
			// 若目前滑动窗口已包含T中全部字符，
			// 则尝试将l右移， 在不影响结果的情况下获得最短子字符串
			while (cnt == T.size())
			{
				if (r - l + 1 < min_size)
				{
					min_l = l;
					min_size = r - l + 1;
				}
				if (flag[S[l]] && ++chars[S[l]] > 0)
				{
					--cnt;
				}
				++l;
			}
		}
	}
	return min_size > S.size() ? "" : S.substr(min_l, min_size);
}


//解法2
class Solution
{
public:
	unordered_map<char, int> ori, cnt;

	bool check()
	{
		for (const auto &p : ori)
		{
			if (cnt[p.first] < p.second)
			{
				return false;
			}
		}
		return true;
	}

	string minWindow(string s, string t)
	{
		for (const auto &c : t)
		{
			++ori[c];
		}

		int l = 0, r = -1;
		int len = INT_MAX, ansL = -1, ansR = -1;

		while (r < int(s.size()))
		{
			if (ori.find(s[++r]) != ori.end())
			{
				++cnt[s[r]];
			}
			while (check() && l <= r)
			{
				if (r - l + 1 < len)
				{
					len = r - l + 1;
					ansL = l;
				}
				if (ori.find(s[l]) != ori.end())
				{
					--cnt[s[l]];
				}
				++l;
			}
		}

		return ansL == -1 ? string() : s.substr(ansL, len);
	}
};
```

### 77求组合方法

给定一个整数 n 和一个整数 k，求在 1 到 n 中选取 k 个数字的所有组合方法

输入是两个正整数 n 和 k，输出是一个二维数组，表示所有组合方式

Input: n = 4, k = 2
Output: [[2,4], [3,4], [2,3], [1,2], [1,3], [1,4]]

```cpp
// 主函数
vector<vector<int>> combine(int n, int k)
{
	vector<vector<int>> ans;
	vector<int> comb(k, 0);
	int count = 0;
	backtracking(ans, comb, count, 1, n, k);
	return ans;
}
// 辅函数
void backtracking(vector<vector<int>> &ans, vector<int> &comb, int &count, int pos, int n, int k)
{
	if (count == k)
	{
		ans.push_back(comb);
		return;
	}
	for (int i = pos; i <= n; ++i)
	{
		comb[count++] = i;							 // 修改当前节点状态
		backtracking(ans, comb, count, i + 1, n, k); // 递归子节点
		--count;									 // 回改当前节点状态
	}
}
```

### 79字母矩阵搜索

给定一个字母矩阵，所有的字母都与上下左右四个方向上的字母相连。给定一个字符串，求
字符串能不能在字母矩阵中寻找到

输入输出样例：
	输入是一个二维字符数组和一个字符串，输出是一个布尔值，表示字符串是否可以被寻找
到
Input: word = "ABCCED", board =
[['A','B','C','E'],
['S','F','C','S'],
['A','D','E','E']]
Output: true

```cpp
// 主函数
bool exist(vector<vector<char>> &board, string word)
{
	if (board.empty())
		return false;
	int m = board.size(), n = board[0].size();
	vector<vector<bool>> visited(m, vector<bool>(n, false)); //访问标记
	bool find = false;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j) //
		{
			backtracking(i, j, board, word, find, visited, 0);
		}
	}
	return find;
}
// 辅函数
void backtracking(int i, int j, vector<vector<char>> &board, string &word, bool &find, vector<vector<bool>> &visited, int pos)
{
	if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size())
	{
		return;
	}
	if (visited[i][j] || find || board[i][j] != word[pos])
	{
		return;
	}
	if (pos == word.size() - 1)
	{
		find = true;
		return;
	}
	visited[i][j] = true; // 修改当前节点状态
	// 递归子节点
	backtracking(i + 1, j, board, word, find, visited, pos + 1);
	backtracking(i - 1, j, board, word, find, visited, pos + 1);
	backtracking(i, j + 1, board, word, find, visited, pos + 1);
	backtracking(i, j - 1, board, word, find, visited, pos + 1);
	visited[i][j] = false; // 回改当前节点状态
}

```

### 81旋转数组查找数字

```cpp
//81 旋转数组查找数字
/*
一个原本增序的数组被首尾相连后按某个位置断开（如 [1,2,2,3,4,5] ! [2,3,4,5,1,2]，在第一
位和第二位断开），我们称其为旋转数组。给定一个值，判断这个值是否存在于这个为旋转数组中

输入输出样例：
输入是一个数组和一个值，输出是一个布尔值，表示数组中是否存在该值
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

*/
bool search(vector<int> &nums, int target)
{
	int start = 0, end = nums.size() - 1;
	while (start <= end)
	{
		int mid = (start + end) / 2;
		if (nums[mid] == target)
		{
			return true;
		}
		if (nums[start] == nums[mid])
		{
			// 无法判断哪个区间是增序的
			++start;
		}
		else if (nums[mid] <= nums[end])
		{
			// 右区间是增序的
			if (target > nums[mid] && target <= nums[end])
			{
				start = mid + 1;
			}
			else
			{
				end = mid - 1;
			}
		}
		else
		{
			// 左区间是增序的
			if (target >= nums[start] && target < nums[mid])
			{
				end = mid - 1;
			}
			else
			{
				start = mid + 1;
			}
		}
	}
	return false;
}

```

### 83删除排序链表重复元素

```cpp
//存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次
/*
输入：head = [1,1,2]
输出：[1,2]

输入：head = [1,1,2,3,3]
输出：[1,2,3]


*/
class Solution
{
public:
    ListNode *deleteDuplicates(ListNode *head)
    {
        if (!head)
        {
            return head;
        }

        ListNode *cur = head;
        while (cur->next) //
        {
            if (cur->val == cur->next->val) //
            {
                //需要释放删除的节点的空间
                ListNode * temp = cur->next;

                cur->next = cur->next->next;

                delete cur->next;//
            }
            else
            {
                cur = cur->next; //
            }
        }

        return head;
    }
};

```

### 88归并两个有序数组

给定两个有序数组，把两个数组合并为一个

输入输出样例：
输入是两个数组和它们分别的元素个数 m 和 n。其中第一个数组的长度被延长至 m + n，多出的
n 位被 0 填补。题目要求把第二个数组归并到第一个数组上，不需要开辟额外空间

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: nums1 = [1,2,2,3,5,6]

```cpp
//
void merge(vector<int> &nums1, int m, vector<int> &nums2, int n)
{
	int pos = m-- + n-- - 1; //pos下标，起始位置为 m + n − 1，最终将目标元素插入pos的位置。每次向前移动 m 或 n 的时候，也要向前移动 pos
	while (m >= 0 && n >= 0)
	{
		nums1[pos--] = nums1[m] > nums2[n] ? nums1[m--] : nums2[n--];
	}

	//如果m<n的情况
	while (n >= 0) //
	{
		nums1[pos--] = nums2[n--];
	}
}

```

### 91DecodeWays

已知字母 A-Z 可以表示成数字 1-26。给定一个数字串，求有多少种不同的字符串等价于这个
数字串

输入输出样例
	输入是一个由数字组成的字符串，输出是满足条件的解码方式总数

Input: "226"
Output: 3
有三种解码方式： BZ(2 26)、 VF(22 6) 或 BBF(2 2 6)

```cpp
//
int numDecodings(string s)
{
	int n = s.length();
	if (n == 0)
		return 0;
	int prev = s[0] - '0';
	if (!prev)
		return 0;
	if (n == 1)
		return 1;
	vector<int> dp(n + 1, 1);
	for (int i = 2; i <= n; ++i)
	{
		int cur = s[i - 1] - '0';
		if ((prev == 0 || prev > 2) && cur == 0)
		{
			return 0;
		}
		if ((prev < 2 && prev > 0) || prev == 2 && cur < 7)
		{
			if (cur)
			{
				dp[i] = dp[i - 2] + dp[i - 1];
			}
			else
			{
				dp[i] = dp[i - 2];
			}
		}
		else
		{
			dp[i] = dp[i - 1];
		}
		prev = cur;
	}
	return dp[n];
}
```

### 92反转链表 II

```cpp
//92. 反转链表 II
/*
给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。 

输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]


输入：head = [5], left = 1, right = 1
输出：[5]


*/

//穿针引线
class Solution
{
private:
    void reverseLinkedList(ListNode *head)
    {
        // 也可以使用递归反转一个链表
        ListNode *pre = nullptr;
        ListNode *cur = head;

        while (cur != nullptr)
        {
            ListNode *next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
    }

public:
    ListNode *reverseBetween(ListNode *head, int left, int right)
    {
        // 因为头节点有可能发生变化，使用虚拟头节点可以避免复杂的分类讨论
        ListNode *dummyNode = new ListNode(-1);
        dummyNode->next = head;

        ListNode *pre = dummyNode;
        // 第 1 步：从虚拟头节点走 left - 1 步，来到 left 节点的前一个节点
        // 建议写在 for 循环里，语义清晰
        for (int i = 0; i < left - 1; i++)
        {
            pre = pre->next;
        }

        // 第 2 步：从 pre 再走 right - left + 1 步，来到 right 节点
        ListNode *rightNode = pre;
        for (int i = 0; i < right - left + 1; i++)
        {
            rightNode = rightNode->next;
        }

        // 第 3 步：切断出一个子链表（截取链表）
        ListNode *leftNode = pre->next;
        ListNode *curr = rightNode->next;

        // 注意：切断链接
        pre->next = nullptr;
        rightNode->next = nullptr;

        // 第 4 步：同第 206 题，反转链表的子区间
        reverseLinkedList(leftNode);

        // 第 5 步：接回到原来的链表中
        pre->next = rightNode;
        leftNode->next = curr;
        return dummyNode->next;
    }
};

//一次遍历「穿针引线」反转链表（头插法）
class Solution
{
public:
    ListNode *reverseBetween(ListNode *head, int left, int right)
    {
        // 设置 dummyNode 是这一类问题的一般做法
        ListNode *dummyNode = new ListNode(-1);
        dummyNode->next = head;
        ListNode *pre = dummyNode;
        for (int i = 0; i < left - 1; i++)
        {
            pre = pre->next;
        }
        ListNode *cur = pre->next;
        ListNode *next;
        for (int i = 0; i < right - left; i++)
        {
            next = cur->next;
            cur->next = next->next;
            next->next = pre->next;
            pre->next = next;
        }
        return dummyNode->next;
    }
};

```

### 94二叉树的中序遍历

```cpp
/*
给定一个二叉树的根节点 root ，返回它的 中序 遍历。
输入：root = [1,null,2,3]
输出：[1,3,2]

示例 2：

输入：root = []
输出：[]

示例 3：

输入：root = [1]
输出：[1]
*/

//递归
class Solution
{
public:
    void inorder(TreeNode *root, vector<int> &res)
    {
        if (!root)
        {
            return;
        }
        inorder(root->left, res);
        res.push_back(root->val);
        inorder(root->right, res);
    }
    vector<int> inorderTraversal(TreeNode *root)
    {
        vector<int> res;
        inorder(root, res);
        return res;
    }
};

//迭代
class Solution
{
public:
    vector<int> inorderTraversal(TreeNode *root)
    {
        vector<int> res;
        stack<TreeNode *> stk;
        while (root != nullptr || !stk.empty())
        {
            while (root != nullptr)
            {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            res.push_back(root->val);
            root = root->right;
        }
        return res;
    }
};


//Morris 中序遍历
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        TreeNode *predecessor = nullptr;

        while (root != nullptr) {
            if (root->left != nullptr) {
                // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
                predecessor = root->left;
                while (predecessor->right != nullptr && predecessor->right != root) {
                    predecessor = predecessor->right;
                }
                
                // 让 predecessor 的右指针指向 root，继续遍历左子树
                if (predecessor->right == nullptr) {
                    predecessor->right = root;
                    root = root->left;
                }
                // 说明左子树已经访问完了，我们需要断开链接
                else {
                    res.push_back(root->val);
                    predecessor->right = nullptr;
                    root = root->right;
                }
            }
            // 如果没有左孩子，则直接访问右孩子
            else {
                res.push_back(root->val);
                root = root->right;
            }
        }
        return res;
    }
};


```

### 99 Recover Binary Search Tree

```cpp
//99 Recover Binary Search Tree
/*
给定一个二叉查找树，已知有两个节点被不小心交换了，试复原此树
输入是一个被误交换两个节点的二叉查找树，输出是改正后的二叉查找树

Input:
	 3
	/ \
   1   4
	  /
	 2
Output:
	2
   / \
  1   4
	 /
	3

*/


// 主函数
void recoverTree(TreeNode *root)
{
	TreeNode *mistake1 = nullptr, *mistake2 = nullptr, *prev = nullptr;
	inorder(root, mistake1, mistake2, prev);
	if (mistake1 && mistake2)
	{
		int temp = mistake1->val;
		mistake1->val = mistake2->val;
		mistake2->val = temp;
	}
}


// 辅函数
void inorder(TreeNode *root, TreeNode *&mistake1, TreeNode *&mistake2, TreeNode *&prev)
{
	if (!root)
	{
		return;
	}

	//
	if (root->left)
	{
		inorder(root->left, mistake1, mistake2, prev);
	}
	if (prev && root->val < prev->val)
	{
		if (!mistake1)
		{
			mistake1 = prev;
			mistake2 = root;
		}
		else
		{
			mistake2 = root;
		}
		cout << mistake1->val;
		cout << mistake2->val;
	}
	prev = root;
	if (root->right)
	{
		inorder(root->right, mistake1, mistake2, prev);
	}
}
```

### 101SymmetricTree

```cpp

//101 Symmetric Tree
/*
判断一个二叉树是否对称

Input:
		1
	   / \
	  2   2
	 / \ / \
	3  4 4  3 
Output: true

对称：
（1）如果两个子树都为空指针，则它们相等或对称
（2）如果两个子树只有一个为空指针，则它们不相等或不对称
（3）如果两个子树根节点的值不相等，则它们不相等或不对称
（4）根据相等或对称要求，进行递归处理


*/
// 主函数
bool isSymmetric(TreeNode *root)
{
	return root ? isSymmetric(root->left, root->right) : true;
}
// 辅函数
bool isSymmetric(TreeNode *left, TreeNode *right)
{
	if (!left && !right)
	{
		return true;
	}
	if (!left || !right)
	{
		return false;
	}
	if (left->val != right->val)
	{
		return false;
	}

	//
	return isSymmetric(left->left, right->right) && isSymmetric(left->right,right->left);
}

```



### 102二叉树的层序遍历

```cpp
//二叉树的层序遍历
/*
给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）


*/
//方法一：广度优先搜索
class Solution
{
public:
    vector<vector<int>> levelOrder(TreeNode *root)
    {
        vector<vector<int>> ret;
        if (!root)
        {
            return ret;
        }

        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty())
        {
            int currentLevelSize = q.size();
            ret.push_back(vector<int>());
            for (int i = 1; i <= currentLevelSize; ++i)
            {
                auto node = q.front();
                q.pop();
                ret.back().push_back(node->val);
                if (node->left)
                    q.push(node->left);
                if (node->right)
                    q.push(node->right);
            }
        }

        return ret;
    }
};

```



### 103二叉树的锯齿形层序遍历

要实现二叉树的锯齿形层序遍历，我们可以使用广度优先搜索（BFS）的方式，同时利用一个布尔变量来标识当前层的遍历方向。具体思路如下：

使用队列来存储当前层的节点。
使用一个布尔变量 left_to_right 来控制当前层的遍历方向。
每次遍历一层时，根据当前层的遍历方向，将节点的值存入一个临时列表中。
当一层遍历完成后，如果当前层是从左到右的，再将临时列表的值直接添加到最终结果中，如果是从右到左的，则将其反转后再添加到结果中。

```cpp
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

std::vector<std::vector<int>> zigzagLevelOrder(TreeNode* root) {
    std::vector<std::vector<int>> result;
    if (root == NULL) {
        return result; // 返回空的结果
    }
    
    std::deque<TreeNode*> queue; // 使用双端队列进行层序遍历
    queue.push_back(root);
    bool left_to_right = true; // 控制遍历方向

    while (!queue.empty()) {
        int level_size = queue.size(); // 当前层的节点数
        std::vector<int> current_level; // 当前层的值
       
        for (int i = 0; i < level_size; ++i) {
            TreeNode* node = queue.front(); // 取出队首节点
            queue.pop_front();
            current_level.push_back(node->val); // 保存当前节点的值
            
            // 按照层的方向将子节点添加到队列
            if (node->left) {
                queue.push_back(node->left);
            }
            if (node->right) {
                queue.push_back(node->right);
            }
        }

        // 如果当前层是从右到左遍历，则需要翻转当前层的结果
        if (!left_to_right) {
            std::reverse(current_level.begin(), current_level.end());
        }

        result.push_back(current_level); // 将当前层结果添加到最终结果
        left_to_right = !left_to_right; // 切换遍历方向
    }

    return result;
}

// 示例用法
int main() {
    // 创建一个简单的二叉树
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);
    
    // 获取锯齿形层序遍历结果
    std::vector<std::vector<int>> result = zigzagLevelOrder(root);
    
    // 打印结果
    for (const auto& level : result) {
        for (int val : level) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // 释放内存（省略实际中的释放代码）
    return 0;
}
```

代码说明

TreeNode 结构体：表示二叉树的节点，包含节点的值及左右子节点指针。
zigzagLevelOrder 函数：实现锯齿形层序遍历，输入为树的根节点，输出为包含每层节点值的二维向量。
主函数：创建一个简单的二叉树，并输出其锯齿形层序遍历的结果。

示例输出

对于上面的二叉树，输出结果将会是：

1 
3 2 
4 5 6 7 

这就是二叉树节点值的锯齿形层序遍历的实现方式。



### 104MaximumDepthOfBinaryTree

```cpp
//104 Maximum Depth of Binary Tree
/*
求一个二叉树的最大深度

*/

//
int maxDepth(TreeNode *root)
{
	return root ? 1 + max(maxDepth(root->left), maxDepth(root->right)) : 0;
}

//深度优先搜索
class Solution
{
public:
	int maxDepth(TreeNode *root)
	{
		if (root == nullptr)
			return 0;

		//返回左结点和右结点中最大
		return max(maxDepth(root->left), maxDepth(root->right)) + 1;
	}
};

//广度优先搜索
class Solution
{
public:
	int maxDepth(TreeNode *root)
	{
		if (root == nullptr)
			return 0;
		queue<TreeNode *> Q;
		Q.push(root);
		int ans = 0;

		//
		while (!Q.empty())
		{
			int sz = Q.size();

			//还有结点就会不断计算，直到最后null
			while (sz > 0)
			{
				TreeNode *node = Q.front();
				Q.pop();
				if (node->left)
					Q.push(node->left);
				if (node->right)
					Q.push(node->right);
				sz -= 1;
			}
			ans += 1;
		}
		return ans;
	}
};

```



### 105 Construct Binary Tree from Preorder and Inorder Traversal

给定一个二叉树的前序遍历和中序遍历结果，尝试复原这个树。已知树里不存在重复值的节
点

输入是两个一维数组，分别表示树的前序遍历和中序遍历结果；输出是一个二叉树

Input: preorder = [4,9,20,15,7], inorder = [9,4,15,20,7]
Output:
		4
	   / \
	  9  20
	     / \
		15  7

```cpp
// 主函数
TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder)
{
	if (preorder.empty())
	{
		return nullptr;
	}
	unordered_map<int, int> hash;
	for (int i = 0; i < preorder.size(); ++i)
	{
		hash[inorder[i]] = i;
	}
	return buildTreeHelper(hash, preorder, 0, preorder.size() - 1, 0);
}
// 辅函数
TreeNode *buildTreeHelper(unordered_map<int, int> &hash, vector<int> &preorder, int s0, int e0, int s1)
{
	if (s0 > e0)
	{
		return nullptr;
	}
	int mid = preorder[s1], index = hash[mid], leftLen = index - s0 - 1;
	TreeNode *node = new TreeNode(mid);
	node->left = buildTreeHelper(hash, preorder, s0, index - 1, s1 + 1);
	node->right = buildTreeHelper(hash, preorder, index + 1, e0, s1 + 2 + leftLen);
	return node;
}
```



### 106从中序和后序遍历序列构造二叉树


根据一棵树的中序遍历与后序遍历构造二叉树

可以假设树中没有重复的元素


中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]

返回：
    3
   / \
  9  20
    /  \
   15   7

```cpp
//递归遍历
// 中序遍历
class Solution
{
    int post_idx;
    unordered_map<int, int> idx_map;

public:
    TreeNode *helper(int in_left, int in_right, vector<int> &inorder, vector<int> &postorder)
    {
        // 如果这里没有节点构造二叉树了，就结束
        if (in_left > in_right)
        {
            return nullptr;
        }

        // 选择 post_idx 位置的元素作为当前子树根节点
        int root_val = postorder[post_idx];
        TreeNode *root = new TreeNode(root_val);

        // 根据 root 所在位置分成左右两棵子树
        int index = idx_map[root_val];

        // 下标减一
        post_idx--;
        // 构造右子树
        root->right = helper(index + 1, in_right, inorder, postorder);
        // 构造左子树
        root->left = helper(in_left, index - 1, inorder, postorder);
        return root;
    }
    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder)
    {
        // 从后序遍历的最后一个元素开始
        post_idx = (int)postorder.size() - 1;

        // 建立（元素，下标）键值对的哈希表
        int idx = 0;
        for (auto &val : inorder)
        {
            idx_map[val] = idx++;
        }
        return helper(0, (int)inorder.size() - 1, inorder, postorder);
    }
};

//迭代
class Solution
{
public:
    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder)
    {
        if (postorder.size() == 0)
        {
            return nullptr;
        }
        auto root = new TreeNode(postorder[postorder.size() - 1]);
        auto s = stack<TreeNode *>();
        s.push(root);
        int inorderIndex = inorder.size() - 1;
        for (int i = int(postorder.size()) - 2; i >= 0; i--)
        {
            int postorderVal = postorder[i];
            auto node = s.top();
            if (node->val != inorder[inorderIndex])
            {
                node->right = new TreeNode(postorderVal);
                s.push(node->right);
            }
            else
            {
                while (!s.empty() && s.top()->val == inorder[inorderIndex])
                {
                    node = s.top();
                    s.pop();
                    inorderIndex--;
                }
                node->left = new TreeNode(postorderVal);
                s.push(node->left);
            }
        }
        return root;
    }
};

```



### 109有序链表转换二叉搜索树





### 110BalancedBinaryTree





### 121股票交易



### 122买卖股票的最佳时机



### 126Word Ladder



### 128LongestConsecutiveSequence





### 135分糖果

























































