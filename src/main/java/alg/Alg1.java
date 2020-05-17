package alg;

import linknode.LinkNode;
import treenode.TreeNode;

import java.util.*;

public class Alg1 {



    public void reverse(LinkNode head)  // 翻转链表
    {
       LinkNode cur = head.next;
        LinkNode pre = head;
        pre.next = null;
        LinkNode tmp = new LinkNode();
        // 翻转链表
        while (cur != null) {
            tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        // 输出
        while (pre != null) {
            System.out.println(pre);
            pre = pre.next;
        }
    }

    public int []num(int []a,int k)//两数之和
    {
        int []res=new int[2];
    HashMap<Integer,Integer> map=new HashMap<Integer, Integer>();
        for(int i=0;i<a.length;i++)
        {
            if(map.containsKey(k-a[i]))
            {
                res[0]=i;
                res[1]=map.get(k-a[i]);

            }
            map.put(a[i],i);
        }
    return res;
}

    public LinkNode addtwonumbers(LinkNode l1,LinkNode l2) //两数相加
    {

    LinkNode pre=new LinkNode(0);
    LinkNode cre=pre;
    int carry=0;
    while(l1!=null||l2!=null)
    {
        int x=l1==null?0:l1.val;
        int y=l2==null?0:l2.val;
        int sum=x+y+carry;
        carry=sum/10;
        sum=sum%10;
        cre.next=new LinkNode(sum);
        cre=cre.next;
        if(l1!=null)
            l1=l1.next;
        if(l2!=null)
            l2=l2.next;
    }
    if(carry==1)
        cre.next=new LinkNode(carry);
    return pre.next;
}

    public int lengthOfLongestSubstring(String s)        //无重复字符的最长子串   滑动窗口法
    {
         int n=s.length();
    Set<Character> set=new HashSet<Character>();
    int i=0,j=0;
    int ans=0;
    while(i<n&&j<n)
    {
        if(!set.contains(s.charAt(j)))
        {
            set.add(s.charAt(j++));
            ans=Math.max(ans,j-i);
        }
        else
        set.remove(s.charAt(i++));
    }
    return ans;
}

    public double findMedianSortedArrays(int[] nums1, int[] nums2)  //寻找两个有序数组的中位数
    {
        int n=nums1.length+nums2.length;
        if(n%2==1)
            return findKthElm(nums1,0,nums1.length-1,nums2,0,nums2.length-1,(n+1)/2);
        else
            return (findKthElm(nums1,0,nums1.length-1,nums2,0,nums2.length-1,n/2)+findKthElm(nums1,0,nums1.length-1,nums2,0,nums2.length-1,n/2+1))/2.0;

    }
    private int findKthElm(int[] nums1, int abeg, int aend, int[] nums2, int bbeg, int bend, int k)
    {
        if(abeg>aend){//如果a数组没有元素，直接返回b数组的第K大、小元素
            return nums2[bbeg+k-1];
        }
        if(bbeg>bend){//如果b数组没有元素，直接返回a数组的第K大、小元素
            return nums1[abeg+k-1];
        }
        int amid=(abeg+aend)/2;
        int bmid=(bbeg+bend)/2;

        if(nums1[amid]<nums2[bmid]){//这里写成a[amid]<b[bmid]是求第k小，改成a[amid]>b[bmid]就是求第K大,本题是求第k小
            if(amid-abeg+bmid-bbeg+2>k){//如果a[mid]<b[bmid]且k小于两个数组大小之和的一半，k一定不在b的右半部分，所以递归在
                //整个数组a和b的左半部分找第K小
                return findKthElm(nums1,abeg,aend,nums2,bbeg,bmid-1,k);
            }
            else{//如果a[mid]<b[bmid]且k大于等于两个数组大小之和的一半，k一定不在a的左半部分，所以递归在整个b数组和a数组的
                //右半部分寻找第(k-a左半部分长度)小
                return findKthElm(nums1,amid+1,aend,nums2,bbeg,bend,k-(amid-abeg+1));
            }

        }
        else{//和以上注解一样
            if(amid-abeg+bmid-bbeg+2>k){
                return findKthElm(nums1,abeg,amid-1,nums2,bbeg,bend,k);
            }
            else{
                return findKthElm(nums1,abeg,aend,nums2,bmid+1,bend,k-(bmid-bbeg+1));
            }
        }
    }

    public String longestPalindrome(String s) //最长回文字串
    {
        int start=0,end=0;
        if(s==null||s.length()<1)
            return "";
        for(int i=0;i<s.length();i++)
        {
            int len1=expandAroundCenter(s,i,i);
            int len2=expandAroundCenter(s,i,i+1);
            int len=Math.max(len1,len2);
            if(len>end-start) //若换成len-1，则若长度相同不替换结果
            {
                start=i-(len-1)/2;
                end=i+len/2;
            }
        }
        return s.substring(start,end+1);
    }
    private int expandAroundCenter(String s, int left, int right)  //中心扩展法
    {
        while(left>=0&&right<s.length()&&s.charAt(left)==s.charAt(right))
        {
            left--;
            right++;
        }
        return right-left-1;

    }

    public String convert(String s, int numRows)  //Z字排列
    {
        if(numRows<2)
            return  s;
        List<StringBuffer> lists=new ArrayList<StringBuffer>();
        for(int i=0;i<numRows;i++)
            lists.add(new StringBuffer());
        int i=0,flag=-1;
        for(char c:s.toCharArray())
        {
            lists.get(i).append(c);
            if(i==0||i==numRows-1)
                flag=-flag;
            i+=flag;
        }
        StringBuilder res=new StringBuilder();
        for(StringBuffer list:lists)
            res.append(list);
        return res.toString();
    }

    public boolean isMatch(String s, String p) //字符串匹配
    {
        if(p.isEmpty())return s.isEmpty();
        boolean first_match=!s.isEmpty()&&(s.charAt(0)==p.charAt(0)||p.charAt(0)=='.');
        if(p.length()>2&&p.charAt(1)=='*')
        {
            return isMatch(s,p.substring(2))||(first_match&&isMatch(s.substring(1),p));
        }
        else
        {
            return first_match&&isMatch(s.substring(1),p.substring(1));
        }
    }

    public int maxArea(int[] height)   //盛水容器
    {
        int maxarea=0,l=0,r=height.length-1;
        while(l<r)
        {
            maxarea=Math.max(maxarea,Math.min(height[l],height[r])*(r-l));
            if(height[l]<height[r])
                l++;
            else
                r--;
        }
        return  maxarea;
    }

    public  List<List<Integer>> threeSum(int[] nums)//三数之和
    {
        List<List<Integer>> ans=new ArrayList<List<Integer>>();
        int len=nums.length;
        if(len<3) return ans;
        Arrays.sort(nums);
        for(int i=0;i<len;i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            ;
            int l = i + 1;
            int r = len - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (l < r && nums[l] == nums[l + 1]) l++;
                    while (l < r && nums[r] == nums[r - 1]) r--;
                    l++;
                    r--;
                } else if (sum > 0) {
                    r--;
                } else if (sum < 0) {
                    l++;
                }
            }
        }
        return ans;
    }

    public List<String> letterCombinations(String digits) //电话号码的字母组合
    {
        LinkedList<String> ans=new LinkedList<String>();
        ans.add("");
        if(digits.isEmpty()) return ans;
        String [ ]maps=new String[]{"0","1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        for(int i=0;i<digits.length();i++)
        {
            int x=Integer.parseInt(String.valueOf(digits.charAt(i)));
            while(ans.peek().length()==i)
            {
                String t=ans.remove();
                for(char s:maps[x].toCharArray())
                    ans.add(t+s);
            }
        }
        return ans;
    }

    public LinkNode removeNthFromEnd(LinkNode head, int n) //删除链表倒数第k个节点
    {
        LinkNode pre=new LinkNode(0);
        pre.next=head;
        LinkNode first=head;
        LinkNode second=head;
        for(int i=0;i<n;i++)
            first=first.next;
        while(first.next!=null)
        {
            first=first.next;
            second=second.next;
        }
        second.next=second.next.next;
        return pre.next;
    }

    public boolean isValid(String s) //有效的括号
    {
        Map<Character,Character> map=new HashMap<Character, Character>();
        map.put('}','{');
        map.put(')','(');
        map.put(']','[');
        Stack<Character> stack=new Stack<Character>();
        for(int i=0;i<s.length();i++)
        {
            char c=s.charAt(i);
            if(map.containsKey(c))
            {
                char topelement=stack.isEmpty()?'#':stack.pop();
                if(topelement!=map.get(c))
                    return  false;
            }
            else
                stack.push(c);
        }
        return stack.isEmpty();

    }

    public LinkNode mergeTwoLists(LinkNode l1, LinkNode l2) //合并两个有序链表
    {
        if(l1==null)
            return  l2;
        else  if(l2==null)
            return l1;
        else if(l1.val<l2.val)
        {
            l1.next=mergeTwoLists(l1.next,l2);
            return l1;
        }
        else
        {
            l2.next=mergeTwoLists(l1,l2.next);
            return l2;
        }

    }

    public List<String> generateParenthesis(int n) //括号生成
    {
        List<String> list=new ArrayList<String>();
        backtrack(list,"",0,0,n);
        return list;
    }
    private void backtrack(List<String> list, String s, int open, int close, int n) {
        if(open+close==2*n)
        {
            list.add(s);
            return;
        }
        if(open<n)
            backtrack(list,s+'(',open+1,close,n);
        if(close<open)
            backtrack(list,s+')',open,close+1,n);
    }//回溯括号生成

    public LinkNode mergeKLists(LinkNode[] lists) //合并k个链表
    {
        int len=lists.length;
        if(len==0)
            return null;
        while(len>1)
        {
            for(int i=0;i<len/2;i++)
            {
                lists[i]=mergeTwoLists(lists[i],lists[len-1]);
            }
            len=(len+1)/2;
        }
        return lists[0];
    }

    public int [] nextPermutation(int[] nums) //下一个排列
    {
        int len=nums.length-2;
        while(len>=0&&nums[len]>=nums[len+1])
            len--;
        if(len>=0)
        {
            int j=nums.length-1;
            while(j>=0&&nums[j]<=nums[len])
                j--;
            swap(nums,len,j);
            return nums;
        }
        reverse(nums,len+1);
        return nums;
        //降序排序
    }
    private  void swap(int []nums,int i,int j) //交换
    {
        int temp=nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }
    private void reverse(int[] nums, int start) //反转数组
    {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    public int longestValidParentheses(String s) //最长有效括号
    {
        Stack<Character> stack=new Stack<Character>();
        int len=s.length();
        int count=0;
        for(int i=0;i<len;i++)
        {
            if(stack.isEmpty())
                stack.push(s.charAt(i));
            else if(s.charAt(i)=='('||(s.charAt(i)==')')&&stack.peek()!='(')
                stack.push(s.charAt(i));
            else
            {
                stack.pop();
                count=count+2;
            }
        }
        return count;
    }

    public int search(int[] nums, int target)  //搜索旋转排序数组
    {
        if(nums==null||nums.length==0)
            return -1;
        int end=nums.length-1;
        int start=0;
        while(start<=end)
        {
            int mid=(start+end)/2;
            if(nums[mid]==target)
                return mid;
            if(nums[start]<=nums[mid])
            {
                if(target>=nums[start]&&target<=nums[mid])
                   end=mid-1;
                else
                    start=mid+1;
            }
            else
            {
                if(target<=nums[end]&&target>=nums[mid])
                    start=mid+1;
                else
                    end=mid-1;
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) //排序数组中查找元素第一个和最后一个位置
    {
        int []res=new int[2];
        res[0]=findfirst(nums,target);
        res[1]=findlast(nums,target);
        return res;
    }
    private int findlast(int[] nums, int target) {
        int low=0;
        int high=nums.length-1;
        while(low<=high)
        {
            int mid=(low+high)/2;
            if(nums[mid]<target)
                low=mid+1;
            else if(nums[mid]>target)
                high=mid-1;
            else
            {
                if(mid==nums.length-1||(nums[mid+1]!=target))
                    return mid;
                else
                    low=mid+1;
            }
        }
        return -1;
    }
    private int findfirst(int[] nums, int target) {
        int low=0;
        int high=nums.length-1;
        while(low<=high)
        {
            int mid=(low+high)/2;
            if(nums[mid]<target)
                low=mid+1;
            else if(nums[mid]>target)
                high=mid-1;
            else
            {
                if(mid==0|(nums[mid-1]!=target))
                    return mid;
                else
                    high=mid-1;
            }
        }
        return -1;
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) //组合总和
    {

        int len=candidates.length;
        if(len==0)
            return null;
        Arrays.sort(candidates);
        List<List<Integer>> res=new ArrayList<List<Integer>>();
        res= findCombinationSum(candidates,target,0,res,new Stack<Integer>());
        return res;
    }
    private List<List<Integer>> findCombinationSum(int[] candidates, int target, int i, List<List<Integer>> res, Stack<Integer> stack) {
        if(target==0)
        {
            res.add(new ArrayList<Integer>(stack));
            return res;
        }
        for(int k=i;k<candidates.length&&target>=candidates[i];k++)
        {
            stack.add(candidates[k]);
            findCombinationSum(candidates,target-candidates[k],k,res,stack);
            stack.pop();
        }
        return res;
    }

    public int trap(int[] height) //接雨水
    {
        int left=0,right=height.length-1;
        int ans=0,left_max=0,right_max=0;
        while(left<right)
        {
            if(height[left]<height[right])
            {
                if (height[left] >= left_max) {
                    left_max = height[left];
                } else {
                    ans += left_max - height[left];
                }
                left++;
            }
            else
            {
                if (height[right] >= right_max) {
                    right_max = height[right];
                } else {
                    ans += right_max-height[right];
                }
               right--;
            }
        }
        return ans;
    }

    public List<List<Integer>> permute(int[] nums) //全排列
    {
        if(nums==null||nums.length==0)
            return null;
        List<List<Integer>> lists=new ArrayList<List<Integer>>();
        int []visit=new int[nums.length];
        backtrack(lists,nums,visit,new ArrayList<Integer>());
        return lists;
    }
    private void backtrack(List<List<Integer>> lists,int []nums, int[] visit, ArrayList<Integer> integers) {
        if(integers.size()==nums.length)
        {
            lists.add(new ArrayList<Integer>(integers));
            return ;
        }
        for(int i=0;i<nums.length;i++)
        {
            if(visit[i]==1)
                continue;
            visit[i]=1;
            integers.add(nums[i]);
            backtrack(lists,nums,visit,integers);
            integers.remove(integers.size()-1);
            visit[i]=0;
        }
    }

    public int [][] rotate(int[][] matrix) //旋转图像,先转置后水平镜像
    {
        int row=matrix.length;
        int col=matrix[0].length;
        int temp=0;
        for(int i=0;i<row;i++)
            for(int j=i;j<col;j++)
            {
                temp=matrix[i][j];
                matrix[i][j]=matrix[j][i];
                matrix[j][i]=temp;
            }
        for(int j=0;j<row;j++)
        for(int i=0;i<col/2;i++)
        {
            temp=matrix[j][i];
            matrix[j][i]=matrix[j][col-1-i];
            matrix[j][col-1-i]=temp;
        }
       return matrix;
    }

    @SuppressWarnings("unchecked")
    public ArrayList groupAnagrams(String[] strs) //字母异位词分组
    {
        if(strs.length==0)
            return null;
        Map<String,List> map=new HashMap<String, List>();
        for(String s:strs)
        {
            char []c=s.toCharArray();
            Arrays.sort(c);
            String key=String.valueOf(c);
            if(!map.containsKey(key))
                map.put(key,new ArrayList());
            map.get(key).add(s);
        }
        return new ArrayList(map.values());
    }


    public int maxSubArray(int[] nums) //最大子序和
    {
        int n=nums.length;
        int max=0;
        for(int i=1;i<n;i++)
        {
            if(nums[i-1]>0) nums[i]+=nums[i-1];
            max=Math.max(nums[i],max);
        }
        return max;

    }

    enum Index
    {
        GOOD,BAD,UNKNOWN
    }
    public boolean canJump(int[] nums) //跳跃游戏
    {
        int len=nums.length;
        Index []memo=new Index[nums.length];
        for(int i=0;i<nums.length;i++)
            memo[i]=Index.UNKNOWN;
        memo[len-1]=Index.GOOD;
        for(int i=len-2;i>=0;i--)
        {
            int furstjump=Math.min(len-1,i+nums[i]);
            for(int j=i+1;j<=furstjump;j++)
            {
                if(memo[j]==Index.GOOD)
                {
                    memo[i]=Index.GOOD;
                    break;
                }
            }
        }
        return memo[0]==Index.GOOD;
    }

    public int[][] merge(int[][] intervals) //合并区间
    {
        LinkedList<int []> res=new LinkedList<int[]>();
        if(intervals==null||intervals.length==0)
            return res.toArray(new int[0][]);
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0]-o2[0];
            }
        });
        for (int[] interval : intervals) {
            if (res.isEmpty() || res.getLast()[1] < interval[0])
                res.add(interval);
            else
                res.getLast()[1] = Math.max(res.getLast()[1], interval[1]);
        }
        return res.toArray(new int [0][0]);
    }

    public int uniquePaths(int m, int n) //不同路径
    {
        int [][]dp=new int[m][n];
        for(int i=0;i<m;i++) dp[i][0]=1;
        for(int i=0;i<n;i++)dp[0][i]=1;
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
         return dp[m-1][n-1];
    }

    public int minPathSum(int[][] grid) //最小路径和
    {
        for(int i=grid.length-1;i>=0;i--)
            for(int j=grid[0].length-1;j>=0;j--)
            {
                if(i==grid.length-1&&j!=grid[0].length-1)
                    grid[i][j]+=grid[i][j+1];
                else if(j==grid[0].length-1&&i!=grid.length-1)
                    grid[i][j]+=grid[i+1][j];
                else if(i!=grid.length-1&&j!=grid[0].length-1)
                    grid[i][j]+=Math.min(grid[i+1][j],grid[i][j+1]);
            }
        return grid[0][0];
    }

    public int climbStairs(int n) //爬楼梯
    {
        if(n==1)
            return 1;
        int []dp=new int[n+1];
        dp[1]=1;
        dp[2]=2;
        for(int i=3;i<=n;i++)
            dp[i]=dp[i-1]+dp[i-2];
        return dp[n];
    }

    public int minDistance(String word1, String word2) //编辑距离
    {
        int len1=word1.length();
        int len2=word2.length();
        int [][]nums=new int[len1+1][len2+1];
        for(int i=0;i<=len1;i++)
            nums[i][0]=i;
        for(int j=0;j<=len2;j++)
            nums[0][j]=j;
        for(int i=1;i<=len1;i++)
            for(int j=1;j<=len2;j++)
            {
                if(word1.charAt(i-1)==word2.charAt(j-1))
                    nums[i][j]=1+Math.min(Math.min(nums[i-1][j],nums[i][j-1]),nums[i-1][j-1]-1);
                else
                    nums[i][j]=1+Math.min(Math.min(nums[i-1][j],nums[i][j-1]),nums[i-1][j-1]);
            }
        return nums[len1-1][len2-1];
    }

    public int []sortColors(int[] nums) //荷兰三色旗问题
    {
        int p0=0,cur=0,p1=nums.length-1,temp=0;
        while(cur<p1)
        {
            if(nums[cur]==0)
            {
                temp=nums[p0];
                nums[p0++]=nums[cur];
                nums[cur++]=temp;
            }
            else if(nums[cur]==2)
            {
                temp=nums[p1];
                nums[p1--]=nums[cur];
                nums[cur++]=temp;
            }
            else
                cur++;
        }
        return nums;
    }

    public String minWindow(String s, String t) //最小覆盖字串
    {
        Map<Character,Integer> map=new HashMap<Character, Integer>();
        int left=0,right=0,ans_left=0,ans_right=0,ans_len=Integer.MAX_VALUE;
        for(int i=0;i<t.length();i++)
            map.put(t.charAt(i),map.getOrDefault(t.charAt(i),0)+1);
        while(right<s.length())
        {
            if(map.containsKey(s.charAt(right)))
            {
                map.put(s.charAt(right),map.get(s.charAt(right))-1);
                while(match(map))
                {
                    int len=right-left+1;
                    if(len<ans_len)
                    {
                        ans_left=left;
                        ans_right=right;
                        ans_len=len;
                    }
                    if(map.containsKey(s.charAt(left)))
                    {
                        map.put(s.charAt(left),map.get(s.charAt(left))+1);
                    }
                    left++;
                }
            }
            right++;
        }
        return s.substring(ans_left,ans_right+1);
    }
    private boolean match(Map<Character, Integer> map) {
        for(Integer it:map.values())
            if(it>0)
                return false;
        return true;
    }

    public List<List<Integer>> subsets(int[] nums) //子集
    {
        List<List<Integer>> res=new ArrayList<>();
        backtrack(0,res,nums,new ArrayList<Integer>());
        return res;
    }
    private void backtrack(int i, List<List<Integer>> res, int[] nums, ArrayList<Integer> tmp) {
        res.add(new ArrayList<>(tmp));
        for(int j=i;j<nums.length;j++)
        {
            tmp.add(nums[j]);
            backtrack(j+1,res,nums,tmp);
            tmp.remove(tmp.size()-1);
        }
    }

    public boolean exist(char[][] board, String word) //单词搜索
    {
        boolean [][]visited=new boolean[board.length][board[0].length];
        for(int i=0;i<board.length;i++)
            for(int j=0;j<board[0].length;j++)
            {
                if(board[i][j]==word.charAt(0)&&backtrack(i,j,0,word,board,visited))
                    return  true;
            }
        return false;
    }
    private boolean backtrack(int i, int j, int k, String word, char[][] board, boolean[][] visited) {
        if(k==word.length())
            return true;
        if(i<0||j<0||i>=board.length||j>=board[0].length||board[i][j]!=word.charAt(k)||visited[i][j])
            return false;
        visited[i][j]=true;
        if(backtrack(i-1,j,k+1,word,board,visited)||backtrack(i+1,j,k+1,word,board,visited)||backtrack(i,j-1,k+1,word,board,visited)||backtrack(i,j+1,k+1,word,board,visited))
            return true;
        visited[i][j]=false;
        return false;
    }

    public int largestRectangleArea(int[] heights) //柱状图中最大的矩形
    {

        return calculateArea(heights,0,heights.length-1);

    }
    private int calculateArea(int[] heights, int start, int end) {
        int minindex=start;
        if(start>end)
            return 0;
        for(int i=start;i<=end;i++)
            if(heights[i]<heights[minindex])
                minindex=i;
         return Math.max((heights[minindex]*(end-start+1)),Math.max(calculateArea(heights,start,minindex-1),calculateArea(heights,minindex+1,end)));
    }

    public int maximalRectangle(char[][] matrix) //最大矩形
    {
        int res=0,tmp=0;
        if(matrix==null||matrix.length==0||matrix[0].length==0)
            return res;
        int row=matrix.length;
        int col=matrix[0].length;
        int mint[][]=new int[row][col];
        for(int i=0;i<col;i++)
            mint[0][i]=matrix[0][i]-48;
        for(int i=1;i<row;i++)
            for(int j=0;j<col;j++)
            {
                if(matrix[i][j]=='1')
                    mint[i][j]=mint[i-1][j]+1;
                else
                    mint[i][j]=0;
            }
         for(int i=0;i<row;i++)
             for(int j=0;j<col;j++)
             {
                 int left=j-1;
                 int right=j+1;
                 int wide=1;
                 while(left>=0&&mint[i][left]>=mint[i][j])
                 {
                     left--;
                     wide++;
                 }
                 while(right<col&&mint[i][right]>=mint[i][j])
                 {
                     right++;
                     wide++;
                 }
                 tmp=wide*(mint[i][j]);
                 if(tmp>res)
                     res=tmp;
             }
         return res;
    }

    public List<Integer> inorderTraversal(TreeNode root) //中序遍历
    {
        List<Integer> res=new LinkedList<>();
        midsearch(res,root);
        return res;
    }
    private void midsearch(List<Integer> res, TreeNode root) {
        if(root.left!=null)
            midsearch(res,root.left);
        res.add(root.val);
        if(root.right!=null)
            midsearch(res,root.right);
    }

    public int numTrees(int n) //不同的二叉搜索树
    {

        int []nums=new int[n+1];
        nums[0]=1;
        nums[1]=1;
        for(int i=2;i<=n;i++)
            for(int j=1;j<=i;j++)
                nums[i]+=nums[j-1]*nums[i-j];
         return nums[n];
    }

    public boolean isValidBST(TreeNode root) //验证二叉搜索树
    {
        if(root.left==null&&root.right==null)
            return true;
        if(root.left!=null&&root.left.val>root.val)
            return false;
        if(root.right!=null&&root.right.val<root.val)
            return false;
        if(isValidBST(root.left)&&isValidBST(root.right))
            return true;
        return false;
    }

    public boolean isSymmetric(TreeNode root) //对称二叉树
    {
        return ismirror(root,root);
    }
    private boolean ismirror(TreeNode root1, TreeNode root2) {
        if(root1==null&&root2==null)
            return true;
        if(root1==null||root2==null)
            return  false;
        return  (root1.val==root2.val)&&ismirror(root1.left,root2.right)&&ismirror(root1.right,root2.left);

    }

    public List<List<Integer>> levelOrder(TreeNode root) //二叉树的层次遍历
    {
        List<List<Integer>> res=new ArrayList<>();
        if(root==null)
            return  res;
        helper(root,0,res);
        return res;
    }
    private void helper(TreeNode root, int i, List<List<Integer>> res) {
        if(res.size()==i)
            res.add(new ArrayList<Integer>());
        res.get(i).add(root.val);
        if(root.left!=null)
            helper(root.left,i+1,res);
        if(root.right!=null)
            helper(root.right,i+1,res);
    }

    public int maxDepth(TreeNode root) //二叉树的最大深度
    {
        //迭代dfs
//        int level=0;
//        if(root==null)
//            return level;
//        Queue<TreeNode> queue=new LinkedList<>();
//        queue.add(root);
//        while(!queue.isEmpty())
//        {
//            int level_len=queue.size();
//            for(int i=0;i<level_len;i++)
//            {
//                TreeNode node=queue.remove();
//                if(node.left!=null)
//                    queue.add(node.left);
//                if(node.right!=null)
//                    queue.add(node.right);
//            }
//            level++;
//        }
//        return level;
        int level=0;
        if(root==null)
            return level;
        else
            return 1+Math.max(maxDepth(root.left),maxDepth(root.right));
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) //前序中序构造二叉树
    {
        if (preorder == null || preorder.length == 0 || inorder == null || inorder.length == 0 || preorder.length != inorder.length) {
            return null;
        }
        return help(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }
    private TreeNode help(int[] preorder, int i, int i1, int[] inorder, int i2, int i3) {
        if(i>i1||i2>i3)
            return null;
        TreeNode node=new TreeNode(preorder[i]);
        int index=0;
        while(inorder[i2+index]!=preorder[i])
            index++;
        node.left=help(preorder,i+1,i+index,inorder,i2,i2+index-1);
        node.right=help(preorder,i+index+1,i1,inorder,i2+index+1,i3);
        return node;
    }

    public void  flatten(TreeNode root)  //二叉树展开为链表
    {
        if(root == null){
            return ;
        }
        //将根节点的左子树变成链表
        flatten(root.left);
        //将根节点的右子树变成链表
        flatten(root.right);
        TreeNode temp = root.right;
        //把树的右边换成左边的链表
        root.right = root.left;
        //记得要将左边置空
        root.left = null;
        //找到树的最右边的节点
        while(root.right != null) root = root.right;
        //把右边的链表接到刚才树的最右边的节点
        root.right = temp;
    }

    public int maxProfit(int[] prices) //买卖股票的最佳时机
    {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice)
                minprice = prices[i];
            else if (prices[i] - minprice > maxprofit)
                maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }



    public int maxPathSum(TreeNode root) //二叉树中d的最大路径和
    {
        int []max={Integer.MIN_VALUE};
        maxPathSums(max,root);
        return max[0];
    }
    private int maxPathSums(int []max, TreeNode root)
    {

        if(root==null)
            return 0;
        int leftMax=Math.max(maxPathSums(max,root.left),0);
        int rightMax=Math.max(maxPathSums(max,root.right),0);
        max[0]=Math.max(max[0],leftMax+rightMax+root.val);
        return root.val+Math.max(leftMax,rightMax);
    }


    public int longestConsecutive(int[] nums)  //最长连续序列,时间复杂度为 O(n)
    {
        Set<Integer> num_set=new HashSet<>();
        for(int numS: nums)
            num_set.add(numS);
        int longLength=0;
        for(int num:num_set)
        {
            if(!num_set.contains(num-1))
            {
                int currentNum = num;
                int currentStreak = 1;
                while (num_set.contains(currentNum+1))
                {
                    currentStreak++;
                    currentNum++;
                }
                longLength=Math.max(longLength,currentStreak);
            }
        }
        return longLength;

    }


    public int singleNumber(int[] nums) //只出现一次的数字
    {
        int res=0;
        for(int num:nums)
            res=res^num;
        return res;
    }

    public boolean wordBreak(String s, List<String> wordDict) //单词拆分
    {
        Set<String> set=new HashSet<>(wordDict);
        boolean []dp=new boolean[s.length()+1];
        dp[0]=true;
        for(int i=1;i<=s.length();i++)
            for(int j=0;j<i;j++)
                if(dp[j]&&set.contains(s.substring(j,i)))
                {
                    dp[i]=true;
                    break;
                }
        return dp[s.length()];
    }


    public boolean hasCycle(LinkNode head) //环形链表
    {
        Set<LinkNode> set=new HashSet<>();
        while(head!=null)
        {
            if(set.contains(head))
                return true;
            else
                set.add(head);
            head=head.next;
        }
        return false;
    }


}
