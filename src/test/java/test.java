import alg.Alg1;
import linknode.LinkNode;
import org.junit.Test;
import sun.awt.image.ImageWatched;
import treenode.TreeNode;

import java.util.*;

public class test {


    @Test
    public  void test1()//两数之和
    {
        Alg1 alg=new Alg1();
        int []a={2,4,5,7,8};
        int []test=alg.num(a,9);
        for (int x:test)
            System.out.println(x);
    }

    @Test
    public  void test2() //两数相加 987+23=1010
    {
        LinkNode l1=new LinkNode(7);
        l1.next=new LinkNode(8);
        l1.next.next=new LinkNode(9);

        LinkNode l2=new LinkNode(3);
        l2.next=new LinkNode(2);

        Alg1 alg=new Alg1();
       LinkNode res=alg.addtwonumbers(l1,l2);

        Stack<LinkNode> s=new Stack<LinkNode>();
        while(res!=null)
        {
            s.add(res);
            res=res.next;
        }

        while(!s.empty())
            System.out.print(s.pop().val);
    }

    @Test
    public void test3()  //无重复字符的最长子串   滑动窗口法
    {
        String s="pwwkew";
        Alg1 alg=new Alg1();
        int res=alg.lengthOfLongestSubstring(s);
        System.out.println(res);
    }

    @Test
    public void test4()  //寻找两个有序数组的中位数
    {
        Alg1 alg=new Alg1();
        int  []nums1={1,3,5};
        int []nums2={2,4,6,8};
        double res=alg.findMedianSortedArrays(nums1,nums2);
        System.out.println(res);
    }

    @Test
    public void test5()   //最长回文字串
    {
        String s="babad";
        Alg1 alg=new Alg1();
        String res=alg.longestPalindrome(s);
        System.out.println(res);
    }

    @Test
    public void test6()    //Z字排列
    {
        String s="LEETCODEISHIRING";
        Alg1 alg=new Alg1();
        String res=alg.convert(s,3);
        System.out.println(res);
    }

    @Test
    public void test7()    //字符串匹配
    {
        String s="aab";
        String p="c*a*b";
        Alg1 alg=new Alg1();
        boolean res=alg.isMatch(s,p);
        System.out.println(res);
    }

    @Test
    public void test8()    //盛最多水的容器
    {
        int []height={1,8,6,2,5,4,8,3,7};
        Alg1 alg=new Alg1();
        int res=alg.maxArea(height);
        System.out.println(res);
    }

    @Test
    public void test9()    //三数之和
    {
        int  []nums={-1, 0, 1, 2, -1, -4};
        Alg1 alg=new Alg1();
        List<List<Integer>> ress=alg.threeSum(nums);
        for( List<Integer> res:ress)
            System.out.println(res);
    }

    @Test
    public void test10()    //电话号码的字母组合
    {
        String s="23";
        Alg1 alg=new Alg1();
        List<String> res=alg.letterCombinations(s);
        for( String t:res)
            System.out.println(t);
    }

    @Test
    public void test11()   //删除链表第k个节点
    {
        LinkNode linkNode=new LinkNode(1);
        linkNode.next=new LinkNode(2);
        linkNode.next.next=new LinkNode(3);
        linkNode.next.next.next=new LinkNode(4);
        linkNode.next.next.next.next=new LinkNode(5);
        Alg1 alg=new Alg1();
        LinkNode slinknode=alg.removeNthFromEnd(linkNode,2);
        while(slinknode!=null)
        {
            System.out.println(slinknode.val);
            slinknode=slinknode.next;
        }
    }

    @Test
    public void test12()     //有效的括号
    {
        String s="{[()]}";
        Alg1 alg=new Alg1();
        boolean istrue=alg.isValid(s);
        System.out.println(istrue);
    }

    @Test
    public void test13()       //合并两个有序链表
    {
        LinkNode linkNode1=new LinkNode(1);
        linkNode1.next=new LinkNode(2);
        linkNode1.next.next=new LinkNode(4);
        LinkNode linkNode2=new LinkNode(1);
        linkNode2.next=new LinkNode(3);
        linkNode2.next.next=new LinkNode(4);
        Alg1 alg=new Alg1();
        LinkNode res=alg.mergeTwoLists(linkNode1,linkNode2);
        while(res!=null)
        {
            System.out.println(res.val);
            res=res.next;
        }
    }

    @Test
    public void test14()     //括号生成
    {
        Alg1 alg=new Alg1();
        List<String>list=alg.generateParenthesis(3);
        for(String res:list)
            System.out.println(res);
    }

    @Test
    public void test15()     //合并k个有序链表
    {
        LinkNode linkNode1=new LinkNode(1);
        linkNode1.next=new LinkNode(4);
        linkNode1.next.next=new LinkNode(5);
        LinkNode linkNode2=new LinkNode(1);
        linkNode2.next=new LinkNode(3);
        linkNode2.next.next=new LinkNode(4);
        LinkNode linkNode3=new LinkNode(2);
        linkNode3.next=new LinkNode(6);
        Alg1 alg=new Alg1();
        LinkNode []linkNodes=new LinkNode[3];
        linkNodes[0]=linkNode1;
        linkNodes[1]=linkNode2;
        linkNodes[2]=linkNode3;
       LinkNode res=alg.mergeKLists(linkNodes);
       while(res!=null)
       {
           System.out.println(res.val);
           res=res.next;
       }
    }

    @Test
    public void test16()     //下一个排列
    {
        Alg1 alg=new Alg1();
        int []nums={1,5,8,4,7,6,5,3,2};
        int []res=alg.nextPermutation(nums);
        for( int i:res)
            System.out.println(i);
    }

    @Test
    public void test17()     //最长有效括号
    {
        String s=")()())(()";
        Alg1 alg=new Alg1();
        int res=alg.longestValidParentheses(s);
        System.out.print(res);
    }

    @Test
    public void test18()      //搜索旋转排序数组
    {
        int []nums={4,5,6,7,0,1,2};
        Alg1 alg=new Alg1();
        int res=alg.search(nums,0);
        System.out.print(res);
    }

    @Test
    public void test19()      //搜索旋转排序数组
    {
        int []nums={45,7,7,8,8,10};
        Alg1 alg=new Alg1();
        int []res=alg.searchRange(nums,8);
        for(int i:res)
            System.out.println(i);
    }

    @Test
    public void test20()      //组合总和
    {
        int []nums={2,3,5};
        Alg1 alg=new Alg1();
        List<List<Integer>> res=alg.combinationSum(nums,8);
        for(List<Integer> it:res)
            System.out.println(it);
    }

    @Test
    public void test21()      //组合总和
    {
        int []nums={0,1,0,2,1,0,1,3,2,1,2,1};
        Alg1 alg=new Alg1();
        int res=alg.trap(nums);
        System.out.println(res);
    }

    @Test
    public void test22()      //全排列
    {
        int []nums={1,2,3};
        Alg1 alg=new Alg1();
       List<List<Integer>> res=alg.permute(nums);
       for(List<Integer> it:res)
           System.out.println(it);
    }

    @Test
    public void test23()      //旋转图像
    {
       int [][]matrix =new int[][]{{1,2,3},{4,5,6},{7,8,9}};
        Alg1 alg=new Alg1();
        int [][]res=alg.rotate(matrix);
        for(int []it:res)
        {
            for(int re:it)
                System.out.print(re+" ");
            System.out.println();
        }


    }

    @Test
    @SuppressWarnings("unchecked")
    public void test24()      //字母异位词分组
    {
        String []s={"eat", "tea", "tan", "ate", "nat", "bat"};
        Alg1 alg=new Alg1();
        List<List<String>> res=alg.groupAnagrams(s);
        for(List<String> it:res)
            System.out.println(it);
    }

    @Test
    public void test25()      //最大子序和
    {
        int []nums={-2,1,-3,4,-1,2,1,-5,4};
        Alg1 alg=new Alg1();
        int res=alg.maxSubArray(nums);
        System.out.println(res);
    }

    @Test
    public void test26()      //跳跃游戏
    {
        int []nums={2,3,1,1,4};
        Alg1 alg=new Alg1();
        boolean res=alg.canJump(nums);
        System.out.println(res);
    }

    @Test
    public void test27()      //合并区间
    {
        int [][]nums={{1,3},{2,6},{8,10},{15,18},{18,20}};
        Alg1 alg=new Alg1();
        int [][]res=alg.merge(nums);
        for(int []it:res)
        {
            for(int i:it)
                System.out.print(i+" ");
            System.out.println();
        }

    }

    @Test
    public void test28()      //不同路径
    {
        Alg1 alg=new Alg1();
        int res=alg.uniquePaths(7,3);
        System.out.println(res);
    }

    @Test
    public void test29()      //最小路径和
    {
        int [][]nums={{1,3,1},{1,5,1},{4,2,1}};
        Alg1 alg=new Alg1();
        int res=alg.minPathSum(nums);
        System.out.println(res);
    }

    @Test
    public void test30()      //爬楼梯
    {
        Alg1 alg=new Alg1();
       int res=alg.climbStairs(3);
        System.out.println(res);

    }

    @Test
    public void test31()      //编辑距离
    {
        String word1="intention";
        String word2="execution";
        Alg1 alg=new Alg1();
        int res=alg.minDistance(word1,word2);
        System.out.println(res);
    }

    @Test
    public void test32()      //荷兰三色旗问题
    {
        int []nums={2,0,2,1,1,0};
        Alg1 alg=new Alg1();
        nums=alg.sortColors(nums);
        for (int res:nums) {
            System.out.println(res);
        }
    }

    @Test
    public void test33()      //最小覆盖子串
    {
        String s="ADOBECODEBANC";
        String t="ABC";
        Alg1 alg=new Alg1();
        String res=alg.minWindow(s,t);
        System.out.println(res);
    }

    @Test
    public void test34()      //子集
    {
        int []nums={1,2,3};
        Alg1 alg=new Alg1();
        List<List<Integer>> res=alg.subsets(nums);
        for(List<Integer> it:res)
            System.out.println(it);
    }

    @Test
    public void test35()      //单词搜索
    {
       char [][]nums={{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
       String word="ABCCED";
        Alg1 alg=new Alg1();
        boolean res=alg.exist(nums,word);
        System.out.println(res);
    }

    @Test
    public void test36()      //柱状图中最大的矩形
    {
        int []nums={2,1,5,6,2,3};
        Alg1 alg=new Alg1();
        int res=alg.largestRectangleArea(nums);
        System.out.println(res);
    }

    @Test
    public void test37()      //最大矩形
    {
        char [][]nums={{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}};
        Alg1 alg=new Alg1();
        int res=alg.maximalRectangle(nums);
        System.out.println(res);
    }

    @Test
    public void test38()      //中序遍历
    {
        TreeNode root=new TreeNode(1);
        root.right=new TreeNode(2);
        root.right.left=new TreeNode(3);
        Alg1 alg=new Alg1();
        List<Integer>res=alg.inorderTraversal(root);
        System.out.println(res);
    }

    @Test
    public void test39()      //不同的二叉搜索树
    {

        Alg1 alg=new Alg1();
        int res=alg.numTrees(3);
        System.out.println(res);
    }

    @Test
    public void test40()      //验证二叉搜索树
    {
        TreeNode root = new TreeNode(5);
        root.left=new TreeNode(1);
        root.right=new TreeNode(4);
        root.right.left=new TreeNode(3);
        root.right.right=new TreeNode(6);
        Alg1 alg=new Alg1();
        boolean res=alg.isValidBST(root);
        System.out.println(res);
    }

    @Test
    public void test41()      //对称二叉树
    {
        TreeNode root = new TreeNode(1);
        root.left=new TreeNode(2);
        root.right=new TreeNode(2);
        root.right.left=new TreeNode(3);
        root.left.right=new TreeNode(3);
        Alg1 alg=new Alg1();
        boolean res=alg.isSymmetric(root);
        System.out.println(res);
    }

    @Test
    public void test42()      //二叉树的层次遍历
    {
        TreeNode root=new TreeNode(3);
        root.left=new TreeNode(9);
        root.right=new TreeNode(20);
        root.right.left=new TreeNode(15);
        root.right.right=new TreeNode(7);
        Alg1 alg=new Alg1();
        List<List<Integer>> res=alg.levelOrder(root);
        System.out.println(res);
    }

    @Test
    public void test43()      //最大深度
    {
        TreeNode root=new TreeNode(3);
        root.left=new TreeNode(9);
        root.right=new TreeNode(20);
        root.right.left=new TreeNode(15);
        root.right.right=new TreeNode(7);
        Alg1 alg=new Alg1();
        int res=alg.maxDepth(root);
        System.out.println(res);
    }

    @Test
    public void test44()      //前序中序构造二叉树
    {
        int []preorder ={3,9,20,15,7};
        int []inorder ={9,3,15,20,7};
        Alg1 alg=new Alg1();
        TreeNode res=alg.buildTree(preorder,inorder);
        System.out.println(res.val);
        System.out.println(res.left.val);
        System.out.println(res.right.val);
        System.out.println(res.right.left.val);
        System.out.println(res.right.right.val);
    }



    @Test
    public void Test45() //二叉树展开为链表
    {
        TreeNode root=new TreeNode(1);
        root.left=new TreeNode(2);
        root.left.left=new TreeNode(3);
        root.left.right=new TreeNode(4);
        root.right=new TreeNode(5);
        root.right.right=new TreeNode(6);
        Alg1 alg=new Alg1();
        alg.flatten(root);
        while(root!=null)
        {
            System.out.print(root.val+" ");
            root=root.right;
        }

    }


    @Test
    public void Test46()  //买卖股票的最佳时机
    {
        int []prices={7,2,1,3,7,4};
        Alg1 alg=new Alg1();
        int res=alg.maxProfit(prices);
        System.out.println(res);
    }

    @Test
    public void Test47() //二叉树中的最大路径和
    {
        TreeNode root=new TreeNode(-10);
        root.left=new TreeNode(9);
        root.right=new TreeNode(20);
        root.right.left=new TreeNode(15);
        root.right.right=new TreeNode(7);
        Alg1 alg=new Alg1();
        int res=alg.maxPathSum(root);
        System.out.println(res);
    }


    @Test
    public void Test48()  //最长连续序列
    {
        Alg1 alg=new Alg1();
        int []num={12,4,21,1,3,2,6,8,7,10,9};
        int res=alg.longestConsecutive(num);
        System.out.println(res);
    }


    @Test
    public void Test49()  //只出现一次的数字
    {
        Alg1 alg=new Alg1();
        int []nums={4,1,2,1,2};
        int res = alg.singleNumber(nums);
        System.out.println(res);
    }


    @Test
    public void Test50() //单词拆分
    {
        Alg1 alg=new Alg1();
        String s="catsandog";
        String []wordDict={"cats", "dog", "sand", "and", "cat"};
        List<String> wordList=new ArrayList<>(Arrays.asList(wordDict));
        boolean res = alg.wordBreak(s, wordList);
        System.out.println(res);
    }


    @Test
    public void hasCycleTest() //环形链表
    {
        LinkNode head=new LinkNode(3);
        head.next=new LinkNode(2);
        head.next.next=new LinkNode(0);
        head.next.next.next= head.next;
        Alg1 alg=new Alg1();
        boolean res = alg.hasCycle(head);
        System.out.println(res);
    }















}
