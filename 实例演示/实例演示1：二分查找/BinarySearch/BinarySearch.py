# --------for循环------------------------------------------------------------
# class Solution:
#     def search(self, nums, target):
#             for i in range(len(nums)):    ## range()函数的用法：生成整数序列
#                 if i <len(nums):    ## len() 函数：数组长度 （为什么要加这个语句呢）
#                     if nums[i] == target:
#                         return i
#                     else:
#                         continue   ## continue的用法：用于循环中忽略当前的一次循环
#                 else:
#                     break
#             return -1
# ---------------------------------------------------------------

# ----------二分查找----------------------------------------------
class Solution:
    def search(self, nums, target) -> int:
        left, right = 0, len(nums) - 1  # 初始化双指针 left, right 分别指向数组首、尾元素，代表查找区间为闭区间 [left, right] ；
        while left <= right:
            mid = (right - left) // 2 + left  # 向下取整，确定中点m (考虑非整数问题)
            num = nums[mid]
            if num == target:
                return mid
            elif num > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1


a = range(1, 10, 2)
print(a)
nums = [-1, 0, 3, 5, 9, 12]
print(len(nums))
print(nums[0])
target = 9
# 类的函数的调用
solution = Solution()
print(solution.search(nums, target))
