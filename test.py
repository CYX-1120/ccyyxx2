def bubble_sort(arr):
    """
    冒泡排序算法实现
    
    参数:
        arr: 待排序的列表
    
    返回:
        排序后的列表
    """
    n = len(arr)
    # 外层循环控制排序轮数
    for i in range(n):
        # 标记本轮是否发生交换
        swapped = False
        # 内层循环进行相邻元素比较和交换
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # 交换相邻元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果本轮没有发生交换，说明已经排序完成
        if not swapped:
            break
    return arr


if __name__ == "__main__":
    # 打印 Hello World
    print("Hello World")
    
    # 测试冒泡排序
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"\n原始数组: {test_arr}")
    
    sorted_arr = bubble_sort(test_arr.copy())
    print(f"排序后数组: {sorted_arr}")
    
    # 测试已排序数组
    sorted_test = [1, 2, 3, 4, 5]
    print(f"\n已排序数组: {sorted_test}")
    print(f"排序后: {bubble_sort(sorted_test.copy())}")
    
    # 测试逆序数组
    reverse_arr = [5, 4, 3, 2, 1]
    print(f"\n逆序数组: {reverse_arr}")
    print(f"排序后: {bubble_sort(reverse_arr.copy())}")
