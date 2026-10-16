def count_pairs(nums, k):
    from collections import Counter
    freq = Counter()
    count = 0
    for num in nums:
        count += freq[k - num]
        freq[num] += 1
    return count


from collections import Counter
import random
random.seed(1)
def ref(nums,k):
    return sum(1 for i in range(len(nums)) for j in range(i+1,len(nums)) if nums[i]+nums[j]==k)
for _ in range(300):
    nums=[random.randint(-5,5) for _ in range(random.randint(0,12))]; k=random.randint(-6,6)
    assert count_pairs(nums,k)==ref(nums,k), (nums,k,count_pairs(nums,k),ref(nums,k))

print('PASS')
