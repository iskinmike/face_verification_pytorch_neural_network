import numpy as np
positive_data = [[1,1],[2,2]]
negative_data = [[3,3],[4,4]]

x = np.vstack([positive_data, negative_data])
# y = np.vstack(positive_lables, negative_lables)

print(x)
print(len(x))



test = [[1,1],[2,2]]

test2 = test[1][::2]

# test4 = test[::4]


print(test)
print(test2)
# print(test4)


