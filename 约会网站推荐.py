import knn

# 1.导入数据
data, label = knn.text_parse('datingTestSet.txt', 3)
# 2.数据预处理
data = knn.normalize(data)
# 3.分类
knn.error(data,label, 3)
