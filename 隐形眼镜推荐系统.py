import decision_tree as dtree

fh = open('lenses.txt')
data = [example.strip().split('\t') for example in fh.readlines()]
labels = ['age', 'prescrip', 'astigmatic', 'tearRate']
mytree = dtree.create_tree(data, labels)
print(mytree)


