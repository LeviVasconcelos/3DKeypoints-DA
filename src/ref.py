#edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         #[4, 8], [5, 9], [3, 7], [2, 6]]
#edges = [
#	[0,1], [1,2], [2,3], [4,5], 
#	[0,7],[7,8],[8,9],[9,10],
#	[17,18],[18,19],[19,20],[20,21],[21,22],[22,23],
#	[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],]
edges = [[0,1], [1,2], [2,3], #right leg
             [0, 4], [4,5], [5,6], #left leg
             [0,7], #spine
             [7,8], #neck
             [7,9], [9,10], [10, 11], #left arm
             [7,12], [12, 13], [13, 14]] #right arm 
human_edges = [(0,1), (1,2), (2,3), (0, 4), (4,5), (5,6), (0,7), (7,8), (7,9), (9,10), (10, 11), (7,12), (12, 13), (13, 14)] 


totalViewsModelNet = 18
totalViewsShapeNet = 18
#nValViews = 18
nValViews = 4 

#J = 10
#J = 32
J = 15 
metaDim = 5 + J

eps = 1e-6
imgSize = 224

#Change data dir here
ShapeNet_dir = '../data/ShapeNet/'
ModelNet_dir = '../data/ModelNet/'
DCNN_dir = '../data/3DCNN/'
Redwood_dir = '../data/Redwood_depth/'
RedwoodRGB_dir = '../data/Redwood_RGB/'
#Humans_dir = '/hardmnt/rebel1/data/data/processed/'
Humans_dir = '/hd/levi/workspace/h36m-fetch/processed/'

ModelNet_version = ''
ShapeNet_version = ''
DCNN_version = ''
Annot_ShapeNet_version = ''
Redwood_version = ''
RedwoodRGB_version = ''
#category = 'Chair'
category = 'Human'
tag = ''
exp_name = '{}{}'.format(category, tag)
nViews=1
