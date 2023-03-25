import numpy as np

import math

from scipy.spatial import distance


class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		pred = np.array([])
		predict = []
		for i in X:
			dist = []
			for j in self.X_train:
				dist.append(self.distance(j, i))
			dist = np.array(dist)
			neighbor = zip(dist, self.y_train)
			best_neighbors = sorted(neighbor)[:self.k]
			predict = self.most_common(j[1] for j in best_neighbors)

			pred = np.append(pred, [predict])
			
		return np.ravel(pred)
			
	def most_common(self, neighbor):
		count = 0
		neighbor_val = None
		for i in neighbor:
			if count == 0:
				neighbor_val = i
			if i == neighbor_val:
				# print(neighbor_val)
				count += 1
			else:
				count -= 1
		return neighbor_val


class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		# None
		x_val = X
		y_val = y
		
		for i in range(steps):
			step = i%(y_val.size)
			dot_prod = x_val[step].dot(self.w)
			if dot_prod>0:
				out = 1
			else: 
				out= 0 
				
			self.w += self.lr * (y_val[step] - out) * X[step]

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		prediction = []
		for d in range(0, len(X)):
			dot_prod = X[d].dot(self.w)
			if dot_prod > 0:
				prediction.append(1)  
			else: 
				prediction.append(0)
			
		return np.ravel(prediction)



class ID3:
	class Node:
		def __init__(self, tree_split):
			self.split = tree_split
			self.children=[] # add children of the tree
			self.label = None 
			self.isleaf = False # if leaf node then end

	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		self.y_train = y
		df = np.concatenate((categorical_data,y[:,None]),axis=1) 
		first,eninfo = self.info_gain(df)
		self.head = self.Node(first)
		self.data_split(first,df,self.head)


	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		outputlist = []
		for row in categorical_data: 
			node = self.head # initialize the tree
			while not node.isleaf:
				attribute = node.split
				row_val = row[attribute]
				children = [child[1] for child in node.children]
				if row_val not in children:
					return node.label
				else:
					for child in node.children:
						if child[1]==row_val:
							node=child[0]
			outputlist.append(node.label)
		return np.array(outputlist)

	
	def info_gain(self,df):
		unique_nodes = np.unique(df[:,:-1]) #take the unique label values from the merged data
		
		right_val,constant = df.shape

		parent_entropy = self.calc_entropy(df)  #calculate the entropy
		gain_list =[]
		i =0 
		while i < (constant-1): # iterate over the columns

			ent_sum = 0
			en_list=[]
			
			for j in unique_nodes:
				no = len(df[df[:,i]==j])
				entropy_num = self.calc_entropy(df[df[:,i]==j])
				ent_sum+=no
				en_list.append((no,entropy_num))
			gain = parent_entropy
			
			for index,value in enumerate(en_list):
				gain-= value[0]*value[1]/ent_sum #calc info gain for each col
			gain_list.append((i,gain)) # add it to the final list 
		
			i+=1
		
		info_gain = sorted(gain_list,key = lambda x:x[1], reverse=True) # get the best info gain value
		return info_gain[0][0],info_gain[0][1]
		

	def data_split(self,attr,data,node):
		
		attribute = np.unique(data[:,:-1][:,attr]) 
		l_count = data[:,-1]
		node.label = max(set(l_count), key = list(l_count).count) #get label count
		
		if len(np.unique(data[:,-1]))==1: # check if depth is > 1
			node.isleaf=True
			node.label = int(np.unique(data[:,-1]))
			
		if len(attribute) == 1:  #check if it is leaf node
			node.isleaf=True # change leaf value
			l_count = data[:,-1]
			node.label = max(set(l_count), key = list(l_count).count)
			
		else:
			splitdata = np.array([(data[data[:,attr]==i],i) for i in attribute]) # split data into smaller batches
			for j in splitdata:
				val,eninfo = self.info_gain(j[0])
				new = self.Node(val)
				node.children.append((new,j[1]))
				self.data_split(val,j[0],new)
				
	def calc_entropy(self, y_train):

		en_val1 = np.count_nonzero(self.y_train == 0) # get no of zeros
		en_val2 = np.count_nonzero(self.y_train == 1) # get no of ones
			
		en = 0
		if (en_val1 or en_val2) == 0:
			return 0 
		for i in [en_val1, en_val2]:
			en +=-1*(i/(en_val1+en_val2)*math.log2(i/(en_val1+en_val2))) # calc total entropy
			return en

