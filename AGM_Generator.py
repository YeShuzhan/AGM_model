import sys, math, random, copy, matplotlib.pyplot as plt



# Just to double check our graph stats look pretty good
def ComputeDegreeDistribution(network):
	degrees = []
	# for vertex, neighbors in network.iteritems():
	for vertex, neighbors in network.items():
		degrees.append(len(neighbors))

	degrees.sort()

	vals = range(len(degrees))
	vals = map(float, vals)
	vals = map(lambda x: 1 - x / (len(degrees)-1), vals)

	return vals, degrees

# Just computes the pearson correlation
def ComputeLabelCorrelation(network, labels):
	mean1 = 0.0
	mean2 = 0.0
	total = 0.0
	# for vertex,neighbors in network.iteritems():
	for vertex,neighbors in network.items():
		for neighbor in neighbors:
			mean1 += labels[vertex]
			mean2 += labels[neighbor]
			total += 1

	mean1 /= total
	mean2 /= total
	std1 = 0.0
	std2 = 0.0
	cov = 0.0
	for vertex,neighbors in network.items():
	# for vertex,neighbors in network.iteritems():
		for neighbor in neighbors:
			std1 += (labels[vertex]-mean1)**2
			std2 += (labels[neighbor]-mean2)**2
			cov += (labels[vertex]-mean1)*(labels[neighbor]-mean2)

	std1 = math.sqrt(std1)
	std2 = math.sqrt(std2)
	return cov / (std1*std2)



# The FCL sampler we'll use for a proposal distribution
class FastChungLu:
	def __init__(self, network):
		self.vertex_list = []
		self.degree_distribution = []
		# for vertex, neighbors in network.iteritems():
		for vertex, neighbors in network.items():
			self.vertex_list.append(vertex) #增加节点
			self.degree_distribution.extend([vertex]*len(neighbors)) #这里的度分布很奇怪，它使用当前节点的度的长度乘上当前节点的ID，形成了一个List，可能对于之后采样有用吧

	def sample_edge(self):
		vertex1 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]
		vertex2 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]

		return vertex1, vertex2

	def sample_graph(self):
		sample_network = {}
		for vertex in self.vertex_list:
			sample_network[vertex] = {}

		ne = 0 #ne其实就是原图的边的数量，那这里也就是随机取ne条边，从O(n²)变成了O(E)
		while ne < len(self.degree_distribution):
			v1, v2 = self.sample_edge()
			if v2 not in sample_network[v1]:
				sample_network[v1][v2] = 1
				sample_network[v2][v1] = 1
				ne+=2

		return sample_network

# A simple A/R that creates the following edge features from the corresponding vertex
# attributes.  Namely, if both are 0, if both are 1, and if both are 2.
class SimpleBernoulliAR:
	# Returns 0/0 -> 0, 0/1->1, 1/0->1, 1/1 -> 2
	def edge_var(self, label1, label2):
		return label1*label1 + label2*label2

	# Requires the true network, a complete sampled network from the proposing distribution
	# then the true labels and a random sample of labels
	# 这里其实也不是我们想象中的学习方法，而是通过对真实网络属性的一个计数以及采样网络的属性的一个计数
	# 由于采样网络的边是随机的，顶点的属性也由于随机化过程导致了不同，所以每一种属性的边的概率相对于真实网络是不同的
	# 那么我们在经过计数之后，可以通过真实网络与采样网络的概率的比率对某条边的接受拒绝概率进行调整
	# 但这里很奇怪的一点在于，原代码中的归一化，是取了一个比率最大值，这会导致对于某种属性组合的边（此处为1/1->2）接受概率为1，该种属性组合的边一定能够被接受
	# 这里我将所有比率加起来，形成真正的归一化，虽然结果差不多，但这才是正常的归一化。
	def learn_ar(self, true_network, sampled_network, true_labels, sample_labels):
		true_counts = {}
		true_probs = {}
		sample_counts = {}
		sample_probs = {}
		self.ratios = {}
		self.acceptance_probs = {}

		# Determine the attribute distribution in the real network
		for vertex, neighbors in true_network.items():
		# for vertex, neighbors in true_network.iteritems():
			for neighbor in neighbors:
				var = self.edge_var(true_labels[vertex], true_labels[neighbor])
				if var not in true_counts:
					# put a small (dirichlet) prior
					true_counts[var] = 1.0
				true_counts[var] += 1
		total = sum(true_counts.values())
		for val, count in true_counts.items():
		# for val, count in true_counts.iteritems():
			true_probs[val] = count / total

		# Determine the attribute distribution in the sampled network
		for vertex, neighbors in sampled_network.items():
		# for vertex, neighbors in sampled_network.iteritems():
			for neighbor in neighbors:
				var = self.edge_var(sample_labels[vertex], sample_labels[neighbor])
				if var not in sample_counts:
					# put a small (dirichlet) prior
					sample_counts[var] = 1.0
				sample_counts[var] += 1.0
		total = sum(sample_counts.values())
		# for val, count in sample_counts.iteritems():
		for val, count in sample_counts.items():
			sample_probs[val] = count / total


		# Create the ratio between the true values and sampled values
		for val in true_counts:
		# for val in true_counts.iterkeys():
			self.ratios[val] = true_probs[val] / sample_probs[val]


		# Normalize to figure out the acceptance probabilities
		max_val = 0.0
		for val in self.ratios.values():
			max_val += val
		# max_val = max(self.ratios.values())
		for val, ratio in self.ratios.items():
		# for val, ratio in self.ratios.iteritems():
			self.acceptance_probs[val] = ratio / max_val


	def accept_or_reject(self, label1, label2):
		if (random.random() < self.acceptance_probs[self.edge_var(label1, label2)]):
			return True

		return False

# The AGM process.  Overall, most of the work is done in either the edge_acceptor or the proposing distribution
class AGM:
	# Need to keep track of how many edges to sample
	def __init__(self, network):
		self.ne = 0
		# for vertex, neighbors in network.iteritems():
		for vertex, neighbors in network.items():
			self.ne += len(neighbors)

	# Create a new graph sample
	#这里就开始合成图了，通过之前得到的接受拒绝采样以及目前各个节点的随机属性分布进行合成
	def sample_graph(self, proposal_distribution, labels, edge_acceptor):
		# 初始化图的结构，这里可以看成一个Map
		sample_network = {}
		for vertex in proposal_distribution.vertex_list:
			sample_network[vertex] = {}
		# 这里初始化所需的边数
		sampled_ne = 0
		while sampled_ne < self.ne:
			# 在原图中进行随机采边
			v1, v2 = proposal_distribution.sample_edge()

			# The rejection step.  The first part is just making sure the edge doesn't already exist;
			# the second actually does the acceptance/not acceptance.  This requires the edge_accept
			# to have been previously trained
			#对采到的边进行判断，首先判断是否已经存在，第二判断该边是否能被概率接受。
			if v2 not in sample_network[v1] and edge_acceptor.accept_or_reject(labels[v1], labels[v2]):
				sample_network[v1][v2] = 1
				sample_network[v2][v1] = 1
				sampled_ne += 2

		return sample_network



if __name__ == "__main__":
	# data location
	data = 'cora/cora_agm_fcl_aid'
	data_dir = 'D:/TestAGm/'

	# edge representation
	network = {}
	# corresponding labels
	labels = {}

	# readin the edge file
	with open(data_dir + data + '.edges') as edge_file:
		for line in edge_file:
			# fields = map(int, line.strip().split('::'))
			fields = list(map(int, line.strip().split('::')))
			
			# ids
			id0 = fields[0]
			id1 = fields[1]

			# Remove self-loops
			if id0 == id1:
				continue

			# Check/create new vertices
			if id0 not in network:
				network[id0] = {}
			if id1 not in network:
				network[id1] = {}

			network[id0][id1] = 1
			network[id1][id0] = 1


	# readin the label file
	with open(data_dir + data + '.lab') as label_file:
		for line in label_file:
			# fields = map(int, line.strip().split('::'))
			fields = list(map(int, line.strip().split('::')))

			# values
			id = fields[0]
			lab = fields[1]

			# only include items with edges
			if id not in network:
				continue

			# assign the labels
			labels[id] = lab

	#这个地方计算Pearson相似度应该只是为了防止null值的出现吧
	print('Initial Graph Correlation'), ComputeLabelCorrelation(network,labels)
	fcl = FastChungLu(network) # FCL的初始化
	fcl_sample = fcl.sample_graph()

	# Random permutation of labels.  This is shorter code than sampling bernoullis for all,
	# and can be replaced if particular labels should only exist with some guaranteed probability
	# for (e.g.) privacy
	sample_labels_keys = copy.deepcopy(list(labels.keys()))
	sample_labels_items = copy.deepcopy(list(labels.values()))
	random.shuffle(sample_labels_items)
	sample_labels = dict(zip(sample_labels_keys, sample_labels_items))


	# Double check that the FCL correlation is negligible (if this is not near 0 there's something wrong)
	print('FCL Graph Correlation'), ComputeLabelCorrelation(fcl_sample, sample_labels)

	# Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
	# and the random sample of neighbors.
	edge_acceptor = SimpleBernoulliAR()
	edge_acceptor.learn_ar(network, fcl_sample, labels, sample_labels)

	# Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
	# the edge_acceptor, and let it go!
	agm = AGM(network)
	agm_sample = agm.sample_graph(fcl, sample_labels, edge_acceptor)

	# This should be fairly close to the initial graph's correlation
	print('AGM Graph Correlation'), ComputeLabelCorrelation(agm_sample, sample_labels)
	
	xs, ys = ComputeDegreeDistribution(network)
	plt.plot(ys, list(xs), label='Original')
	# plt.plot(list(xs), ys, label='Original')
	xs, ys = ComputeDegreeDistribution(fcl_sample)
	plt.plot(ys, list(xs), label='FCL')
	# plt.plot(list(xs), ys, label='FCL')
	xs, ys = ComputeDegreeDistribution(agm_sample)
	plt.plot(ys, list(xs), label='AGM-FCL')
	plt.ylim(0.00001,1.0)
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Degree')
	plt.ylabel('CCDF')
	plt.show()
print("WTF")
