
class huffman_main:
	def __init__(self):
		self.bit = {}
		self.freq_dic = {}
	class node:
		def __init__(self, freq, symbol, left=None, right=None):
			# frequency of symbol
			self.freq = freq

			# symbol name (character)
			self.symbol = symbol

			# node left of current node
			self.left = left

			# node right of current node
			self.right = right

			# tree direction (0/1)
			self.huff = ''	

	def printNodes(self, node, bit, val=''):
		newVal = val + str(node.huff)

	
		if(node.left):
			self.printNodes(node.left, bit, newVal)
		if(node.right):
			self.printNodes(node.right, bit, newVal)
    	#bit_num[node] = newVal 

        
		if(not node.left and not node.right):
			print(f"{node.symbol} -> {newVal}")
			self.bit[node.symbol] = newVal
			
        #print(f"{node.symbol} -> {freq_dic[node.symbol] * len(newVal)}")
        #after_compression += freq_dic[node.symbol] * len(newVal)
    #print("after compression: {}".format(after_compression))	 		


	def main(self, args):

		list = args
	#for arg in args:
		#print(arg)
		for i in range(len(list)):
			num = list[i]
			if num not in self.freq_dic.keys():
				self.freq_dic[num] = 1
			else:
				self.freq_dic[num] += 1    
		#print(freq_dic)

	# list containing unused nodes
		nodes = []
	#bit_num = {}

		for num in self.freq_dic.keys():
			nodes.append(self.node(self.freq_dic[num], num))    

		while len(nodes) > 1:
		# sort all the nodes in ascending order
		# based on theri frequency
			nodes = sorted(nodes, key=lambda x: x.freq)
			left = nodes[0]
			right = nodes[1]

			left.huff = 0
			right.huff = 1

			newNode = self.node(left.freq+right.freq, left.symbol+right.symbol, left, right)

			nodes.remove(left)
			nodes.remove(right)
			nodes.append(newNode)

		#data_length_bef_compression = len(list) * 8
		#print("before compression: {}".format(data_length_bef_compression))
		self.printNodes(nodes[0], self.bit)
		#sum_bits = 0
		#for i in self.bit.keys():
			#sum_bits += self.bit[i]
		#print("after compression: {}".format(sum_bits))
		return self.bit

#if __name__ == '__main__':
	#main(8, 8, 34, 5, 10, 34, 6, 43, 127, 10, 10, 8, 10, 34, 10)