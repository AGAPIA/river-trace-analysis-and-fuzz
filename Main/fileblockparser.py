import re
import operator
from functools import reduce
from random import randint
from igraph import *
from typing import Dict, List, Tuple
import copy


# This is intended for parsing a blocks file.
# First,instantiate the parser then call run function
# It will return a dictionary in the following format:
# {'blockOffset' : [(VariableIndex, [interval1, interval2])]}. E.g. {'000012AC': [(66, [96, 104])], '0000125C': [(65, [96, 104])], '00002D5B': [(70, [120, 128])], '000025FC': [(78, [120, 128])], '00002D72': [(72, [120, 128])]}
# The variable index is not important. It will have one entry per block, possibly with multiple intervals
class FileBlockParser:

	def __init__(self):

		self.normal_pats = [r' ', r'\n', r'I\[[0-9]+\]\ <=']
		self.concat1_pats = [r'I\[([0-9]+)\] \+\+', r' ', r'\n', r'I\[[0-9]+\]\ <=']
		self.concat2_pats = [r'\+\+ I\[([0-9]+)\]', r' ', r'\n', r'I\[[0-9]+\]\ <=']
		self.compo1_pats = [r'(?:CF:I\[[0-9]+\] \||I\[[0-9]+\] \|)', r' ', r'\n', r'I\[[0-9]+\]\ <=']
		self.compo2_pats = [r'(?:\| ZF:I\[[0-9]+|\| I\[[0-9]+\]', r' ', r'\n', r'I\[[0-9]+\]\ <=']

		self.combined_pat = r'|'.join(map(r'(?:{})'.format, self.normal_pats))
		self.concat1_combined_pat = r'|'.join(map(r'(?:{})'.format, self.concat1_pats))
		self.concat2_combined_pat = r'|'.join(map(r'(?:{})'.format, self.concat2_pats))
		self.compo1_combined_pat = r'|'.join(map(r'(?:{})'.format, self.compo1_pats))
		self.compo2_combined_pat =r'|'.join(map(r'(?:{})'.format, self.compo2_pats))

		# TODO: the graph can be huge. Make sure to remove nodes dynamically in the end
		self.DAG = Graph()
		
		self.current: str = ''
		self.jumped: str = ''

		# Because we are mainly interested on the variables after stdin marker - the ones that are really used in the code, not in tainting the input
		self.firstVariableIndexAfterStdin = -1

		self.remaining="remaining"
		
	def run(self, blocksFile):
		

		# The content of 'details' would be something like: <class 'tuple'>: ('I[90]', '0007F849', '0007F866', '0007F857')
		# First - variable I[x]
		# The other 3: current offset, next offset if branch taken, next offset if branch not taken

		# For a sequence of  BLOCKS_INFO1  VARIABLES_INFO BLOCKS_INFO2
		# Note that VARIABLE_INFO are used inside BLOCKS_INFO2 for its branching condition
		# More, only the LAST variable in BLOCKS_INFO2 is used in branching condition
		# So this means that only that variable is affecting the input and data flow!
		# TODO: there is a small issue here because we should use pair (file + offset) ; what if two files have same offset ?

		'''
		details: List[Tuple[str, str, str, str]] = self.get_details("tainted_output.txt")
		groups: Dict[str, List[str]] = {}

		for i, entry in enumerate(details):
			if i - 1 >= 0 and details[i][1] == details[i - 1][2]:
				self.current: str = details[i - 1][0] 
				self.jumped: str = details[i][0]
				if self.current not in groups:
					groups[self.current] = [self.jumped]
				else:
					groups[self.current].append(self.jumped)
		'''

		# Create DAG 		# Find "real" vars
		# DAG will contain a node for each variable. If variable is I[X] then node X-1 corresponds to it
		# DAG.v[X]['sequence'] corresponds to the interval used by variable X from the input sequence
		# Generally when I[X] depends on other variables Y and / or Z, we connect X to those in DAG and unify the used input sequence.
		with open(blocksFile, "r") as file:

			# I[index] => what is the right expression for variable I[index+1] in log file OR False if nothing
			I = list()

			# A list of constant variables indices
			I_const = list()

			self.DAG.add_vertices(1)
			sequence = "sequence"		# This is the sequence of bits used from the initial input by variable
			remaining = "remaining"		# This is the part of the sequence that is not used (yet) by any variable which have an edge to this variable
			sequence_null = "sequence_null"
			null_vertices = []
			self.DAG.vs[0][sequence_null] = ['null']

			# The index of the p variable. The idea behind these is that p variables will flow in order
			# p[x] means dword x from input (i.e. bits [x*32, (x+1)*32) )
			# The first part of the input contains a copy from p to I variables because of the tainting input analysis.
			p_index = 0

			# The index of the variable analyzed. Note that variables flow in consecutive and increasing order in the out txt file.
			# Also, NOTE THAT index = 0 corresponds to I[1] , and in general I[X] corresponds to index X-1 in DAG
			index = 0

			variableIndexToBlockWhereUsed = {} # TODO: add module too !
			self.firstVariableIndexAfterStdin = -1
			pendingVariablesIndex = [] # The current variables that needs to have a block where used

			# TODO: optimize for minimal search of text here !
			for line in file:
				# Check to see if I[X] <= const value . If yes, add X to constant table
				if re.search(r'const', line):
					self.constant(sequence_null, index, I, I_const, line)
					index += 1

				# Consider only lines that have I[X] something in the beggining...
				elif "I[" in line:
					initialVarIndex = index
					if self.firstVariableIndexAfterStdin != -1: # Did the code section already started ?
						pendingVariablesIndex.append(initialVarIndex)

					# ???????????????
					#if str(re.search(r'(I\[[0-9]+\]) <=', line).group(1)) in [elem for items in groups.values() for elem in items]:
					#	self.constant(sequence_null, index, I, I_const, line)
					#	index += 1

					# Search if p[x][..] appears in the line
					if re.search(r'p\[[0-9]+\]', line):
						I.append(re.sub(self.combined_pat, '', line))
						self.dword(sequence, index, p_index)
						p_index += 1
						index += 1

					# There is a concatenation operator for the expression evaluated "X <= Y ++ Z "
					elif re.search(r'\+\+', line):
						# Extract the left and right indices
						left = int(re.search(r'I\[([0-9]+)\] \+\+', line).group(1))
						right = int(re.search(r'\+\+ I\[([0-9]+)\]', line).group(1))

						# If Y and Z are in constant => X is constant
						if left in I_const and right in I_const:
							self.both_const(sequence_null, index, I, I_const, line)
							index += 1

						# If X or right constant make the connection in the graph and sequence update for X using only the non const variable
						elif left in I_const:
							I.append(re.sub(self.concat1_combined_pat, '', line))
							self.left_const(sequence, index, I, right)
							index += 1

						elif right in I_const:
							I.append(re.sub(self.concat2_combined_pat, '', line))
							self.right_const(sequence, index, I, left)
							index += 1
						# If both are non-const, connect X to both Y and Z and unify the intervals used by Y and Z into X
						else:
							I.append(re.sub(self.combined_pat, '', line))
							self.concat(sequence, index, I)
							index += 1

					# Search for pattern I[X] <= I[Y][start:length]
					elif re.search(r'I\[[0-9]+\]\[[0-9]+\:[0-9]+\]', line):

						# if Y is in the const list, it means that X is same a constatnt
						if int(re.search(r'I\[([0-9]+)\]\[[0-9]+\:[0-9]+\]', line).group(1)) in I_const:
							self.both_const(sequence_null, index, I, I_const, line)
							index += 1
						# If not in cost, add right side to I list and extract the sequence
						# Add to DAG.v[index][sequence] = start-end bits used
						else:
							I.append(re.sub(self.combined_pat, '', line))
							self.extract(sequence, index, I)
							index += 1
					
					else:  # This branch is similar to ++ one, but it solves the operator "|". It is just duplicated code unfortunately

						if re.search(r'..:I\[[0-9]+\] \|', line):
							left = int(re.search(r'..:I\[([0-9]+)\] \|', line).group(1))
						else:
							left = int(re.search(r'I\[([0-9]+)\] \|', line).group(1))
						
						if re.search(r'\| ..:I\[[0-9]+\]', line):
							right = int(re.search(r'\| ..:I\[([0-9]+)\]', line).group(1))
						else:
							right = int(re.search(r'\| I\[([0-9]+)\]', line).group(1))

						if left in I_const and right in I_const:
							I_const.append(int(re.match(r'I\[([0-9]+)\] <=', line).group(1)))
							self.both_const(sequence_null, index, I, I_const, line)
							index += 1

						elif left in I_const:
							I.append(re.sub(self.compo1_combined_pat, '', line))
							self.left_const(sequence, index, I, right)
							index += 1

						elif right in I_const:
							I.append(re.sub(self.compo2_combined_pat, '', line))
							self.right_const(sequence, index, I, left)
							index += 1

						else:
							I.append(re.sub(self.combined_pat, '', line))
							self.compose(sequence, index, I, left, right)
							index += 1

					# Remaining part is a copy of the sequence
					self.DAG.vs[initialVarIndex][remaining] = copy.deepcopy(self.DAG.vs[initialVarIndex][sequence])

				elif "stdin" in line:
					self.firstVariableIndexAfterStdin = index
				else:
					# Check if this is a code block definition line
					splitLine = line.split("+");
					if len(splitLine) < 3:

						#assert "I was expecting this to be a code line. {}".format(splitLine)
						continue

					currentBlockUsed = splitLine[1].strip().split(" ")[0]

					# Each pending variable will be added to this block
					for var in pendingVariablesIndex:
						variableIndexToBlockWhereUsed[var] = currentBlockUsed

					pendingVariablesIndex = []

		allBlocksUsedByVariables = set(blockIter for blockIter in variableIndexToBlockWhereUsed.values())
		blocksToVariableList : Dict[str, List[int, List[int]]] = {}  # X[block] = list of all variable leaves using this block and their sequence intevals   X[block][(varIndex1, [e1 e2 e3 e4 .... en])] with the semnification that sequences are [e1,e2], [e3, e4], etc
		for x in allBlocksUsedByVariables:
			blocksToVariableList[x] = []

		# Consider only the leaf nodes - the ones with degree 1. Eliminate the rest
		# LeafNodes have the format: [ (variable index), (start:end)] where (start:end) represent the interval of bits affected by this variable
		# TODO: add the block offset where this var is being used to match between them - this is the REAL variable index actually
		for index_vertex in range(1, index):
			if self.DAG.degree(index_vertex) == 0:
				null_vertices.append(self.DAG.vs[index_vertex])
				index -= 1
			elif index_vertex >= self.firstVariableIndexAfterStdin:
				# Are there any tainted bits used by this variable ?
				if self.DAG.vs[index_vertex][remaining]: #self.DAG.degree(index_vertex) == 1
					blockUsedByThisVariable = variableIndexToBlockWhereUsed[index_vertex]
					blocksToVariableList[blockUsedByThisVariable].append((index_vertex, self.DAG.vs[index_vertex][sequence]))

		#self.DAG.delete_vertices(null_vertices)
		#self.DAG.delete_vertices(index)
		#del self.DAG.vs[sequence_null]
		
		#print(blocksToVariableList)
		#print(allBlocksUsedByVariables)
		#print(blocksToVariableList)
		#print(variableIndexToBlockWhereUsed)

		# From each block, unify the variables into a single one since they represent the same thing
		for block in blocksToVariableList.keys():
				allVariablesInBlock = blocksToVariableList[block]
				if len(allVariablesInBlock) == 0:

					continue

				representativeVariable = allVariablesInBlock[0][0]

				blockSequence = []
				for varInfo in allVariablesInBlock:
					varIndex = varInfo[0]
					blockSequence.extend(self.DAG.vs[varIndex][sequence])

				intervals = self.union(blockSequence)
				blocksToVariableList[block] = [(representativeVariable, intervals)]


		#print(blocksToVariableList)
		# Sanity check - one entry per block, intervals are not empty
		for blockId, blockData in blocksToVariableList.items():
			if not blockData:
				assert False
				continue
			if not blockData[0]:
				assert False
				continue

		return blocksToVariableList

		# Plotting
		#layout = self.DAG.layout("tree")
		#plot(self.DAG, layout = layout, bbox = (4000,1500))

	# This function make union
	# between intervals of bytes
	def union(self,intervals):

		intervals_ = [[intervals[i],intervals[i+1]] for i in range(0,len(intervals),2)]

		union_interv = []
		for begin,end in sorted(intervals_):
			if union_interv and union_interv[-1][1] >= begin - 1:
				union_interv[-1][1] = max(union_interv[-1][1], end)
			else:
				union_interv.append([begin, end])

		union_interv = reduce(operator.concat, union_interv)

		return union_interv

	# General form: I[x] <= p[i](v)
	def dword(self, sequence, index, dw_index):
		#print('\nat node {0} we found a dword/ p[i]'.format(index+1))

		self.DAG.add_vertices(1)
		r = (dw_index + 1) * 32
		l = r - 32
		self.DAG.vs[index][sequence] = [l, r]

		#print(self.DAG.vs[index])

	# General form: I[x] <= I[y] ++ I[z]
	def concat(self, sequence, index, I):
		#print('\nat node {0} we found concatenation'.format(index+1))
		
		self.DAG.add_vertices(1)
		Node = []
		Node = [int(i) for i in re.match(r'I\[([0-9]+)\]\+\+I\[([0-9]+)\]', I[index]).group(1,2)]  # extract [y,z]

		#print('with nodes {0} and {1}'.format(Node[0], Node[1]))

		NodeLeft = Node[0] - 1
		NodeRight = Node[1] - 1
		
		self.DAG.add_edges([(index, NodeLeft), (index, NodeRight)])  # add edge from index to both y and z
		self.DAG.vs[index][sequence] = []

		#print(len(self.DAG.vs[Node[0]-1][sequence]))
		#print(len(self.DAG.vs[Node[1]-1][sequence]))

		# Add the intervals used in both then unify them
		for elem in self.DAG.vs[NodeLeft][sequence]:
			self.DAG.vs[index][sequence].append(elem)
			#print(self.DAG.vs[index])

		for elem in self.DAG.vs[NodeRight][sequence]:
			self.DAG.vs[index][sequence].append(elem)
			#print(self.DAG.vs[index])

		self.DAG.vs[index][sequence] = self.union(self.DAG.vs[index][sequence])

		# Nothing remains from the two  children
		self.DAG.vs[NodeLeft][self.remaining] = None
		self.DAG.vs[NodeRight][self.remaining] = None

		#print(self.DAG.vs[index])

	# General form: I[x] <= I[y][a:b]
	# Here will be an ampler explanation:
	# First, we need to find the start bit and the length of the intervals
	# While final of interval - the start bit < necessary bits
	# Update nes_bits; go to next interval; bit_start becomes first bit
	#   of the next interval
	def extract(self, sequence, X, I):
		#print('\nat node {0} we found extraction'.format(index+1))

		self.DAG.add_vertices(1)
		substr = [int(i) for i in re.match(r'I\[([0-9]+)\]\[([0-9]+)\:([0-9]+)\]', I[X]).group(1, 2, 3)]  # Extract list [Y, a, b ]

		Y = substr[0] - 1 # Remember that all nodes are shifted with one less...I[1] is index 0 actually # TODO: please refactor this..
		
		#print('from node I[{0}] the start bit: {1}, take next {2} bits'.format(Y+1, substr[1], substr[2]))
		#print('The sequence of node I[{0}] : {1}'.format(Y+1, self.DAG.vs[Y][sequence]))

		self.DAG.add_edges([(X, Y)])
		self.DAG.vs[X][sequence] = []

		index_interv = 0
		bit_start = substr[1]
		len_interv = []

		for index5 in range(0,len(self.DAG.vs[Y][sequence]),2):
			len_interv.append(self.DAG.vs[Y][sequence][index5 + 1] - self.DAG.vs[Y][sequence][index5])

		numIntervals = len(len_interv)
			
		#print('the lenght of the interval is: {0}'.format(len_interv))

		while self.DAG.vs[Y][sequence][index_interv + 1] - self.DAG.vs[Y][sequence][index_interv] <= bit_start and index_interv + 1 < numIntervals:
			bit_start = bit_start - self.DAG.vs[Y][sequence][index_interv + 1] + self.DAG.vs[Y][sequence][index_interv]
			index_interv += 2

		bit_start += self.DAG.vs[Y][sequence][index_interv]

		self.DAG.vs[X][sequence].append(bit_start)

		#print('the start bit is: {0}'.format(bit_start))

		nes_bits = substr[2]

		#print('necessary bits: {0}'.format(nes_bits))
		#print('the last bit of current interval: {0}'.format(self.DAG.vs[Y][sequence][index_interv + 1]))

		while self.DAG.vs[Y][sequence][index_interv + 1] - bit_start < nes_bits  and index_interv + 1 < numIntervals:
			nes_bits = nes_bits - self.DAG.vs[Y][sequence][index_interv + 1] + bit_start
			self.DAG.vs[X][sequence].append(self.DAG.vs[Y][sequence][index_interv + 1])
			index_interv += 2
			self.DAG.vs[X][sequence].append(self.DAG.vs[Y][sequence][index_interv])
			bit_start = self.DAG.vs[Y][sequence][index_interv]

		self.DAG.vs[X][sequence].append(bit_start + nes_bits)

		# From Y, cut the intervals used now in X.
		intervalsUsedByX = self.DAG.vs[X][sequence]
		invervalsUsedByY = self.DAG.vs[Y][self.remaining]
		for i in range(0, len(intervalsUsedByX), 2):
			begin_used = intervalsUsedByX[i]
			end_used = intervalsUsedByX[i + 1]

			invervalsUsedByY = self.cutSubInterval(invervalsUsedByY, [begin_used, end_used])
		
		#print(self.DAG.vs[index])

	# General form: I[x] <= (maybe a flag) I[y] | (maybe a flag) I[z] (maybe something)
	def compose(self, sequence, index, I, *argv):
		#print('\nat node {0} we found compose'.format(index+1))
		
		self.DAG.add_vertices(1)
		Node = []
		Node.append(argv[0])
		Node.append(argv[1])

		NodeLeft = Node[0] - 1
		NodeRight = Node[1] - 1

		#print('with nodes {0} and {1}'.format(Node[0], Node[1]))
		
		self.DAG.add_edges([(index, NodeLeft), (index, NodeRight)])
		self.DAG.vs[index][sequence] = []

		#print(len(self.DAG.vs[Node[0]-1][sequence]))
		#print(len(self.DAG.vs[Node[1]-1][sequence]))

		for elem in self.DAG.vs[NodeLeft][sequence]:
			self.DAG.vs[index][sequence].append(elem)
			#print(self.DAG.vs[index])

		for elem in self.DAG.vs[NodeRight][sequence]:
			self.DAG.vs[index][sequence].append(elem)
			#print(self.DAG.vs[index])

		self.DAG.vs[index][sequence] = self.union(self.DAG.vs[index][sequence])

		# Nothing remains from the two  children
		self.DAG.vs[NodeLeft][self.remaining] = None
		self.DAG.vs[NodeRight][self.remaining] = None

		#print(self.DAG.vs[index])

	# General form: I[x] <= const 0x...(v)
	def constant(self, sequence_null, *argv):
		#print('\nat node {0} we found a constant'.format(argv[0]+1))
		
		argv[2].append(int(re.match(r'I\[([0-9]+)\]', argv[3]).group(1)))
		self.DAG.add_vertices(1)
		self.DAG.vs[argv[0]][sequence_null] = ['null']
		argv[1].append('false')
		
		#print(self.DAG.vs[argv[0]])

	# 3 functions to deal with consts from concat and compose
	def both_const(self, sequence_null, *argv):
		#print('\nat node {0} we found everything const'.format(argv[0]+1))
		
		argv[2].append(int(re.match(r'I\[([0-9]+)\] <=', argv[3]).group(1)))
		self.DAG.add_vertices(1)
		self.DAG.vs[argv[0]][sequence_null] = ['null']
		argv[1].append('false')
		
		#print(self.DAG.vs[argv[0]])

	def left_const(self, sequence, *argv):
		#print('\nat node {0} we found a left side const'.format(argv[0]+1))
		
		self.DAG.add_vertices(1)
		self.DAG.vs[argv[0]][sequence] = self.DAG.vs[argv[2] - 1][sequence]
		self.DAG.add_edges([(argv[0], argv[2] - 1)])
		
		#print(self.DAG.vs[argv[0]])

	def right_const(self, sequence, *argv):
		#print('\nat node {0} we found a right side const'.format(argv[0]+1))

		self.DAG.add_vertices(1)
		self.DAG.vs[argv[0]][sequence] = self.DAG.vs[argv[2] - 1][sequence]
		self.DAG.add_edges([(argv[0], argv[2] - 1)])
		
		#print(self.DAG.vs[argv[0]])

	def get_block_info_lines(self, file_path: str) -> List[str]:
			input = open(file_path, "r")

			# TODO: use iterator, there could be so many lines
			lines = [line for line in input]
			input.close()
			info_lines: List[str] = []
			for i, line in enumerate(lines):
				if "<=" not in line:
					continue
				if i + 1 < len(lines) and "<=" in lines[i + 1]:
					continue
				if i + 1 >= len(lines):
					break
				if "stdin" in lines[i + 1]:
					continue
				info_lines.append(
					(lines[i].split("<=")[0].strip(), lines[i + 1])
				)
			return info_lines
    
	def get_current(self, line: str) -> str:
		return line.split("+")[1].strip().split(" ")[0]
    
	def get_jumped(self, line: str) -> str:
		return line.split("+")[2].strip().split(" ")[0]
    
	def get_non_jumped(self, line: str) -> str:
		return line.split("+")[3].strip().split(" ")[0]
    
	def get_details(self, file_path: str) -> List[Tuple[str, str, str, str]]:
		return [
			(line[0], self.get_current(line[1]), self.get_jumped(line[1]), self.get_non_jumped(line[1]))
			for line in self.get_block_info_lines(file_path)[0]
		]


	# Cuts a subinterval from the given intervals
	# Returns the new intervals list
	# Should use binary search but expecting intervals to be of len 2-3 in average
	def cutSubInterval(self, intervals, subinterval):
		assert len(subinterval) == 2 # Expecting [begin, end]
		begin_sub = subinterval[0]
		end_sub = subinterval[1]
		newintervals = []

		lenIntervals = len(intervals)
		if lenIntervals == 0:
			return []

		# Is entire intervals cut ?
		if begin_sub <= intervals[0] and end_sub >= intervals[-1]:
			return []

		# Is nothing cut ?
		if end_sub < intervals[0] or begin_sub > intervals[-1]:
			return []

		# Find the indices that intersect subtree in ends
		foundLeft = -1
		foundRight = -1
		for i in range(0, lenIntervals, 2):
			begin_i = intervals[i]
			end_i = intervals[i+1]

			if begin_sub < end_i and foundLeft == -1:
				foundLeft = i

			if end_sub < end_i  and foundRight == -1:
				foundRight = i

		if foundLeft == -1:
			foundLeft = 0

		if foundRight == -1:
			foundRight = lenIntervals - 2

		# Add left side unaffected by cut
		for i in range(0, foundLeft, 2):
			newintervals.extend([intervals[i], intervals[i+1]])

		# Add the affected intervals i not empty
		begin_aff = intervals[foundLeft]
		end_aff = begin_sub
		if begin_aff < end_aff:
			newintervals.extend([begin_aff, end_aff])
		#if foundLeft != foundRight:
		begin_aff = max(end_sub, intervals[foundRight])
		end_aff = intervals[foundRight + 1]
		if begin_aff < end_aff:
			newintervals.extend([begin_aff, end_aff])

		# Add right side unaffected by cut
		for i in range(foundRight + 2, lenIntervals, 2):
			newintervals.extend([intervals[i], intervals[i + 1]])

		return newintervals
