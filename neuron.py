import numpy as np

class Neuron:
	def __init__(self, data, weights):
		self.data = data
		self.weights = weights

	#slenksine funkc
	def slenkstine(self,a):
		return 1 if a > 0 else 0

	#sigmoidine funkc
	def sigmoidine(self,a):
		return 1/(1+np.exp(-a))

	def sum(self,data):
		a = 0
		for i in range(len(data) - 1):
			a += data[i] * self.weights[i]
		return a


	#neurono apmokymo funkcija(2,3 punktas)
	#slenkst = True - 2 punktas, False - 3 punktas
	def train(self,iterations = 1000, l_rate = 1, slenkst = False, done = False):
		for it in range(iterations):
			for i in range(len(self.data)):
				a = self.sum(self.data[i])
				y = self.slenkstine(a) if slenkst else self.sigmoidine(a)

				#apskaičiuojama paklaida
				error = self.data[i][3] - y
				if y != self.data[i][3]:
					for j in range(len(self.weights)):
						self.weights[j] = self.weights[j] + (l_rate * error * self.data[i][j])
					



	#1 punktas, by default skaičiuoja pirmą duomenų vektorių
	def calculateOutput(self, data):
		a = self.sum(data)
		return self.slenkstine(a)



def main():
			#bias x1   x2  y
	data = [[1.0,-0.2, 0.5,0.0],
			[1.0, 0.2,-0.5,0.0],
			[1.0, 0.8, -0.8,1.0],
			[1.0, 0.8, 0.8,1.0]]

	weights = [0,0,0]

	neuron = Neuron(data, weights)
	neuron.train()
	print(neuron.calculateOutput(data[3]))

if __name__ == '__main__':
	main()