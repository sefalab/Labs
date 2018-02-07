#!/usr/bin/python
#Richard Klein 2017
import cv2
import numpy as np
import numpy.random as rnd
from scipy.stats import multivariate_normal
import glob
import pickle
import pdb

fg = []
bg = []
for i in range(0,1):
	print "Image %d" % i
	file_img = "./Training/t%d.jpg" % i
	file_lbl = "./lbls/l%d.png" % i
	img = cv2.imread(file_img)
	lbl = cv2.imread(file_lbl, cv2.IMREAD_GRAYSCALE)
	h,w,ch = img.shape
	#fg.append(img[lbl==255])
	#bg.append(img[lbl==0])
	for r in range(h):
		for c in range(w):
			if lbl[r,c] == 255:
				fg.append(img[r,c,:])
			else:
				bg.append(img[r,c,:])

fg = np.array(fg)
bg = np.array(bg)

prior = len(fg) / float(len(fg) + len(bg))

def random_cov(D):
	# Will ensure symmetric pos. def.
	out = rnd.random((D,D))
	out *= out.T
	out += 100*D*np.eye(D)
	return out

class GMM():
	def __init__(self, data, K):
		self.data = data
		# Dimensionality
		self.D = len(data[0])
		# Data Size
		self.n = len(data)
		# Num Gaussians
		self.K = K
		# Init Uniform Lambdas
		self.lam = np.ones(K)/K
		# Init K Means [Dx1]
		self.mu = rnd.random((K,self.D))*255.0
		# Init K cov matrices [DxD]
		self.cov = np.array([random_cov(self.D) for i in range(K)])
		# Init Responsibilities [n x K]
		self.r = np.zeros((self.n,self.K))
		
	def estep(self):
		# E-Step - Update Responsibility
		K = self.K
		n = self.n
		# Setup new dists
		self.setup()

		for i in range(n):
			sum = 0.0
			for k in range(K):
				self.r[i,k] = self.lam[k]*(self.norm[k].pdf(self.data[i]))
			self.r[i,:] /= self.r[i,:].sum()
			
	def mstep(self):
		# M-Step - Update Norm Params
		K = self.K
		D = self.D
		n = self.n
		sum_full = float(self.r.sum())
		r_sum = self.r.sum(0)		
		self.lam = r_sum/sum_full

		for k in range(K):
			self.mu[k] = self.r[:,k].dot(self.data)/r_sum[k]
		for k in range(K):
			tmp = np.zeros((D,D))
			for i in range(n):
				t = self.data[i,:] - self.mu[k]
				tmp += self.r[i,k]*np.outer(t,t)
			#print(tmp)
			self.cov[k] = tmp/r_sum[k]
			print("R_sum[k]: %f" % r_sum[k])
		
	def step(self):
		old_mu = self.mu.copy()
		print("E-Step")
		self.estep()
		print("M-Step")
		self.mstep()
		d = np.linalg.norm(old_mu - self.mu)
		print(d)
		return d

	def train(self, tol):
		d = tol
		while d >= tol:
			d = self.step()
		self.setup()

	def setup(self):
		K = self.K
		n = self.n
		self.norm = []
		for k in range(K):
			self.norm.append(multivariate_normal(mean=self.mu[k], cov=self.cov[k]))		

	def probs(self, x):
		K = self.K
		n = self.n
		
		out = 0.0		
		for k in range(K):
			out += self.lam[k]*(self.norm[k].pdf(x))
		return out	

b = GMM(bg, 2)
f = GMM(fg, 3)

print("BG")
b.train(20)

print("FG")
f.train(20)

#For single pixels
def prob(x, b, f, prior):
	p1 = f.probs(x)
	p2 = b.probs(x)
	l1 = prior
	l2 = 1 - prior
	return (p1*l1)/(p1*l1 + p2*l2)

#For full images
def prob2(x, b, f, prior):
	p1 = f.probs(x)
	p2 = b.probs(x)
	l1 = prior
	l2 = 1 - prior
	return np.divide(p1*l1, p1*l1 + p2*l2)

#sss = {'f' : f, 'b' : b}
#pickle.dump( sss, open( "save.pkl", "wb" ) )

file_img = "./Training/t%d.jpg" % 1
img = cv2.imread(file_img)
p = prob2(img, b, f, prior)
cv2.imshow("p", p)
cv2.waitKey()

