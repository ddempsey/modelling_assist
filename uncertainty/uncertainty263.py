import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from ipywidgets import interact, fixed, interactive_output, HBox, Button, VBox, Output, IntSlider, Checkbox, FloatSlider, FloatLogSlider, Dropdown
TEXTSIZE = 16
from IPython.display import clear_output
import time
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as colmap
from copy import copy
from scipy.stats import multivariate_normal

# general figures
def plot_priors(thmin, thmax, thmean, thstd):
	f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
	ax1.set_ylim([0,10])
	x = np.linspace(-0.01,1.01,1001)
	ymax = 1./(thmax-thmin)
	y = ymax + 0.*x
	y[np.where((x<thmin)|(x>thmax))] = 0.
	ax1.plot(x,y,'k-')
	ax1.set_xticks([thmin,thmax])
	ax1.set_xticklabels([r'$\theta_{min}$',r'$\theta_{max}$'])
	ax1.set_yticks([ymax])
	ax1.set_yticklabels([r'$(\theta_{max}-\theta_{min})^{-1}$'])
	ax1.set_xlim([-0.05,1.05])
		
	y = np.exp(-(x-thmean)**2/(2*thstd**2))/np.sqrt(2*np.pi*thstd**2)
	ax2.set_xlim([0.2,0.8])
	ax2.set_ylim([0,15])
	ax2.plot(x,y,'k-')
	ax2.set_xticks([thmean])
	ax2.set_xticklabels([r'$\bar{\theta}$'])
	ax2.set_yticks([1./np.sqrt(2*np.pi*thstd**2)])
	ax2.set_yticklabels([r'$(2\pi\sigma_\theta^2)^{-1/2}$'])
		
	plt.show()
def priors():
	thmin = FloatSlider(value=0.099, description=r'$\theta_{min}$', min = -0.001, max = 0.499, step = 0.1, continuous_update = False, readout=False)
	thmax = FloatSlider(value=0.901, description=r'$\theta_{max}$', min = 0.501, max = 1.001, step = 0.1, continuous_update = False, readout=False)
	thmean = FloatSlider(value=0.5, description=r'$\bar{\theta}$', min = 0.3, max = 0.7, step = 0.1, continuous_update = False, readout=False)
	thstd = FloatSlider(value=0.05, description=r'$\sigma_\theta$', min = 0.03, max = 0.07, step = 0.02, continuous_update = False, readout=False)
	io = interactive_output(plot_priors, {'thmin':thmin,'thmax':thmax,'thmean':thmean,'thstd':thstd})
	return VBox([HBox([thmin,thmean]), HBox([thmax,thstd]), io])

# linear model
def f(x,m,c): 
    return m*x+c
def err(x,var): 
    return np.random.randn(len(x))*np.sqrt(var)
def plot_observations(N_obs, bestModel, trueModel, var, seed):#, true_model, RMS_fit, error_dist):
	# define a model
	x = np.linspace(0,1,101)

	# model parameters
	m0 = 2.       # true gradient
	c0 = 3.       # true intercept
	
	# compute the "true" model, using the "true" parameters
	y = f(x,m0,c0)

	# seed the random number generator so we get the same numbers each time
	np.random.seed(seed)
	
	# define some values of the independent variable at which we will be taking our "observations"
	xo = np.linspace(0,1,12)[1:-1]

	# compute the observations - "true" model + random error (drawn from normal distribution)
	yo = f(xo,m0,c0) + err(xo,var)

	# initialize figure window and axes
	fig,ax = plt.subplots(1,1,figsize=(12,6))
	
	# plot the observations
	i = np.min([len(xo), N_obs])
	ln2 = ax.plot(xo[:i],yo[:i],'wo', mec = 'k', mew = 1.5, ms = 5, label = r'observations', zorder = 10)
	
	# add "best-fit" model if appropriate
	if bestModel:
		# find best-fit model
		p2,pc = curve_fit(f, xo[:i], yo[:i], [1,1])
		# plot model
		ax.plot(x,f(x,*p2),'r-', label = 'best model')
	
	# plot the "true" model    
	if trueModel:
		ln1 = ax.plot(x,y,'b-', label = 'true process',zorder = 10)
	
	# add normal distributions to plot
	yvar = 15.*np.sqrt(var)
	ye = np.linspace(-yvar,yvar,101)*0.2
	ye2 = np.linspace(-yvar,yvar,101)*0.25
	# loop over observations
	for xoi, yoi in zip(xo[:i],yo[:i]):
		# normal dist
		xi = 0.05*np.exp(-(ye)**2/var)+xoi
		# add to plot
		ax.plot(xi, ye+f(xoi,m0,c0), 'k-', lw = 0.5, zorder = 0)
		ax.plot(xi*.0+xoi, ye2+f(xoi,m0,c0), '-', lw = 0.5, zorder = 0, color = [0.5, 0.5, 0.5])

	# plot upkeep + legend
	ax.set_xlim(ax.get_xlim())
	ax.legend(loc=2, prop={'size':TEXTSIZE})
	ax.set_ylim([1,7])
	ax.set_xlim([0,1])
	ax.set_xlabel('$x$',size = TEXTSIZE)
	ax.set_ylabel('$y$',size = TEXTSIZE)
	for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontsize(TEXTSIZE)
def observation_error():
	#out = Output()
	seed = 13
	rolldice = Button(description='ROLL THE DICE', tooltip='randomise the random number generator')
	Nsldr = IntSlider(value=5, description='$N_{obs}$', min = 2, max = 10, step = 1, continuous_update = False)
	trueModel = Checkbox(value = False, description='True Process')
	bestModel = Checkbox(value = False, description='Best (LSQ) Model')
	varsldr = FloatLogSlider(value=0.1, base=10, description='$\sigma_i^2$', min = -2, max = 0, step = 1, continuous_update = False)
	sdf = fixed(seed)
	
	np.random.seed(13)
	def on_button_clicked(b):
		sdf.value = int(time.time())		
		
	rolldice.on_click(on_button_clicked)
	io = interactive_output(plot_observations, {'N_obs':Nsldr,'trueModel':trueModel,'bestModel':bestModel,'var':varsldr,'seed':sdf})
	
	return VBox([HBox([Nsldr, bestModel, trueModel, rolldice, varsldr]), io])
def get_obs(seed, Nobs, mtrue, ctrue,var):
	np.random.seed(seed)
	xo = np.linspace(0,1,Nobs+2)[1:-1]
	yo = f(xo,mtrue,ctrue) + err(xo,var)
	return xo,yo
def plot_posterior(m,c,p):
	x = np.linspace(0,1,101)
	m0,c0 = [2,3]
	var = 0.1
	y = f(x,m0,c0)
	xo,yo = get_obs(13,10,m0,c0,var)
	mf,cf = curve_fit(f, xo, yo, [1,1])[0]

	# initialize figure window and axes
	fig = plt.figure(figsize=(12,6))
	ax1 = plt.axes([0.05, 0.15, 0.35, 0.7])
	ax2 = plt.axes([0.55, 0.15, 0.35, 0.7])
	ax3 = plt.axes([0.95, 0.15, 0.02, 0.7])
	ax4 = fig.add_subplot(122, projection='3d')
	dx = 0.15; dy = 0.25
	ax4.set_position([0.555,0.155,dx,dy])
	ax4.set_xticks([])
	ax4.set_yticks([])
	ax4.set_zticks([])
	ax4.set_facecolor(cm.jet(0))
	#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
	ax1.plot(xo,yo,'wo', mec = 'k', mew = 1.5, ms = 8, label = r'observations', zorder = 10)
	ax1.plot(x,f(x,mf,cf),'r-', label = 'best model')
	ax1.plot(x,y,'b-',label='true process')
	ax1.plot(x,f(x,m,c),'g-')
	
	# show 
	CS = ax2.imshow(np.flipud(p.P), cmap=cm.jet, extent = [p.mmin,p.mmax,p.cmin,p.cmax], aspect='auto')
	plt.colorbar(CS, cax = ax3)
	ax2.plot(m0,c0,'bo', label=r'$\theta_0$',ms = 12, mec='w', mew=3)
	ax2.plot(mf,cf,'ro', label=r'$\hat{\theta}_0$',ms = 12, mec='w', mew=3)
	ax2.plot(m, c, 'go', label=r'$\theta$',ms = 12, mec='w', mew=3)
	ax2.legend(loc=1, prop={'size':TEXTSIZE})
	ax4.plot_surface(p.M, p.C, p.P, rstride=1, cstride=1, cmap=cm.jet, lw = 0.5, zorder = 10)
	
	# plot upkeep + legend
	ax1.set_xlim(ax1.get_xlim())
	ax1.legend(loc=2, prop={'size':TEXTSIZE})
	ax1.set_ylim([1,7])
	ax1.set_xlim([0,1])
	ax1.set_xlabel('$x$',size = TEXTSIZE)
	ax1.set_ylabel('$y$',size = TEXTSIZE)
	ax2.set_xlabel('$m$',size = TEXTSIZE)
	ax2.set_ylabel('$c$',size = TEXTSIZE)
	ax3.set_xlabel('\n'+r'$P(\theta)$',size=TEXTSIZE,rotation=0.)
	for ax in [ax1,ax2,ax3]:
		for t in ax.get_xticklabels()+ax.get_yticklabels(): 
			t.set_fontsize(TEXTSIZE)
def posterior(Nm, Nc):
	mtrue,ctrue = [2,3]
	cmin,c0,cmax = [2.55,3.05,3.55]
	mmin,m0,mmax = [1.3,2.1,2.9]
	m = FloatSlider(value=m0, description=r'$m$', min = mmin, max = mmax, step = (mmax-mmin)/Nm, continuous_update = False)
	c = FloatSlider(value=c0, description=r'$c$', min = cmin, max = cmax, step = (cmax-cmin)/Nc, continuous_update = False)
	p = Posterior(cmin=cmin,cmax=cmax,Nc=Nc,mmin=mmin,mmax=mmax,Nm=Nm,ctrue=ctrue,mtrue=mtrue,var=0.1)
	io = interactive_output(plot_posterior, {'m':m,'c':c,'p':fixed(p)})
	return VBox([HBox([m,c]),io])
class Posterior(object):
	def __init__(self,**kwargs):
		for k in kwargs.keys():
			self.__setattr__(k, kwargs[k])
		self.grid_search()
		self.fit_mvg()
	def grid_search(self):
		xo,yo = get_obs(13,10,self.mtrue,self.ctrue,self.var)
		m = np.linspace(self.mmin,self.mmax,self.Nm); dm = m[1]-m[0]
		c = np.linspace(self.cmin,self.cmax,self.Nc); dc = c[1]-c[0]
		self.dm = dm
		self.dc = dc
		M,C = np.meshgrid(m,c)
		# compute objective function
			# empty vector, correct size, for storing computed objective function
		S = 0.*M.flatten() 
			# for each parameter combination in the grid search
		for i,theta in enumerate(zip(M.flatten(), C.flatten())):
				# unpack parameter vector
			mi,ci = theta
				# compute objective function
			S[i]=np.sum((yo-f(xo,mi,ci))**2)/self.var
			# reshape objective function to meshgrid dimensions
		S = np.array(S).reshape([len(c), len(m)])
			# compute posterior
		self.P = np.exp(-S/2.)
		self.P /= np.sum(self.P)*dm*dc
		self.M = M
		self.C = C
	def fit_mvg(self):
		mv, cv, pv = [vi.flatten() for vi in [self.M,self.C,self.P]]
		self.m1 = np.sum(pv*mv)*self.dm*self.dc
		self.c1 = np.sum(pv*cv)*self.dm*self.dc
			# variances
		smm = np.sum(pv*(mv-self.m1)**2)*self.dm*self.dc
		scc = np.sum(pv*(cv-self.c1)**2)*self.dm*self.dc
		scm = np.sum(pv*(mv-self.m1)*(cv-self.c1))*self.dm*self.dc
			# matrix
		self.cov = np.array([[smm,scm],[scm,scc]])
class UberPosterior(object):
	def __init__(self, **kwargs):
		for k in kwargs.keys():
			self.__setattr__(k, kwargs[k])
		self.xo,self.yo = get_obs(13,10,2,3,self.var)
		for model in ['linear','log','power','sin']:
			self.fit(model)
	def fit(self, model):
		if model is 'linear':
			self.f = linear
			self.Nargs = 2
		elif model is 'power':
			self.f = powerlaw
			self.Nargs = 3
		elif model is 'log':
			self.f = logarithmic
			self.Nargs = 3
		elif model is 'sin':
			self.f = sinusoid
			self.Nargs = 3
		self.grid_search()
		self.__setattr__(model+'_mean', self.mean)
		self.__setattr__(model+'_cov', self.cov)
		self.__setattr__(model, self.f)
		self.__setattr__(model+'_P', self.P)
		self.__setattr__(model+'_PVS', self.PVS)
		self.__setattr__(model+'_dps', self.dps)
	def grid_search(self):
		# get best fit parameters for model
		pi = np.ones(self.Nargs)
		if self.f is sinusoid:	
			pi = [3, 2, 3]
		p,pcov = curve_fit(self.f, self.xo, self.yo, pi, sigma=np.sqrt(self.var/2.)+0.*self.xo, absolute_sigma=True)
		self.mean = p
		self.cov = pcov
		# setup search grid
		pvs = []
		self.dps = []
		for i,pi in enumerate(p):
			pvs.append(np.linspace(pi/3.,pi*3., self.N))
			self.dps.append(abs(pvs[-1][1]-pvs[-1][0]))
		self.PVS = np.meshgrid(*pvs)
		
		# compute objective function
			# empty vector, correct size, for storing computed objective function
		S = 0.*self.PVS[0].flatten() 
			# for each parameter combination in the grid search
		for i,theta in enumerate(zip(*[PVSI.flatten() for PVSI in self.PVS])):
			# compute objective function
			S[i]=np.sum((self.yo-self.f(self.xo,*theta))**2)/self.var
			# reshape objective function to meshgrid dimensions
		S = np.array(S).reshape([len(pv) for pv in pvs])
			# compute posterior
		self.P = np.exp(-S/2.)
		self.P /= np.sum(self.P)*np.product(self.dps)
	def get_samples(self,option,N):
		# use rejection sampling on the posterior
		s = []
		P = self.__getattribute__(option+'_P')
		PVS = self.__getattribute__(option+'_PVS')
		dps = self.__getattribute__(option+'_dps')
		pmax = np.max(P)
		inds = np.where(P>pmax/1000.)
		P2 = P[inds]
		PVS2 = [pvsi[inds] for pvsi in PVS]
		N2 = len(P2)
		while len(s) < N:
			i = np.random.randint(0, N2)
			r = np.random.rand()*pmax
			if P2[i] > r:
				s.append([pvsi[i]+dpsi*(np.random.rand()-0.5) for pvsi,dpsi in zip(PVS2,dps)])
		return s
def plot_predictions(zoom, N, xf, p):
	fig = plt.figure(figsize=(15,5))
	ax1 = plt.axes([0.05, 0.15, 0.25, 0.7])
	ax2 = plt.axes([0.37, 0.15, 0.25, 0.7])
	ax3 = plt.axes([0.69, 0.15, 0.25, 0.7])
	
	x = np.linspace(0,5.5,101)
	m0,c0 = [2,3]
	xo,yo = get_obs(13,10,m0,c0,p.var)
	mf,cf = curve_fit(f, xo, yo, [1,1])[0]

	# get samples
	np.random.seed(13)
	s = multivariate_normal.rvs(mean = [p.m1, p.c1], cov = p.cov, size = int(N))
	if N == 1: 
		s = [s,]
	
	ax1.plot(xo,yo,'wo', mec = 'k', mew = 1.5, ms = 8, label = r'obs.', zorder = 10)
	ax1.plot(x,f(x,mf,cf),'r-', label = 'best model',zorder = 1)
	ax1.plot(x,f(x,m0,c0),'b-',label='true process',zorder = 1)
	
	CS = ax2.imshow(np.flipud(p.P), cmap=cm.jet, extent = [p.mmin,p.mmax,p.cmin,p.cmax], aspect='auto')
	ax2.plot(m0,c0,'bo', label=r'$\theta_0$',ms = 12, mec='w', mew=3, zorder=3)
	ax2.plot(mf,cf,'ro', label=r'$\hat{\theta}_0$',ms = 12, mec='w', mew=3, zorder = 3)
	xlim = ax2.get_xlim(); ax2.set_xlim(xlim)
	ylim = ax2.get_ylim(); ax2.set_ylim(ylim)
	
	alpha = np.min([0.5,10./N])
	yfs = []
	for i,si in enumerate(s):
		ax1.plot(x,f(x,*si),'k-', zorder = 0, lw = 0.5, alpha = alpha)
		ax2.plot(*si, 'kx', mew = 2, ms = 8)
		yfs.append(f(xf,*si))
	ax2.plot([],[], 'kx', mew = 2, ms = 8, label = 'sample')
	ax1.plot([],[],'k-', zorder = 0, lw = 0.5, label='sample')
	
	bins = np.linspace(np.min(yfs)*0.999, np.max(yfs)*1.001, int(np.sqrt(N))+1)
	h,e = np.histogram(yfs, bins)
	h = h/(np.sum(h)*(e[1]-e[0]))
	ax3.bar(e[:-1],h,e[1]-e[0], color = [0.5,0.5,0.5])
	ax3.set_xlim([4,20])
	ax3.set_ylim([0,1])
	
	if N>10:
		yf = f(xf, mf, cf)
		ax3.axvline(yf, label='best model',color = 'r', linestyle = '-')
		y0 = f(xf, m0, c0)
		ax3.axvline(y0, label='true process',color = 'b', linestyle = '-')
		
		yf5,yf95 = np.percentile(yfs, [5,95])
		ax3.axvline(yf5, label='90% interval',color = 'k', linestyle = '--')
		ax3.axvline(yf95, color = 'k', linestyle = '--')
		
	
	ax1.set_xlim(ax1.get_xlim())
	ax1.axvline(xf, color = 'k', linestyle=':', label = '$x_f$')
	ax1.legend(loc=4, prop={'size':TEXTSIZE-1})
	ax2.legend(loc=3, prop={'size':TEXTSIZE})
	ax3.legend(loc=1, prop={'size':TEXTSIZE})
	ax1.set_ylim([0,15])
	ax1.set_xlim([0,5.5])
	if zoom:
		ax1.set_ylim([1,7])
		ax1.set_xlim([0,1])
	ax1.set_xlabel('$x$',size = TEXTSIZE)
	ax1.set_ylabel('$y$',size = TEXTSIZE)
	ax2.set_xlabel('$m$',size = TEXTSIZE)
	ax2.set_ylabel('$c$',size = TEXTSIZE)
	ax3.set_xlabel('$y_f$',size = TEXTSIZE)
	ax3.set_ylabel('$P(y_f)$',size = TEXTSIZE)
		
	for ax in [ax1,ax2,ax3]:
		for t in ax.get_xticklabels()+ax.get_yticklabels(): 
			t.set_fontsize(TEXTSIZE)
def prediction(var):
	mtrue,ctrue = [2,3]
	cmin,c0,cmax = [2.55,3.05,3.55]
	mmin,m0,mmax = [1.3,2.1,2.9]
	cmin,c0,cmax = [1.55,3.05,4.55]
	mmin,m0,mmax = [0.3,2.1,4.9]
	p = Posterior(cmin=cmin,cmax=cmax,Nc=101,mmin=mmin,mmax=mmax,Nm=101,ctrue=ctrue,mtrue=mtrue,var=var)
	zoom = Checkbox(value = False, description='zoom')
	Nsamples = FloatLogSlider(value = 16, base=4, description='samples', min = 0, max = 5, step = 1, continuous_update=False)
	xf = FloatSlider(value=3, description=r'$x_f$', min = 2, max = 5, step = 0.5, continuous_update = False)
	io = interactive_output(plot_predictions, {'zoom':zoom,'N':Nsamples,'xf':xf,'p':fixed(p)})
	return VBox([HBox([zoom,Nsamples,xf]),io])
def linear(x, *p): return p[0]*x + p[1]
def logarithmic(x, *p): return p[0]+p[1]*np.log10(x+p[2])
def powerlaw(x, *p): return p[0]+p[1]*x**p[2]
def sinusoid(x, *p): return p[0]+p[1]*np.sin(p[2]*(x-0.1))
def plot_structural(zoom, option, xf, p):
	fig = plt.figure(figsize=(12,6))
	ax1 = plt.axes([0.05, 0.15, 0.35, 0.7])
	ax2 = plt.axes([0.55, 0.15, 0.35, 0.7])
	
	if option == 1:
		f2 = powerlaw
		col = 'm'
		option = 'power'
	elif option == 2:
		f2 = logarithmic
		col = 'r'
		option = 'log'
	elif option == 3:
		f2 = sinusoid
		col = 'g'
		option = 'sin'
		
	N = 256
	
	x = np.linspace(p.xo[0],5.5,101)
	m0,c0 = [2,3]
	xo,yo = get_obs(13,10,m0,c0,p.var)
	mf,cf = curve_fit(f, xo, yo, [1,1])[0]

	np.random.seed(13)
	s = multivariate_normal.rvs(mean = p.linear_mean, cov = p.linear_cov, size = int(N))
	if N == 1: 
		s = [s,]
	mean = p.__getattribute__(option+'_mean')
	s2 = p.get_samples(option, N)
	if N == 1: 
		s2 = [s2,]
	
	ax1.plot(xo,yo,'wo', mec = 'k', mew = 1.5, ms = 8, label = r'obs.', zorder = 10)
	ax1.plot(x,f(x,m0,c0),'b-',label='true process',lw=2,zorder = 1)
	S = np.sum((p.yo-f(p.xo,mf,cf))**2/p.var)
	ax1.plot(x,f(x,mf,cf),'k-',label='best linear'+' (S={:2.1f})'.format(S),lw=2,zorder = 1)
	
	alpha = np.min([0.5,10./N])
	yfs = []; yfs2=[]
	for si,si2 in zip(s,s2):
		#print(si2)
		ax1.plot(x,f(x,*si),'k-', zorder = 0, lw = 0.5, alpha = alpha)
		yfs.append(f(xf,*si))
		ax1.plot(x,f2(x,*si2),col+'-', zorder = 0, lw = 0.5, alpha = alpha)
		yfs2.append(f2(xf,*si2))
	S = np.sum((p.yo-f2(p.xo,*mean))**2/p.var)
	ax1.plot(x,f2(x,*mean),col+'-',label='best '+option+' (S={:2.1f})'.format(S),lw=2,zorder = 1)
	
	bins = np.linspace(np.min(yfs)*0.999, np.max(yfs)*1.001, int(np.sqrt(N))+1)
	bins = np.linspace(0,20,61)
	h,e = np.histogram(yfs, bins)
	h = h/(np.sum(h)*(e[1]-e[0]))
	ax2.bar(e[:-1],h,e[1]-e[0], color = 'k', alpha=0.5, edgecolor='k')
	
	#bins2 = np.linspace(np.min(yfs2)*0.999, np.max(yfs2)*1.001, int(np.sqrt(N))+1)
	h,e = np.histogram(yfs2, bins)
	h = h/(np.sum(h)*(e[1]-e[0]))
	ax2.bar(e[:-1],h,e[1]-e[0], color = col, alpha=0.5, edgecolor='k')
	
	ax2.set_xlim([0,20])
	ax2.set_ylim([0,1])
	
	if N>10:
		y0 = f(xf, m0, c0)
		ax2.axvline(y0, label='true process',color = 'b', linestyle = '-')
		
		yf5,yf95 = np.percentile(yfs, [5,95])
		ax2.axvline(yf5, label='90% linear',color = 'k', linestyle = '--')
		ax2.axvline(yf95, color = 'k', linestyle = '--')
		
		yf5,yf95 = np.percentile(yfs2, [5,95])
		ax2.axvline(yf5, label='90% '+option,color = col, linestyle = '--')
		ax2.axvline(yf95, color = col, linestyle = '--')
		
	ax1.set_xlim(ax1.get_xlim())
	ax1.axvline(xf, color = 'k', linestyle=':')#, label = '$x_f$')
	ax1.legend(loc=2, prop={'size':TEXTSIZE-3})
	ax2.legend(loc=1, prop={'size':TEXTSIZE})
	ax1.set_ylim([0,15])
	ax1.set_xlim([0,5.5])
	if zoom:
		ax1.set_ylim([1,7])
		ax1.set_xlim([0,1])
	ax1.set_xlabel('$x$',size = TEXTSIZE)
	ax1.set_ylabel('$y$',size = TEXTSIZE)
	ax2.set_xlabel('$y_f$',size = TEXTSIZE)
	ax2.set_ylabel('$P(y_f)$',size = TEXTSIZE)
		
	for ax in [ax1,ax2,]:
		for t in ax.get_xticklabels()+ax.get_yticklabels(): 
			t.set_fontsize(TEXTSIZE)
def structural():
	var = 0.1
	mtrue,ctrue = [2,3]
	cmin,c0,cmax = [2.55,3.05,3.55]
	mmin,m0,mmax = [1.3,2.1,2.9]
	#p = Posterior(cmin=cmin,cmax=cmax,Nc=31,mmin=mmin,mmax=mmax,Nm=31,ctrue=ctrue,mtrue=mtrue,var=var)
	p = UberPosterior(N=41, var=var)
	zoom = Checkbox(value = False, description='zoom')
	options = Dropdown(options = {'power-law':1, 'logarithmic':2, 'sinusoidal':3}, value = 2, description='alternative model')
	zoom = Checkbox(value = False, description='zoom')
	xf = FloatSlider(value=3, description=r'$x_f$', min = 2, max = 5, step = 0.5, continuous_update = False)
	io = interactive_output(plot_structural, {'zoom':zoom,'option':options,'xf':xf,'p':fixed(p)})
	return VBox([HBox([zoom,options,xf]),io])
