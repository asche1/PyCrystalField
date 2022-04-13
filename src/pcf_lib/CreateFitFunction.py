# Create function for fitting
# Allen Scheie
# February, 2017

import itertools

def makeFitFunction(function, fitargs, **kwargs):
	"""return function with only fitargs as a concatinated variable,
	as required by scipy.minimize
	combined into a single variable. To be used in minimize routines."""
	
	# A) Get function arguments
	numargs = function.__code__.co_argcount
	funcargs = [x for x in function.__code__.co_varnames[:numargs] if x != 'self']
	nonFitArgs = {x : kwargs[x] for x in kwargs if x not in fitargs}
	# B) Warn user of discarded arguments (if any fitargs not in kwargs)
	nonArgs = [x for x in fitargs if x not in kwargs]
	if len(nonArgs)>0: print('  Warning: ',nonArgs,'not in function arguments. Discarding',nonArgs)
	fitargs = [x for x in fitargs if x not in nonArgs]

	# C) Define starting values for fitargs
	p0 = []
	for fa in fitargs:
		kwa = kwargs[fa]
		try:
			for k in kwa:
				p0.append(k)
		except TypeError:  #This happens if 'kwa' is an int or float
			p0.append(kwa)

	lengths = []
	index = 0
	for fa in fitargs:
		fitval = kwargs[fa]
		try: 
			lengths.append([index, index+len(fitval)])
			index += len(fitval)
		except TypeError:  #This happens if 'fitval' is an int or float
			lengths.append([index, index+1])
			index +=1
		

	# D) Create new function
	scope = locals()
	fitfunc = eval('lambda x: function('+
		', '.join([fa+'=x['+':'.join(str(l) for l in lengths[i])+']' for i, 
					fa in enumerate(fitargs)])+
		', **nonFitArgs)', scope)

	# E) Create function which splits the result back into variables
	def resultfunc(x):
	 	return {fa : x[lengths[i][0]: lengths[i][1]] for i, fa in enumerate(fitargs)}

	return fitfunc, p0, resultfunc






def makeCurveFitFunction(function, fitargs, **kwargs):
	"""return function with only fitargs as a concatinated variable,
	as required by scipy.optimize.curve_fit
	combined into a single variable. To be used in minimize routines."""
	
	# A) Get function arguments
	numargs = function.__code__.co_argcount
	funcargs = [x for x in function.__code__.co_varnames[:numargs] if x != 'self']
	nonFitArgs = {x : kwargs[x] for x in kwargs if x not in fitargs}
	# B) Warn user of discarded arguments (if any fitargs not in kwargs)
	nonArgs = [x for x in fitargs if x not in kwargs]
	if len(nonArgs)>0: print('  Warning: ',nonArgs,'not in function arguments. Discarding',nonArgs)
	fitargs = [x for x in fitargs if x not in nonArgs]

	# C) Define starting values for fitargs
	p0 = []
	for fa in fitargs:
		kwa = kwargs[fa]
		try:
			for k in kwa:
				p0.append(k)
		except TypeError:  #This happens if 'kwa' is an int or float
			p0.append(kwa)

	lengths = []
	index = 0
	for fa in fitargs:
		fitval = kwargs[fa]
		try: 
			lengths.append([index, index+len(fitval)])
			index += len(fitval)
		except TypeError:  #This happens if 'fitval' is an int or float
			lengths.append([index, index+1])
			index +=1
		

	print('lambda '+', '.join([fa[:-1] for fa in fitargs])+' : function('+\
		', '.join([fa+'='+fa[:-1] for fa in fitargs])+', **nonFitArgs)')
	print(fitargs, nonFitArgs)
	# D) Create new function
	scope = locals()
	fitfunc = eval('lambda '+', '.join([fa[:-1] for fa in fitargs])+' : function('+\
		', '.join([fa+'='+fa[:-1] for fa in fitargs])+', **nonFitArgs)')

	# E) Create function which splits the result back into variables
	def resultfunc(x):
	 	return {fa[:-1] : x[lengths[i][0]: lengths[i][1]] 
	 			if (lengths[i][1]-lengths[i][0]) > 1 else x[lengths[i][0]]
	 			for i, fa in enumerate(fitargs)}

	return fitfunc, p0, resultfunc
