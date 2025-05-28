import numpy as np


def stringUncertainty(value, unc):
	'''returns a string with uncertainty in parenthesis behind the main value'''
	try:
		sigdigits = int(-np.log10(unc)+1)
		if unc > 10:
			if str(unc)[0] != '1':
				sigdigits -= 1
		if (unc >= 1) and (unc < 10):
			if str(unc)[0] == '1':
				sigdigits += 1
				roundedunc = str(np.around(unc, sigdigits))
			else:
				roundedunc = str(np.around(unc, sigdigits))[0]
			roundedval = np.around(value, sigdigits)
			return ('{:.'+str(sigdigits)+'f}').format(roundedval)+'('+roundedunc+')'
		roundedval = np.around(value, sigdigits)
		roundedunc = str(np.around(unc, sigdigits))[-1]
		# if str(np.around(unc, sigdigits+1))[-2] == '1':
		if roundedunc == '1':
			roundedval = np.around(value, sigdigits+1)
			roundedunc = str(int(np.around(unc, sigdigits+1)*10**(sigdigits+1)))
			if roundedunc == '100':
				sigdigits -= 1
				roundedunc = '10'
			if roundedunc[0] == '9':
				return ('{:.'+str(sigdigits+1)+'f}').format(roundedval)+'('+roundedunc[0]+')'
			else:
				return ('{:.'+str(sigdigits+1)+'f}').format(roundedval)+'('+roundedunc+')'
		elif sigdigits < 0:
			return '{}'.format(int(roundedval)) +'('+str(int(np.around(unc, sigdigits)))+')'
		else:
			return ('{:.'+str(sigdigits)+'f}').format(roundedval)+'('+roundedunc+')'
	except OverflowError:
		return str(value)


def stringUncertaintyAniso(value, maxval, minval):
	try:
		uncp = maxval-value
		uncm = value - minval
		smallestUnc = np.min([uncp, uncm])
		try:
			sigdigits = int(-np.log10(smallestUnc)+1)
		except ValueError: return str(value)
		if (smallestUnc >= 1):
			if str(smallestUnc)[0] == '1':
				sigdigits += 1

		if str(np.around(smallestUnc, sigdigits))[-1] == '1':
			sigdigits += 1

		roundedval = np.around(value, sigdigits)
		roundeduncp = str(np.around(uncp, sigdigits))
		roundeduncm = str(np.around(uncm, sigdigits))

		if sigdigits > 0:
			return ('{:.'+str(sigdigits)+'f}').format(roundedval)+'^{+'+roundeduncp+'}_{-'+roundeduncm+'}'
		else:
			return '{}'.format(int(roundedval)) +'^{+'+str(int(np.around(uncp, sigdigits)))+'}_{-'+\
						str(int(np.around(uncm, sigdigits)))+'}'

	except OverflowError:
		return str(value)

### Test it
# print( stringUncertainty(4.000235358278, 0.00834) )
