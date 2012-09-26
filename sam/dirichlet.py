#!/usr/bin/python
from __future__ import division
from sam import *

from math import gamma
from numpy import multiply

def dirichlet_pdf(alpha):
    k = len(alpha)
    gamma_sum = gamma(sum(alpha))
    product_gamma = reduce(multiply, [gamma(a) for a in alpha])
    beta = product_gamma / gamma_sum
    
    return lambda x: reduce(multiply, [x[i]**(alpha[i]-1) for i in xrange(k)])

def main(alphas):
    d = dirichlet_pdf(alphas)
    
    smooth = [1/3,1/3,1/3]
    sparse = [0.98,0.01,0.01]
    
    print
    print 'sparse > smooth' if d(sparse) > d(smooth) else 'smooth > sparse' if not about(d(smooth),d(sparse)) else 'sparse ~ smooth'
    print '.'
    print 'alphas', alphas
    print 'smooth %.1e' % d(smooth)
    print 'sparse %.1e' % d(sparse)

main([1/2,1/2,1/2])
main([1,1,1])
main([2,2,2])

