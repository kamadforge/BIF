###########
#orange skin,

#switchg nn - netter point estimate
#no switch nn - sampling better around 0.25, pe also distiguiahs four featyres but about 0.2 each

#switch nn, 110, epochs 20, lr 0.1, pe, nokl
importance = [0.25,0.248, 0.242, 0.26,  0.,    0.,    0.,    0.,    0.,    0.]

#no switch, sampling, nokl
importance = [0.245, 0.238, 0.27,  0.247, 0.,    0.,    0.,    0.,    0.,    0.   ],

##########
#nonlinear

#no switch
importance = [0.565, 0.   , 0.172, 0.262, 0. ,   0. ,   0. ,   0. ,   0. ,   0.]

switch nn
importance = [0.768, 0.  ,  0.  ,  0.232, 0.  ,  0.   , 0. ,   0.  ,  0.   , 0.]

########
#xor
#no switch
importance = [0.493, 0.506, 0.,    0. ,   0. ,   0. ,   0. ,   0.,    0.,    0.]

#switch nn

importance = [0.571, 0.429, 0. ,   0. ,   0.  ,  0. ,   0. ,   0. ,   0.,    0.]