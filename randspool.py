import numpy as np

def order2prime(order):
    if order == 6:
        return 100003
    elif order == 5:
        return 10007
    else:
        return int(10**order) + 1

class randspool:
    def __init__(self, cache_len=64433, genfun=np.random.standard_normal,
        order=None):
        """Init new spooler.
        
        cache_len: size to cache. should be prime?
            http://www.primos.mat.br/primeiros_10000_primos.txt
        
        
        """
        if order is not None:
            cache_len = order2prime(order)
        self.genfun = genfun
        self.cache = self._generate(cache_len)
        self._index = 0
    
    def _generate(self, cache_len):
        return self.genfun(size=(cache_len,))
    
    def get(self, shape=None, copy=False):
        N = reduce(np.multiply, shape)
        stop_index = self._index + N
        if shape is None:
            shape = (1,)
        
        if stop_index <= len(self.cache):
            res = self.cache[self._index:stop_index]
        else:
            # only works for a single overrun..
            res = np.concatenate([
                self.cache[self._index:],
                self.cache[:N-(len(self.cache)-self._index)]
                ])
        res = res.reshape(shape)
        self._index = np.mod(stop_index, len(self.cache)) # in case of == 

        
        
        
        if copy is True:
            res = res.copy()
        
        return res
    
    
