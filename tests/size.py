import sys
from types import ModuleType, FunctionType
from gc import get_referents
import numpy as np
import torch

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType



def getsize(obj):
	"""sum size of object & members."""
	if isinstance(obj, BLACKLIST):
		raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
	seen_ids = set()
	size = 0
	objects = [obj]
	while objects:
		need_referents = []
		for obj in objects:
			if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
				seen_ids.add(id(obj))
				size += sys.getsizeof(obj)
				need_referents.append(obj)
		objects = get_referents(*need_referents)
	return size

def main():
	print(getsize(np.array([])))
	print(getsize(np.array([1])))
	print(getsize(np.array([1.])))
	print(getsize(np.array([1,2,3])))
	print(getsize(np.array([1.53641723,2.123123,343265734.123135345])))
	print(getsize(np.float32(1)))
	print(getsize(np.int32(1)))
	print(getsize(np.random.randn(300,100000).astype(np.float32)))
	a = torch.tensor([])
	print(a.element_size() * a.nelement())
	a = torch.tensor([1])
	print(a.element_size() * a.nelement())
	a = torch.tensor([1,2,3])
	print(a.element_size() * a.nelement())
	a = torch.randint(0,100,(1000,1000))
	print(a.element_size() * a.nelement())


if __name__ == "__main__":
	main()