import json
from pathlib import Path
import os
import importlib
from .base_backend import BaseBackend, UnittestBaseBackend
from .base_imagebackend import BaseImageBackend
import inspect

# get the config json to read which backend to use
base_path = Path(__file__).parent
config_path = (base_path / "../config.json").resolve()
with open(config_path,'rb') as fp:
	config = json.load(fp)
backend_name = config['backend']
imagebackend_name = config['imagebackend']
floatx = config['floatx']

#list available backends
available_backends = []
for file_name in os.listdir(base_path):
	file_end = '_backend.py'
	if (file_name[-len(file_end):]==file_end) and (file_name[:-len(file_end)]!='base'):
		available_backends.append(file_name[:-len(file_end)])

available_imagebackends = []
for file_name in os.listdir(base_path):
	file_end = '_imagebackend.py'
	if (file_name[-len(file_end):]==file_end) and (file_name[:-len(file_end)]!='base'):
		available_imagebackends.append(file_name[:-len(file_end)])

#check the choosen backend from the config.json file exit in available backends
if backend_name not in available_backends:
	raise ModuleNotFoundError(f"No module named '{backend_name}'."\
	" make sure the name of the backend in the config.json file is correct."\
	f" use one of the following available backends: {available_backends}" ,name=backend_name, path = __file__)

if imagebackend_name not in available_imagebackends:
	raise ModuleNotFoundError(f"No module named '{imagebackend_name}'."\
	" make sure the name of the backend in the config.json file is correct."\
	f" use one of the following available backends: {available_imagebackends}" ,name=imagebackend_name, path = __file__)

#import the correct backend module and the correct classes
backend_module = importlib.import_module(f'.{backend_name}_backend', __name__)
for module_object_name in dir(backend_module):
	module_object = getattr(backend_module,module_object_name)
	if inspect.isclass(module_object):

		if issubclass(module_object, BaseBackend) and (module_object is not BaseBackend):
			backend_class = module_object()
			backend_class.default_dtype = getattr(backend_class, floatx)

		if issubclass(module_object, UnittestBaseBackend) and (module_object is not UnittestBaseBackend):
			unittest_backend = module_object




imagebackend_module = importlib.import_module(f'.{imagebackend_name}_imagebackend', __name__)
for module_object_name in dir(imagebackend_module):
	module_object = getattr(imagebackend_module,module_object_name)
	if inspect.isclass(module_object):

		if issubclass(module_object, BaseImageBackend) and (module_object is not BaseImageBackend):
			imagebackend_class = module_object()


#import all the backend class functions as this module functions
blacklist_items = ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl']
imported_funcs = dict()
for name in set(dir(backend_class)) - set(blacklist_items):
	imported_funcs[name] = getattr(backend_class, name)

globals().update(imported_funcs)

globals()['image'] = imagebackend_class