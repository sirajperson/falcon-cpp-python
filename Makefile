update:
	poetry install
	git submodule update --init --recursive

update.vendor:
	cd vendor/ggllm.cpp && git pull origin master

build:
	python3 setup.py develop

build.cuda:
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 python3 setup.py develop

build.opencl:
	CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 python3 setup.py develop

build.openblas:
	CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 python3 setup.py develop

build.blis:
	CMAKE_ARGS="-DLLAMA_OPENBLAS=on -DLLAMA_OPENBLAS_VENDOR=blis" FORCE_CMAKE=1 python3 setup.py develop

build.metal:
	CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 python3 setup.py develop

build.sdist:
	python3 setup.py sdist

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

clean:
	- cd vendor/ggllm.cpp && make clean
	- cd vendor/ggllm.cpp && rm llamacpp.so
	- rm -rf _skbuild
	- rm falcon_cpp/*.so
	- rm falcon_cpp/*.dylib
	- rm falcon_cpp/*.metal
	- rm falcon_cpp/*.dll
	- rm falcon_cpp/*.lib

.PHONY: \
	update \
	update.vendor \
	build \
	build.cuda \
	build.opencl \
	build.openblas \
	build.sdist \
	deploy.pypi \
	deploy.gh-docs \
	clean