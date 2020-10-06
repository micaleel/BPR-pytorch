.ONESHELL:

install:
	conda create -n torch-bpr python=3.8 --yes
	@conda env update -n torch-bpr -f environment.yml
	@./cuda.sh

uninstall:
	conda env remove -n torch-bpr