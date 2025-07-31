CURRENT_DATE = $(shell date '+%Y%m%dT%H%M%S')
MODEL_DIRECTORY = output/models/$(CURRENT_DATE)
EXEC=python3 main.py
TRAIN=${EXEC} train
PREDICT=${EXEC} predict

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: clean-code
clean-code:
	black .

.PHONY: clear
clear:
	rm -rf tb_logs/
	rm -rf lightning_logs/
	rm -rf resources/mnist_full/MNIST/
	rm -rf qaradoq-p2.tar.tb2
	find . \( -name __pycache__ -o -name "*.pyc" \) -delete

.PHONY: board
board:
	tensorboard --logdir=tb_logs/

.PHONY: tarball
tarball: clear
	mkdir qaradoq-p2
	cp -r main.py Makefile README.md requirements.txt resources qml docs qaradoq-p2
	zip -r qaradoq-p2.zip qaradoq-p2
	rm -rf qaradoq-p2

.PHONY: board
pca:
	${TRAIN} --topology=pca --batch-size=-1 --epochs=1 --topology-extra={components:12}

.PHONY: mnist-tristan
mnist-tristan:
	${TRAIN} --topology=tristan_qt --batch-size 128 --epochs 5 --output-directory=${MODEL_DIRECTORY}-tristan-qt

.PHONY: mnist-dagonet
mnist-dagonet:
	${TRAIN} --topology=dagonet_qt --batch-size 128 --epochs 5 --output-directory=${MODEL_DIRECTORY}-dagonet-qt