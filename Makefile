.PHONY: proto master slave test

proto:
	python -m ml_crowd.proto.build

master:
	python -m ml_crowd.master.server --model $(MODEL) --port 50051

slave:
	python -m ml_crowd.slave.client --master localhost:50051 --port $(PORT)

test:
	python -m pytest tests -q
