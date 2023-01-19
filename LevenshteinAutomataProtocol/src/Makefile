install:
	pip install pyzmq phe numpy pytest

test-communication:
	pytest ../test/communication_test.py -k test_server -q &
	pytest ../test/communication_test.py -k test_client -q
