install:
	pip install pyzmq phe numpy pytest

test-communication:
	pytest ./test/communication_test.py -k test_server -q &
	pytest ./test/communication_test.py -k test_client -q

test-ot:
	pytest ./test/ot_test.py -k TestEncoder -q
	pytest ./test/ot_test.py -k test_sender -q &
	pytest ./test/ot_test.py -k test_receiver -q
