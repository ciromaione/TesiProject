install:
	pip install -r requirements.txt

test-communication:
	pytest ./test/communication_test.py -k test_server -q &
	pytest ./test/communication_test.py -k test_client -q

test-ot:
	pytest ./test/ot_test.py -k TestEncoder -q
	pytest ./test/ot_test.py -k test_sender -q &
	pytest ./test/ot_test.py -k test_receiver -q

test-odfae:
	pytest ./test/odfae_test.py -k TestUtilsFunctions -q
	pytest ./test/odfae_test.py -k test_garbler -q &
	pytest ./test/odfae_test.py -k test_evaluator -q

test-psi:
	pytest ./test/lev_da_psi_test.py -k test_alice -q &
	pytest ./test/lev_da_psi_test.py -k test_bob -q
