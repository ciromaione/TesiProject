import numpy as np
from src.ot import encode_message, decode_message, OTSender, OTReceiver
from phe import paillier
from src.communication import ServerSocket, ClientSocket


class TestEncoder:
    def test_encoding(self):
        arr = np.array([
            np.random.bytes(100),
            np.random.bytes(100),
            np.random.bytes(100)
        ])
        data = arr[0] + arr[1] + arr[2]
        chunk1 = data[0:256]
        assert len(chunk1) == 256
        chunk2 = data[256:]
        assert len(chunk2) == 44
        res = {
            "lc": 44,
            "vals": [
                int.from_bytes(chunk1, 'big'),
                int.from_bytes(chunk2, 'big')
            ]
        }

        res2 = encode_message(arr)
        assert res == res2

    def test_decoding(self):
        arr = np.array([
            np.random.bytes(100),
            np.random.bytes(100),
            np.random.bytes(100)
        ])
        data = arr[0] + arr[1] + arr[2]
        chunk1 = data[0:256]
        assert len(chunk1) == 256
        chunk2 = data[256:]
        assert len(chunk2) == 44
        d = {
            "lc": 44,
            "vals": [
                int.from_bytes(chunk1, 'big'),
                int.from_bytes(chunk2, 'big')
            ]
        }

        res = decode_message(d, 100)
        assert (res == arr).all()

    def test_enc_dec(self):
        arr = np.array([
            np.random.bytes(100),
            np.random.bytes(100),
            np.random.bytes(100),
            np.random.bytes(100),
            np.random.bytes(100),
            np.random.bytes(100)
        ])

        enc = encode_message(arr)
        assert enc['lc'] == 88
        assert len(enc['vals']) == 3

        dec = decode_message(enc, 100)
        assert (dec == arr).all()

        arr = np.array([b'asdfg', b'hklpo'])
        enc = encode_message(arr)
        dec = decode_message(enc, 5)
        assert (dec == arr).all()


pk = paillier.PaillierPublicKey(
    4705964732092977493086560386626438048649770880106897606640175205624197690750367272440666727504085607422359977335364696343395617147790470047733181992863396832514046398178993554292914294607596219141734268595540841764868526547598736674979799677053344349025047288187883881695725168351269815356657204595180275392828995931784858863832323860404328784960107613095366915473222390134631179939277264553302486521711492033377089927053545136665203140994076281291195138959457341970151637024559053874199454615327612566393737681740426222549933002787272137607562298344970110347950396603331436673422279458426930732797166388049887143349262585827154707654438498208237696601002730548857968225543428481703759812857485951990913001916951022415219873772852840379251239884929469753627445213041367043517569889112807457946245543286970212231684157609315549247222376857125705895589430693584324384034004523384628795228251136860868267269973394155433419135699)
sk = paillier.PaillierPrivateKey(
    pk,
    2140796533195830358354160316221708910835711537152069320108748635547855374428236639648423715008741750787380806223756721022258223927767632823953488522226447288394932121696251209764926046152337675673111322872830211047274905609258739334049529536788371311192889102709819475724501651502717029020262523285846043126177950205206625357609239048974070841111945769782752881431049900675137245894518977864684741384370743226287631765101608402655738245944799723125521193828835307,
    2198230732870164444031948375988524765946016321007894680092417213162154083126545029263731250047524707028001391114371399815442343989961053815928658686728739115722980585107539875742203472802699196804429316314227090040807559064191500674725578641384808612595233056457770616333195165587359377004850148080167155984738499283023210493500136419669357896187997588597782317502482187482650945108220776542434677421412566829729683055497690499223459410023354372837538048075177657
)
garbled_arrays = [
    np.array([
        [b'qwert', b'yuiop', b'asdfg'],
        [b'hjkl?', b'zxcvb', b'nm,.-']
    ]),
    np.array([
        [b'fjsow', b'ewop3', b'eui3h'],
        [b',sieh', b'dkjei', b'387hf']
    ]),
    np.array([
        [b'djoef', b'winde', b'pinyd'],
        [b'ai93j', b'w3e40', b'49hnj']
    ])
]

n = 3
choices = [1, 1, 2]
enc_len = 5

expected_result = np.array([
    [b'yuiop', b'ewop3', b'pinyd'],
    [b'zxcvb', b'dkjei', b'49hnj']
])


def test_sender():
    socket = ServerSocket(9000)
    try:
        sender = OTSender(n, socket, pk)
        sender.send_secrets(garbled_arrays)
    finally:
        socket.socket.close()


def test_receiver():
    socket = ClientSocket('localhost', 9000)
    try:
        recv = OTReceiver(n, socket, pk, sk, enc_len)
        res = recv.recv_secrets(choices)
        assert (expected_result == res).all()
    finally:
        socket.socket.close()
