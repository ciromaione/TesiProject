import sys

from phe import paillier
import numpy as np
import numpy.typing as npt
import LevenshteinAutomataProtocol.src.communication as com

CHUNKS_LEN = 256


def encode_message(array: npt.NDArray) -> dict:
    data = b''
    for el in array:
        data += el
    last_chunk_len = len(data) % CHUNKS_LEN
    values = []
    for i in range(0, len(data), CHUNKS_LEN):
        values.append(int.from_bytes(data[i: i + CHUNKS_LEN], 'big'))
    return {
        "lc": last_chunk_len,
        "vals": values
    }


def decode_message(data: dict, len_enc: int) -> npt.NDArray:
    values = data["vals"]
    last = values.pop()
    raw = b''
    for v in values:
        try:
            raw += v.to_bytes(CHUNKS_LEN, 'big')
        except OverflowError:
            print("value: ", v)
    raw += last.to_bytes(data["lc"], 'big')
    mess = [raw[i:i + len_enc] for i in range(0, len(raw), len_enc)]
    return np.array(mess)


class OTSender:
    def __init__(self, n: int, socket: com.ServerSocket, pk: paillier.PaillierPublicKey):
        self.n = n
        self.socket = socket
        self.public_key = pk

    def send_secrets(self, matrix: npt.NDArray):
        choice_bits = self.socket.recv()
        secrets = tuple(encode_message(matrix[:, i]) for i in range(self.n))

        n_ciphertexts = len(secrets[0]["vals"])
        ciphertexts = [self.public_key.encrypt(0) for _ in range(n_ciphertexts)]

        for b, s in zip(choice_bits, secrets):
            for i in range(n_ciphertexts):
                ciphertexts[i] += b * s["vals"][i]

        last_chunk_size = self.public_key.encrypt(0)
        for i, b in enumerate(choice_bits):
            last_chunk_size += b * secrets[i]["lc"]

        self.socket.send({
            "lc": last_chunk_size,
            "vals": ciphertexts
        })



class OTReceiver:
    def __init__(
            self,
            n: int,
            socket: com.ClientSocket,
            pk: paillier.PaillierPublicKey,
            sk: paillier.PaillierPrivateKey,
            len_encoding_states: int
    ):
        self.public_key = pk
        self.secret_key = sk
        self.n = n
        self.socket = socket
        self.len_enc_states = len_encoding_states

    def recv_secret(self, choice) -> npt.NDArray:
        encoded_choice = [self.public_key.encrypt(1 if i == choice else 0) for i in range(self.n)]
        ciphertext = self.socket.send_wait(encoded_choice)
        lc = self.secret_key.decrypt(ciphertext["lc"])
        values = [self.secret_key.decrypt(v) for v in ciphertext["vals"]]
        return decode_message({
            "lc": lc,
            "vals": values
        }, self.len_enc_states)


if __name__ == '__main__':
    pk = paillier.PaillierPublicKey(
        4705964732092977493086560386626438048649770880106897606640175205624197690750367272440666727504085607422359977335364696343395617147790470047733181992863396832514046398178993554292914294607596219141734268595540841764868526547598736674979799677053344349025047288187883881695725168351269815356657204595180275392828995931784858863832323860404328784960107613095366915473222390134631179939277264553302486521711492033377089927053545136665203140994076281291195138959457341970151637024559053874199454615327612566393737681740426222549933002787272137607562298344970110347950396603331436673422279458426930732797166388049887143349262585827154707654438498208237696601002730548857968225543428481703759812857485951990913001916951022415219873772852840379251239884929469753627445213041367043517569889112807457946245543286970212231684157609315549247222376857125705895589430693584324384034004523384628795228251136860868267269973394155433419135699)
    sk = paillier.PaillierPrivateKey(
        pk,
        2140796533195830358354160316221708910835711537152069320108748635547855374428236639648423715008741750787380806223756721022258223927767632823953488522226447288394932121696251209764926046152337675673111322872830211047274905609258739334049529536788371311192889102709819475724501651502717029020262523285846043126177950205206625357609239048974070841111945769782752881431049900675137245894518977864684741384370743226287631765101608402655738245944799723125521193828835307,
        2198230732870164444031948375988524765946016321007894680092417213162154083126545029263731250047524707028001391114371399815442343989961053815928658686728739115722980585107539875742203472802699196804429316314227090040807559064191500674725578641384808612595233056457770616333195165587359377004850148080167155984738499283023210493500136419669357896187997588597782317502482187482650945108220776542434677421412566829729683055497690499223459410023354372837538048075177657
    )
    matrix2 = np.array([
        [b'qwert', b'yuiop', b'asdfg'],
        [b'hjkl?', b'zxcvb', b'nm,.-']
    ])

    n = 3
    choice = 2
    enc_len = 5


    def test_sender():
        socket = com.ServerSocket(9000)
        sender = OTSender(n, socket, pk)
        ciphertext = sender.send_secrets(matrix2)
        print(ciphertext)
        lc = sk.decrypt(ciphertext["lc"])
        values = [sk.decrypt(v) for v in ciphertext["vals"]]
        print("lc", lc)
        print("values", values)


    def test_receiver():
        socket = com.ClientSocket('localhost', 9000)
        recv = OTReceiver(n, socket, pk, sk, enc_len)
        res = recv.recv_secret(choice)
        print(res)
        assert res == matrix2[:, choice]


    if sys.argv[1] == 'server':
        test_sender()

    else:
        test_receiver()
