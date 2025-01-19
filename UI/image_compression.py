import numpy as np
import math
from PIL import Image
import time


def bitiVbyte(bits):
    polje = bytearray()
    byte = 0
    for index, bit in enumerate(bits):
        byte = (byte << 1) | bit
        if (index + 1) % 8 == 0:
            polje.append(byte)
            byte = 0
    if len(bits) % 8 != 0:
        byte = byte << (8 - len(bits) % 8)
        polje.append(byte)
    return polje


def IZ_BMP_V_2D(file_path):

    img = Image.open(file_path)
    img = img.convert('L')  # Converts to 8-bit grayscale (0-255)
    pixel_array = np.array(img)
    return pixel_array


def DATA_V_BMP(array, output_file_path):

    array = np.clip(array, 0, 255).astype(np.uint8)
    img = Image.fromarray(array, mode='L')
    img.save(output_file_path, format='BMP')




byte = 0
bit_stevec = 0
lavfar = []
index = 0

def PISI_V_BIN(f, bits: str):
    global byte, bit_stevec
    for bit in bits:
        byte = (byte << 1) | int(bit)  # Add the bit to the current byte
        bit_stevec += 1

        # If we have 8 bits, write the byte to the file
        if bit_stevec == 8:
            f.write(bytes([byte]))
            byte = 0
            bit_stevec = 0


def PREDVIDEVANJE(testna_slikica):
    X, Y = testna_slikica.shape
    predvidenje = np.zeros(X * Y, dtype=np.int16)

    for x in range(X):
        for y in range(Y):
            if x == 0 and y == 0:

                predvidenje[y * X + x] = int(testna_slikica[0][0])
            elif x == 0:

                predvidenje[y * X + x] = int(testna_slikica[0][y - 1]) - int(testna_slikica[0][y])
            elif y == 0:

                predvidenje[y * X + x] = int(testna_slikica[x - 1][0]) - int(testna_slikica[x][0])
            else:
                A = int(testna_slikica[x - 1][y])
                B = int(testna_slikica[x][y - 1])
                C = int(testna_slikica[x - 1][y - 1])

                if C >= max(A, B):
                    predvidenje[y * X + x] = min(A, B) - int(testna_slikica[x][y])
                elif C <= min(A, B):
                    predvidenje[y * X + x] = max(A, B) - int(testna_slikica[x][y])
                else:
                    predvidenje[y * X + x] = (A + B - C) - int(testna_slikica[x][y])

    return predvidenje


def NASTAVI_HEADER(f, X, C0, C_LAST, n):
    # appenda v file
    PISI_V_BIN(f, format(X, f'0{16}b'))
    PISI_V_BIN(f, format(C0, f'0{8}b'))
    PISI_V_BIN(f, format(C_LAST, f'0{32}b'))
    PISI_V_BIN(f, format(n, f'0{32}b'))

    return None


def IC(f, B, C, L, H):
    if (H - L) > 1:
        if C[H] != C[L]:
            m_val = math.floor(0.5 * (H + L))
            g_val = math.ceil(math.log2(C[H] - C[L] + 1))
            tmp = C[m_val] - C[L]

            PISI_V_BIN(f, format(tmp, f'0{g_val}b'))

            if L < m_val:
                IC(f, B, C, L, m_val)
            if m_val < H:
                IC(f, B, C, m_val, H)


def KOMPRESIJA(img, file):
    with open(file, 'wb') as f:
        X, Y = img.shape  # Extract dimensions
        E = np.array(PREDVIDEVANJE(img), dtype=np.int16)

        # print("to je E")
        # print(E)
        velikost_n = X * Y
        N = np.zeros(len(E), dtype=np.uint32)
        N[0] = E[0]

        for i in range(1, velikost_n):
            if E[i] >= 0:
                N[i] = 2 * E[i]
            else:
                N[i] = 2 * abs(E[i]) - 1
        # print("to je N")
        # print(N)
        C: np.ndarray = np.zeros(len(N), dtype=np.uint32)
        C[0] = N[0]
        for i in range(1, velikost_n):
            C[i] = C[i - 1] + N[i]
        # print("to je C")
        # print(C)
        B = NASTAVI_HEADER(f, X, C[0], C[velikost_n - 1], velikost_n)
        IC(f, B, C, 0, velikost_n - 1)

        global bit_stevec, byte
        if bit_stevec > 0:
            byte <<= (8 - bit_stevec);

            f.write(bytes([byte]))
            byte = 0;
            bit_stevec = 0;

    return B;


def BERI_IZ_BIN_FILE(file_path):
    with open(file_path, 'rb') as f:

        podatki = f.read()


    seznam = []
    for byte in podatki:

        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            seznam.append(bit)


    return np.array(seznam, dtype=np.uint8)


def DEIC( C, L, H):
    global lavfar, index
    if H - L > 1:
        if C[H] == C[L]:
            for i in range(L + 1, H):
                C[i] = C[L]
        else:
            m_val = math.floor(0.5 * (H + L))
            g_val = math.ceil(math.log2(C[H] - C[L] + 1))

            value = int(''.join(map(str, lavfar[index:g_val+index])), 2)
            C[m_val] = value + C[L]

            #del lavfar[:g]
            index +=g_val
            if L < m_val:
                DEIC( C, L, m_val)
            if m_val < H:
                DEIC( C, m_val, H)

    return C


def INIT_C(C0, C_last, size):
    C = [0] * size
    C[0] = C0
    C[-1] = C_last
    return C


def DEKODIRAJ_HEADER(B):
    visinaSlike = int(''.join(str(bit) for bit in B[:16]), 2)
    C0 = int(''.join(str(bit) for bit in B[16:24]), 2)
    C_LAST = int(''.join(str(bit) for bit in B[24:56]), 2)
    size = int(''.join(str(bit) for bit in B[56:88]), 2)

    B = B[88:]

    return visinaSlike, C0, C_LAST, size, B


def IVERZ_PREDVIDEVANJA(X, Y, E):
    P = np.zeros((X, Y), dtype=np.int16)

    for x in range(X):
        for y in range(Y):
            if x == 0 and y == 0:
                P[x][y] = E[y * X + x]
            elif x == 0:
                P[x][y] = P[0][y - 1] - E[y * X + x]
            elif y == 0:
                P[x][y] = P[x - 1][0] - E[y * X + x]
            else:
                A = P[x - 1][y]
                B = P[x][y - 1]
                C = P[x - 1][y - 1]

                if C >= max(A, B):
                    P[x][y] = min(A, B) - E[y * X + x]
                elif C <= min(A, B):
                    P[x][y] = max(A, B) - E[y * X + x]
                else:
                    P[x][y] = (A + B - C) - E[y * X + x]

    return P


def DEKOMPRESIRAJ(B):
    v, c0, clqst, size, B = DEKODIRAJ_HEADER(B)

    Y = int(size / v)

    C = INIT_C(c0, clqst, size)
    global lavfar
    lavfar = list(B)
    C = DEIC( C, 0, size - 1)

    N = [0] * len(C)
    N[0] = C[0]
    for i in range(1, size):
        N[i] = C[i] - C[i - 1]

    E = [0] * len(N)
    E[0] = N[0]
    for i in range(1, size):
        if N[i] % 2 == 0:
            E[i] = int(N[i] / 2)
        else:
            E[i] = int(-(N[i] + 1) / 2)

    P = IVERZ_PREDVIDEVANJA(v, Y, E)

    return v, Y, P

testna_slikica = np.array([
    [23, 21, 21, 23, 23],
    [24, 22, 22, 20, 24],
    [23, 22, 22, 19, 23],
    [26, 25, 21, 19, 22]
], dtype=np.int16)

if __name__ == '__main__':
    # Predict values
    ime_izhod = 'kompresirana_slika.bin'
    slika = "Maltese.bmp"
    slikazavn = "izhod\slikica.bmp"

    podatki_slike = IZ_BMP_V_2D(slika)
    start_time = time.time()
    KOMPRESIJA(podatki_slike, ime_izhod)
    end_time = time.time()  # Record the end time

    procesiranje = end_time - start_time
    print(f"Execution time: {procesiranje:.6f} seconds")

    cas1 = time.time()
    dekodiran = BERI_IZ_BIN_FILE(ime_izhod)

    v, c0, p = DEKOMPRESIRAJ(dekodiran)
    cas2 = time.time()  # Record the end time

    procesiranje2 = cas2 - cas1
    print(f"Execution time: {procesiranje2:.6f} seconds")
    DATA_V_BMP(p, slikazavn)
    print("konc")



