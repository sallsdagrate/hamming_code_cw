"""
This file is part of Lab 4 (Hamming Codes), assessed coursework for the module
COMP70103 Statistical Information Theory. 

You should submit an updated version of this file, replacing the
NotImplementedError's with the correct implementation of each function. Do not
edit any other functions.

Follow further instructions in the attached .pdf and .ipynb files, available
through Scientia.
"""
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import numpy.random as rn
from itertools import product

alphabet = "abcdefghijklmnopqrstuvwxyz01234567890 .,\n"
digits = "0123456789"


def char2bits(char: chr) -> np.array:
    '''
    Given a character in the alphabet, returns a 8-bit numpy array of 0,1 which represents it
    '''
    num = ord(char)
    if num >= 256:
        raise ValueError("Character not recognised.")
    bits = format(num, '#010b')
    bits = [int(b) for b in bits[2:]]
    return np.array(bits)


def bits2char(bits) -> chr:
    '''
    Given a 7-bit numpy array of 0,1 or bool, returns the character it encodes
    '''
    bits = ''.join(bits.astype(int).astype(str))
    num = int(bits, base=2)
    return chr(num)


def text2bits(text: str) -> np.ndarray:
    '''
    Given a string, returns a numpy array of bool with the binary encoding of the text
    '''
    text = text.lower()
    text = [t for t in text if t in alphabet]
    bits = [char2bits(c) for c in text]
    return np.array(bits, dtype=bool).flatten()


def bits2text(bits: np.ndarray) -> str:
    '''
    Given a numpy array of bool or 0 and 1 (as int) which represents the
    binary encoding a text, return the text as a string
    '''
    if np.mod(len(bits), 8) != 0:
        raise ValueError("The length of the bit string must be a multiple of 8.")
    bits = bits.reshape(int(len(bits) / 8), 8)
    chrs = [bits2char(b) for b in bits]
    return ''.join(chrs)


def parity_matrix(m: int) -> np.ndarray:
    """
    m : int
      The number of parity bits to use

    return : np.ndarray
      m-by-n parity check matrix
    """

    n = 2 ** m - 1
    # f"{i:0{m}b}" means convert i to a binary number of length m with padded 0s if needed
    return np.array([np.array(list(f"{i:0{m}b}"[::-1]), dtype=int) for i in range(1, n + 1)]).T


def hamming_generator(m: int) -> np.ndarray:
    """
    m : int
      The number of parity bits to use

    return : np.ndarray
      k-by-n generator matrix
    """
    n = get_n(m)
    k = n - m
    Gt = np.zeros((n, k))

    data_indices, parity_indices = get_data_and_parity_indices(m)

    Gt[data_indices, :] = np.identity(k)
    Gt[parity_indices, :] = parity_matrix(m)[:, data_indices]

    return Gt.T


def hamming_encode(data: np.ndarray, m: int) -> np.ndarray:
    """
    data : np.ndarray
      array of shape (k,) with the block of bits to encode

    m : int
      The number of parity bits to use

    return : np.ndarray
      array of shape (n,) with the corresponding Hamming codeword
    """
    assert (data.shape[0] == 2 ** m - m - 1)

    return ((hamming_generator(m).T @ data) % 2).astype(int).astype(bool)


def hamming_decode(code: np.ndarray, m: int) -> np.ndarray:
    """
    code : np.ndarray
      Array of shape (n,) containing a Hamming codeword computed with m parity bits
    m : int
      Number of parity bits used when encoding

    return : np.ndarray
      Array of shape (k,) with the decoded and corrected data
    """
    assert (np.log2(len(code) + 1) == int(np.log2(len(code) + 1)) == m)
    z = ((parity_matrix(m) @ code) % 2).astype(int)
    flipped_bit = int("".join(np.flip(z.astype(str), 0)), 2) - 1
    if flipped_bit >= 0:
        code[flipped_bit] = 1 - code[flipped_bit]
    data_indices, _ = get_data_and_parity_indices(m)
    return code[data_indices]


def decode_secret(msg: np.ndarray) -> str:
    """
    msg : np.ndarray
      One-dimensional array of binary integers

    return : str
      String with decoded text
    """
    m = 4  # <-- Your guess goes here
    blocks = msg.copy().reshape(-1, get_n(m))
    decoded = np.array([hamming_decode(c, m) for c in blocks]).flatten()
    return bits2text(decoded)


def binary_symmetric_channel(data: np.ndarray, p: float) -> np.ndarray:
    """
    data : np.ndarray
      1-dimensional array containing a stream of bits
    p : float
      probability by which each bit is flipped

    return : np.ndarray
      data with a number of bits flipped
    """

    raise NotImplementedError


def decoder_accuracy(m: int, p: float) -> float:
    """
    m : int
      The number of parity bits in the Hamming code being tested
    p : float
      The probability of each bit being flipped

    return : float
      The probability of messages being correctly decoded with this
      Hamming code, using the noisy channel of probability p
    """

    raise NotImplementedError


def get_n(m):
    return 2 ** m - 1


def get_data_and_parity_indices(m):
    n = get_n(m)
    parity_indices = [2 ** i - 1 for i in range(m)]
    data_indices = [i for i in range(n) if i not in parity_indices]
    return data_indices, parity_indices
