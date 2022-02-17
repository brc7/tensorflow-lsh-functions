import tensorflow as tf
import numpy as np
from lsh_functions import *

class StringMinhashTest(tf.test.TestCase):

    def testCollisionProbability(self):
        x = tf.convert_to_tensor(['a', 'b', 'c'])
        y = tf.convert_to_tensor(['a', 'b', 'd'])
        expected_p = 0.5
        num_hashes = 1000
        lsh = StringMinHash(num_hashes)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testBatchInputs(self):
        x = tf.convert_to_tensor(['a', 'b', 'c'])
        y = tf.convert_to_tensor(
            [['a', 'b', 'c'],
             ['a', 'b', 'd'],
             ['e', 'f', 'g']])
        num_hashes = 10
        lsh = StringMinHash(num_hashes)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        self.assertAllEqual(x_h, y_h[0,:])

    def testDuplicates(self):
        x = tf.convert_to_tensor(['a', 'b', 'c', 'c', 'b', 'a'])
        y = tf.convert_to_tensor(['a', 'b', 'c'])
        num_hashes = 10
        lsh = StringMinHash(num_hashes)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        self.assertAllEqual(x_h, y_h)

    def testRepeatableHashing(self):
        x = tf.convert_to_tensor(
            [['a', 'b', 'c', 'd', 'e', 'f'],
             ['g', 'h', 'i', 'j', 'k', 'l'],
             ['m', 'n', 'o', 'p', 'q', 'r'],
             ['s', 't', 'u', 'v', 'w', 'x']])
        num_hashes = 10
        lsh_1 = StringMinHash(num_hashes, seed=4234)
        lsh_2 = StringMinHash(num_hashes, seed=4234)
        x_h1 = lsh_1.hash(x)
        x_h2 = lsh_2.hash(x)
        self.assertAllEqual(x_h1, x_h2)



class IntMinhashTest(tf.test.TestCase):

    def testCollisionProbability(self):
        x = tf.convert_to_tensor([1, 2, 3], dtype=tf.int64)
        y = tf.convert_to_tensor([1, 2, 4], dtype=tf.int64)
        expected_p = 0.5
        num_hashes = 1000
        lsh = IntMinHash(num_hashes)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testBatchInputs(self):
        x = tf.convert_to_tensor([1, 2, 3], dtype=tf.int64)
        y = tf.convert_to_tensor(
            [[1, 2, 3],
             [1, 2, 4],
             [5, 6, 7]], dtype=tf.int64)
        num_hashes = 10
        lsh = IntMinHash(num_hashes)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        self.assertAllEqual(x_h, y_h[0,:])

    def testDuplicates(self):
        x = tf.convert_to_tensor([1, 2, 3, 3, 2, 1], dtype=tf.int64)
        y = tf.convert_to_tensor([1, 2, 3], dtype=tf.int64)
        num_hashes = 10
        lsh = IntMinHash(num_hashes)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        self.assertAllEqual(x_h, y_h)

    def testRepeatableHashing(self):
        x = tf.convert_to_tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8],
             [9, 10, 11, 12, 13, 14, 15, 16],
             [17, 18, 19, 20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29, 30, 31, 32]], dtype=tf.int64)
        num_hashes = 10
        lsh_1 = IntMinHash(num_hashes, seed=1234)
        lsh_2 = IntMinHash(num_hashes, seed=1234)
        x_h1 = lsh_1.hash(x)
        x_h2 = lsh_2.hash(x)
        self.assertAllEqual(x_h1, x_h2)


class PStableHashTest(tf.test.TestCase):

    def testCollisionProbabilityL1(self):
        x = tf.convert_to_tensor([1.0, 0.5, 1.0, 1.0, 0.5])
        y = tf.convert_to_tensor([1.0, 0.0, 0.0, 0.5, -0.5])
        dim = 5
        num_hashes = 1000
        # Expected collision probability for the L1 LSH function is:
        # 2 * atan(scale / dist)/pi - dist/(scale*pi)*ln(1 + (scale/dist)**2)
        expected_p = 0.1042
        lsh = PStableHash(dim, num_hashes, p = 1.0, scale = 1.0)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testCollisionProbabilityL2(self):
        x = tf.convert_to_tensor([1.0, 0.5, 1.0, 1.0, 0.5])
        y = tf.convert_to_tensor([1.0, 0.0, 0.0, 0.5, -0.5])
        dim = 5
        num_hashes = 1000
        # Expected collision probability for the L2 LSH function is:
        # 1 - 2*phi(-scale/dist)
        # - 2.0/(sqrt(2*pi)*(scale/dist)) * (1 - exp(-0.5*(scale/dist)**2)))
        # where phi is the CDF of the standard normal distribution.
        expected_p =  0.2442
        lsh = PStableHash(dim, num_hashes, p = 2.0, scale = 1.0)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testCollisionProbabilityPStable(self):
        # It is hard to calculate the exact collision probability for
        # p-stable LSH, so instead we simply verify the locality-sensitive
        # property.
        p = 0.4
        x = tf.convert_to_tensor([1.0, 0.5, 1.0, 1.0, 0.5])
        y = tf.convert_to_tensor([1.0, 0.0, 0.0, 0.5, -0.5])
        z = tf.convert_to_tensor([100.0, -100.0, 50.0, 1.0, 0.5])
        dim = 5
        num_hashes = 1000
        lsh = PStableHash(dim, num_hashes, p = p, scale = 1.0)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        z_h = lsh.hash(z)
        p_xy = np.count_nonzero(x_h == y_h) / num_hashes
        p_xz = np.count_nonzero(x_h == z_h) / num_hashes
        self.assertGreater(p_xy, p_xz)

    def testBatchInputs(self):
        p = 1.2
        x = tf.convert_to_tensor([1.5, 0.0, -1.5])
        y = tf.convert_to_tensor(
            [[1.5, 0.0, -1.5],
             [-1.5, 0.0, 1.5],
             [1.2, 0.5, 2.0]])
        dim = 3
        num_hashes = 10
        lsh = PStableHash(dim, num_hashes, p = p, scale = 1.0)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        self.assertAllEqual(x_h, y_h[0,:])


class SRPHashTest(tf.test.TestCase):

    def testCollisionProbability1Bit(self):
        # x and y are orthogonal
        x = tf.convert_to_tensor([1.0, 0.1, 0.0, -1.0, -0.5])
        y = tf.convert_to_tensor([1.0, 0.5, 1.0, 1.0, 0.1])
        dim = 5
        num_hashes = 1000
        # Expected collision probability for the SRP hash function is:
        # 1 - 1/pi * angle(x, y)
        expected_p = 0.5
        lsh = SRPHash(dim, num_hashes, num_bits=1)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testCollisionProbability2Bit(self):
        # x and y are orthogonal
        x = tf.convert_to_tensor([1.0, 0.1, 0.0, -1.0, -0.5])
        y = tf.convert_to_tensor([1.0, 0.5, 1.0, 1.0, 0.1])
        dim = 5
        num_hashes = 1000
        expected_p = 0.25
        lsh = SRPHash(dim, num_hashes, num_bits=2)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testCollisionProbability8Bit(self):
        # Test with a really large number of bits.
        x = tf.convert_to_tensor([1.0, 0.6, 0.9, 1.2, -0.2, 2.1, 1.0, -1.3])
        y = tf.convert_to_tensor([1.0, 0.5, 1.0, 1.0, 0.1, 2.0, 1.0, -1.25])
        dim = 8
        num_hashes = 1000
        # x and y have collision probability 0.962615 per bit
        expected_p = 0.7373
        lsh = SRPHash(dim, num_hashes, num_bits=8)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        empirical_p = np.count_nonzero(x_h == y_h) / num_hashes
        self.assertAlmostEqual(empirical_p, expected_p, delta = 0.05)

    def testBatchInputs(self):
        x = tf.convert_to_tensor([1.0, -1.0, 0.5])
        y = tf.convert_to_tensor(
            [[1.0, -1.0, 0.5],
             [0.0, 0.2, -0.5],
             [1.0, 1.0, -1.0],
             [2.0, -1.0, 2.0],
             [0.0, 0.0, 0.0]])
        dim = 3
        num_hashes = 10
        lsh = SRPHash(dim, num_hashes, num_bits = 32)
        x_h = lsh.hash(x)
        y_h = lsh.hash(y)
        self.assertAllEqual(x_h, y_h[0,:])


if __name__ == '__main__':
    tf.test.main()