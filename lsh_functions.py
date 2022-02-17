import tensorflow as tf
import string
import random
import math


class HashModP(tf.Module):
    """Universal hash for integers via affine-mod-prime transformation."""
    def __init__(
            self,
            num_hashes,
            prime = 2**31 - 1,
            seed = None
            ):
        """Hash function constructor.

        Arguments:
            num_hashes: Integer. Number of hash values to compute for each
                input.
            prime: Integer. Large prime number used to compute the modulus.
            seed: Integer. Seed for repeatable hashing across class instances.
        """
        self._num_hashes = num_hashes
        self._prime = prime
        self._seed = seed
        initializer_a = tf.random_uniform_initializer(
            minval=1, maxval=self._prime - 1, seed=self._seed)
        initializer_b = tf.random_uniform_initializer(
            minval=0, maxval=self._prime - 1, seed=self._seed)
        self._a = tf.Variable(
            initializer_a(shape=[self._num_hashes], dtype=tf.int64))
        self._b = tf.Variable(
            initializer_b(shape=[self._num_hashes], dtype=tf.int64))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
    def hash(self, x):
        """Calculates hashes for input tensor x.

        Arguments:
            x: tf.Tensor of rank R, type tf.int64 and arbitrary shape
            [N0, N1, ... NR].

        Returns:
            A rank R+1 tf.Tensor of shape [N0, N1, ... NR, num_hashes] that
            contains tf.int64 hash codes.
        """
        affine = tf.tensordot(x, self._a, axes=0) + self._b
        return tf.math.floormod(affine, self._prime)

class HashBitMix(tf.Module):
    """Universal hash for integers via mixing with bitwise operations."""
    def __init__(
            self, 
            num_hashes,
            seed = None
            ):
        """Hash function constructor.

        Arguments:
            num_hashes: Integer. Number of hash values to compute for each
                input.
            seed: Integer. Seed for repeatable hashing across class instances.
        """
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1 but is %s" % num_hashes)

        self._num_hashes = num_hashes
        self._seed = seed
        self._c1 = tf.convert_to_tensor(0xbf58476d1ce4e5b9, dtype=tf.uint64)
        self._c2 = tf.convert_to_tensor(0x94d049bb133111eb, dtype=tf.uint64)
        initializer_xor = tf.random_uniform_initializer(
            minval=1, maxval=2**31 - 1, seed=self._seed)
        self._xor_constants = tf.Variable(
            initializer_xor(shape=[self._num_hashes], dtype=tf.int64))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
    def hash(self, x):
        """Calculates hashes for input tensor x.

        Arguments:
            x: tf.Tensor of rank R, type tf.int64 and arbitrary shape
            [N0, N1, ... NR].

        Returns:
            A rank R+1 tf.Tensor of shape [N0, N1, ... NR, num_hashes] that
            contains tf.int64 hash codes.
        """
        x = tf.stack(
            [tf.bitwise.bitwise_xor(x, self._xor_constants[i])
            for i in range(self._num_hashes)], axis=-1)
        # Use uint64 mixer (constants found via annealing by David Stafford).
        x = tf.bitcast(x, tf.uint64)  # Does not copy data.
        x = tf.bitwise.bitwise_xor(x, tf.bitwise.right_shift(x, 30))
        x = x * self._c1
        x = tf.bitwise.bitwise_xor(x, tf.bitwise.right_shift(x, 27))
        x = x * self._c2
        x = tf.bitwise.bitwise_xor(x, tf.bitwise.right_shift(x, 31))
        return tf.bitcast(x, tf.int64)


class IntMinHash(tf.Module):
    """Minhash for sequences of integer tokens."""
    def __init__(
            self,
            num_hashes,
            seed = None
            ):
        """Hash function constructor.

        Arguments:
            num_hashes: Integer. Number of minhash values to compute.
            seed: Integer. Seed for repeatable hashing across class instances.
        """
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1 but is %s" % num_hashes)

        self._num_hashes = num_hashes
        self._seed = seed
        self._universal_hash = HashBitMix(self._num_hashes, seed=self._seed)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
    def hash(self, x):
        """Calculates minhashes for input tensor x.

        Arguments:
            x: tf.Tensor of type tf.int64 and shape [batch_size, dimensions]
            or shape [dimensions].

        Returns:
            tf.Tensor of [batch_size, num_hashes] or [num_hashes] hash codes
            of type tf.int64. The shape depends on the input shape.
        """
        hashes = self._universal_hash.hash(x)
        return tf.math.reduce_min(hashes, axis=-2)


class StringMinHash(tf.Module):
    """Minhash for sequences of string tokens."""
    def __init__(
            self,
            num_hashes,
            num_buckets = 2**32-1,
            salt_len = 4,
            seed = None
            ):
        """Hash function constructor.

        Arguments:
            num_hashes: Integer. Number of minhash values to compute.
            num_buckets: Integer. Number of possible hash values for the
                internal universal hash that is used to compute the minhashes.
            salt_len: Integer. Size of random salt appended to each input.
                Note: the number of unique minhash functions is salt_len**100.
                We randomly select functions, so we suggest salt_len >= 4.
            seed: Integer. Seed for repeatable hashing across class instances.
        """
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1 but is %s" % num_hashes)
        if num_buckets < 2:
            raise ValueError("num_buckets must be >= 2 "
                             "but is %s" % num_buckets)
        if salt_len < 1:
            raise ValueError("salt_len must be >= 1 but is %s" % salt_len)

        self._num_hashes = num_hashes
        self._num_buckets = num_buckets
        self._seed = seed
        # Initialize salts with default Python generator, saving and restoring
        # the state to avoid side effects for other users of the global RNG.
        if seed:
            state = random.getstate()
            random.seed(seed)
        self._salts = [''.join(random.choice(string.printable)
                       for _ in range(salt_len))
                       for _ in range(num_hashes)]
        if seed:
            random.setstate(state)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
    def hash(self, x):
        """Calculates minhashes for input tensor x.

        Arguments:
            x: tf.Tensor of type tf.string and shape [batch_size, dimensions]
            or shape [dimensions].

        Returns:
            tf.Tensor of [batch_size, num_hashes] or [num_hashes] hash codes
            of type tf.int64. The shape depends on the input shape.
        """
        salted_strings = [x + s for s in self._salts]
        hashed_strings = [tf.strings.to_hash_bucket_fast(s, self._num_buckets)
                          for s in salted_strings]
        min_hashes = [tf.math.reduce_min(h, axis=-1) for h in hashed_strings]
        return tf.stack(min_hashes, axis=-1)


class PStableHash(tf.Module):
    """Locality-sensitive hash for vectors in Lp-norm spaces."""
    def __init__(
            self,
            dimension,
            num_hashes,
            scale = 1.0,
            p = 2.0,
            seed = None
            ):
        """Hash function constructor.

        Arguments:
            dimension: Integer. Size of each input to the hash function.
            num_hashes: Integer. Number of hash values to compute per input.
            scale: Float. Scale factor for the metric space. Larger values of
                scale will result in larger hash partitions (and thus,
                collisions between vectors that are further away).
            p: Float in range [0.4, 2.0]. Value of p for the Lp norm space.
                For example, p = 1.0 and 2.0 produce LSH for the Manhattan and
                Euclidean distances, respectively.
            seed: Integer. Seed for repeatable hashing across class instances.
        """
        if p < 0.4 or p > 2.0:
            raise ValueError("p must be in [0.4, 2.0] but is %s" % p)
        if dimension < 1:
            raise ValueError("dimension must be >= 1 but is %s" % dimension)
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1 but is %s" % num_hashes)
        if scale <= 0:
            raise ValueError("scale must be > 0 but is %s" % scale)

        self._dim = dimension
        self._num_hashes = num_hashes
        self._p = p
        self._seed = seed
        self._scale = scale
        self._projections = self._init_projections()
        self._bias = self._init_bias()

    def _init_projections(self):
        # Only runs once during initialization - should not be a tf.function.
        if self._p == 2.0:  # Special case: Gaussian-distributed projection
            initializer = tf.random_normal_initializer(
                mean=0.0, stddev=1.0, seed=self._seed)
            return tf.Variable(
                initializer(shape=[self._dim, self._num_hashes]))
        elif self._p == 1.0:  # Special case: Cauchy-distributed projection
            initializer = tf.random_uniform_initializer(
                minval=0, maxval=1, seed=self._seed)
            x = tf.Variable(initializer(shape=[self._dim, self._num_hashes]))
            return tf.math.tan(math.pi * (x - 0.5))
        else:
            # Uses method from http://dimacs.rutgers.edu/~graham/code.html.
            # The initialization bounds are necessary to prevent overflow.
            initializer_0 = tf.random_uniform_initializer(
                minval=1e-8, maxval=1-1e-8, seed=self._seed)
            seed_1 = self._seed + 1 if self._seed else None
            initializer_1 = tf.random_uniform_initializer(
                minval=1e-6, maxval=1-1e-6, seed=seed_1)
            x0 = tf.Variable(
                initializer_0(shape=[self._dim, self._num_hashes]))
            x1 = tf.Variable(
                initializer_1(shape=[self._dim, self._num_hashes]))
            theta = math.pi * (x0 - 0.5)
            w = -1.0 * tf.math.log(x1)
            left_denom = tf.math.pow(tf.math.cos(theta), 1.0 / self._p)
            left = tf.math.sin(self._p * theta) / left_denom
            right_base = tf.math.divide(tf.math.cos(theta*(1.0 - self._p)), w)
            right = tf.math.pow(right_base, (1.0 - self._p) / self._p)
            return tf.math.multiply(left, right)

    def _init_bias(self):
        initializer = tf.random_uniform_initializer(
            minval=0.0, maxval=self._scale, seed=self._seed)
        return tf.Variable(initializer(shape=[self._num_hashes]))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def hash(self, x):
        """Calculates hash codes for input x.

        Arguments:
            x: tf.Tensor of type tf.float32 and shape [batch_size, dimensions]
            or shape [dimensions].

        Returns:
            tf.Tensor of [batch_size, num_hashes] or [num_hashes] hash codes
            of type tf.int64. The shape depends on the input shape.
        """
        projs = tf.tensordot(x, self._projections, axes=((-1), (0)))
        # proj is [batch_size, num_hashes], use broadcasting to add bias.
        affine_projs = (projs + self._bias) / self._scale
        rounded_projs = tf.math.floor(affine_projs)
        return tf.cast(rounded_projs, tf.int64)


class SRPHash(tf.Module):
    """signed random projection hash for the angular distance metric."""
    def __init__(
            self,
            dimension,
            num_hashes,
            num_bits = 1,
            seed = None
            ):
        """Hash function constructor.

        Arguments:
            dimension: Integer. Size of inputs to the hash function.
            num_hashes: Integer. Number of random projections to compute.
            num_bits: Integer. Number of bits to use in each hash function.
                The hash will output integer values in [0, 2**num_bits - 1].
            seed: Integer. Seed for repeatable hashing across class instances.
        """
        if dimension < 1:
            raise ValueError("dimension must be >= 1 but is %s" % dimension)
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1 but is %s" % num_hashes)
        if num_bits < 1:
            raise ValueError("num_bits must be >= 1 but is %s" % num_bits)

        self._dim = dimension
        self._num_hashes = num_hashes
        self._num_bits = num_bits
        self._seed = seed
        initializer = tf.random_normal_initializer(seed=self._seed)
        self._projections = tf.Variable(
            initializer(shape=[self._dim, self._num_hashes, self._num_bits]))
        self._powers_of_two = tf.Variable(
            [2**i for i in range(self._num_bits)], dtype=float)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def hash(self, x):
        """Calculates hash codes for input x.

        Arguments:
            x: tf.Tensor of type tf.string and shape [batch_size, dimensions]
            or shape [dimensions].

        Returns:
            tf.Tensor of [batch_size, num_hashes] or [num_hashes] hash codes
            of type tf.int64. Hash codes are in range [0, 2**num_bits - 1].
        """
        # x.shape: [batch_size, dimensions]
        # self._projections.shape: [dimensions, num_hashes, num_bits]
        projs = tf.tensordot(x, self._projections, axes=((-1), (0)))
        # projs.shape: [batch_size, num_hashes, num_bits]
        signs = tf.clip_by_value(tf.math.sign(projs), 0.0, 1.0)
        # signs.shape: [batch_size, num_hashes, num_bits]
        hash_vals = tf.tensordot(self._powers_of_two, signs, axes=((0), (-1)))
        return tf.cast(hash_vals, tf.int64)

