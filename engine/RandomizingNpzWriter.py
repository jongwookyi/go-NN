#!/usr/bin/python
import numpy as np
import random
import os

class RandomizingNpzWriter:
    def __init__(self, out_dir, names, shapes, dtypes, Nperfile, buffer_len):
        assert buffer_len >= Nperfile
        assert len(names) == len(shapes) == len(dtypes)
        self.out_dir = out_dir
        self.names = names
        self.shapes = shapes
        self.dtypes = dtypes
        self.Nperfile = Nperfile
        self.buffer_len = buffer_len
        self.examples = []
        self.filenum = 0

    def push_example(self, example):
        print "pushing (before push, len(examples) = %d)" % len(self.examples)
        assert len(example) == len(self.names)
        for i in xrange(len(example)):
            assert example[i].dtype == self.dtypes[i]
        self.examples.append(example)
        if len(self.examples) >= self.buffer_len:
            self.write_npz_file()

    def drain(self):
        print "draining..."
        while len(self.examples) >= self.Nperfile:
            self.write_npz_file()
        print "finished draining. %d examples left unwritten." % len(self.examples)

    def write_npz_file(self):
        assert len(self.examples) >= self.Nperfile

        print "write_npz_file: len(self.examples) =", len(self.examples)

        # put Nperfile random examples at the end of the list
        for i in xrange(self.Nperfile):
            a = len(self.examples) - i - 1
            if a > 0:
              b = random.randint(0, a-1)
              self.examples[a], self.examples[b] = self.examples[b], self.examples[a]

        # pop Nperfile examples off the end of the list
        # put each component into a separate numpy batch array
        save_dict = {}
        for c in xrange(len(self.names)):
            batch_shape = (self.Nperfile,) + self.shapes[c]
            batch = np.empty(batch_shape, dtype=self.dtypes[c])
            for i in xrange(self.Nperfile):
                batch[i,:] = self.examples[-1-i][c]
            save_dict[self.names[c]] = batch

        del self.examples[-self.Nperfile:]

        filename = os.path.join(self.out_dir, "examples.%d.%d" % (self.Nperfile, self.filenum))
        print "writing", filename
        np.savez_compressed(filename, **save_dict)
        self.filenum += 1


if __name__ == '__main__':
    writer = RandomizingNpzWriter('/tmp/npz_writer',
            names=['some_ints', 'some_floats'],
            shapes=[(2,2), (2,)],
            dtypes=[np.int32, np.float32],
            Nperfile=2, buffer_len=4)

    writer.push_example((1*np.ones((2,2),dtype=np.int32), 1*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((2*np.ones((2,2),dtype=np.int32), 2*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((3*np.ones((2,2),dtype=np.int32), 3*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((4*np.ones((2,2),dtype=np.int32), 4*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((5*np.ones((2,2),dtype=np.int32), 5*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((6*np.ones((2,2),dtype=np.int32), 6*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((7*np.ones((2,2),dtype=np.int32), 7*np.array([1.0, 1.0], dtype=np.float32)))
    writer.push_example((8*np.ones((2,2),dtype=np.int32), 8*np.array([1.0, 1.0], dtype=np.float32)))
    writer.drain()






