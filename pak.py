# Copyright (c) 2018 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

__all__ = (
    'Filesystem',
    'MalformedPakFile',
)


import collections
import collections.abc
import glob
import logging
import os
import struct


_PakEntry = collections.namedtuple('_PakEntry', ('pak_file', 'offset', 'size'))


class MalformedPakFile(Exception):
    pass


class Filesystem(collections.abc.Mapping):
    """Interface to a .pak file based filesystem."""

    def _read(self, f, n):
        b = f.read(n)
        if len(b) < n:
            raise MalformedPakFile("File ended unexpectedly")
        return b

    def _read_fname(self, f):
        fname = self._read(f, 56).decode('ascii')
        if '\0' in fname:
            fname = fname[:fname.index('\0')]
        return fname

    def _read_header(self, f):
        try:
            magic = self._read(f, 4)
            if magic != b"PACK":
                raise MalformedPakFile("Invalid magic number")
            return struct.unpack("<II", self._read(f, 8))
        except EOFError:
            raise MalformedPakFile("File too short")

    def _generate_entries(self, pak_file):
        with open(pak_file, "rb") as f:
            logging.info("Reading %s", pak_file)
            file_table_offset, file_table_size = self._read_header(f)
            f.seek(file_table_offset)
            i = 0
            while i < file_table_size:
                fname = self._read_fname(f)
                logging.debug("Indexed %s", fname)
                offset, size = struct.unpack("<II", self._read(f, 8))
                yield fname, _PakEntry(pak_file, offset, size)
                i += 64

    def __init__(self, pak_dir):
        pak_files = sorted(glob.glob(os.path.join(pak_dir, "*.pak")))
        self._index = {fname: entry for pak_file in pak_files for fname, entry in self._generate_entries(pak_file)}

    def __getitem__(self, fname):
        entry = self._index[fname]
        with open(entry.pak_file, "rb") as f:
            f.seek(entry.offset)
            return self._read(f, entry.size)

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)


if __name__ == "__main__":
    import sys

    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(logging.DEBUG)

    fs = Filesystem(sys.argv[1])

