# -*- coding: utf-8 -*-
#!/usr/bin/env python


class Diff(object):
    def __init__(self, v=0.):
        self.value = v
        self.diff = 0.

    def update(self, v):
        if self.value:
            diff = (v / self.value - 1)
            self.diff = diff
        self.value = v


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])