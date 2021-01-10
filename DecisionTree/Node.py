from __future__ import annotations

from typing import List


class Node:
    def __int__(self, *argv):
        self.setChildren(argv)
        self.result = None
feature        self.featureIndex = None

    def __init__(self, featureName):
        self.featureName = featureName
        self.result = None
        self.children = []
        self.featureIndex = None

    def setChildren(self, *args):
        self.children = []
        for arg in args:
            self.children.append(arg)
    def addToChildren(self, child: Node):
        self.children.append(child)