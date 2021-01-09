from __future__ import annotations

from typing import List


class Node:
    featureIndex: int
    featureName: str
    children: List[Node]
    result = None

    def __int__(self, *argv):
        self.setChildren(argv)

    def __init__(self, featureName):
        self.featureName = featureName
        self.children = []

    def setChildren(self, *args):
        self.children = []
        for arg in args:
            self.children.append(arg)
    def addToChildren(self, child: Node):
        self.children.append(child)