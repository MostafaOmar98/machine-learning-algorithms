from __future__ import annotations

from typing import List


class Node:
    def __int__(self, *argv):
        self.setChildren(argv)
        self.result = None
        self.featureIndex = None

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

    def getHeight(self,currentNode: Node, currentHeight):
        if len(currentNode.children) == 0:
            return currentHeight
        mx = 0
        for child in currentNode.children:
            mx = max(mx, self.getHeight(child, currentHeight + 1))
        return mx
    def getSize(self,currentNode: Node):
        if len(currentNode.children) == 0:
            return 0
        ret = 0
        for child in currentNode.children:
            ret = ret + 1 + self.getSize(child)
        return ret