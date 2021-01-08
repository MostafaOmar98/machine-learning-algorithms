from __future__ import annotations

from typing import List


class Node:
    featureIndex: int
    featureName: str
    children: List[Node]
    result = None

    def __int__(self, *argv):
        self.setChildren()

    def setChildren(self, *args):
        for arg in args:
            self.children.append(arg)
