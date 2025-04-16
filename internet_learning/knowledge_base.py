# internet_learning/knowledge_base.py
# Simple in-memory knowledge base
class KnowledgeBase:
    def __init__(self):
        self.knowledge = []
    def add(self, item):
        self.knowledge.append(item)
    def query(self, keyword):
        return [k for k in self.knowledge if keyword in str(k)]
