# main.py
from core.agent import Agent
from core.rules import GlobalRules, never_delete_data
from multiagent.collaboration import Collaboration
from internet_learning.web_search import WebSearch
from internet_learning.video_learning import VideoLearning
from internet_learning.knowledge_base import KnowledgeBase

def main():
    # Define global rules
    rules = GlobalRules([never_delete_data])
    # Create agents
    agent1 = Agent("Agent1", rules)
    agent2 = Agent("Agent2", rules)
    agents = [agent1, agent2]
    # Collaboration
    collab = Collaboration(agents)
    # Knowledge base
    kb = KnowledgeBase()
    # Internet learning
    web = WebSearch()
    video = VideoLearning()
    # Example usage
    web.search("neuro-fuzzy systems")
    video.learn_from_video("https://youtube.com/example")
    kb.add({"topic": "neuro-fuzzy", "content": "example"})
    print(kb.query("neuro"))

if __name__ == "__main__":
    main()
