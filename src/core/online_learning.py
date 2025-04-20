"""
online_learning.py

Provides OnlineLearnerMixin for agents to learn from web resources.
"""



class OnlineLearnerMixin:
    """
    Mixin for enabling agents to learn from web resources with flexible parsing and robust integration.
    """

    # Registry for parsing functions by content type or file extension
    parsing_registry = {
        "application/json": lambda data: __import__("json").loads(data),
        "text/csv": lambda data: [line.split(",") for line in data.strip().split("\n")],
        "text/plain": lambda data: data,
        "text/html": lambda data: data,  # Placeholder: can use BeautifulSoup if needed
    }

    def learn_from_url(self, url, parse_fn=None, content_type=None):
        """
        Fetch data from a URL and update agent knowledge.
        Auto-select or use a provided parsing function.
        """
        import requests

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.text
            # Auto-select parsing function if not provided
            if parse_fn is None:
                ctype = (
                    content_type
                    or response.headers.get("Content-Type", "").split(";")[0].strip()
                )
                parse_fn = self.parsing_registry.get(ctype, lambda x: x)
            parsed = parse_fn(data)
            self.integrate_online_knowledge(parsed)
        except Exception as e:
            print(f"[OnlineLearnerMixin] Error learning from {url}: {e}")

    def integrate_online_knowledge(self, knowledge):
        """
        Integrate fetched knowledge into the agent.
        Override for custom behavior. Should validate knowledge against laws if applicable.
        """
        # Example: store as attribute
        self.online_knowledge = knowledge
        # Optionally: validate or update model/rules
