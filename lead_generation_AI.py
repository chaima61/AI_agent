import os
import json
import time
from typing import List, Dict, Any, Optional
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import requests

# Loading environment variables
load_dotenv()

class LeadGenerationAgent:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the Lead Generation Agent with API keys."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set it as an environment variable or pass it directly.")
            
        # Initializing the LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=self.openai_api_key
        )
        
        # Setting up agent with tools
        self.setup_agent()
        
    def setup_agent(self):
        """Set up the agent with necessary tools using the newer LangChain format."""
        tools = [
            Tool(
                name="search_twitter",
                func=self.search_twitter,
                description="Search for Twitter accounts related to a keyword or topic."
            ),
            Tool(
                name="search_instagram",
                func=self.search_instagram,
                description="Search for Instagram accounts related to a keyword or topic."
            ),
            Tool(
                name="analyze_social_profile",
                func=self.analyze_social_profile,
                description="Analyze a social media profile to determine if they're a potential influencer."
            )
        ]
        
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Lead Generation AI specialized in discovering relevant influencers for brands.
            
            Your goal is to find potential influencers that match a given brand keyword or niche.
            Follow these steps:
            1. Search for relevant accounts on Twitter and Instagram
            2. Analyze profiles to determine if they're suitable influencers
            3. Compile a list of the best matches with relevant data
            
            Present your final results in a clear, organized format with the most promising influencers first."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
            }
            | prompt
            | self.llm.bind_tools(tools)
            | OpenAIFunctionsAgentOutputParser()
        )
        
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def search_twitter(self, query: str) -> str:
        """
        Search for Twitter accounts related to a keyword.
        In a real implementation this would use the Twitter API.
        """
        print(f"Searching Twitter for: {query}")
        
        try:
            # Attempt to use Twitter API if you have keys set up
            # This is a placeholder for where actual API calls would go
            # For now we'll use simulated data
            
            # Simulated response - in production replace with actual API call
            simulated_results = self._simulate_twitter_results(query)
            return json.dumps(simulated_results)
            
        except Exception as e:
            print(f"Error with Twitter search: {e}")
            return json.dumps({"error": str(e)})
    
    def search_instagram(self, query: str) -> str:
        """
        Search for Instagram accounts related to a keyword.
        In a real implementation this would use the Instagram API.
        """
        print(f"Searching Instagram for: {query}")
        
        try:
            # Attempt to use Instagram API if you have keys set up
            # This is a placeholder for where actual API calls would go
            # For now we'll use simulated data
            
            # Simulated response - in production replace with actual API call
            simulated_results = self._simulate_instagram_results(query)
            return json.dumps(simulated_results)
            
        except Exception as e:
            print(f"Error with Instagram search: {e}")
            return json.dumps({"error": str(e)})
    
    def analyze_social_profile(self, profile_data: str) -> str:
        """
        Analyze a social profile to determine if they're a good influencer match.
        Uses LLM to evaluate profile data and determine relevance.
        """
        try:
            profile = json.loads(profile_data) if isinstance(profile_data, str) else profile_data
            
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze this social media profile to determine if they would be a good influencer for marketing purposes.
                
                Please evaluate:
                1. Relevance to their niche
                2. Audience size and engagement rates
                3. Content quality and consistency
                4. Professional presentation
                5. Potential brand alignment
                
                Return a JSON with:
                - overall_score (0-100)
                - strengths (list)
                - weaknesses (list)
                - recommendation (boolean if they should be considered a potential lead)"""),
                ("human", "{profile_data}")
            ])
            
            chain = analysis_prompt | self.llm
            result = chain.invoke({"profile_data": json.dumps(profile)})
            
            # Extract the content from the AI message
            if isinstance(result, AIMessage):
                return result.content
            return str(result)
            
        except Exception as e:
            print(f"Error analyzing profile: {e}")
            return json.dumps({"error": str(e)})
    
    def _simulate_twitter_results(self, query: str) -> List[Dict[str, Any]]:
        """Simulate Twitter search results for demonstration purposes."""
        # In a real implementation, this would be replaced with actual Twitter API calls
        niche_keywords = query.lower().split()
        
        accounts = []
        base_accounts = [
            {
                "username": f"{niche_keywords[0]}Expert",
                "display_name": f"{niche_keywords[0].title()} Expert",
                "followers": random.randint(5000, 100000),
                "following": random.randint(500, 2000),
                "bio": f"Passionate about {' '.join(niche_keywords)}. Creator and consultant.",
                "posts_count": random.randint(500, 5000),
                "engagement_rate": round(random.uniform(1.5, 5.2), 2),
                "location": random.choice(["New York", "Los Angeles", "London", "Toronto", "Sydney"]),
                "website": f"https://www.{niche_keywords[0]}expert.com",
                "verified": random.choice([True, False]),
                "recent_posts": [
                    f"Just shared my top 10 tips for {' '.join(niche_keywords)}!",
                    f"New {niche_keywords[0]} workshop coming next month!",
                    f"Collaborating with amazing brands in the {' '.join(niche_keywords)} space"
                ]
            },
            {
                "username": f"Daily{niche_keywords[0].title()}",
                "display_name": f"Daily {niche_keywords[0].title()} Tips",
                "followers": random.randint(10000, 500000),
                "following": random.randint(200, 1000),
                "bio": f"Sharing daily content about {' '.join(niche_keywords)}. Join our community!",
                "posts_count": random.randint(1000, 8000),
                "engagement_rate": round(random.uniform(2.0, 7.5), 2),
                "location": random.choice(["Miami", "Chicago", "Berlin", "Tokyo", "Melbourne"]),
                "website": f"https://www.daily{niche_keywords[0]}.com",
                "verified": random.choice([True, False]),
                "recent_posts": [
                    f"How I grew my {niche_keywords[0]} business from zero to six figures",
                    f"The truth about {' '.join(niche_keywords)} that nobody tells you",
                    f"My favorite {niche_keywords[0]} products for 2025"
                ]
            },
            {
                "username": f"{niche_keywords[0]}Coach",
                "display_name": f"{niche_keywords[0].title()} Coach",
                "followers": random.randint(20000, 200000),
                "following": random.randint(300, 3000),
                "bio": f"Certified {niche_keywords[0]} coach. Author. Speaker. Helping you achieve your {' '.join(niche_keywords)} goals.",
                "posts_count": random.randint(2000, 10000),
                "engagement_rate": round(random.uniform(1.8, 6.0), 2),
                "location": random.choice(["San Francisco", "Austin", "Vancouver", "Paris", "Dublin"]),
                "website": f"https://www.{niche_keywords[0]}coach.net",
                "verified": random.choice([True, False]),
                "recent_posts": [
                    f"Just launched my new {niche_keywords[0]} course - 50% off this week only!",
                    f"Interview with the top {niche_keywords[0]} experts in the industry",
                    f"My {niche_keywords[0]} transformation story"
                ]
            }
        ]
        
        # Creating personalized variations based on the query
        for i in range(5):
            base_account = random.choice(base_accounts).copy()
            # Adding some randomization
            base_account["username"] = f"{base_account['username']}{random.randint(1, 999)}"
            base_account["followers"] = random.randint(5000, 500000)
            base_account["engagement_rate"] = round(random.uniform(1.0, 8.0), 2)
            accounts.append(base_account)
            
        return accounts
    
    def _simulate_instagram_results(self, query: str) -> List[Dict[str, Any]]:
        """Simulate Instagram search results for demonstration purposes."""
        # In a real implementation, this would be replaced with actual Instagram API calls
        niche_keywords = query.lower().split()
        
        accounts = []
        base_accounts = [
            {
                "username": f"{niche_keywords[0]}.lifestyle",
                "display_name": f"{niche_keywords[0].title()} Lifestyle",
                "followers": random.randint(10000, 1000000),
                "following": random.randint(500, 5000),
                "bio": f"📸 {niche_keywords[0].title()} content creator\n✨ Sharing {' '.join(niche_keywords)} inspiration\n🔗 Check out my latest collab",
                "posts_count": random.randint(100, 2000),
                "engagement_rate": round(random.uniform(2.0, 10.0), 2),
                "location": random.choice(["Miami", "Bali", "Los Angeles", "New York", "London"]),
                "website": f"https://linkin.bio/{niche_keywords[0]}lifestyle",
                "verified": random.choice([True, False]),
                "recent_posts": [
                    {"type": "image", "likes": random.randint(500, 50000), "comments": random.randint(10, 1000)},
                    {"type": "video", "views": random.randint(1000, 100000), "comments": random.randint(20, 2000)},
                    {"type": "image", "likes": random.randint(500, 50000), "comments": random.randint(10, 1000)}
                ],
                "highlights": [f"{niche_keywords[0]} Tips", "Brand Collabs", "My Favorites"]
            },
            {
                "username": f"the.{niche_keywords[0]}.guru",
                "display_name": f"The {niche_keywords[0].title()} Guru",
                "followers": random.randint(50000, 2000000),
                "following": random.randint(200, 1000),
                "bio": f"💫 Making {' '.join(niche_keywords)} accessible for everyone\n🏆 Featured in @forbes\n👇 My latest {niche_keywords[0]} program",
                "posts_count": random.randint(500, 3000),
                "engagement_rate": round(random.uniform(3.0, 8.0), 2),
                "location": random.choice(["California", "New York", "Toronto", "Sydney", "London"]),
                "website": f"https://{niche_keywords[0]}guru.com",
                "verified": random.choice([True, False]),
                "recent_posts": [
                    {"type": "carousel", "likes": random.randint(5000, 100000), "comments": random.randint(100, 5000)},
                    {"type": "reels", "views": random.randint(10000, 1000000), "comments": random.randint(200, 10000)},
                    {"type": "image", "likes": random.randint(3000, 80000), "comments": random.randint(50, 3000)}
                ],
                "highlights": ["Before/After", "Success Stories", "Programs"]
            },
            {
                "username": f"{niche_keywords[0]}_daily",
                "display_name": f"{niche_keywords[0].title()} Daily",
                "followers": random.randint(20000, 500000),
                "following": random.randint(500, 2000),
                "bio": f"🔥 Daily {niche_keywords[0]} content\n💪 Join my {niche_keywords[0]} challenge\n📱 DM for collabs",
                "posts_count": random.randint(300, 5000),
                "engagement_rate": round(random.uniform(2.5, 7.0), 2),
                "location": random.choice(["Austin", "Chicago", "Vancouver", "Berlin", "Melbourne"]),
                "website": f"https://www.{niche_keywords[0]}daily.co",
                "verified": random.choice([True, False]),
                "recent_posts": [
                    {"type": "image", "likes": random.randint(1000, 30000), "comments": random.randint(50, 2000)},
                    {"type": "reels", "views": random.randint(5000, 500000), "comments": random.randint(100, 5000)},
                    {"type": "image", "likes": random.randint(1000, 30000), "comments": random.randint(50, 2000)}
                ],
                "highlights": ["Products I Love", "My Routine", "FAQs"]
            }
        ]
        
        # Creating personalized variations based on the query
        for i in range(5):
            base_account = random.choice(base_accounts).copy()
            # Adding some randomization
            base_account["username"] = f"{base_account['username']}{random.randint(1, 999)}"
            base_account["followers"] = random.randint(10000, 2000000)
            base_account["engagement_rate"] = round(random.uniform(1.5, 12.0), 2)
            accounts.append(base_account)
            
        return accounts
    
    def generate_leads(self, brand_keyword: str, platform: str = "both", limit: int = 5) -> Dict[str, Any]:
        """
        Main method to generate influencer leads based on a brand keyword.
        
        Args:
            brand_keyword: The niche or topic to search for influencers
            platform: Which platform to search ("twitter", "instagram", or "both")
            limit: Maximum number of leads to return
            
        Returns:
            Dictionary with search results and analyzed leads
        """
        # Formatting the input for the agent
        input_text = f"""
        Find potential influencers for the brand keyword: "{brand_keyword}"
        Platform(s) to search: {platform}
        Number of leads needed: {limit}
        
        For each potential influencer, provide:
        1. Username and platform
        2. Follower count and engagement rate
        3. Content relevance to "{brand_keyword}"
        4. Why they would be a good fit for brand collaboration
        """
        
        # Executing the agent
        result = self.agent_executor.invoke({"input": input_text})
        
        
        return {
            "query": brand_keyword,
            "platform": platform,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": result
        }


# Example usage
if __name__ == "__main__":
    
    os.environ["OPENAI_API_KEY"] = "open_ai_key"
    
    # Initializing the agent
    agent = LeadGenerationAgent()
    
    # Generating leads for a brand keyword
    results = agent.generate_leads("fitness influencers", platform="both", limit=5)
    
    # results
    print(json.dumps(results, indent=2))