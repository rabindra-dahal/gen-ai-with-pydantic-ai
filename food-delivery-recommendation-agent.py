import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

# 1. Load configuration and silence framework splash banner
load_dotenv()

# -----------------------------------------------------------------------------
# Step 2: Define Structured Schemas using Pydantic
# -----------------------------------------------------------------------------

class RecommendedItem(BaseModel):
    dish_name: str = Field(description="Name of the food item")
    price_npr: int = Field(description="Price of the dish in Nepalese Rupees (NPR)")
    reasoning: str = Field(description="Brief explanation of why this dish fits the user's mood/criteria")

class RestaurantRecommendation(BaseModel):
    restaurant_name: str = Field(description="Name of the restaurant offering the items")
    rating: float = Field(description="User review rating out of 5.0")
    items: List[RecommendedItem] = Field(description="Top recommended options matching user request")

class FoodDeliveryResponse(BaseModel):
    summary: str = Field(description="A warm conversational summary introduction personalized to the user's craving")
    recommendations: List[RestaurantRecommendation] = Field(description="Grouped recommendations by local restaurants")
    total_estimated_delivery_time_mins: int = Field(description="An estimated average travel time for these zones")

# -----------------------------------------------------------------------------
# Step 3: Initialize the Pydantic AI Agent wrapped with Gemini 3.5 Flash Lite
# -----------------------------------------------------------------------------

delivery_agent = Agent(
    model='google:gemini-3.5-flash-lite', 
    output_type=FoodDeliveryResponse,
    instructions=(
        "You are an expert local food delivery recommendation assistant. "
        "Your task is to read the user's current food craving, location, or dietary restrictions "
        "and suggest the best matching meals available from nearby restaurants using your internal tools. "
        "Keep your tones friendly, engaging, and precise."
    )
)

# -----------------------------------------------------------------------------
# Step 4: Attach a Dynamic Function Tool to Fetch Live Menu Catalog Data
# -----------------------------------------------------------------------------

# Update your tool to include RunContext as the first argument
@delivery_agent.tool
def search_local_menu_database(ctx: RunContext[None], cuisine_type: str) -> str:
    """
    Queries the local inventory and restaurant directory for available menus.
    Use this tool whenever you need to find real meals matching a specific cuisine keyword.
    """
    mock_database = {
        "nepali": [
            {"restaurant": "Gaun Ghar Chulo", "rating": 4.1, "dish": "Traditional Nepali Veg Thali", "price": 350},
            {"restaurant": "Bodhi Villa", "rating": 4.0, "dish": "Mutton Bhutuwa", "price": 669},
            {"restaurant": "Bodhi Villa", "rating": 4.0, "dish": "Duck Choyela", "price": 555}
        ],
        "asian": [
            {"restaurant": "Bodhi Villa", "rating": 4.0, "dish": "Spicy Minced Chicken with Rice Noodles", "price": 650},
            {"restaurant": "Daali Restaurant", "rating": 4.3, "dish": "Pan-Fried Chicken Momos", "price": 280}
        ],
        "fast food": [
            {"restaurant": "Daali Restaurant", "rating": 4.3, "dish": "Crunchy Zinger Burger combo", "price": 420},
            {"restaurant": "Gaun Ghar Chulo", "rating": 4.1, "dish": "Spicy Fries with Dipping Sauce", "price": 185}
        ]
    }
    
    normalized_query = cuisine_type.lower()
    if "nepal" in normalized_query or "local" in normalized_query:
        matches = mock_database["nepali"]
    elif "momo" in normalized_query or "noodle" in normalized_query or "asian" in normalized_query:
        matches = mock_database["asian"]
    else:
        matches = mock_database["fast food"]
        
    return f"Available database results for '{cuisine_type}': {str(matches)}"


# -----------------------------------------------------------------------------
# Step 5: Execute Routine Pipeline
# -----------------------------------------------------------------------------

def run_recommendation_pipeline():
    user_craving = "I am looking for some authentic local Nepali non-veg food or spicy appetizers."
    print(f"User Query: '{user_craving}'\nProcessing Agent Loop...")
    
    # Run agent loop synchronously (handles internal tool call execution seamlessly)
    result = delivery_agent.run_sync(user_prompt=user_craving)
    
    # Output properties are fully validated and mapped right into your Pydantic model
    validated_response: FoodDeliveryResponse = result.output
    
    print("\n=======================================================")
    print(f" AI Assistant Summary: {validated_response.summary}")
    print(f" Est. Delivery Duration: {validated_response.total_estimated_delivery_time_mins} minutes")
    print("=======================================================")
    
    for restaurant in validated_response.recommendations:
        print(f"\n {restaurant.restaurant_name} ( {restaurant.rating}/5.0)")
        for item in restaurant.items:
            print(f"   • {item.dish_name} | Price: {item.price_npr} NPR")
            print(f"      Reason: {item.reasoning}")

if __name__ == "__main__":
    run_recommendation_pipeline()
