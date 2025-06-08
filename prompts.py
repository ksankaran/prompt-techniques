from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

examples = [
    HumanMessage(
        "what is the weather forecast for cumming ga?", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "get_weather_forecast", "args": {"lat": "34.208464", "long": "-84.137575"}, "id": "1"}
        ],
    ),
    ToolMessage("It is sunny in Cumming, GA", tool_call_id="1"),
    AIMessage(
        "I have successfully obtained the forecast for Cumming, GA. It is sunny there.",
        name="example_assistant",
    ),
]

system = """I will help user with weather information using the tool provided.
You will be provided with a tool to get the weather forecast based on latitude and longitude.
You can use the tool by calling it with the latitude and longitude values.

You will receive a query from the user, and if it is related to weather forecast, use the tool to provide the weather information.
If not, respond with a message indicating that you can only provide weather information."""

few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

zero_shot_prompt = PromptTemplate.from_template("""You are a helpful assistant that provides weather information.
If the user asks about the weather, use the tool to get the forecast.
If not, respond with a message indicating that you can only provide weather information.
User query: {query}
""")


chain_of_thought_prompt = PromptTemplate.from_template("""You are a helpful assistant that identifies pattern in number sequence and respond with next number in the sequence. 
For example, the user enters the numbers 1, 3, 5, 7, 9. Let's think step by step about the pattern in these numbers. 
1, 3, 5, 7 and 9 are odd numbers with a step count of 2. You will respond with the next number in the sequence, which is 11. 
User query: {query}
""")

meta_prompt = PromptTemplate.from_template("""Problem: [question to be asked by user]
**Solution Structure**
1. Identify the problem in the question.
2. Break down the problem into smaller parts.
3. If problem is related to mathematical calculations, do the calculation and bold the answer.
4. If problem is related to weather forecast, use the tool to get the forecast. Bold all the temperature values in the response.
5. If problem is related to general knowledge, provide the answer based on your knowledge and bold relevant information.
Problem: {query}
""")

generated_knowledge_prompt = PromptTemplate.from_template("""You are a helpful assistant who can answer questions about traffic rules
that deals with right of way or who has to yield to which car or what should drivers do on the road."
You will be provided with a question and you need to answer it based on the following rules.
If you are unsure, you can ask for clarification.
At intesections, follow the below rules:\n
1. If there is a stop sign, the vehicle must stop.\n
2. If there is a yield sign, the vehicle must yield to oncoming traffic.\n
3. If there are no signs, the vehicle on the right has the right of way.\n
4. If there are traffic lights, follow the traffic light rules.\n
5. If there is a roundabout, yield to traffic in the roundabout.\n
6. If there is a pedestrian crossing, yield to pedestrians.\n
7. If there is a 4 way stop, the first vehicle to arrive has the right of way.\n
8. If you are entering or exiting a parking lot, yield to pedestrians and other vehicles already in the lot.\n
At crosswalks or intersections, follow the below rules:\n
1. If there is a pedestrian crossing, yield to pedestrians.\n
Merging or entering roads, follow the below rules:\n
1. If you are merging onto a highway, yield to traffic already on the highway.\n
2. If you are entering a road from a driveway or parking lot, yield to traffic on the road.\n
3. If you are entering from a privately owned road, yield to traffic on the public road.\n
If you see emergency vehicles, follow the below rules:\n
1. Always yield to emergency vehicles with flashing lights or sirens.\n
2. If you are stopped at a red light or stop sign, do not proceed until the emergency vehicle has passed.\n
3. If you are driving on a highway, move to the right lane and slow down to allow the emergency vehicle to pass.\n
In other situations, follow the below rules:\n
1. If you are being overtaken by another vehicle, yield to the right of overtaking vehicle \n
2. If you are on left lane on a multi-lane highway, yield to the right lane if another vehicle is trying to overtake you.\n
"User asked: {query}"
""")

self_consistency_prompt = PromptTemplate.from_template("""User: If the car mileage is 20 miles per gallon, and the car is going to travel 100 miles, how many gallons of gas will it need?
Assistant: Let's think step by step about the problem.
The car mileage is 20 miles per gallon, which means it can travel 20 miles on 1 gallon of gas.
To find out how many gallons of gas the car will need to travel 100 miles, we can divide the total distance by the mileage:
    100 miles / 20 miles per gallon = 5 gallons of gas.
So, the car will need 5 gallons of gas to travel 100 miles.
User: If the car mileage is 20 miles per gallon, and the car consumes 25% more gas when driving uphill, how many gallons of gas will it need to travel 50 miles uphill and 50 miles downhill?
Assistant: Let's think step by step about the problem.
The car mileage is 20 miles per gallon, which means it can travel 20 miles on 1 gallon of gas.
When driving uphill, the car consumes more gas, so we need to account for that.
When driving uphill, the car consumes 25% more gas, which means it will consume 20 miles / (1 + 0.25) = 16 miles per gallon.
To find out how many gallons of gas the car will need to travel 50 miles uphill, we can divide the distance by the mileage:
50 miles / 16 miles per gallon = 3.125 gallons of gas.
When driving downhill, the car will consume the normal mileage of 20 miles per gallon.
To find out how many gallons of gas the car will need to travel 50 miles downhill, we can divide the distance by the mileage:
50 miles / 20 miles per gallon = 2.5 gallons of gas.
So, the total gallons of gas needed for the trip is:
    3.125 gallons (uphill) + 2.5 gallons (downhill) = 5.625 gallons of gas.
User: If the car mileage is 20 miles per gallon, and the car consumes 25% more gas when driving uphill and 10% less gas when driving downhill, how many gallons of gas will it need to travel 50 miles uphill and 50 miles downhill?
Assistant: Let's think step by step about the problem.
The car mileage is 20 miles per gallon, which means it can travel 20 miles on 1 gallon of gas.
When driving uphill, the car consumes more gas, so we need to account for that.
When driving uphill, the car consumes 25% more gas, which means it will consume 20 miles / (1 + 0.25) = 16 miles per gallon.
To find out how many gallons of gas the car will need to travel 50 miles uphill, we can divide the distance by the mileage:
    50 miles / 16 miles per gallon = 3.125 gallons of gas.
When driving downhill, the car consumes less gas, so we need to account for that.
When driving downhill, the car consumes 10% less gas, which means it will consume 20 miles / (1 - 0.10) = 22.22 miles per gallon.
To find out how many gallons of gas the car will need to travel 50 miles downhill, we can divide the distance by the mileage:
    50 miles / 22.22 miles per gallon = 2.25 gallons of gas.
So, the total gallons of gas needed for the trip is:
    3.125 gallons (uphill) + 2.25 gallons (downhill) = 5.375 gallons of gas.
User: {query}
Assistant: 
""")