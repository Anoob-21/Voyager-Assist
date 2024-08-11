
#import whisper
import openai
#from langchain.llms import OpenAI
from langchain.agents import initialize_agent
#from langchain.agents.agent_toolkits import ZapierToolkit
#from langchain.utilities.zapier import ZapierNLAWrapper
import os
from serpapi import GoogleSearch
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from typing import List
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import os
from serpapi import GoogleSearch
import pandas as pd

class Airport(BaseModel):
        name: str
        id: str
        time: str

class Flight(BaseModel):
        departure_airport: Airport
        arrival_airport: Airport
        duration: int
        airplane: str
        airline: str
        airline_logo: str
        travel_class: str
        flight_number: str
        legroom: str
        extensions: List[str]

class FlightResult(BaseModel):
        flights: List[Flight]
        total_duration: int
        price: int
        type: str
        airline_logo: str
        
def email_summary(file):
    # Transcribe audio file
    #transcribed_text = transcribe_audio(file)

    # Extract flight search parameters
    #flight_params = extract_flight_params(transcribed_text)

    # Perform flight search
    #results = perform_flight_search(flight_params)

    # Filter results for American Airlines
    #filtered_results = filter_results(results, 'American')

    # Create DataFrame from filtered results
    #df = create_dataframe(filtered_results)

    #print(df)
    return "Email sent"

def transcribe_audio(uploaded_file):
    client = openai.Client()
    with open(uploaded_file,"rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            #response_format="text"
            )
    
    print("Transcription complete")
    print(transcription.text)
    return transcription.text

def extract_flight_params(transcribed_text):
    class FlightSearchParams(BaseModel):
        engine: str = Field(description="This should be set to google_flights")
        hl: str = Field(description="This should be set to en")
        gl: str = Field(description="This should be set to us")
        departure_id: str = Field(description="This is the airport code of the departure location")
        arrival_id: str = Field(description="This is the airport code of the arrival location")
        outbound_date: str = Field(description="This should be set to departure date with year set as 2024")
        return_date: str = Field(description=" This should be set to return date with year set as 2024")
        currency: str = Field(description="This should be set to USD")
        type: str = Field(description="This should be set to 1")
        travel_class: str = Field(description="This should be set to 1")
        adults: str = Field(description="This should be set to number of adults in the trip")
        children: str = Field(description="This should be set to number of children in the trip")
        stops: str = Field(description="This should be set to 2")
        api_key: str = "secret_api_key"

    extraction_template = """
    From the following transcribed text, extract the following information:
    
    transcribed text: {transcribed_text}
    {format_instructions}
    
    """
    pydantic_parser = PydanticOutputParser(pydantic_object=FlightSearchParams)
    format_instructions = pydantic_parser.get_format_instructions()
    updated_prompt = ChatPromptTemplate.from_template(template=extraction_template)
    messages = updated_prompt.format_messages(transcribed_text=transcribed_text,
                                              format_instructions=format_instructions)
    llm_model = "gpt-3.5-turbo"
    chat = ChatOpenAI(temperature=0.0, model=llm_model)
    format_response = chat(messages)
    vacation = pydantic_parser.parse(format_response.content)

    vacation_dict = vacation.dict()
    #vacation_dict["api_key"] = "Test"
    print("Pydantic parsing complete")
    return vacation_dict

def perform_flight_search(flight_params):
    search = GoogleSearch(flight_params)
    results = search.get_dict()
    print("Flight search complete")
    print(results)
    return results

def filter_results(results, airline):
    
    #best_flights = results.get('best_flights')
    #print(best_flights)
    best_flights = results.get('best_flights')
    other_flights = results.get('other_flights')
    all_flights = best_flights + other_flights
    print("All Flights")
    print(all_flights)

    #keys_to_remove = ['departure_token', 'carbon_emissions']
    #best_flights = [{k: v for k, v in flight.items() if k not in keys_to_remove} for flight in best_flights]
    print("Filtered results")
    filtered_results = [result for result in best_flights  for flight in result['flights']]
    print(filtered_results)
    #filtered_results = [result for result in best_flights if any(flight.get('airline') == airline for flight in result['flights'])]
    flight_data =[]
    for flight_result in filtered_results:
        if 'price' not in flight_result:
          flight_result['price'] = 0
        flight_result = FlightResult.parse_obj(flight_result)
        for flight in flight_result.flights:
            hours, minutes = divmod(flight_result.total_duration, 60)  # Convert total_duration from minutes to hours and minutes
            df_dict = {
                'Flight Number': flight.flight_number,
                #'Airline Logo': f'<html><img src="{flight_result.airline_logo}" width="100" height="50" alt="Airline Logo"></html>',
                'Departure Airport': flight.departure_airport.name,
                'Arrival Airport': flight.arrival_airport.name,
                'Departure Time': flight.departure_airport.time,
                'Arrival Time': flight.arrival_airport.time,
                'Total Duration': f'{hours} hours {minutes} minutes',  # Format total_duration as 'X hours Y minutes'
                'Airline': flight.airline,
                #'Airline Logo': flight_result.airline_logo,
                'Travel Class': flight.travel_class,
                'Travel Type': flight_result.type,
                'Price': flight_result.price 
            }
            flight_data.append(df_dict)
    print("Filtering complete")
    print("flight_data")
    print(flight_data)
    
    return flight_data

def create_dataframe(filtered_results,airline=None):
    if airline:
        #[{'Flight Number': 'UA 1335', 'Departure Airport': "Chicago O'Hare International Airport", 'Arrival Airport': 'Orlando International Airport', 'Departure Time': '2024-07-15 17:50', 'Arrival Time': '2024-07-15 21:45', 'Total Duration': '2 hours 55 minutes', 'Airline': 'United', 'Travel Class': 'Economy', 'Travel Type': 'Round trip', 'Price': 873}, 
        #filtered_results = [result for result in filtered_results if any(flight.get('airline') == airline for flight in result['flights'])]
        #filtered_results = [result for result in filtered_results if any(a == airline for a in result.get('Airline', []))]
        filtered_results = [result for result in filtered_results if result.get('Airline') == airline]
    df = pd.DataFrame(filtered_results, columns=['Flight Number','Departure Airport', 'Arrival Airport', 'Departure Time', 'Arrival Time', 'Total Duration', 'Airline', 'Travel Class', 'Travel Type', 'Price'])
    #for index, row in df.iterrows():
        #   df['Airline Logo']= df['Airline Logo'].apply(display_image)
    print("Dataframe created")
    print(df)
    return df
    

# get from https://platform.openai.com/
os.environ["OPENAI_API_KEY"] = "Test"

# get from https://nla.zapier.com/docs/authentication/ after logging in):
#os.environ["ZAPIER_NLA_API_KEY"] = "sk-ak-dV0owIyoCEbQJVzZv5F1S5JG77"


def email_summary(file):


    # large language model
    #llm = OpenAI(temperature=0)

    # Initializing zapier
    #zapier = ZapierNLAWrapper()
    #toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    # The agent used here is a "zero-shot-react-description" agent. 
    # Zero-shot means the agent functions on the current action only — it has no memory. 
    # It uses the ReAct framework to decide which tool to use, based solely on the tool's description.
    #agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react """-description", verbose=True)
    
    client = openai.Client()

   # audio_file = open("/path/to/file/speech.mp3", "rb")
    with open("VoyagerAssistCall.mp4","rb") as audio_file:
        transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        #response_format="text"
        )
    print(transcription.text)
    transcribed_text = transcription.text
    
    class FlightSearchParams(BaseModel):
        engine: str = Field(description="This should be set to google_flights")
        hl: str = Field(description="This should be set to en")
        gl: str = Field(description="This should be set to us")
        departure_id: str  = Field(description="This is the airport code of the departure location")
        arrival_id: str = Field(description="This is the airport code of the arrival location")
        outbound_date: str = Field(description="This should be set to departure date with year set as 2024")
        return_date: str = Field(description=" This should be set to return date with year set as 2024")
        currency: str = Field(description="This should be set to USD")
        type: str = Field(description="This should be set to 1")
        travel_class: str  = Field(description="This should be set to 1")
        adults: str  = Field(description="This should be set to number of adults in the trip")
        children: str = Field(description="This should be set to number of children in the trip")
        stops: str  = Field(description="This should be set to 1")
        api_key: str = "secret_api_key"
    
    # specify a model, here its BASE
    #model = whisper.load_model("base")
    #whisper.
    # transcribe audio file
    #result = model.transcribe(file)
    #print(result["text"])

    # Send email using zapier
    #agent.run("Send an Email to sharathraju489@gmail.com via gmail summarizing the following text provided below : "+result["text"])
    extraction_template = """
    From the following transcribed text, extract the following information:
    
    transcribed text: {transcribed_text}
    {format_instructions}
    
    """
    pydantic_parser = PydanticOutputParser(pydantic_object=FlightSearchParams)
    format_instructions = pydantic_parser.get_format_instructions()
    updated_prompt = ChatPromptTemplate.from_template(template=extraction_template)
    messages = updated_prompt.format_messages(transcribed_text=transcribed_text,
                                           format_instructions=format_instructions)
    llm_model = "gpt-3.5-turbo"
    chat = ChatOpenAI(temperature=0.0, model=llm_model)
    format_response = chat(messages)
    vacation = pydantic_parser.parse(format_response.content)
    print("Vacation obj")
    print(vacation)
    print(type(vacation))
   # vacation.api_key = "8Test"
    vacation_dict = vacation.dict()
    vacation_dict["api_key"] = "Test"
    search = GoogleSearch(vacation_dict)
    results = search.get_dict()
    print("Results type")
    for key, value in results.items():
        print(f"Key: {key}, Type of value: {type(value)}")
        print(type(results))
    print("Results")
    print(results)
    best_flights = results.get('best_flights')
    keys_to_remove = ['departure_token', 'carbon_emissions']  # Replace with the actual keys you want to remove
    best_flights = [{k: v for k, v in flight.items() if k not in keys_to_remove} for flight in best_flights]

    print("Best Flights Type")
    print(type(best_flights))
    #print(best_flights)
    
    print("Best Flights")
    for flight in best_flights:
        for key, value in flight.items():
            print(f"Key: {key}, Type of value: {type(value)}")
            def get_nested_keys(dictionary):
                keys = []
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        nested_keys = get_nested_keys(value)
                        keys.extend([f"{key}.{nested_key}" for nested_key in nested_keys])
                    else:
                        keys.append(key)
                return keys

            nested_keys = get_nested_keys(flight)
            print(nested_keys)
    print(best_flights)
    filtered_results = [result for result in best_flights if any(flight.get('airline') == 'American' for flight in result['flights'])]
    #filtered_results = [flight for flight in best_flights if flight.get('airline') == 'American']
    print("Filtered for American Airlines")
    print(filtered_results)
    df = pd.DataFrame(filtered_results)
    print("Dataframe")
    print(df)
         
    #new_dict = {'best flights': best_flights}
    #print("Best Flights")
    #for key, value in best_flights.items():
     #   print(f"Key: {key}, Type of value: {type(value)}")
      #  print(f"Key: {key}, Value:{value}")
    #print(new_dict)
    #filtered_results = [flight for flight in best_flights if flight.get('airline') == 'American']
    #print("Filtered for American Airlines")
    #print(filtered_results)
    
    
    #for key, value in results.items():
     #print(f"Key: {key}, Type of value: {type(value)}")
     #print(f"Key: {key}, Value:{value}")
     #   filtered_results = {key: value for key, value in results.items() if value.get('airline') == 'American'}
        
      #  print(filtered_results)
        
        
    #search = GoogleSearch(params)
    #results = search.get_dict()

    class Airport(BaseModel):
        name: str
        id: str
        time: str

    class Flight(BaseModel):
        departure_airport: Airport
        arrival_airport: Airport
        duration: int
        airplane: str
        airline: str
        airline_logo: str
        travel_class: str
        flight_number: str
        ticket_also_sold_by: List[str]
        legroom: str
        extensions: List[str]
        
    class FlightResult(BaseModel):
        flights: List[Flight]
        total_duration: int
        price: int
        type: str
        airline_logo: str
    result_json = [{'flights': [{'departure_airport': {'name': 'Austin-Bergstrom International Airport', 'id': 'AUS', 'time': '2024-07-15 07:40'}, 'arrival_airport': {'name': 'Orlando International Airport', 'id': 'MCO', 'time': '2024-07-15 11:26'}, 'duration': 166, 'airplane': 'Boeing 737', 'airline': 'American', 'airline_logo': 'https://www.gstatic.com/flights/airline_logos/70px/AA.png', 'travel_class': 'Economy', 'flight_number': 'AA 1274', 'ticket_also_sold_by': ['Alaska'], 'legroom': '30 in', 'extensions': ['Average legroom (30 in)', 'Wi-Fi for a fee', 'In-seat power & USB outlets', 'Stream media to your device', 'Carbon emissions estimate: 581 kg']}], 'total_duration': 166, 'price': 1265, 'type': 'Round trip', 'airline_logo': 'https://www.gstatic.com/flights/airline_logos/70px/AA.png'}]

    flight_result = FlightResult.parse_obj(filtered_results)
    
    #Flight Number	Departure Airport	Arrival Airport	Departure Time	Arrival Time	TotalDuration	Airline	AirlineLogo	Travel Class	Travel Type	Price
    df = pd.DataFrame(columns=['Flight Number', 'Departure Airport', 'Arrival Airport', 'Departure Time', 'Arrival Time', 'Total Duration', 'Airline', 'Airline Logo', 'Travel Class', 'Travel Type', 'Price'])
    for flight in flight_result.flights:
        
        # Use pd.concat instead of append
        df = pd.concat([df, pd.DataFrame({
            'Flight Number': flight.flight_number,
            'Departure Airport': flight.departure_airport.name,
            'Arrival Airport': flight.arrival_airport.name,
            'Departure Time': flight.departure_airport.time,
            'Arrival Time': flight.arrival_airport.time,
            'Total Duration': flight_result.total_duration,
            'Airline': flight.airline,
            'Airline Logo': flight_result.airline_logo,
            'Travel Class': flight.travel_class,
            'Travel Type': flight_result.type,
            'Price': flight_result.price
        })], ignore_index=True)

    print(df)
   
   
