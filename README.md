# Voyager-Assist
Flight search assistant using Generative AI-  audio transcription and natural language processing of transcribed text. 

Tech Stack:Streamlit,LangChain

**WorkFlow**
**User Audio Input**
•	User records their flight requirements (e.g., "I need a flight from New York to London next Friday, returning the following Monday")
•	User uploads the audio file (.mp3 or .mp4) to the Voyager Assist web interface

**Audio Transcription:** System transcribe the audio file,onverts speech to text with high accuracy, handling various accents and background noise

Pyndantic Models are used to create a model for key flight search parameters.Parameters are : Departure location,Arrival location,Departure date,Return date (if applicable),Number of passengers,ther preferences (e.g., direct flights, preferred airlines), 
Result Processing:System extracts key information: flight numbers, times, durations, prices, airlines from the returned data.

Data Presentation: Processed flight data is converted into a Pandas DataFrame.DataFrame is displayed in the Streamlit interface, showing a table of flight options.User can view all flight options in the table
