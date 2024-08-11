from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv
from utils import *
from os import write
#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
#from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

# Define the Streamlit app
def main():
    st.title("Voyager Assist - A Flight Search Assistant")

    # Upload  files
    uploaded_file = st.file_uploader("Upload recorded .mp3/.mp4 files", type=["mp3","mp4"], accept_multiple_files=False)

    if uploaded_file:
        st.write("Uploaded File...")

        # Display uploaded files and buttons in a tabular form
        #for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        print(file_name)

            #col1, col2, col3 = st.columns([0.1, 1, 2])
            #with col1:
             #   st.write("-")
            #with col2:
             #   st.write(file_name)
            #with col3:
        send_button = st.button(f"Transcribe audio in {file_name}")

        if send_button:
            #email_summary(file_name)
            transcribed_text = transcribe_audio(file_name)
            with st.container():
                    
                st.write("Text from audio as follows:")
                #transcribed_text = transcribed_text.tex
                #st.write_stream(transcribed_text)
                st.write(transcribed_text)
                        
                    #st.write(transcribed_text)
            with st.container():
                st.write("Extracting flight parameters from generated text...")
                flight_params = extract_flight_params(transcribed_text)
                with st.expander("Flight Parameters"):
                    st.write(flight_params)
                #st.expander(flight_params)
                #st.text_area("Flight Parameters", flight_params,width=300,height=300)
                results = perform_flight_search(flight_params)
                filtered_results = filter_results(results)
                #filtered_results = filter_results(results, 'United')
                df = create_dataframe(filtered_results, 'United')
                st.dataframe(df)    
            
                    #st.success(f"Send email for: {file_name}")


# Run the Streamlit app
if __name__ == "__main__":
    main()

