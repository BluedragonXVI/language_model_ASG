import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from streamlit.report_thread import get_report_ctx
import pydeck as pdk
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os



# get the database URL from heroku app
DATABASE_URL = os.environ['DATABASE_URL']
# get a unique session ID that can used at postgres primary key 
def get_session_id() -> str:
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id # postgres name convention
    return session_id

# functions to read/write states of user input and dataframes
def write_state(column, value, engine, session_id):
    engine.execute("UPDATE %s SET %s='%s'" % (session_id, column, value))

def write_state_df(df:pd.DataFrame, engine, session_id):
    df.to_sql('%s' % (session_id),engine,index=False,if_exists='replace',chunksize=1000)

def read_state(column, engine, session_id):
    state_var = engine.execute("SELECT %s FROM %s" % (column, session_id))
    state_var = state_var.first()[0]
    return state_var

def read_state_df(engine, session_id):
    try:
        df = pd.read_sql_table(session_id, con=engine)
    except:
        df = pd.DataFrame([])
    return df

if __name__ == '__main__':

    # create PostgreSQL client using configuration file
    #username:str = config.username 
    #password:str = config.password
    #db_name:str = config.db_name 
    engine = create_engine(DATABASE_URL, connect_args={'sslmode':'require'})

    # retrieve session ID
    session_id = get_session_id()

    # create state tables of session
    engine.execute("CREATE TABLE IF NOT EXISTS %s (size text)" % (session_id))
    len_table = engine.execute("SELECT COUNT(*) FROM %s" % (session_id))
    len_table = len_table.first()[0]
    if len_table == 0:
        engine.execute("INSERT INTO %s (size) VALUES ('1')" % (session_id))

    # can now create pages
    #page = st.sidebar.selectbox("Select page:", ("About", "What is SLAM?", "Active Neural SLAM", "Autonomous Drone Platform"))
    page = st.sidebar.selectbox("Select page:", ("About", "Language Models", "Evaluation"))

    
    # Import README markdown file 
    read_me_file_name = "README.md"
    read_me_file_path = os.path.join(os.getcwd(), read_me_file_name)
    read_me_file = open(read_me_file_path)
    read_me = read_me_file.read()


    if page == "About":
        st.markdown(read_me)
        
    elif page == "Language Models":
        # stateful input feature below
        size = st.text_input("Stateful input for future updates", read_state("size", engine, session_id))
        write_state("size", size, engine, session_id)
        size = int(read_state("size", engine, session_id))

        if st.button("Click"):
            data = [[0 for (size) in range((size))] for y in range((size))]
            df = pd.DataFrame(data)
            write_state_df(df, engine, session_id + "_df")

        if not (read_state_df(engine, session_id + "_df").empty):
            df = read_state_df(engine, session_id + "_df")
            st.write(df)
            
    elif page == "Evaluation":
        st.header("Work in progress")

  
        
