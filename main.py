import streamlit as st
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from streamlit.report_thread import get_report_ctx

# get a unique session ID that can used at postgres primary key 
def get_session_id() -> str:
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id # postgres name convention
    return session_id

# functions to read/write states of user input and dataframes
def write_state(column, value, engine, session_id):
    engine.execute("UPDATE %s SET %s='%s'" % (session_id, column, value))

def write_state_df(df, engine, session_id):
    df.to_sql('%s' % (session_id, engine, index=False, if_exists='replace', chunksize=1000))

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