import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from streamlit.report_thread import get_report_ctx
import altair as alt
import pydeck as pdk
import config
import math
import os
from bokeh.io import output_file, show
from bokeh.models import (Ellipse, GraphRenderer, StaticLayoutProvider,
                          BoxSelectTool, Circle, EdgesAndLinkedNodes,
                          Range1d, Plot, MultiLine, Label, LabelSet, ColumnDataSource)
from bokeh.palettes import Spectral8
from bokeh.plotting import figure, from_networkx


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

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data

data = load_data(100000)

if __name__ == '__main__':

    # create PostgreSQL client using configuration file
    username:str = config.username 
    password:str = config.password
    db_name:str = config.db_name 
    engine = create_engine("postgresql+psycopg2://"+username+':'+password+"@localhost:5432/"+db_name)

    # retrieve session ID
    session_id = get_session_id()

    # create state tables of session
    engine.execute("CREATE TABLE IF NOT EXISTS %s (size text)" % (session_id))
    len_table = engine.execute("SELECT COUNT(*) FROM %s" % (session_id))
    len_table = len_table.first()[0]
    if len_table == 0:
        engine.execute("INSERT INTO %s (size) VALUES ('1')" % (session_id))

    # can now create pages
    page = st.sidebar.selectbox("Select page:", ("About", "What is SLAM?", "Active Neural SLAM", "Autonomous Drone Platform"))

    
    # Import README markdown file 
    read_me_file_name = "README.md"
    read_me_file_path = os.path.join(os.getcwd(), read_me_file_name)
    read_me_file = open(read_me_file_path)
    read_me = read_me_file.read()
    # Import SLAM info markdown files 
    WIS_file_name = "WIS.md"
    WIS_file_path = os.path.join(os.getcwd(), WIS_file_name)
    WIS_file = open(WIS_file_path)
    WIS = WIS_file.read()

    # page config and actions
    if page == "About":
        st.markdown(read_me)
    elif page == "What is SLAM?":
        st.markdown(WIS)
        num_nodes = 10
        node_coords = [(1,8), (3,10), (6,10), (10,10), (4,8), (8,8), (3,5), (6,5), (10,5), (4,3)]
        x = [x for x,y in node_coords]
        y = [y for x,y in node_coords]
        xs, ys = [0,1,1,2,2,2,3,3,9,9,9], [1,2,6,3,4,7,8,5,6,7,8] # vertices
        node_labels = ["u[t-1]", "x[t-1]", "x[t]", "x[t+1]", "u[t]", "u[t+1]", "z[t-1]", "z[t]", "z[t+1]", "m"]
        node_indices = list(range(num_nodes))
        plot = figure(title='', x_range=(0,11), y_range=(2,11), tools='', toolbar_location=None)
        plot.axis.visible = False
        plot.background_fill_color = "beige"
        plot.background_fill_alpha = 0.5
        graph = GraphRenderer()
        graph.node_renderer.data_source.add(node_indices, 'index')
        graph.node_renderer.data_source.add(Spectral8, 'color')
        graph.node_renderer.glyph = Circle(radius=0.7, fill_color="color", line_color="color", fill_alpha=0.5, line_alpha=0.5)
        graph.edge_renderer.glyph = MultiLine(line_color="#000000", line_width=0.7, line_alpha=0.8, line_dash="dashed")
        graph.edge_renderer.data_source.data = dict(start=xs, end=ys)
        #graph.edge_renderer.data_source.data = dict(start=[0]*num_nodes, end=node_indices)
        graph.node_renderer.data_source.data['name'] = node_labels
        source = ColumnDataSource(data=dict(xs=x, ys=y, names=node_labels))
        # start of layout code
        labels = LabelSet(x='xs', y='ys', text='names', level='glyph', x_offset=0, y_offset=0, source=source, render_mode='canvas')
        graph_layout = dict(zip(node_indices, zip(x, y)))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        plot.renderers.append(graph)
        selected_circles = plot.circle([1,4,8,3,6,10], [8,8,8,5,5,5])
        glyphs = selected_circles.glyph
        glyphs.size = 80
        glyphs.fill_alpha = 0.5
        glyphs.line_color = "#060f1c"
        glyphs.line_dash = [30, 3]
        glyphs.line_width = 5
        plot.add_layout(labels)
        st.write(plot)

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
    elif page == "Active Neural SLAM":
        st.title("Active Neural SLAM")
        st.header("Work in progress")

    if page == "Autonomous Drone Platform":
        st.title("Autonomous Drone Platform")
        st.header("Work in progress")
        
