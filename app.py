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
from bokeh.models import Ellipse, GraphRenderer, StaticLayoutProvider
from bokeh.palettes import Spectral8
from bokeh.plotting import figure

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
        # Create the graphical SLAM model
        N = 8
        node_indices = list(range(N))
        plot = figure(title='Graph Layout Demonstration', x_range=(-1.1,1.1), y_range=(-1.1,1.1), tools='', toolbar_location=None)
        graph = GraphRenderer()
        graph.node_renderer.data_source.add(node_indices, 'index')
        graph.node_renderer.data_source.add(Spectral8, 'color')
        graph.node_renderer.glyph = Ellipse(height=0.1, width=0.2, fill_color='color')
        graph.edge_renderer.data_source.data = dict(start=[0]*N, end=node_indices)

        ### start of layout code
        circ = [i*2*math.pi/8 for i in node_indices]
        x = [math.cos(i) for i in circ]
        y = [math.sin(i) for i in circ]

        graph_layout = dict(zip(node_indices, zip(x, y)))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        plot.renderers.append(graph)
        output_file('graph.html')
        #show(plot)
        st.write(plot)

        size = st.text_input("Matrix size", read_state("size", engine, session_id))
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
        # CREATING FUNCTION FOR MAPS
        def map(data, lat, lon, zoom):
            st.write(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={
                    "latitude": lat,
                    "longitude": lon,
                    "zoom": zoom,
                    "pitch": 50,
                },
                layers=[
                    pdk.Layer(
                        "HexagonLayer",
                        data=data,
                        get_position=["lon", "lat"],
                        radius=100,
                        elevation_scale=4,
                        elevation_range=[0, 1000],
                        pickable=True,
                        extruded=True,
                    ),
                ]
            ))

        # LAYING OUT THE TOP SECTION OF THE APP
        row1_1, row1_2 = st.beta_columns((2,3))

        with row1_1:
            st.title("NYC Uber Ridesharing Data")
            hour_selected = st.slider("Select hour of pickup", 0, 23)

        with row1_2:
            st.write(
            """
            ##
            Examining how Uber pickups vary over time in New York City's and at its major regional airports.
            By sliding the slider on the left you can view different slices of time and explore different transportation trends.
            """)

        # FILTERING DATA BY HOUR SELECTED
        data = data[data[DATE_TIME].dt.hour == hour_selected]

        # LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
        row2_1, row2_2, row2_3, row2_4 = st.beta_columns((2,1,1,1))

        # SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
        la_guardia= [40.7900, -73.8700]
        jfk = [40.6650, -73.7821]
        newark = [40.7090, -74.1805]
        zoom_level = 12
        midpoint = (np.average(data["lat"]), np.average(data["lon"]))

        with row2_1:
            st.write("**All New York City from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
            map(data, midpoint[0], midpoint[1], 11)

        with row2_2:
            st.write("**La Guardia Airport**")
            map(data, la_guardia[0],la_guardia[1], zoom_level)

        with row2_3:
            st.write("**JFK Airport**")
            map(data, jfk[0],jfk[1], zoom_level)

        with row2_4:
            st.write("**Newark Airport**")
            map(data, newark[0],newark[1], zoom_level)

        # FILTERING DATA FOR THE HISTOGRAM
        filtered = data[
            (data[DATE_TIME].dt.hour >= hour_selected) & (data[DATE_TIME].dt.hour < (hour_selected + 1))
            ]

        hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]

        chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

        # LAYING OUT THE HISTOGRAM SECTION

        st.write("")

        st.write("**Breakdown of rides per minute between %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))

        st.altair_chart(alt.Chart(chart_data)
            .mark_area(
                interpolate='step-after',
            ).encode(
                x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
                y=alt.Y("pickups:Q"),
                tooltip=['minute', 'pickups']
            ).configure_mark(
                opacity=0.5,
                color='red'
            ), use_container_width=True)

    if page == "Autonomous Drone Platform":
        st.title("Autonomous Drone Platform")
