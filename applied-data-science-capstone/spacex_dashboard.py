"""SpaceX Dash app. Run analysis first, then python spacex_dashboard.py."""
from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from dash import Dash,dcc,html,Input,Output
from flask import send_file
DATA=Path(__file__).parent/"spacex_output"
geo=pd.read_csv(DATA/"spacex_launch_geo.csv")
geo["Booster Version Category"]=geo["Booster Version"].str.extract(r"F9\s+([^\s]+)",expand=False).fillna("Unknown")
def pie(site,limits):
 d=geo[geo["Payload Mass (kg)"].between(*limits)]
 if site=="ALL":
  s=d.groupby("Launch Site")["class"].sum().reset_index()
  return px.pie(s,names="Launch Site",values="class",title="Successful launches by site")
 s=d.loc[d["Launch Site"]==site,"class"].value_counts().rename_axis("Outcome").reset_index(name="Launches")
 s["Outcome"]=s["Outcome"].map({0:"Failure",1:"Success"})
 return px.pie(s,names="Outcome",values="Launches",title=site+" launch outcomes")
def scatter(site,limits):
 d=geo[geo["Payload Mass (kg)"].between(*limits)].copy()
 if site!="ALL":d=d[d["Launch Site"]==site]
 d["Outcome"]=d["class"].map({0:"Failure",1:"Success"})
 return px.scatter(d,x="Payload Mass (kg)",y="class",color="Booster Version Category",symbol="Outcome",hover_data=["Launch Site"],title="Payload versus launch outcome",labels={"class":"Landing outcome"}).update_layout(height=520,legend={"orientation":"h","yanchor":"top","y":-0.25,"x":0},margin={"l":55,"r":15,"t":70,"b":150})
external_prefix=os.environ.get("DASH_REQUESTS_PREFIX","/")
app=Dash(__name__,requests_pathname_prefix=external_prefix)
app.layout=html.Div([
 html.H1("SpaceX launch and landing outcomes"),
 html.P("Historical course dashboard data: 56 launches. This is distinct from the 90-launch modeling sample."),
 html.Label("Launch site"),
 dcc.Dropdown(id="site",options=[{"label":"All sites","value":"ALL"}]+[{"label":s,"value":s} for s in sorted(geo["Launch Site"].unique())],value="ALL",clearable=False),
 html.Label("Payload range (kg)"),
 dcc.RangeSlider(id="mass",min=0,max=10000,step=500,value=[0,10000],marks={i:str(i) for i in range(0,10001,2000)}),
 html.Div([dcc.Graph(id="pie"),dcc.Graph(id="scatter")],style={"display":"grid","gridTemplateColumns":"1fr 1fr"}),
 html.P("The all-site pie shows shares of successful launches. A selected-site pie shows that site's success and failure counts."),
 html.A("Launch map",href=external_prefix+"map"),html.Span(" / "),html.A("Proximity map",href=external_prefix+"proximity")
],style={"fontFamily":"Arial","margin":"25px","color":"#17364F"})
@app.callback(Output("pie","figure"),Output("scatter","figure"),Input("site","value"),Input("mass","value"))
def update(site,mass):return pie(site,mass),scatter(site,mass)
@app.server.route("/map")
def map_view():return send_file(DATA/"launch_map.html")
@app.server.route("/proximity")
def proximity_view():return send_file(DATA/"proximity_map.html")
if __name__=="__main__":app.run(host="0.0.0.0",port=8050,debug=False)
