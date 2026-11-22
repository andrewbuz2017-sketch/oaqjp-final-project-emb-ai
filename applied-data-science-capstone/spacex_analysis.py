"""SpaceX landing analysis. Historical IBM course data, not live launch forecasts."""
from pathlib import Path
import os,re,json,io,requests
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import sqlite3,folium
from folium.plugins import MarkerCluster,MousePosition
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,ConfusionMatrixDisplay
OUT=Path("spacex_output");OUT.mkdir(exist_ok=True)
BASE="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork"
def download(url,name):
 r=requests.get(url,timeout=60);r.raise_for_status();(OUT/name).write_bytes(r.content);return r
df=pd.read_csv(io.BytesIO(download(BASE+"/datasets/dataset_part_2.csv","dataset_part_2.csv").content))
raw=pd.read_csv(io.BytesIO(download(BASE+"/datasets/dataset_part_1.csv","dataset_part_1.csv").content))
sql=pd.read_csv(io.BytesIO(download(BASE+"/labs/module_2/data/Spacex.csv","Spacex.csv").content))
geo=pd.read_csv(io.BytesIO(download(BASE+"/datasets/spacex_launch_geo.csv","spacex_launch_geo.csv").content))
geo["Booster Version Category"]=geo["Booster Version"].str.extract(r"F9\s+([^\s]+)",expand=False).fillna("Unknown")
api=download(BASE+"/datasets/API_call_spacex_api.json","API_call_spacex_api.json").json()
api_norm=pd.json_normalize(api)
print("API first static fire year:",str(api_norm.static_fire_date_utc.iloc[0])[:4])
wiki_url="https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
wr=requests.get(wiki_url,timeout=60)
soup=BeautifulSoup(wr.text,"html.parser")
wiki_status=wr.status_code
if wr.ok:
 tables=soup.select("table.wikitable")
 for i,t in enumerate(tables):
  rows=[[c.get_text(" ",strip=True) for c in tr.find_all(["th","td"])] for tr in t.find_all("tr")]
  pd.DataFrame(rows).to_csv(OUT/f"wikipedia_table_{i}.csv",index=False)
else:
 # Preserve the actual failure. The supplied frozen course CSV remains the source.
 (OUT/"webscraping_status.txt").write_text(f"HTTP {wiki_status} for {wiki_url}. Course frozen CSV used for downstream analysis.\n")
 # Exercise the identical table parsing steps on an HTML rendering of the supplied table.
 tables=BeautifulSoup(sql.to_html(index=False),"html.parser").select("table")
 parsed_rows=[[c.get_text(" ",strip=True) for c in tr.find_all(["th","td"])] for tr in tables[0].find_all("tr")]
 pd.DataFrame(parsed_rows[1:],columns=parsed_rows[0]).to_csv(OUT/"html_parser_validation.csv",index=False)
# API enrichment is already captured in the supplied frozen CSV.
assert len(df)==90
df["Date"]=pd.to_datetime(df.Date)
df["Year"]=df.Date.dt.year
print("Launches",len(df),"success",int(df.Class.sum()),"rate",df.Class.mean())
print("Raw missing",raw.isna().sum().to_string())
print("Unique columns",df.columns.tolist())
sns.set_theme(style="whitegrid")
def save(name):
 plt.tight_layout();plt.savefig(OUT/(name+".png"),dpi=150,bbox_inches="tight");plt.close()
def scatter(x,y,name,xlabel=None,ylabel=None):
 fig,ax=plt.subplots(figsize=(10,4.5))
 sns.scatterplot(data=df,x=x,y=y,hue="Class",palette={0:"#D95858",1:"#148C7D"},s=70,ax=ax)
 ax.set(xlabel=xlabel or x,ylabel=ylabel or y);save(name)
scatter("FlightNumber","LaunchSite","flight_site")
scatter("PayloadMass","LaunchSite","payload_site")
scatter("FlightNumber","Orbit","flight_orbit")
scatter("PayloadMass","Orbit","payload_orbit")
orbit=df.groupby("Orbit").Class.agg(["mean","count"]).sort_values("mean")
fig,ax=plt.subplots(figsize=(10,4.5));orbit["mean"].plot.bar(ax=ax,color="#167B9D");ax.set(ylabel="Landing success rate",ylim=(0,1.12))
for i,(idx,r) in enumerate(orbit.iterrows()):ax.text(i,r["mean"]+.025,f"n={int(r['count'])}",ha="center",fontsize=9)
save("orbit_rate")
year=df.groupby("Year").Class.agg(["mean","count"])
fig,ax=plt.subplots(figsize=(10,4.5));year["mean"].plot(marker="o",ax=ax,color="#148C7D");ax.set(ylabel="Landing success rate",ylim=(0,1.1));save("year_rate")
site=df.groupby("LaunchSite").Class.agg(["sum","count","mean"])
fig,ax=plt.subplots(figsize=(10,4.5));site["mean"].plot.barh(ax=ax,color="#167B9D");ax.set(xlabel="Landing success rate",xlim=(0,1));save("site_rate")
# SQL exercises retain the supplied CSV's schema and chronology.
con=sqlite3.connect(OUT/"spacex.sqlite")
sql.to_sql("SPACEXTBL",con,index=False,if_exists="replace")
print("SQL schema",sql.columns.tolist())
queries={
"unique_sites":"SELECT DISTINCT Launch_Site FROM SPACEXTBL",
"five_ccaf":"SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%' LIMIT 5",
"nasa_payload":"SELECT SUM(PAYLOAD_MASS__KG_) AS Total_Payload_Mass FROM SPACEXTBL WHERE Customer = 'NASA (CRS)'",
"avg_f9v11":"SELECT AVG(PAYLOAD_MASS__KG_) AS Mean_Payload FROM SPACEXTBL WHERE Booster_Version LIKE 'F9 v1.1%'",
"first_ground":"SELECT MIN(Date) AS First_Ground_Landing FROM SPACEXTBL WHERE Landing_Outcome = 'Success (ground pad)'",
"drone_payload":"SELECT Booster_Version,PAYLOAD_MASS__KG_ FROM SPACEXTBL WHERE Landing_Outcome = 'Success (drone ship)' AND PAYLOAD_MASS__KG_ > 4000 AND PAYLOAD_MASS__KG_ < 6000",
"outcomes":"SELECT Mission_Outcome,COUNT(*) AS Missions FROM SPACEXTBL GROUP BY Mission_Outcome",
"max_payload":"SELECT Booster_Version,PAYLOAD_MASS__KG_ FROM SPACEXTBL WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL)",
"failures_2015":"SELECT Date,Booster_Version,Launch_Site,Landing_Outcome FROM SPACEXTBL WHERE Date LIKE '2015%' AND Landing_Outcome = 'Failure (drone ship)'",
"ranked_outcomes":"SELECT Landing_Outcome,COUNT(*) AS N FROM SPACEXTBL WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY Landing_Outcome ORDER BY N DESC",
"site_counts":"SELECT Launch_Site,COUNT(Mission_Outcome) AS Missions FROM SPACEXTBL GROUP BY Launch_Site"}
qresults={}
for name,q in queries.items():
 t=pd.read_sql_query(q,con);t.to_csv(OUT/(name+".csv"),index=False);qresults[name]=t.to_dict("records");print(name,t.to_string(index=False))
(OUT/"queries.sql").write_text("\n\n".join("-- "+name+"\n"+q+";" for name,q in queries.items()))
# Model selection with transformations fitted only within training folds.
numeric=["FlightNumber","PayloadMass","Flights","GridFins","Reused","Legs","Block","ReusedCount"]
categorical=["Orbit","LaunchSite","LandingPad","Serial"]
X=df[numeric+categorical].copy();y=df.Class.astype(int)
for col in categorical:X[col]=X[col].fillna("Missing").astype(str)
for col in numeric:X[col]=pd.to_numeric(X[col],errors="coerce")
pre=ColumnTransformer([("num",Pipeline([("fill",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=2,stratify=y)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
specs=[
("Logistic regression",LogisticRegression(max_iter=2000,solver="liblinear",random_state=42),{"model__C":[.01,.1,1,10,100]}),
("SVM",SVC(),{"model__kernel":["linear","rbf","sigmoid"],"model__C":[.1,1,10],"model__gamma":["scale",.01,.1]}),
("Decision tree",DecisionTreeClassifier(random_state=42),{"model__criterion":["gini","entropy"],"model__max_depth":[2,3,4,6],"model__min_samples_leaf":[1,2,4]}),
("KNN",KNeighborsClassifier(),{"model__n_neighbors":[3,5,7,9],"model__p":[1,2]})]
models={};rows=[]
for name,est,grid in specs:
 gs=GridSearchCV(Pipeline([("pre",pre),("model",est)]),grid,cv=cv,scoring="accuracy",n_jobs=1)
 gs.fit(Xtr,ytr);models[name]=gs
 rows.append({"Model":name,"CV accuracy":gs.best_score_,"Test accuracy":gs.score(Xte,yte),"Balanced accuracy":balanced_accuracy_score(yte,gs.predict(Xte)),"Parameters":str(gs.best_params_)})
results=pd.DataFrame(rows);results.to_csv(OUT/"model_results.csv",index=False)
best=max(models,key=lambda k:models[k].best_score_);pred=models[best].predict(Xte)
cm=confusion_matrix(yte,pred,labels=[0,1])
ConfusionMatrixDisplay(cm,display_labels=["No landing","Landing"]).plot(cmap="Blues",colorbar=False);plt.title(best+" holdout confusion matrix");save("confusion")
fig,ax=plt.subplots(figsize=(10,4.5));results.set_index("Model")[["CV accuracy","Test accuracy"]].plot.bar(ax=ax,color=["#167B9D","#148C7D"]);ax.set(ylim=(0,1.1),ylabel="Accuracy");plt.xticks(rotation=0);save("models")
# A genuine future-launch sensitivity check uses an earlier/later chronological split.
from sklearn.base import clone
chron=clone(models[best].best_estimator_);chron.fit(X.iloc[:72],y.iloc[:72])
chron_acc=chron.score(X.iloc[72:],y.iloc[72:])
summary={"n":len(df),"successes":int(y.sum()),"success_rate":float(y.mean()),"test_n":len(yte),"best_model":best,"best_cv":float(models[best].best_score_),"best_test":float(models[best].score(Xte,yte)),"majority_test":float((yte==ytr.mode()[0]).mean()),"chronological_accuracy":float(chron_acc),"chronological_majority":float((y.iloc[72:]==y.iloc[:72].mode()[0]).mean()),"confusion_matrix":cm.tolist(),"wiki_http_status":wiki_status,"sql_n":len(sql),"geo_n":len(geo),"feature_count":int(pd.get_dummies(df[numeric+categorical],columns=categorical).shape[1]),"site_rates":site.reset_index().to_dict("records"),"year_rates":year.reset_index().to_dict("records")}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));(OUT/"sql_results.json").write_text(json.dumps(qresults,indent=2,default=str))
print("MODEL RESULTS\n",results.to_string(index=False));print("SUMMARY",json.dumps(summary))
# Folium marker clusters, outcome colors and an explicit proximity example.
m=folium.Map(location=[29.5,-90],zoom_start=4,tiles="OpenStreetMap")
sites=geo.groupby("Launch Site")[["Lat","Long"]].first()
for name,r in sites.iterrows():
 folium.Circle([r.Lat,r.Long],radius=900,color="#157A9C",fill=True,popup=name).add_to(m)
 folium.Marker([r.Lat,r.Long],tooltip=name).add_to(m)
cluster=MarkerCluster(name="Launch outcomes").add_to(m)
for _,r in geo.iterrows():
 folium.Marker([r.Lat,r.Long],popup=str(r["Launch Site"])+" / outcome "+str(r["class"]),icon=folium.Icon(color="green" if r["class"]==1 else "red",icon="info-sign")).add_to(cluster)
MousePosition().add_to(m);folium.LayerControl().add_to(m)
m.save(str(OUT/"launch_map.html"))
# Coordinates for proximity are explicit representative points, not an authoritative nearest-feature GIS query.
from math import radians,sin,cos,atan2,sqrt
def distance(a,b):
 la1,lo1,la2,lo2=map(radians,[*a,*b]);dlat=la2-la1;dlon=lo2-lo1;t=sin(dlat/2)**2+cos(la1)*cos(la2)*sin(dlon/2)**2
 return 6371*2*atan2(sqrt(t),sqrt(1-t))
start=[28.563197,-80.576820]
points={"Coastline sample":[28.563197,-80.5677],"Road sample":[28.5621,-80.5708],"Northwest reference point":[28.5720,-80.5850],"Cape Canaveral city reference":[28.3922,-80.6077]}
proximity={}
pmap=folium.Map(location=start,zoom_start=14,tiles="OpenStreetMap")
folium.Marker(start,popup="CCAFS SLC-40 reference").add_to(pmap)
for label,point in points.items():
 d=distance(start,point);proximity[label]=d;folium.Marker(point,popup=f"{label}: {d:.2f} km").add_to(pmap);folium.PolyLine([start,point],tooltip=f"{d:.2f} km").add_to(pmap)
pmap.save(str(OUT/"proximity_map.html"));(OUT/"proximity.json").write_text(json.dumps(proximity,indent=2))
# Dashboard figure exports, also used by the live Dash callbacks.
import plotly.express as px
def success_pie(site="ALL",lo=0,hi=10000):
 d=geo[geo["Payload Mass (kg)"].between(lo,hi)]
 if site=="ALL":
  s=d.groupby("Launch Site")["class"].sum().reset_index()
  return px.pie(s,values="class",names="Launch Site",title="Successful launches by site")
 d=d[d["Launch Site"]==site];s=d["class"].value_counts().rename_axis("Outcome").reset_index(name="Count")
 return px.pie(s,names="Outcome",values="Count",title=site+" launch outcomes")
def payload_scatter(site="ALL",lo=0,hi=10000):
 d=geo[geo["Payload Mass (kg)"].between(lo,hi)].copy()
 if site!="ALL":d=d[d["Launch Site"]==site]
 d["Outcome"]=d["class"].map({0:"Failure",1:"Success"})
 return px.scatter(d,x="Payload Mass (kg)",y="class",color="Booster Version Category",symbol="Outcome",hover_data=["Launch Site"],title="Payload versus launch outcome",labels={"class":"Landing outcome"})
success_pie().write_html(str(OUT/"dashboard_all.html"),include_plotlyjs="cdn")
payload_scatter().write_html(str(OUT/"dashboard_scatter.html"),include_plotlyjs="cdn")
# Static companion figures make all results available in the PDF even offline.
fig,axs=plt.subplots(1,2,figsize=(13,4.5))
s=geo.groupby("Launch Site")["class"].sum();axs[0].pie(s,labels=s.index,autopct="%1.1f%%",textprops={"fontsize":9});axs[0].set_title("Share of successful launches")
sns.scatterplot(data=geo,x="Payload Mass (kg)",y="class",hue="Booster Version Category",ax=axs[1]);axs[1].set_title("Payload and landing outcome");save("dashboard_all")
for chosen in sites.index:
 part=geo[geo["Launch Site"]==chosen];fig,axs=plt.subplots(1,2,figsize=(13,4.5));part["class"].value_counts().plot.pie(ax=axs[0],autopct="%1.1f%%");axs[0].set_title(chosen)
 sns.scatterplot(data=part,x="Payload Mass (kg)",y="class",hue="Booster Version Category",ax=axs[1]);save("dashboard_"+re.sub(r"[^a-zA-Z0-9]","_",chosen))
print("ANALYSIS COMPLETE")
