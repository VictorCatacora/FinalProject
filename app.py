
import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from flask import (
    Flask,
    render_template,
    jsonify,
    request)
from flask_sqlalchemy import SQLAlchemy

connection_string = "postgres:bsb4ever@localhost:5432/Region5"
engine = create_engine(f'postgresql://{connection_string}')

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

kpidaily_nodes = Base.classes.kpidaily_nodes
cmdata_nodes = Base.classes.cmdata_nodes
kpidaily_cell = Base.classes.kpidaily_cell

app = Flask(__name__)

db = SQLAlchemy(app)

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")

@app.route("/kpisite")
def kpisite():
    """Return the sitesheet."""
    return render_template("site.html")

@app.route("/kpicell")
def kpicell():
    """Return the cellsheet."""
    return render_template("cell.html")

@app.route("/sitemaps")
def maps():
    """Return the mapsheet."""
    return render_template("maps.html")

@app.route("/tilt")
def tilt():
    """Return the mapsheet."""
    return render_template("tilt.html")

@app.route("/send")
def tiltresult():
    """Return the mapsheet."""
    return render_template("tiltresult.html")

@app.route("/kpicell/cellspmdata/<siteid>")
def graphskpicell(siteid):
    session = Session(engine)
    kpi_cells_df=pd.read_sql_query('select * from kpidaily_cell', con=engine)
    session.close()
    kpi_cellsbyNode_df=kpi_cells_df.loc[kpi_cells_df["nodebname"] == siteid, :]
    kpi_groupbyNode=kpi_cellsbyNode_df.groupby("cellname")
    kpi_groupbyNode=kpi_groupbyNode.mean()
    kpi_groupbyNode=pd.DataFrame(kpi_groupbyNode, columns=["availability","cs_call_completion","ps_call_completion","throughput_kbps","propagation_mts","quality_ecno","users_total","coverage_rscp","rtwp","traffic","traffic_mb","goodquality","users_data"])
    kpi_groupbyNode=kpi_groupbyNode.reset_index()
    kpi_groupbyNode=kpi_groupbyNode.reset_index()
    kpi_groupbyNode= kpi_groupbyNode.sort_values(by=['cellname'])
    kpi_groupbyNode_json=kpi_groupbyNode.to_json(orient='records')
   
    return kpi_groupbyNode_json

@app.route("/kpicell/pmdata/<cellid>")
def graphskpicellid(cellid):
    session = Session(engine)
    kpi_cells_df=pd.read_sql_query('select * from kpidaily_cell', con=engine)
    session.close()
    kpi_groupbyNode=kpi_cells_df.loc[kpi_cells_df["cellname"] == cellid, :]

    data ={
        "cellname": kpi_groupbyNode.cellname.tolist(),
        "cs_call_completion": kpi_groupbyNode.cs_call_completion.tolist(),
        "ps_call_completion": kpi_groupbyNode.ps_call_completion.tolist(),
        "throughput_kbps": kpi_groupbyNode.throughput_kbps.tolist(),
        "propagation_mts": kpi_groupbyNode.propagation_mts.tolist(),
        "quality_ecno": kpi_groupbyNode.quality_ecno.tolist(),
        "users_total": kpi_groupbyNode.users_total.tolist(),
        "coverage_rscp": kpi_groupbyNode.coverage_rscp.tolist(),
        "rtwp": kpi_groupbyNode.rtwp.tolist(),
        "traffic": kpi_groupbyNode.traffic.tolist(),
        "traffic_mb": kpi_groupbyNode.traffic_mb.tolist(),
        "goodquality": kpi_groupbyNode.goodquality.tolist(),
        "users_data": kpi_groupbyNode.users_data.tolist(),
        "dates": kpi_groupbyNode.dates.tolist(),
    }
   
    return jsonify(data)

@app.route("/kpisite/CData/<siteid>")
def sample_nodes(siteid):
    session = Session(engine)

    """Return a list of all site names"""
    cmdata_nodes_df= pd.read_sql_query('select * from cmdata_nodes', con=engine)
    session.close()
    cmdata_nodes_df=cmdata_nodes_df.loc[:, ["rnc","site", "site_name","cell","azimuth","latitude","longitude","uarfcn","height"]]
    sites_filter = cmdata_nodes_df.loc[cmdata_nodes_df["site"] == siteid, :]
    sites_filter = sites_filter.sort_values("cell")
    sites_json=sites_filter.to_json(orient='records')
   
    return sites_json

@app.route("/kpisite/pmdata/<siteid>")
def graphskpisite(siteid):
    session = Session(engine)
    kpi_nodes_df=pd.read_sql_query('select * from kpidaily_nodes', con=engine)
    session.close()
    kpi_nodes_sitekpi_df=kpi_nodes_df.loc[kpi_nodes_df["nodebname"] == siteid, :]
    data ={
        "dates": kpi_nodes_sitekpi_df.dates.tolist(),
        "cs_call_completion": kpi_nodes_sitekpi_df.cs_call_completion.tolist(),
        "ps_call_completion": kpi_nodes_sitekpi_df.ps_call_completion.tolist(),
        "throughput_kbps": kpi_nodes_sitekpi_df.throughput_kbps.tolist(),
        "propagation_mts": kpi_nodes_sitekpi_df.propagation_mts.tolist(),
        "quality_ecno": kpi_nodes_sitekpi_df.quality_ecno.tolist(),
        "users_total": kpi_nodes_sitekpi_df.users_total.tolist(),
        "coverage_rscp": kpi_nodes_sitekpi_df.coverage_rscp.tolist(),
        "rtwp": kpi_nodes_sitekpi_df.rtwp.tolist(),
        "traffic": kpi_nodes_sitekpi_df.traffic.tolist(),
        "traffic_mb": kpi_nodes_sitekpi_df.traffic_mb.tolist(),
        "goodquality": kpi_nodes_sitekpi_df.goodquality.tolist(),
        "users_data": kpi_nodes_sitekpi_df.users_data.tolist(),
        "nodebname": kpi_nodes_sitekpi_df.nodebname.tolist(),
    }
   
    return jsonify(data)

@app.route("/kpisite/table/<siteid>")
def kpi_nodes(siteid):
    session = Session(engine)

    """Return a list of all site names"""
    kpi_nodes_df=pd.read_sql_query('select * from kpidaily_nodes', con=engine)
    session.close()
    site_kpi = kpi_nodes_df.loc[kpi_nodes_df["nodebname"] == siteid, :]
    data ={
        "dates": site_kpi.dates.tolist(),
        "cs_call_completion": site_kpi.cs_call_completion.tolist(),
        "ps_call_completion": site_kpi.ps_call_completion.tolist(),
        "throughput_kbps": site_kpi.throughput_kbps.tolist(),
        "propagation_mts": site_kpi.propagation_mts.tolist(),
        "quality_ecno": site_kpi.quality_ecno.tolist(),
        "users_total": site_kpi.users_total.tolist(),
        "coverage_rscp": site_kpi.coverage_rscp.tolist(),

    }
    return jsonify(data)

@app.route("/pie")
def rnctraffic():
      """Return `pie data`"""
      session = Session(engine)
      kpi_nodes_df=pd.read_sql_query('select * from kpidaily_nodes', con=engine)
      session.close()
      kpi_nodes_rnc_df=kpi_nodes_df.groupby("rnc")
      kpi_nodes_rnc_df=kpi_nodes_rnc_df.sum()
      kpi_nodes_byrnc_df=pd.DataFrame(kpi_nodes_rnc_df, columns=["traffic","traffic_mb","users_data","users_total"])
      kpi_nodes_byrnc_df=kpi_nodes_byrnc_df.reset_index()

      data = {
        "traffic": kpi_nodes_byrnc_df.traffic.tolist(),
        "rnc": kpi_nodes_byrnc_df.rnc.tolist(),
        "traffic_mb":kpi_nodes_byrnc_df.traffic_mb.tolist(),
        "users_data":kpi_nodes_byrnc_df.users_data.tolist(),
        "users_total":kpi_nodes_byrnc_df.users_total.tolist()
      }
    
      return jsonify(data)

@app.route("/barrnc")
def rncbar():
      """Return bar data`"""
      session = Session(engine)
      kpi_nodes_df=pd.read_sql_query('select * from kpidaily_nodes', con=engine)
      session.close()
      kpi_nodes_rnc_df=kpi_nodes_df.groupby("rnc")
      kpi_nodes_rnc_df=kpi_nodes_rnc_df.mean()
      kpi_nodes_byrnc_df=pd.DataFrame(kpi_nodes_rnc_df, columns=["cs_call_completion","ps_call_completion","throughput_kbps","goodquality"])
      kpi_nodes_byrnc_df=kpi_nodes_byrnc_df.reset_index()

      databarrnc = {
        "cs_call_completion": kpi_nodes_byrnc_df.cs_call_completion.tolist(),
        "rnc": kpi_nodes_byrnc_df.rnc.tolist(),
        "ps_call_completion":kpi_nodes_byrnc_df.ps_call_completion.tolist(),
        "throughput_kbps":kpi_nodes_byrnc_df.throughput_kbps.tolist(),
        "goodquality":kpi_nodes_byrnc_df.goodquality.tolist()
      }
    
      return jsonify(databarrnc)
      

@app.route("/kpisite/site/<rncid>")
def site(rncid):
    # Create our session (link) from Python to the DB
    session = Session(engine)

    """Return a list of all site names"""
    # Query all passengers
    cmdata_nodes_df= pd.read_sql_query('select * from cmdata_nodes', con=engine)
    cmdata_nodes_df=cmdata_nodes_df.loc[:, ["rnc", "site"]]
    sites = cmdata_nodes_df.loc[cmdata_nodes_df["rnc"] == rncid, :]
    sites = sites.sort_values("site")
    site_list=sites['site'].values.tolist()
    all_site = list(dict.fromkeys(site_list))
    session.close()
    return jsonify(all_site)

@app.route("/kpicell/<siteid>")
def cell(siteid):
    # Create our session (link) from Python to the DB
    session = Session(engine)

    """Return a list of all site names"""
    # Query all passengers
    pmdata_nodes_df= pd.read_sql_query('select * from kpidaily_cell', con=engine)
    pmdata_nodes_df=pmdata_nodes_df.loc[:, ["nodebname", "cellname"]]
    cells = pmdata_nodes_df.loc[pmdata_nodes_df["nodebname"] == siteid, :]
    cells = cells.sort_values("cellname")
    cells_list=cells['cellname'].values.tolist()
    all_cells = list(dict.fromkeys(cells_list))
    session.close()
    return jsonify(all_cells)

@app.route("/topoffender/<rncid>")
def topoffender(rncid):
    # Create our session (link) from Python to the DB
    session = Session(engine)
    kpi_nodes_df=pd.read_sql_query('select * from kpidaily_nodes', con=engine)
    session.close()

    """Return a list of all site names"""
    # Query all passengers
    kpi_nodes_site_df = kpi_nodes_df.loc[kpi_nodes_df["rnc"] == rncid, :]
    kpi_nodes_site_df=kpi_nodes_site_df.groupby("nodebname")
    kpi_nodes_site_df=kpi_nodes_site_df.mean()
    kpi_nodes_site_df=pd.DataFrame(kpi_nodes_site_df, columns=["cs_call_completion","ps_call_completion","throughput_kbps","goodquality"])
    kpi_nodes_site_df=kpi_nodes_site_df.reset_index()
    kpi_topoffender_site_pd=kpi_nodes_site_df.sort_values(by=['throughput_kbps'])
    kpi_topoffender_site_pd=kpi_topoffender_site_pd.head(20)
    kpitop = {
        "cs_call_completion": kpi_topoffender_site_pd.cs_call_completion.tolist(),
        "nodebname": kpi_topoffender_site_pd.nodebname.tolist(),
        "ps_call_completion":kpi_topoffender_site_pd.ps_call_completion.tolist(),
        "throughput_kbps":kpi_topoffender_site_pd.throughput_kbps.tolist(),
        "goodquality":kpi_topoffender_site_pd.goodquality.tolist()
      }
    return jsonify(kpitop)

@app.route("/RNC")
def rnc():
    # Create our session (link) from Python to the DB
    session = Session(engine)

    """Return a list of all rnc names"""
    # Query all passengers
    results = session.query(cmdata_nodes.rnc).all()

    session.close()

    # Convert list of tuples into normal list
    all_rnc = list(np.ravel(results))
    all_rnc = list(dict.fromkeys(all_rnc))
    ##all_rnc=all_rnc.sort()
    return jsonify(all_rnc)

@app.route("/sitename")
def sitename(rncid):
    # Create our session (link) from Python to the DB
    session = Session(engine)

    """Return a list of all site names"""
    # Query all passengers
    cmdata_nodes_df= pd.read_sql_query('select * from cmdata_nodes', con=engine)
    cmdata_nodes_df=cmdata_nodes_df.loc[:, ["rnc", "site_name"]]
    sites = cmdata_nodes_df.loc[cmdata_nodes_df["rnc"] == rncid, :]
    sitename_list=sites['site_name'].values.tolist()
    all_sitename = list(dict.fromkeys(sitename_list))
    session.close()
    return jsonify(all_sitename)



@app.route("/CData")
def cmdata():
    with open('csvjson.json', 'r') as myfile:
        data=myfile.read()

    # session = Session(engine)

    # """Return a list of all site names"""
    # cmdata_nodes_df= pd.read_sql_query('select * from cmdata_nodes', con=engine)
    # session.close()
    # cmdata_nodes_json=cmdata_nodes_df.to_json(orient='records')
   
 
    return data 

@app.route("/send", methods=["GET", "POST"])
def send():

    import pandas as pd

    if request.method == "POST":
        cs_call_completion = request.form["cs_call_completion"]
        ps_call_completion = request.form["ps_call_completion"]
        throughput_kbps = request.form["throughput_kbps"]
        tp_85 = request.form["tp_85%"]
        rtwp = request.form["rtwp"]
        coverage_rscp = request.form["coverage_rscp"]
        quality_ecno = request.form["quality_ecno"]
        goodquality = request.form["goodquality"]
        traffic = request.form["traffic"]
        users_data = request.form["users_data"]
        users_total = request.form["users_total"]
        traffic_load = request.form["traffic_load"]
        height = request.form["height"]
        mechtilt = request.form["mechtilt"]
        hbwidth = request.form["hbwidth"]
        vbwidth = request.form["vbwidth"]
        banda = request.form["inputbanda"]
        cpich = request.form["inputcpich"]
        maxpower = request.form["inputmaxpower"]
        morphology = request.form["inputmorphology"]

        if cs_call_completion =="":
            cs_call_completion="98.5"
        if ps_call_completion =="":
            ps_call_completion = "98.5"
        if throughput_kbps =="":
            throughput_kbps = "1000"
        if tp_85 =="":
            tp_85 = "500"
        if rtwp =="":
            rtwp = "-95.0"
        if coverage_rscp =="":
            coverage_rscp = "-85.0"
        if quality_ecno =="":
            quality_ecno = "-9.0"
        if goodquality =="":
            goodquality = "85"
        if traffic =="":
            traffic = "15.0"
        if users_data =="":
            users_data = "30"
        if users_total =="":
            users_total = "60.0"
        if ps_call_completion =="":
            ps_call_completion = "98.5"
        if traffic_load =="":
            traffic_load = "40"
        if height =="":
            height = "24.0"
        if mechtilt =="":
            mechtilt = "4"
        if hbwidth =="":
            hbwidth = "60.0"
        if vbwidth =="":
            vbwidth = "7.5"
        if banda =="":
            banda = "4413"
        if cpich =="":
            cpich = "330"
        if maxpower =="":
            maxpower = "430"
        if morphology =="":
            morphology = "DU"        

        form_data= {
            "cs_call_completion": [cs_call_completion],
            "ps_call_completion": [ps_call_completion],
            "throughput_kbps": [throughput_kbps],
            "rtwp": [rtwp],
            "traffic": [traffic],
            "quality_ecno": [quality_ecno],
            "goodquality": [goodquality],
            "users_data": [users_data],
            "users_total": [users_total],
            "coverage_rscp": [coverage_rscp],
            "banda":[banda],
            "cpich":[cpich],
            "maxpower":[maxpower],
            "traffic_load": [traffic_load],
            "tp_85%": [tp_85],
            "morphology":[morphology],
            "height": [height],                 
            "mechtilt": [mechtilt],
            "hbwidth": [hbwidth],
            "vbwidth": [vbwidth],
        }

        df = pd.DataFrame(form_data)
        df['cs_call_completion'] = df['cs_call_completion'].astype(float)
        df['ps_call_completion'] = df['ps_call_completion'].astype(float)
        df['throughput_kbps'] = df['throughput_kbps'].astype(float)
        df['rtwp'] = df['rtwp'].astype(float)
        df['traffic'] = df['traffic'].astype(float)
        df['quality_ecno'] = df['quality_ecno'].astype(float)
        df['goodquality'] = df['goodquality'].astype(float)
        df['users_data'] = df['users_data'].astype(float)
        df['users_total'] = df['users_total'].astype(float)
        df['coverage_rscp'] = df['coverage_rscp'].astype(float)
        df['banda'] = df['banda'].astype(float)
        df['cpich'] = df['cpich'].astype(float)
        df['maxpower'] = df['maxpower'].astype(float)
        df['traffic_load'] = df['traffic_load'].astype(float)
        df['tp_85%'] = df['tp_85%'].astype(float)
        df['height'] = df['height'].astype(float)
        df['mechtilt'] = df['mechtilt'].astype(float)
        df['hbwidth'] = df['hbwidth'].astype(float)
        df['vbwidth'] = df['vbwidth'].astype(float)


    import csv
    import requests
    import numpy as np
    from pprint import pprint
    import scipy.stats as stats
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import to_categorical
    
    morphology = df.loc[0,"morphology"]
    inputdata = df.drop(columns=["morphology"])
    y_test = "y_test.csv"
    y_test = pd.read_csv(y_test)
    y_test_urban = "y_test_urban.csv"
    y_test_urban = pd.read_csv(y_test_urban)
    y_test_rural = "y_test_rural.csv"
    y_test_rural = pd.read_csv(y_test_rural)
    if morphology == "DU":
         # update file name with student file
        filename = 'deep_learning_denseurban_v2.sav'
        loaded_model = joblib.load(filename)
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)
        encoded_y_test = label_encoder.transform(y_test)
        y_test_categorical = to_categorical(encoded_y_test)
        encoded_predictions = loaded_model.predict_classes(inputdata)
        prediction_labels = label_encoder.inverse_transform(encoded_predictions)

    elif morphology == "UR":
        filename = 'deep_learning_urban_v2.sav'
        loaded_model = joblib.load(filename)
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test_urban)
        encoded_y_test_urban = label_encoder.transform(y_test_urban)
        y_test_urban_categorical = to_categorical(encoded_y_test_urban)
        encoded_predictions = loaded_model.predict_classes(inputdata)
        prediction_labels = label_encoder.inverse_transform(encoded_predictions)
    
    else :
        filename = 'deep_learning_rural_v2.sav'
        loaded_model = joblib.load(filename)
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test_rural)
        encoded_y_test_rural= label_encoder.transform(y_test_rural)
        y_test_rural_categorical = to_categorical(encoded_y_test_rural)
        encoded_predictions = loaded_model.predict_classes(inputdata)
        prediction_labels = label_encoder.inverse_transform(encoded_predictions)

    print(prediction_labels)
    tilt_result =str(prediction_labels[0])


    return render_template("tiltresult.html", tiltcalculator=tilt_result,
    cs_call_completion=cs_call_completion,ps_call_completion=ps_call_completion,
    throughput_kbps=throughput_kbps,tp_85=tp_85,rtwp=rtwp,coverage_rscp=coverage_rscp,
    quality_ecno=quality_ecno, goodquality=goodquality,traffic=traffic,
    users_data=users_data,users_total=users_total,
    traffic_load=traffic_load,height=height,mechtilt=mechtilt,
    hbwidth =hbwidth ,vbwidth=vbwidth,banda=banda,cpich=cpich,
    maxpower=maxpower,morphology=morphology)

if __name__ == "__main__":
    app.run()
