import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output

from fbprophet import Prophet
from datetime import datetime, timedelta

# gobals
step_num = 100

# Load data
df = pd.read_csv('data/stockdata2.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df['Date'])


# Initialize the app
app = dash.Dash(__name__,
                assets_folder = 'assets')

app.config.suppress_callback_exceptions = True


# Do hard coded data prep - #TODO move to seperate file
df_sub = df.copy()
stock = "AAPL"
x=df_sub[df_sub['stock'] == stock].index,
y=df_sub[df_sub['stock'] == stock][['Date','value']]
y.columns = ["ds","y"]
y["ds"] = pd.to_datetime(y["ds"])


def simulate_outlier_detection(
    data = None,
    delta_test = 3,
delta_train = 28,
curr_date = pd.to_datetime("2010-01-01")
):

    y = data.copy()
    # generate dates
    maxcut_date = curr_date - timedelta(days=delta_test)
    mincut_date = curr_date - timedelta(days=delta_train)

    # generate train data
    ys = y[(y.ds <= maxcut_date) & (y.ds >= mincut_date)]

    # generate pred dates
    future_dates = pd.DataFrame({"ds":[curr_date - timedelta(days=i) for i in range(delta_test)]})
    future_dates.sort_values("ds",inplace=True)
    future_dates.index = future_dates.ds
    pred_dates = pd.concat([ys[["ds"]],future_dates],axis=0)



    model1=Prophet(interval_width=0.95) 
    m = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.1)

    m.fit(ys)
    #future = m.make_future_dataframe(periods=delta_test)
    forecast = m.predict(pred_dates)

    # find outlier
    dfj =  pd.merge(left=forecast,right=y,on="ds")
    dfj_cut = dfj[(dfj.ds > maxcut_date) & (dfj.ds <= curr_date)]
    df_outlier = dfj_cut[(dfj_cut.y <dfj_cut.yhat_lower) | (dfj_cut.y > dfj_cut.yhat_upper)]

    return df_outlier[["ds","y"]]

date_range = [pd.to_datetime("2010-01-01")+timedelta(days=i) for i in range(7)]
df_outlier = [simulate_outlier_detection(data=y, curr_date = i) for i in date_range]






def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list


                      
app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('DASH - STOCK PRICES'),
                                 html.P('Visualising time series with Plotly - Dash.'),
                                 html.P('Pick one or more stocks from the dropdown below.'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='stockselector', options=get_options(df['stock'].unique()),
                                                      multi=True, value=[df['stock'].sort_values()[0]],
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'}),                                
                                
                                 dcc.Interval(id='auto-stepper',
                                 interval=1*100, # in milliseconds
                                 n_intervals=0
                                    ),
                                 
                                    dcc.Slider(
                                        id = "steper",
                                        min=1,
                                        max=step_num,
                                        value=1
                                    )
                                ]

                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True)
                             ])
                              ])
        ]

)


# Callback for timeseries price
@app.callback(Output('timeseries', 'figure'),
              [Input('stockselector', 'value')])
def update_graph(selected_dropdown_value):
    trace1 = []
    df_sub = df
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df_sub[df_sub['stock'] == stock].index,
                                 y=df_sub[df_sub['stock'] == stock]['value'],
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
              ),

              }

    return figure

@app.callback(
    dash.dependencies.Output('steper', 'value'),
    [dash.dependencies.Input('auto-stepper', 'n_intervals')])
def on_click(n_intervals):
    if n_intervals is None:
        return 0
    else:
        return (n_intervals+1)%step_num

if __name__ == '__main__':
    app.run_server(debug=True)