import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

def plot_ichimoku_cloud(df, n):
    # Set colours for up and down candles
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'
    # create list to hold dictionary with data for our first series to plot
    # (which is the candlestick element itself)
    data = [dict(
        type='candlestick',
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        x=df.index,
        yaxis='y2',
        name='Ichimoku-cloud',
        increasing=dict(line=dict(color=INCREASING_COLOR)),
        decreasing=dict(line=dict(color=DECREASING_COLOR)),
    )]

    # Create empty dictionary for later use to hold settings and layout options
    layout = dict(title = 'KumoCloud', xaxis = dict(autoscale=True, type='date'), yaxis = dict(autoscale=True))

    # create our main chart "Figure" object which consists of data to plot and layout settings
    fig = dict(data=data, layout=layout)

    # Assign various seeting and choices - background colour, range selector etc
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True))
    fig['layout']['yaxis'] = dict(domain=[0, 0.2], showticklabels=False)
    fig['layout']['yaxis2'] = dict(domain=[0.2, 0.8])
    fig['layout']['legend'] = dict(orientation='h', y=0.9, x=0.3, yanchor='bottom')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

    # Populate the "rangeselector" object with necessary settings
    rangeselector = dict(
        visible=True,
        x=0, y=0.9,
        bgcolor='rgba(150, 200, 250, 0.4)',
        font=dict(size=13),
        buttons=list([
            dict(count=1,
                 label='reset',
                 step='all'),
            dict(count=1,
                 label='1yr',
                 step='year',
                 stepmode='backward'),
            dict(count=3,
                 label='3 mo',
                 step='month',
                 stepmode='backward'),
            dict(count=1,
                 label='1 mo',
                 step='month',
                 stepmode='backward'),
            dict(step='all')
        ]))

    fig['layout']['xaxis']['rangeselector'] = rangeselector

    # Append the Ichimoku elements to the plot
    fig['data'].append(dict(x=df.index, y=df['KUMO_TK_%s' % str(n)], type='scatter', mode='lines',
                            line=dict(width=1),
                            marker=dict(color='#33BDFF'),
                            yaxis='y2', name='conv_line'))

    fig['data'].append(dict(x=df.index, y=df['KUMO_KJ_%s' % str(n)], type='scatter', mode='lines',
                            line=dict(width=1),
                            marker=dict(color='#F1F316'),
                            yaxis='y2', name='base_line'))

    fig['data'].append(dict(x=df.index, y=df['KUMO_SKA_%s' % str(n)].shift(n), type='scatter', mode='lines',
                            line=dict(width=1),
                            marker=dict(color='#228B22'),
                            yaxis='y2', name='senkou_span_a'))

    fig['data'].append(dict(x=df.index, y=df['KUMO_SKB_%s' % str(n)].shift(n), type='scatter', mode='lines',
                            line=dict(width=1), fill='tonexty',
                            marker=dict(color='#FF3342'),
                            yaxis='y2', name='senkou_span_b'))

    fig['data'].append(dict(x=df.index, y=df['close'].shift(-n), type='scatter', mode='lines',
                            line=dict(width=1),
                            marker=dict(color='#D105F5'),
                            yaxis='y2', name='chikou_span'))
    # Set colour list for candlesticks
    colors = []
    for i in range(len(df.close)):
        if i != 0:
            if df.close[i] > df.close[i - 1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    iplot(fig, filename='candlestick-ichimoku')


def plot_ohlc_with_indicators(df, ind_fields=[]):
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'
    data = []
    trace1 = go.Candlestick(
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        x=df.index,
        yaxis='y',
        name='K-line',
        increasing=dict(line=dict(color=INCREASING_COLOR)),
        decreasing=dict(line=dict(color=DECREASING_COLOR)),
    )
    data.append(trace1)
    for ind in ind_fields:
        trace = go.Scatter(x=list(df.index),
                           y=list(df[ind]))
        data.append(trace)

    layout = dict(
        title='Time series with range slider and selectors',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='YTD',
                         step='year',
                         stepmode='todate'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )

    fig = go.FigureWidget(data=data, layout=layout)
    return fig