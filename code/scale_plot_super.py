#!/usr/bin/python
# Programmer : Jiayin Hong
# Created: 22 Feb 2023 10:26:08 AM

#### import dependent modules ####
import numpy as np
import plotly.graph_objects as go

def scale_plot_super(datax, datay, data_label, param_sub, param_linear, plot_title, FLAG_save=False, **kwargs):
    """
    This function takes metrics passed in as parameters and
    makes plot to show how various metrics scale with chromosome size
    **kwargs allows passing a variable number of keyword arguments to the function
    """

    ###### unwrap parameters ######
    sub_slope = param_sub['Slope']
    sub_intercept = param_sub['Intercept']
    sub_r = param_sub['Rsquared']
    sub_p = param_sub['Pvalue']
    if bool(param_linear):
        slope = param_linear['Slope']
        intercept = param_linear['Intercept']
        r = param_linear['Rsquared']
        p = param_linear['Pvalue']

    ###### get line formula ######
    if sub_intercept<0:
        line_sublinear = f'y={sub_slope:.2f}x{sub_intercept:.2f}, R\N{SUPERSCRIPT TWO}={sub_r:.2f}, p={sub_p:.2e}'
    else:
        line_sublinear = f'y={sub_slope:.2f}x+{sub_intercept:.2f}, R\N{SUPERSCRIPT TWO}={sub_r:.2f}, p={sub_p:.2e}'
    
    if bool(param_linear):    
        if intercept<0:
            line_linear = f'y={slope:.2f}x{intercept:.2f}, R\N{SUPERSCRIPT TWO}={r:.2f}, p={p:.2e}'
        else:
            line_linear = f'y={slope:.2f}x+{intercept:.2f}, R\N{SUPERSCRIPT TWO}={r:.2f}, p={p:.2e}'

    ###### data prep ######
    if (datay==0).any():  # to avoid log10(0)
        datay = datay+1
    X_var = np.log10(datax.astype(float))
    Y_var = np.log10(datay.astype(float))
    x_range = np.linspace(X_var.min(), X_var.max(), 100)
    X2 = np.column_stack([np.ones(x_range.shape[0]), x_range])
    y_range_sublinear = np.dot(X2, [sub_intercept, sub_slope])
    if bool(param_linear):
        y_range_linear = np.dot(X2, [intercept, slope])
    
    ###### figure set up & add traces ######
    fig = go.Figure()
    # add trace of scatter data points
    fig.add_traces(data=go.Scatter(x=X_var, y=Y_var, marker_symbol='circle-open',
                                    mode='markers+text',
                                   marker_size=9, text=data_label, textposition='top center',
                                    showlegend=False))

    # add trace of regression fit lines
    fig.add_traces(data=go.Scatter(x=x_range, y=y_range_sublinear, name=line_sublinear, line=dict(color='firebrick',dash='solid')))
    if bool(param_linear):
        fig.add_traces(data=go.Scatter(x=x_range, y=y_range_linear, name=line_linear, line=dict(color='darkturquoise',dash='dash')))
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01,font=dict(family="Courier",size=16,color="black")))
    fig.update_layout(title=plot_title + 'in log-log scale')
    fig.update_layout(autosize=False,width=1000,height=600)
    fig.update_layout(font=dict(size=18))
    # add labels for x & y axes
    if 'xaxis_title' in kwargs.keys():
        fig.update_xaxes(title_text=kwargs['xaxis_title'])
    if 'yaxis_title' in kwargs.keys():
        fig.update_yaxes(title_text=kwargs['yaxis_title'])   
    fig.show()
    if FLAG_save:   # whether or not save the plot
        if 'save_dir' in kwargs.keys(): 
            fig.write_image(kwargs['save_dir'] + plot_title + "in log-log scale.pdf")
        else: # save in default plot directory
            fig.write_image('/Users/jh2313/JupyterProjects/plot/' + plot_title + "in log-log scale.pdf")



