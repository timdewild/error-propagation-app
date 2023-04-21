import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


@st.cache_data()
def get_values():
    x = np.linspace(0, 10, 100)
    a = 0.5
    x_min, x_max = 0, 10
    y_min, y_max = 0, 1.2
    return x, a, x_min, x_max, y_min, y_max

@st.cache_data()
def get_pdf(mu, sigma):
    x = np.linspace(mu-4*sigma, mu+4*sigma, 100)
    pdf = norm.pdf(x, mu, sigma)
    return x, pdf

@st.cache_data()
def f(x,a):
    return 1 - np.exp(-a*x)

@st.cache_data()
def df(x,a):
    return a*np.exp(-a*x)

@st.cache_data()
def ft(x,x0,a):
    return f(x0,a) + df(x0,a) * (x-x0)

@st.cache_data()
def sigma_y(x,sigma_x,a):
    return df(x,a)*sigma_x

@st.cache_data()
def plot_figure_init():

    # get values
    x, a, x_min, x_max, y_min, y_max = get_values()

    #--- CREATE FIGURE ---#

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)

    gs = fig.add_gridspec(2, 2,  width_ratios=(1, 4), height_ratios=(4, 1),
                        left=0.1, right=0.9, bottom=0.1, top=0.9)

    # Create the Axes.
    ax = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[1, 1], sharex=ax)
    ax_y = fig.add_subplot(gs[0, 0], sharey=ax)

    # hide x and y ticklabels of side plots plot
    ax_x.tick_params(bottom=False, labelbottom = False, left = True, labelleft = False, top = True)
    ax_x.invert_yaxis()

    ax_y.tick_params(bottom=True, labelbottom=False, left=False, labelleft=False, right=True, labelright = False)
    ax_y.invert_xaxis()

    # set x_lim and y_lim main plot
    x_min, x_max, y_min, y_max = 0, 10, 0, 1.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # set x and y labels
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    #--- FIXED CONTENT ---#

    #--- main axis ---#

    # f(x)
    ax.plot(x,f(x,a), c = 'k')

    # plot taylor expansion f(x)
    tay, = ax.plot([],[], c='darkgrey', linestyle='dotted')

    # vertical lines 
    ux, = ax.plot([],[], linestyle='solid', color = 'tab:blue', lw=1)
    ux_m, = ax.plot([],[], linestyle='--', color = 'tab:blue', lw=1)
    ux_p, = ax.plot([],[], linestyle='--', color = 'tab:blue', lw=1)

    # horizontal lines
    uy, = ax.plot([],[], linestyle='solid', color = 'tab:red', lw=1)
    uy_m, = ax.plot([],[], linestyle='--', color = 'tab:red', lw=1)
    uy_p, = ax.plot([],[], linestyle='--', color = 'tab:red', lw=1)

    #--- x axis ---#
    pdf_x, = ax_x.plot([],[], color='tab:blue')

    #--- y axis ---#
    pdf_y, = ax_y.plot([],[], color='tab:red')

    axes = [ax, ax_x, ax_y]
    vlines = [ux, ux_m, ux_p, uy, uy_m, uy_p]
    curves = [tay, pdf_x, pdf_y]

    return fig, axes, vlines, curves

def plot_figure_update(mu, sigma, axes, vlines, curves):

    # unpack variables 
    ax, ax_x, ax_y = axes
    ux, ux_m, ux_p, uy, uy_m, uy_p = vlines
    tay, pdf_x, pdf_y = curves

    # obtain values
    x, a, x_min, x_max, y_min, y_max = get_values()

    # set y_lim ax_x
    ax_x.set_ylim(1.2/np.sqrt(2*np.pi*sigma**2), 0)

    # set x_lim ax_y
    ax_y.set_xlim(1.2/np.sqrt(2*np.pi*sigma_y(mu, sigma, a)**2), 0)

    # plot taylor expansion x,ft(x, mu, a)
    tay.set_data(x,ft(x, mu, a))

    # x axis pdf
    X, Y = get_pdf(mu,sigma)
    pdf_x.set_data(X,Y)

    # y axis pdf
    XX, YY = get_pdf(f(mu,a), sigma_y(mu, sigma, a))
    pdf_y.set_data(YY,XX)

    # vertical lines
    ux.set_data([mu, mu], [y_min, ft(mu, mu, a)])
    ux_m.set_data([mu-sigma, mu-sigma], [y_min, ft(mu-sigma, mu, a)])
    ux_p.set_data([mu+sigma, mu+sigma], [y_min, ft(mu+sigma, mu, a)])

    # horizontal lines 
    uy.set_data([x_min, mu], [ft(mu, mu, a), ft(mu, mu, a)])
    uy_m.set_data([x_min, mu-sigma], [ft(mu-sigma, mu, a), ft(mu-sigma, mu, a)])
    uy_p.set_data([x_min, mu+sigma], [ft(mu+sigma, mu, a), ft(mu+sigma, mu, a)])

def plotly_test_section():
    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, 5, 0.1):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=np.arange(0, 10, 0.001),
                y=np.sin(step * np.arange(0, 10, 0.01))))

    # Make 10th trace visible
    fig.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    return fig


def run(): 
    # # x, a = get_values()

    # # st.write("""
    # # Some LaTeX text $y\equiv f(x)$
    # # """)

    mu = st.slider(
        "mu", value=float(5), min_value=float(1), max_value=float(7), step=float(0.1)
    )
    sigma = st.slider(
        "sigma",
        value=float(0.3),
        min_value=float(0.1),
        max_value=float(0.5),
        step=float(0.05),
    )

    # y = get_pdf(x, mu, sigma)
    
    fig, axes, vlines, curves = plot_figure_init()
    
    plot_figure_update(mu, sigma, axes, vlines, curves)

    st.pyplot(fig)
    # # fig = plotly_test_section()
    # # st.plotly_chart(fig)

    # a = st.slider(
    #     "a", value=float(1), min_value=float(1), max_value=float(3), step=float(0.5)
    #     )

    # x = np.linspace(0,2,50)

    # fig = go.Figure(data=go.Scatter(x=x, y=a*x))
    # fig.update_layout(yaxis_range=[0,6])

    # st.plotly_chart(fig)
    
    


run()
