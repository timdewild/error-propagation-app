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
    x = np.linspace(mu-4*sigma, mu+4*sigma, 50)
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

    uxx, = ax_x.plot([],[], linestyle='solid', color = 'tab:blue', lw=1, label = '$m_x$')
    uxx_m, = ax_x.plot([],[], linestyle='--', color = 'tab:blue', lw=1, label = '$m_x\pm s_x$')
    uxx_p, = ax_x.plot([],[], linestyle='--', color = 'tab:blue', lw=1)

    ax_x.legend(handles=[uxx, uxx_m])


    #--- y axis ---#
    pdf_y, = ax_y.plot([],[], color='tab:red')

    uyy, = ax_y.plot([],[], linestyle='solid', color = 'tab:red', lw=1, label = '$m_y$')
    uyy_m, = ax_y.plot([],[], linestyle='--', color = 'tab:red', lw=1, label = '$m_y\pm s_y$')
    uyy_p, = ax_y.plot([],[], linestyle='--', color = 'tab:red', lw=1)

    ax_y.legend(handles=[uyy, uyy_m])

    #--- COLLECT GRAPH OBJECTS ---#

    axes = [ax, ax_x, ax_y]
    vlines = [ux, ux_m, ux_p, uy, uy_m, uy_p]
    vplines = [uxx, uxx_m, uxx_p, uyy, uyy_m, uyy_p]
    curves = [tay, pdf_x, pdf_y]

    return fig, axes, vlines, vplines, curves

def plot_figure_update(mu, sigma, axes, vlines, vplines, curves):

    # unpack variables 
    ax, ax_x, ax_y = axes
    ux, ux_m, ux_p, uy, uy_m, uy_p = vlines
    uxx, uxx_m, uxx_p, uyy, uyy_m, uyy_p = vplines
    tay, pdf_x, pdf_y = curves

    # obtain values
    x, a, x_min, x_max, y_min, y_max = get_values()

    pdf_x_max = 1/np.sqrt(2*np.pi*sigma**2)
    pdf_y_max = 1/np.sqrt(2*np.pi*sigma_y(mu, sigma, a)**2)

    # set y_lim ax_x
    ax_x.set_ylim(1.2 * pdf_x_max, 0)

    # set x_lim ax_y
    ax_y.set_xlim(1.2 * pdf_y_max, 0)

    # plot taylor expansion x,ft(x, mu, a)
    tay.set_data(x,ft(x, mu, a))

    # x axis pdf
    X, Y = get_pdf(mu,sigma)
    pdf_x.set_data(X,Y)

    # y axis pdf
    XX, YY = get_pdf(f(mu,a), sigma_y(mu, sigma, a))
    pdf_y.set_data(YY,XX)

    # vertical lines main axis
    ux.set_data([mu, mu], [y_min, ft(mu, mu, a)])
    ux_m.set_data([mu-sigma, mu-sigma], [y_min, ft(mu-sigma, mu, a)])
    ux_p.set_data([mu+sigma, mu+sigma], [y_min, ft(mu+sigma, mu, a)])

    # horizontal lines main axis
    uy.set_data([x_min, mu], [ft(mu, mu, a), ft(mu, mu, a)])
    uy_m.set_data([x_min, mu-sigma], [ft(mu-sigma, mu, a), ft(mu-sigma, mu, a)])
    uy_p.set_data([x_min, mu+sigma], [ft(mu+sigma, mu, a), ft(mu+sigma, mu, a)])

    # vertical lines x axis
    uxx.set_data([mu, mu], [0, pdf_x_max])
    uxx_m.set_data([mu-sigma, mu-sigma], [0, 0.6065*pdf_x_max])
    uxx_p.set_data([mu+sigma, mu+sigma], [0, 0.6065*pdf_x_max])

    # vertical lines x axis
    uyy.set_data([0, pdf_y_max], [ft(mu, mu, a), ft(mu, mu, a)])
    uyy_m.set_data([0, 0.6065*pdf_y_max], [ft(mu-sigma, mu, a), ft(mu-sigma, mu, a)])
    uyy_p.set_data([0, 0.6065*pdf_y_max], [ft(mu+sigma, mu, a), ft(mu+sigma, mu, a)])
  




def run(): 
    # # x, a = get_values()

    # # st.write("""
    # # Some LaTeX text $y\equiv f(x)$
    # # """)

    col1, col2 = st.columns(2)

    with col1:
        mu = st.slider(
            "mean estimate $m_x$", value=float(5), min_value=float(1), max_value=float(7), step=float(0.1)
        )
    with col2: 
        sigma = st.slider(
            "error estimate $s_x$",
            value=float(0.3),
            min_value=float(0.1),
            max_value=float(0.5),
            step=float(0.05),
        )

    # y = get_pdf(x, mu, sigma)
    
    fig, axes, vlines, vplines, curves = plot_figure_init()
    
    plot_figure_update(mu, sigma, axes, vlines, vplines, curves)

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
