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
    ax.plot(x,f(x,a), c = 'k', label='$f(x)$')

    # plot taylor expansion f(x)
    tay, = ax.plot([],[], c='darkgrey', linestyle='solid', label='tangent line $f_1(x)$')

    ax.legend()

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

    uxx, = ax_x.plot([],[], linestyle='solid', color = 'tab:blue', lw=1, label = '$x^\prime$')
    uxx_m, = ax_x.plot([],[], linestyle='--', color = 'tab:blue', lw=1, label = '$x^\prime\pm \Delta x$')
    uxx_p, = ax_x.plot([],[], linestyle='--', color = 'tab:blue', lw=1)

    ax_x.legend(handles=[uxx, uxx_m])


    #--- y axis ---#
    pdf_y, = ax_y.plot([],[], color='tab:red')

    uyy, = ax_y.plot([],[], linestyle='solid', color = 'tab:red', lw=1, label = '$y^\prime$')
    uyy_m, = ax_y.plot([],[], linestyle='--', color = 'tab:red', lw=1, label = '$y^\prime\pm \Delta y$')
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
    #pdf_x.set_data(X,Y)
    ax_x.fill_between(X,Y, color='tab:blue', alpha = 0.3)

    # y axis pdf
    XX, YY = get_pdf(f(mu,a), sigma_y(mu, sigma, a))
    #pdf_y.set_data(YY,XX)
    ax_y.fill_between(YY,XX, color='tab:red', alpha = 0.3)


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

    with st.sidebar:
        st.write("""
            # Control Panel
            ### Input Parameters
            Specify the measured value and its error here.
        """)
        mu = st.slider(
            "$x^\prime$", 
            value=float(5), 
            min_value=float(1), 
            max_value=float(7), 
            step=float(0.1)
        )

        sigma = st.slider(
            "$\Delta x$",
            value=float(0.3),
            min_value=float(0.1),
            max_value=float(0.5),
            step=float(0.05),
        )
        st.write("""
            ### Function
            The function $f$ used in this example is:

            $$
            f(x) = 1 - e^{-x/2}.
            $$ 
        """)

        st.write("""
            ### Output Parameters
            The derived value and its error are:
        """)
        st.write(f"$y^\prime = {round(f(mu,0.5),3)}$")
        st.write(f"$\Delta y = {round(sigma_y(mu,sigma,0.5),3)}$")

    fig, axes, vlines, vplines, curves = plot_figure_init()
    
    plot_figure_update(mu, sigma, axes, vlines, vplines, curves)
  
    st.pyplot(fig)
    
    st.title('Error Propagation')
    st.write("""
        ### Introduction
        In experiments, the quantity you measure is often not the final quantity that you want to determine. Instead, the latter is related to the former via a function (which is for instance a physical law). 
        The measured qauntity comes with a measurement error, but how does this error *propagate* into the final quantity? 

        More concretely, suppose the quantity $y$ is related to the measured quantity $x$ via $y=f(x)$. Let the measured value of $x$ be denoted by $x'$ and its error by $\Delta x$, what is the error $\Delta y$ in the derived quantity $y'=f(x')$?

        You know the answer: if $\Delta x \ll x'$, we can approximate the error in $y'$ as:

        $$
        \Delta y \simeq \\frac{df}{dx}\\bigg\\vert_{x'}\;\Delta x,
        $$

        where the notation indicates the derivative has to be evaluated at $x=x'$. In the lecture notes (section 2.1), this result is derived using the discretized definition of the derivative. In this application, we wish to provide a graphical interpretation of this result. 

        ### Animation Setup
        We assume that the (limiting) distribution of the measured quantity $x$ is normal, centered around the measured value $x^\prime$ with standard deviation equal to the measurement error $\Delta x$.
        This distribution is shown in light blue in the bottom panel.

        Now, we would like to know how the distribution of the measured quantity, $x$, *propagates* through the function $f(x)$ to produce the distribution of $y$. If the function $f$ is non-linear, finding the distribution of $y$ can be very complex. Therefore, we use the following simplified approach:

        1. We construct a straight line tangent to $f$ at $x'$, see the gray line in the central panel. Mathematically, this line corresponds to the first order Taylor expansion $f_1$ (evaluated at $x^\prime$):

        $$
        f_1(x) = f(x') + \\frac{df}{dx}\\bigg\\vert_{x'} \\times (x-x^\prime).
        $$

        2. Using this tangent line, we can now graphically determine the values $y'\pm \Delta y$, as indicated by the thin dashed lines. Mathematically, this corresponds to evaluating $y'\pm \Delta y$ using the first order Taylor expansion. 
        For example:

        $$
        y'+\Delta y \simeq f_1(x'+\Delta x) = f(x') + \\frac{df}{dx}\\bigg\\vert_{x'} \Delta x.
        $$

        3. The error $\Delta y$ is equal to the vertical separation between the red dashed lines corresponding to $y'\pm \Delta y$ and the red solid line corresponding to $y'$. Mathematically, we can use $y'=f(x')$ in the previous equation to obtain: 

        $$
        \Delta y \simeq \\frac{df}{dx}\\bigg\\vert_{x'}\;\Delta x. 
        $$

        4. We make the simplifying approximation that the distribution of $y$ is also normal, centered around $y'$ with standard deviation equal to $\Delta y$. This distribution is shown in light red to the left of the central panel. 
       

    """)

    

run()
