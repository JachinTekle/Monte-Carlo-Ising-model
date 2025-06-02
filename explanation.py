import streamlit as st

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(r"""
# Monte Carlo Ising Model

### 2D Ising Model

In classical statistical mechanics, we study the behavior of systems with many interacting components.  
The Ising model is a simple yet powerful model used to understand phase transitions — such as the transition between magnetized and non-magnetized states in ferromagnetic materials.  

In most ordinary materials, the associated magnetic dipoles of atoms have a random orientation.  
As a result, there is no overall macroscopic magnetic moment. However, in certain materials such as iron, a magnetic moment emerges due to a preferred alignment of atomic spins.  

This phenomenon, known as **spontaneous magnetization**, arises from interactions between neighboring spins that tend to align in the same direction.  
The Ising model provides a simple theoretical framework to study this behavior.  

---

The Ising model consists of *N* systems (or "spins") located on a lattice, where each spin

$$
\sigma_i \in \{1, -1\}
$$

can be in one of two states: up (+1) or down (–1).  

The **energy** of the system depends on the alignment of neighboring spins and the influence of thermal fluctuations.  

By studying the Ising model — especially in two dimensions — one can gain insight into **phase transitions** and **critical phenomena**, such as the abrupt loss of magnetization at a certain critical temperature.
""")