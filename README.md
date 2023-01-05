# Online mixEHR Algorithm

Online LDA proposed by \cite{hoffman2010online} is based on online stochastic optimization and can handily analyze massive document collections, including those arriving in a stream in a more efficient way. Here we extended the algorithm to handle different data categories for the same document (see \figureref{fig:plate} in the Appendix for a graphical representation of the model) to obtain an online version of the mixEHR model. % \citep{li2020inferring}.

%The mixEHR model is a natural extension of the standard LDA model \citep{blei2003latent},  handling different data categories for the same document (see \figureref{fig:plate} in the Appendix for a graphical representation of the model).  Instead of a single $\beta$, we have a set of $\{\beta^1, \dots, \beta^T\}$ (where $T$ stands for the total number of data categories). Each data category has its own word distribution.

We used Variational Bayesian (VB) inference to approximate the true posterior by a simpler distribution $q(z^1,\dots, z^T, \theta, \beta^1,\dots, \beta^T)$. Using similar notations to \cite{hoffman2010online}, the Evidence Lower BOund (ELBO) is as follows:
\begin{small}
\begin{align*}
& \log p(w^1,\dots, w^T|\alpha, \eta^1, \dots, \eta^T) \\
 \geq & \mathcal{L}(w^1,\dots, w^T,\phi^1, \dots,\phi^T,\gamma, \lambda^1, \dots, \lambda^T) \\
  \overset{\Delta}{=} & \mathbf{E}_q[\log p(w^1,\dots, w^T, z^1,\dots, z^T, \theta, \\
  &\qquad \qquad \beta^1, \dots, \beta^T |\alpha, \eta^1, \dots, \eta^T) ] \\
 & \qquad - \mathbf{E}_q[\log q(z^1,\dots, z^t, \theta, \beta^1, \dots, \beta^T)].
\end{align*}
\end{small}

Similarly, we chose a fully factorized distribution $q$ of the form
\begin{small}
\begin{align*}
    q(z_{di}^t = k) &= \phi_{dw_{di}k}^t, \; t = 1, \dots, T; \\
    q(\theta_d) &= \text{Dirichlet}(\theta_d; \gamma_d); \\
    q(\beta_k^t) &= \text{Dirichlet}(\beta_k^t; \lambda_k^t), \; t = 1, \dots, T.
\end{align*}
\end{small}

Thanks to the factorization, the updates of $\mathbf{\phi}$ and $\mathbf{\lambda}$ remain the same. The update of $\gamma$ becomes:
\begin{small}
\begin{equation*}
    \gamma_{dk} = \alpha + \frac{1}{T} \sum_t \sum_w n_{dw}^t \phi_{dwk}^t,
\end{equation*}
\end{small}
which is the average updates among different data categories.

We then obtained an extension of the Algorithm 2 in \cite{hoffman2010online}.
