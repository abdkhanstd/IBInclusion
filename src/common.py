"""Shared estimation and inference helpers.

All linear models are estimated by OLS after absorbing fixed effects and controls with the
Frisch-Waugh-Lovell theorem: the outcome and the regressor of interest are residualised on a design
matrix of fixed-effect dummies and controls, and the coefficient of interest is the slope of the
residualised outcome on the residualised regressor. Inference uses

  * CR1 cluster-robust standard errors (clusters = countries) with G-1 degrees of freedom;
  * CR3 (cluster jackknife) standard errors, re-estimating the model without each cluster;
  * the wild cluster restricted bootstrap-t (WCR, Webb six-point weights): p-values for H0 beta = b0 and
    confidence intervals by test inversion, so p-values and intervals agree; equivalence tests use the
    two one-sided WCR tests at the equivalence bounds;
  * randomisation inference for first-difference designs (permuting the treatment change);
  * Holm step-down adjustment within outcome families.
"""
import numpy as np
import pandas as pd
from scipy import stats

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
B_DEFAULT = 99999


def design(df, fe=(), controls=()):
    """Dummy matrix for the fixed effects (drop-first per factor) plus controls and an intercept."""
    parts = [np.ones((len(df), 1))]
    names = ["const"]
    for f in fe:
        if isinstance(f, (list, tuple)):
            key = df[list(f)].astype(object).fillna("NA").astype(str).apply(lambda r: "|".join(r.values), axis=1)
            label = "x".join(f)
        else:
            key = df[f].astype(object).fillna("NA").astype(str)
            label = f
        d = pd.get_dummies(key, drop_first=True, dtype=float)
        if d.shape[1]:
            parts.append(d.values)
            names += [f"{label}={c}" for c in d.columns]
    for c in controls:
        parts.append(df[[c]].values.astype(float))
        names.append(c)
    return np.hstack(parts), names


def residualise(v, Z):
    coef, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ coef


def rank(Z):
    return np.linalg.matrix_rank(Z)


class FWLFit:
    """Scalar coefficient of x on y after absorbing fixed effects and controls, with cluster inference."""

    def __init__(self, df, y, x, fe=(), controls=(), cluster="iso3", scale=1.0):
        d = df.dropna(subset=[y, x] + list(controls)).reset_index(drop=True)
        self.df, self.y, self.x, self.fe, self.controls = d, y, x, fe, list(controls)
        self.n = len(d)
        Z, _ = design(d, fe, controls)
        self.k = rank(Z) + 1
        self.yt = residualise(d[y].values.astype(float), Z)
        self.xt = residualise(d[x].values.astype(float), Z)
        self.cluster = cluster
        self.groups = pd.factorize(d[cluster])[0]
        self.G = int(self.groups.max() + 1)
        self.scale = scale
        sxx = self.xt @ self.xt
        self.beta = (self.xt @ self.yt) / sxx
        self.resid = self.yt - self.xt * self.beta
        self.adj = (self.G / (self.G - 1)) * ((self.n - 1) / max(self.n - self.k, 1))
        sc = np.bincount(self.groups, weights=self.xt * self.resid, minlength=self.G)
        self.se = np.sqrt(self.adj * (sc @ sc)) / sxx
        self.t = self.beta / self.se if self.se > 0 else np.nan
        self.p = 2 * stats.t.sf(abs(self.t), self.G - 1)
        self.A = np.bincount(self.groups, weights=self.xt * self.yt, minlength=self.G)
        self.C = np.bincount(self.groups, weights=self.xt * self.xt, minlength=self.G)

    # ---------------------------------------------------------------- analytic and jackknife
    def ci(self, level=0.95):
        q = stats.t.ppf(0.5 + level / 2, self.G - 1)
        return self.beta - q * self.se, self.beta + q * self.se

    def jackknife(self):
        """CR3: re-estimate without each cluster (fixed effects re-absorbed)."""
        bs = []
        for g in np.unique(self.df[self.cluster]):
            sub = self.df[self.df[self.cluster] != g]
            try:
                f = FWLFit(sub, self.y, self.x, self.fe, self.controls, self.cluster)
                bs.append(f.beta)
            except Exception:
                continue
        bs = np.array(bs)
        G = len(bs)
        se = np.sqrt((G - 1) / G * np.sum((bs - bs.mean()) ** 2))
        p = 2 * stats.t.sf(abs(self.beta / se), self.G - 1) if se > 0 else np.nan
        return se, p

    # ---------------------------------------------------------------- wild cluster restricted bootstrap
    def weights(self, B=B_DEFAULT, seed=0):
        rng = np.random.default_rng(seed)
        return WEBB[rng.integers(0, 6, size=(B, self.G))]

    def _moments(self, wG):
        """Bootstrap score sums are linear in the null value: S(b0) = P - b0 Q with P = w*A - a*C and
        Q = w*C - c*C, so the bootstrap t only needs a, c and the three quadratic forms of P and Q per draw."""
        if getattr(self, "_mkey", None) is wG:
            return self._mom
        sxx = self.C.sum()
        a = wG @ self.A / sxx
        c = wG @ self.C / sxx
        P = wG * self.A[None, :] - a[:, None] * self.C[None, :]
        Q = wG * self.C[None, :] - c[:, None] * self.C[None, :]
        self._mom = (a, c, np.sum(P * P, axis=1), np.sum(P * Q, axis=1), np.sum(Q * Q, axis=1), sxx)
        self._mkey = wG
        return self._mom

    def _tstar(self, beta0, wG):
        a, c, pp, pq, qq, sxx = self._moments(wG)
        d = a - beta0 * c
        ss = np.clip(pp - 2 * beta0 * pq + beta0 ** 2 * qq, 0, None)
        se_star = np.sqrt(self.adj * ss) / sxx
        with np.errstate(divide="ignore", invalid="ignore"):
            return d / se_star

    def wcr_p(self, beta0, wG, side="two"):
        """WCR bootstrap p-value for H0: beta = beta0 (side: 'two', 'greater' for H1 beta > beta0, 'less')."""
        ts = self._tstar(beta0, wG)
        t0 = (self.beta - beta0) / self.se
        if side == "two":
            return float(np.mean(np.abs(ts) >= abs(t0)))
        if side == "greater":
            return float(np.mean(ts >= t0))
        return float(np.mean(ts <= t0))

    def wcr_ci(self, wG, level=0.95):
        alpha = 1 - level

        def bound(direction):
            step = self.se
            inner, outer = self.beta, self.beta + direction * step
            it = 0
            while self.wcr_p(outer, wG) >= alpha and it < 60:
                inner = outer
                step *= 1.6
                outer = self.beta + direction * step
                it += 1
            for _ in range(40):
                mid = 0.5 * (inner + outer)
                if self.wcr_p(mid, wG) >= alpha:
                    inner = mid
                else:
                    outer = mid
            return 0.5 * (inner + outer)
        return bound(-1), bound(+1)

    def wcr_set(self, wG, level=0.95, width=12.0, npts=241):
        """Bootstrap confidence set by test inversion on a grid of null values (beta +/- width CR1 SEs), refined by
        bisection at each crossing. Returns a list of (lo, hi) segments; more than one segment means the set is
        not convex. The grid is widened until the set is closed inside it."""
        alpha = 1 - level
        for _ in range(4):
            grid = self.beta + np.linspace(-width, width, npts) * self.se
            ok = np.array([self.wcr_p(b0, wG) >= alpha for b0 in grid])
            if not ok[0] and not ok[-1]:
                break
            width *= 2
        segs = []
        i = 0
        while i < npts:
            if ok[i]:
                j = i
                while j + 1 < npts and ok[j + 1]:
                    j += 1
                lo = self._cross(grid[i - 1], grid[i], wG, alpha) if i > 0 else grid[i]
                hi = self._cross(grid[j + 1], grid[j], wG, alpha) if j + 1 < npts else grid[j]
                segs.append((lo, hi))
                i = j + 1
            else:
                i += 1
        return segs

    def _cross(self, out_pt, in_pt, wG, alpha):
        for _ in range(30):
            mid = 0.5 * (out_pt + in_pt)
            if self.wcr_p(mid, wG) >= alpha:
                in_pt = mid
            else:
                out_pt = mid
        return 0.5 * (out_pt + in_pt)

    def g_star(self):
        """Effective number of clusters (Carter, Schnepel and Steigerwald 2017) with zero intra-cluster
        correlation: G / (1 + CV^2 of the cluster leverages C_g)."""
        c = self.C
        gam = np.mean((c - c.mean()) ** 2) / c.mean() ** 2
        return float(self.G / (1 + gam))

    def cr2(self):
        """CR2 (bias-reduced linearisation) standard error with Bell-McCaffrey / Satterthwaite degrees of freedom
        (Pustejovsky and Tipton 2018). Returns (se, df, p)."""
        d = self.df
        Z, _ = design(d, self.fe, self.controls)
        X = np.column_stack([Z, d[self.x].values.astype(float)])
        Xp = np.linalg.pinv(X)
        H = X @ Xp
        n = self.n
        M = np.eye(n) - H
        sxx = self.xt @ self.xt
        Om = np.zeros((n, self.G))
        V = 0.0
        for g in range(self.G):
            idx = np.where(self.groups == g)[0]
            Hgg = H[np.ix_(idx, idx)]
            w, U = np.linalg.eigh(np.eye(len(idx)) - Hgg)
            inv = np.where(w > 1e-10, 1 / np.sqrt(np.clip(w, 1e-10, None)), 0.0)
            A = U @ np.diag(inv) @ U.T
            a = A @ self.xt[idx] / sxx
            V += float(a @ self.resid[idx]) ** 2
            e = np.zeros(n)
            e[idx] = a
            Om[:, g] = M.T @ e
        S = Om.T @ Om
        df = np.trace(S) ** 2 / np.trace(S @ S)
        se = np.sqrt(V)
        p = 2 * stats.t.sf(abs(self.beta / se), df) if se > 0 else np.nan
        return float(se), float(df), float(p)

    def wild(self, B=B_DEFAULT, seed=0, level=0.95):
        wG = self.weights(B, seed)
        lo, hi = self.wcr_ci(wG, level)
        return dict(p_wcr=self.wcr_p(0.0, wG), ci_lo=lo, ci_hi=hi)

    def equivalence(self, bound, wG):
        """Two one-sided WCR tests of H0: beta <= -bound and H0: beta >= +bound (bound on the model scale)."""
        p_low = self.wcr_p(-bound, wG, side="greater")
        p_high = self.wcr_p(bound, wG, side="less")
        return float(max(p_low, p_high))

    # ---------------------------------------------------------------- summary
    def summary(self, name="", B=B_DEFAULT, seed=0, bound=None, jack=True, ci90=False, extra_b0=(), sets=True,
                cr2=False):
        lo, hi = self.ci()
        Cg = self.C
        top = int(np.argmax(Cg))
        out = dict(name=name, beta=self.beta * self.scale, se=self.se * self.scale, t=self.t, p=self.p,
                   ci_lo=lo * self.scale, ci_hi=hi * self.scale, n=self.n, G=int(self.G),
                   lev_max=float(Cg[top] / Cg.sum()),
                   lev_top=str(self.df.iloc[np.where(self.groups == top)[0][0]][self.cluster]),
                   mde80=2.8 * self.se * self.scale, g_star=self.g_star())
        if B:
            wG = self.weights(B, seed)
            out["p_wcr"] = self.wcr_p(0.0, wG)
            if sets:
                s95 = self.wcr_set(wG, 0.95)
                out.update(wb_lo=min(a for a, _ in s95) * self.scale, wb_hi=max(b for _, b in s95) * self.scale,
                           wb_nseg=len(s95),
                           wb_set=" U ".join(f"[{a * self.scale:.2f}, {b * self.scale:.2f}]" for a, b in s95))
                out["mde80_wcr"] = 2.8 * (out["wb_hi"] - out["wb_lo"]) / 3.92
            else:
                l95, h95 = self.wcr_ci(wG, 0.95)
                out.update(wb_lo=l95 * self.scale, wb_hi=h95 * self.scale)
            if ci90 or bound is not None:
                if sets:
                    s90 = self.wcr_set(wG, 0.90)
                    out.update(wb90_lo=min(a for a, _ in s90) * self.scale, wb90_hi=max(b for _, b in s90) * self.scale,
                               wb90_nseg=len(s90))
                else:
                    l90, h90 = self.wcr_ci(wG, 0.90)
                    out.update(wb90_lo=l90 * self.scale, wb90_hi=h90 * self.scale)
            if bound is not None:
                out["p_equiv"] = self.equivalence(bound / self.scale, wG)
                out["bound"] = bound
            for b0 in extra_b0:
                out[f"p_wcr_b{b0:g}"] = self.wcr_p(b0 / self.scale, wG)
        if jack:
            se3, p3 = self.jackknife()
            out.update(se_cr3=se3 * self.scale, p_cr3=p3)
        if cr2:
            se2, df2, p2 = self.cr2()
            out.update(se_cr2=se2 * self.scale, df_cr2=df2, p_cr2=p2)
        return out


def ri_slope(dy, dx, B=99999, seed=0):
    """Randomisation inference for the OLS slope of dy on dx (with intercept): permute dx across units."""
    rng = np.random.default_rng(seed)
    dy = np.asarray(dy, float)
    dx = np.asarray(dx, float)
    xc = dx - dx.mean()
    yc = dy - dy.mean()
    b0 = (xc @ yc) / (xc @ xc)
    idx = np.argsort(rng.random((B, len(xc))), axis=1)
    bs = xc[idx] @ yc / (xc @ xc)
    return float(b0), float(np.mean(np.abs(bs) >= abs(b0) - 1e-12))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
