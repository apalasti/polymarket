# %% [markdown]
# # Specification
#
# In this project I collected data from polymarket (a prediction market allowing people to bet on almost anything), specifically I collected data from 15 minute crypto markets which predict whether the crypto price will go up or down in the next 15 minutes compared to now.
#
# The full project is available [here](https://github.com/apalasti/polymarket), and the data collected can be downloaded from [here](https://drive.google.com/file/d/1cxc8K4J5-sYDDa3o-caFs953byHdf9oT/view?usp=sharing).
#
#
# ## Prediction Markets
#
# Prediction markets are exchange platforms where participants can buy and sell contracts whose payoffs depend on the outcome of uncertain future events (such as elections, sports games, or financial metrics). Each contract is tied to a specific outcome and typically trades at a price between 0 and 1, which can be interpreted as the market’s consensus probability of that outcome occurring.
#
# These markets aggregate information from all traders about the likelihood of various outcomes, often leading to highly accurate forecasts. Traders attempt to profit by buying undervalued contracts (betting for an outcome) or selling overvalued ones (betting against), based on their information or predictions.
#
# When the actual outcome is realized, contracts paying out on the correct outcome settle at their full value (1.0), and all others become worthless (0.0). This mechanism rewards accurate predictions and encourages the flow of information into market prices.
#
# ## Task definition
#
# The main goal of the project is to build a model that will predict the outcome of a **15 minute bitcoin market** based on it's current state. This can also include extending the dataset with new features, uncovering signals, using feature engineering methodologies, etc. to improve the created model.
#
# But in the first phase the goal is to do exploratory data analysis and to answer these interesting questions:
# - How accurate is the crowd in predicting the real outcome?
# - How quickly does the crowd figure out the answer before the 15 minutes are up?
# - Based on the gap between the start and current asset price, what probability does the crowd assign to `UP` and `DOWN` outcomes?
# - When the Current Price suddenly spikes on the crypto exchange, how many seconds or minutes does it take for the Polymarket probability to catch up?

# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from IPython import get_ipython

    _in_ipython = get_ipython() is not None
except (ImportError, NameError):
    _in_ipython = False
if not _in_ipython:

    def _noop_fig_show(_self, *args, **kwargs):
        return None

    go.Figure.show = _noop_fig_show

# %%
df = pd.read_parquet("../data/crypto-prices/all_crypto_price_history.parquet")
df = df[df["slug"].str.startswith("btc")]

df["ob_updated_at"] = pd.to_datetime(df["timestamp"], unit="ms")
df["start_of_market"] = pd.to_datetime(df["slug"].str.split("-").str[-1].astype(int), unit="s")
df.drop(["outcome", "timestamp"], axis=1, inplace=True)

df.info(verbose=True)

# %%
start_date = df["start_of_market"].min()
end_date = df["start_of_market"].max()
days = pd.date_range(start=start_date, end=end_date, freq='D')

markets_per_day = pd.DataFrame(
    df["start_of_market"].unique().floor("D").value_counts().reindex(days, fill_value=0)
)
px.bar(
    x=markets_per_day.index,
    y=markets_per_day["count"],
    labels={"x": "Date", "y": "Market Count"},
    title="Markets per Day",
)

# %% [markdown]
#
# ## Features
# - `slug`: textual identifier of the market
# - `start_of_market`: start of the 15‑minute market, decoded from the slug’s embedded epoch (anchor for `seconds`)
# - `seconds`: Elapsed time into the current 15‑minute market window (seconds since that window’s start)
# - `outcome_price`: we are always **just following the `UP` market** since from that the other side can be derived, this is the final price of the `UP` event, so 0 if the `DOWN` event happened and 1 if `UP` happened
# - `price`: spot price of the underlying crypto at this row’s time
# - `bid_[1-5]_price`: price levels of the **1st–5th** best bids on the contract’s book
# - `bid_[1-5]_size`: quantity available at each corresponding bid level
# - `ask_[1-5]_price`: price levels of the **1st–5th** best asks
# - `ask_[1-5]_size`: quantity available at each corresponding ask level
# - `ob_updated_at`: time of this order book snapshot
#
# Note: markets are from 2025 Oct 15 to 2026 Mar 8, with gaps between them.

# %%

start_prices = df.sort_values("seconds").groupby("start_of_market")["price"].first()
df["price_diff"] = df["price"] - df["start_of_market"].map(start_prices)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Spot price", "Price difference from market start"),
)
fig.add_trace(
    go.Histogram(x=df["price"], nbinsx=100, showlegend=False),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(x=df["price_diff"], nbinsx=100, showlegend=False),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="Spot price", row=1, col=1)
fig.update_xaxes(title_text="Price difference", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_layout(title_text="Price distributions")
fig.show()

# %%

def price_over_time(slug: str):
    df_single_market = df[df["slug"] == slug]
    fig = px.line(
        df_single_market,
        x="seconds",
        y="price",
        title=f"Market Over Time for {slug} ({df_single_market['start_of_market'].iloc[0]})",
        labels={"seconds": "Elapsed Seconds", "price": "Spot Price"},
    )
    fig.add_scatter(
        x=df_single_market["seconds"],
        y=df_single_market["ask_1_price"],
        name="UP Price",
        yaxis="y2",
    )
    fig.update_layout(
        yaxis2=dict(
            title="UP Price",
            overlaying='y',
            side='right'
        ),
        showlegend=False
    )
    return fig

slug = "btc-updown-15m-1770332400"
fig = price_over_time(slug)
fig.show()

# %% [markdown]
# **How accurate is the crowd in predicting the real outcome?**
#
# To answer this question we need to assess the performance of the crowd. We'll use the `ask_1_price` as the crowd's suggested probability of the `UP` event, and predict `UP` if this is greater than 0.5. To evaluate the accuracy of this prediction model, we compare these predictions to the true outcomes and measure how often the crowd's predictions match the actual results.
#
# We will also do the same with a constructed variable `p_up_mid = (bid_1_price + ask_1_price) / 2.0` to see which has more predictive power.

# %%
df["p_up"] = df["ask_1_price"]
df["p_up_mid"] = (df["bid_1_price"] + df["ask_1_price"]) / 2.0

crowd: pd.DataFrame = df.loc[df["p_up"].notna()].copy()
crowd["spread"] = crowd["ask_1_price"] - crowd["bid_1_price"]
crowd["p_up_correct"] = (crowd["p_up"] > 0.5).astype(np.int8) == crowd[
    "outcome_price"
].astype(np.int8)
crowd["p_up_mid_correct"] = (crowd["p_up_mid"] > 0.5).astype(np.int8) == crowd[
    "outcome_price"
].astype(np.int8)

has_both = crowd["bid_1_price"].notna() & crowd["ask_1_price"].notna()
crowd["confident_prob"] = np.nan
crowd.loc[has_both, "confident_prob"] = np.where(
    crowd.loc[has_both, "ask_1_price"] > 0.5,
    crowd.loc[has_both, "ask_1_price"],
    1.0 - crowd.loc[has_both, "bid_1_price"],
)

by_sec = (
    crowd.groupby("seconds", sort=True)
    .agg(
        p_up_acc=("p_up_correct", "mean"),
        p_up_mid_acc=("p_up_mid_correct", "mean"),
        price_mean=("confident_prob", "mean"),
        price_std=("confident_prob", "std"),
    )
    .reset_index()
)

fig = px.line(
    by_sec,
    x="seconds",
    y=["p_up_acc", "p_up_mid_acc", "price_mean"],
    labels={"seconds": "Seconds into 15m window"},
    title="Crowd accuracy vs mean confident probability over time",
)
fig.update_traces(
    yaxis="y2", name="Mean price of more confident contract",
    selector=dict(name="price_mean"),
)
fig.update_traces(
    selector=dict(name="p_up_acc"),
    name="Binary accuracy (ask > 0.5 ⇒ UP)",
)
fig.update_traces(
    selector=dict(name="p_up_mid_acc"),
    name="Binary accuracy (mid > 0.5 ⇒ UP)",
)
fig.add_trace(
    go.Scatter(
        x=by_sec["seconds"],
        y=(by_sec["price_mean"] + by_sec["price_std"]).clip(0, 1),
        yaxis="y2",
        mode="lines",
        line=dict(width=1, color="rgba(44, 160, 44, 0.50)"),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=by_sec["seconds"],
        y=(by_sec["price_mean"] - by_sec["price_std"]).clip(0, 1),
        yaxis="y2",
        mode="lines",
        line=dict(width=1, color="rgba(44, 160, 44, 0.50)"),
        showlegend=False,
        hoverinfo="skip",
        fill="tonexty",
        fillcolor="rgba(44, 160, 44, 0.12)",
    )
)
fig.update_layout(
    yaxis=dict(title="Accuracy", range=[0.4, 1], tickformat=".0%"),
    yaxis2=dict(
        title="Price",
        range=[0.4, 1],
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        title_text="",  # Remove legend title text/variable name
    ),
)
fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7)
fig.show()

# %% [markdown]
# The plot above illustrates how the crowd's accuracy evolves over time. The mean price of the more confident contract is also shown and closely tracks the accuracy curves, indicating that the crowd is generally well-calibrated with little evidence of systematic mispricing. Toward the end of the 15-minute window, there is a small divergence, with the crowd assigning slightly lower probabilities than the actual outcomes, but this deviation is minimal, less than 0.05.
#

# %%
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
crowd["p_bin"] = pd.cut(crowd["p_up"], bins=bin_edges, include_lowest=True)
reliability = (
    crowd.groupby("p_bin", observed=True)
    .agg(mean_pred=("p_up", "mean"), emp_freq=("outcome_price", "mean"), count=("p_up", "size"))
    .reset_index()
)
reliability["bin_str"] = reliability["p_bin"].astype(str)

fig_cal = px.scatter(
    reliability,
    x="mean_pred",
    y="emp_freq",
    size="count",
    hover_data=["count", "mean_pred", "emp_freq", "bin_str"],
    labels={"mean_pred": "Mean Predicted P(UP)", "emp_freq": "Empirical UP Frequency", "count": "Bin Count", "bin_str": "Bin Range"},
)
fig_cal.add_shape(
    type="line",
    x0=0, y0=0, x1=1, y1=1,
    line=dict(color="black", dash="dash"),
    name="perfect calibration"
)
fig_cal.update_layout(
    title="Reliability: mean predicted P(UP) vs empirical UP frequency",
    xaxis_title="Mean predicted P(UP) in bin",
    yaxis_title="Fraction of resolved UP events",
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
    showlegend=False,
)
fig_cal.show()

# %% [markdown]
# The plot above is a **reliability (calibration) curve** for the crowd’s implied probabilities. Observations are sorted into ten bins by `ask_1_price` (our `p_up`), which we treat as the market’s probability of `UP`. For each bin we plot the **mean predicted** probability on the horizontal axis and the **empirical frequency** of resolved `UP` outcomes (`outcome_price` averaged over rows in that bin) on the vertical axis. Marker size reflects how many snapshots fall in the bin.
#
# The dashed diagonal is **perfect calibration**: if the crowd is calibrated, points should lie on this line—e.g. whenever the market averages about 0.6 for `UP`, `UP` should occur about 60% of the time in that bin. In practice the points sit very close to the diagonal across the full [0, 1] range, which means the quoted prices are a trustworthy guide to realized frequencies rather than systematically shifted. Any small departures from the line (e.g. slightly below the diagonal in the mid-probability range) would indicate mild **overconfidence**—predicted probabilities a bit higher than the realized `UP` rate—but the overall picture matches the time-series plot: the crowd is **well calibrated**, with no large systematic bias.

# %% [markdown]
# ## How quickly does the crowd figure out the answer?
#
# We combine two views. First, for each 15m market we define **persistent correct time**: the earliest second such that the binary side implied by `ask_1_price` **never disagrees with** the realized outcome from that second through the **last observed snapshot** in the window. This is when the crowd has "locked in" the winning side for the rest of the data we see. If the last quote is still on the wrong side, we set this time to **900** (end of the window) and record **`censored_at_end`** for that market.
#
# Second, we plot **mean Brier score** $\left((p - y)^2\right)$ versus `seconds`: this curve shows how fast **probability error** decays through the window.

# %%
def _persistent_correct_one_market(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("seconds")
    seconds = g["seconds"].to_numpy(dtype=float)
    pred = (g["ask_1_price"].to_numpy() > 0.5).astype(np.int8)
    y = int(g["outcome_price"].iloc[0])
    wrong = pred != y
    if not wrong.any():
        return pd.Series(
            {"time_persistent_correct": float(seconds[0]), "censored_at_end": False}
        )
    last_wrong = int(np.nonzero(wrong)[0][-1])
    if last_wrong == len(wrong) - 1:
        return pd.Series({"time_persistent_correct": 900.0, "censored_at_end": True})
    return pd.Series(
        {
            "time_persistent_correct": float(seconds[last_wrong + 1]),
            "censored_at_end": False,
        }
    )


_slugs, _groups = zip(*crowd.groupby("slug", sort=False))
market_timing = pd.DataFrame(
    (_persistent_correct_one_market(g) for g in _groups),
    index=pd.Index(_slugs, name="slug"),
)
crowd["brier"] = (crowd["ask_1_price"] - crowd["outcome_price"]) ** 2
by_sec_brier = (
    crowd.groupby("seconds", sort=True)["brier"].mean().reset_index(name="mean_brier")
)

quantile_levels = (0.1, 0.25, 0.5, 0.75, 0.9)
t_quants = market_timing["time_persistent_correct"].quantile(list(quantile_levels))
censored_share = float(market_timing["censored_at_end"].mean())

palette = px.colors.qualitative.Plotly
fig_speed = go.Figure(
    data=[
        go.Scatter(
            x=by_sec_brier["seconds"],
            y=by_sec_brier["mean_brier"],
            mode="lines",
            name="Mean Brier (ask vs outcome)",
        )
    ]
)
ymax = float(by_sec_brier["mean_brier"].max())
for i, q in enumerate(quantile_levels):
    xv = float(t_quants[q])
    fig_speed.add_shape(
        type="line",
        x0=xv,
        x1=xv,
        y0=0,
        y1=ymax,
        line=dict(color=palette[i % len(palette)], width=1.5, dash="dash"),
    )
    fig_speed.add_annotation(
        x=xv,
        y=ymax,
        text=f"p{int(round(q * 100)):d}",
        showarrow=False,
        yshift=4,
        font=dict(size=10, color=palette[i % len(palette)]),
    )

fig_speed.update_layout(
    title=(
        "Mean Brier vs time; quantiles of persistent correct time "
        f"(censored_at_end: {censored_share:.1%})"
    ),
    xaxis_title="Seconds into 15m window",
    yaxis_title="Mean Brier",
    yaxis=dict(rangemode="tozero"),
    #showlegend=True,
    margin=dict(t=80),
)
fig_speed.show()

# %% [markdown]
# The figure is very consistent with the previous ones. Reading the quantile levels for persistent correct time, we obtain:
# - 10%: 0.0 seconds
# - 25%: 66.0 seconds
# - 50%: 314.0 seconds
# - 75%: 661.0 seconds
# - 90%: 839.0 seconds
#
# Which we can interpret by, in case of 75% of markets we observed they don't change and predict the correct outcome under 661 seconds
