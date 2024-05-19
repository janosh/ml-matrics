# %%
import gzip
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pymatgen.core import Composition, Structure
from tqdm import tqdm

from pymatviz import pmv_dark_template
from pymatviz.enums import Key
from pymatviz.histograms import spacegroup_hist
from pymatviz.io import save_fig
from pymatviz.powerups import add_identity_line
from pymatviz.ptable import count_elements, ptable_heatmap, ptable_heatmap_splits
from pymatviz.sunburst import spacegroup_sunburst


px.defaults.template = pmv_dark_template
pio.templates.default = pmv_dark_template
module_dir = os.path.dirname(__file__)


# %%
with gzip.open(f"{module_dir}/20240214-MatPES-178070-r2SCAN.json.gz", "rt") as file:
    r2scan_data = json.load(file)
# examples/dataset_exploration/matpes/20240214-MatPES-187023-PBE.json.gz
with gzip.open(f"{module_dir}/20240214-MatPES-183027-PBE.json.gz", "rt") as file:
    pbe_data = json.load(file)

n_r2scan, n_pbe = 178_070, 183_027
for data, expected in ((r2scan_data, n_r2scan), (pbe_data, n_pbe)):
    if len(data) != expected:
        raise ValueError(f"{expected=}, {len(data)=}")


# %%
df_r2scan = pd.DataFrame(r2scan_data).T
df_pbe = pd.DataFrame(pbe_data).T

for df in (df_r2scan, df_pbe):
    df.index.name = Key.mat_id
    df.rename(columns={"magmom": Key.magmoms}, inplace=True)  # noqa: PD002
    df[Key.structure] = df[Key.structure].map(Structure.from_dict)
    df[Key.formula] = [struct.formula for struct in df[Key.structure]]
    df[Key.n_sites] = df[Key.structure].map(len)


# %% plot histogram of energies
fig = go.Figure()

fig.add_histogram(x=df_r2scan[Key.energy], name="r2scan", opacity=0.8)
fig.add_histogram(x=df_pbe[Key.energy], name="pbe", opacity=0.8)

fig.update_layout(xaxis_title=Key.energy.label, margin=dict(l=5, r=5, t=5, b=5))
fig.layout.legend.update(x=0, y=1)
fig.show()
# save_fig(fig, "matpes-energy-hist.pdf")

# @janosh 2024-05-15: initially surprised by the difference in r2scan/pbe energy distros
# how could energy differences between two similar chemistries always be similar across
# r2SCAN and PBE if the distribution for r2SCAN is so much wider


# %% seems fine. parity plot reveals this nice collection of bands which looks like
# within each chemical system, you indeed get consistent energy differences. just across
# different systems, the zero-level energies differ wildly
fig = go.Figure()

fig.add_scatter(
    x=df_r2scan[Key.energy].head(1000),
    y=df_pbe.loc[df_r2scan.index][Key.energy].head(1000),
    mode="markers",
)
fig.layout.xaxis.title = f"r2SCAN {Key.energy.label}"
fig.layout.yaxis.title = f"PBE {Key.energy.label}"
add_identity_line(fig, retain_xy_limits=True)
fig.show()


# %% plot histograms of mean forces per atom for both functionals
total_force_col = "Σ|force<sub>i</sub>| per-atom"
df_pbe[total_force_col] = df_pbe[Key.forces].map(lambda arr: np.abs(arr).sum(axis=1))
df_r2scan[total_force_col] = df_r2scan[Key.forces].map(
    lambda arr: np.abs(arr).sum(axis=1)
)

fig = go.Figure()

fig.add_histogram(x=df_r2scan[total_force_col].explode(), name="r2scan", opacity=0.8)
fig.add_histogram(x=df_pbe[total_force_col].explode(), name="pbe", opacity=0.8)

fig.update_layout(xaxis_title=total_force_col, margin=dict(l=5, r=5, t=5, b=5))
fig.layout.legend.update(x=0, y=1)
fig.update_yaxes(type="log")
fig.show()
# save_fig(fig, "matpes-forces-hist.pdf")


# %% plot element counts
r2scan_elem_counts = locals().get("r2scan_elem_counts")
if r2scan_elem_counts is None:
    r2scan_elem_counts = count_elements(df_r2scan[Key.formula])
ax = ptable_heatmap(r2scan_elem_counts)

pbe_elem_counts = locals().get("pbe_elem_counts")
if pbe_elem_counts is None:
    pbe_elem_counts = count_elements(df_pbe[Key.formula])
ax = ptable_heatmap(pbe_elem_counts)


# %% calculate per element energies
frac_comp_col = "fractional composition"
for df in (df_r2scan, df_pbe):
    df[frac_comp_col] = [
        Composition(comp).fractional_composition for comp in tqdm(df[Key.formula])
    ]

df_r2scan_frac_comp = pd.DataFrame(
    comp.as_dict() for comp in df_r2scan[frac_comp_col]
).set_index(df_r2scan.index)
df_pbe_frac_comp = pd.DataFrame(
    comp.as_dict() for comp in df_pbe[frac_comp_col]
).set_index(df_pbe.index)

if any(df_r2scan_frac_comp.sum(axis=1).round(6) != 1):
    raise ValueError("composition fractions don't sum to 1")

df_per_elem = pd.DataFrame()
r2scan_col = "r2SCAN energy"
col_name = Key.cohesive_energy_per_atom
df_per_elem[r2scan_col] = (
    df_r2scan_frac_comp * df_r2scan[col_name].to_numpy()[:, None]
).mean()
pbe_col = "PBE energy"
df_per_elem[pbe_col] = (df_pbe_frac_comp * df_pbe[col_name].to_numpy()[:, None]).mean()


# %% cohesive energies should (and do) look nearly identical between r2scan and pbe
per_elem_cohesive_energy = {
    key: list(dct.values()) for key, dct in df_per_elem.to_dict(orient="index").items()
}

fig = ptable_heatmap_splits(
    per_elem_cohesive_energy, cbar_title=f"{col_name.label} (eV)"
)


# %% which elements have a higher share of missing r2scan data
ax = ptable_heatmap(
    (pbe_elem_counts - r2scan_elem_counts) / pbe_elem_counts,
    fmt=".1%",
    cbar_fmt=".0%",
    cbar_title="Fraction of missing PBE calcs missing r2SCAN",
)

save_fig(ax, "matpes-missing-r2scan-data-ptable.pdf")


# %% per-elem mean abs magmoms
df_per_elem_magmoms = pd.DataFrame(
    {site.specie.symbol: abs(site.properties["magmom"]) for site in struct}
    for struct in df_r2scan[Key.structure]
).mean()

ax = ptable_heatmap(
    df_per_elem_magmoms, cbar_title=r"Mean |magmom| ($\mu_B$)", fmt=".1f"
)
save_fig(ax, "matpes-magmoms-ptable.pdf")


# %% spacegroup distribution
df_r2scan[Key.spacegroup] = [
    struct.get_space_group_info()[1]
    for struct in tqdm(df_r2scan[Key.structure], desc="r2scan spacegroups")
]


# %%
fig = spacegroup_sunburst(
    df_r2scan[Key.spacegroup],
    title="r2SCAN spacegroup distribution",
    show_counts="percent",
)
fig.layout.title.update(text=f"{n_r2scan:,} r2SCAN spacegroups", x=0.5, y=0.97)
fig.layout.margin = dict(l=0, r=0, b=0, t=30)
fig.show()
save_fig(fig, "matpes-r2scan-spacegroup-sunburst.pdf")


# %% spacegroup histogram
fig = spacegroup_hist(
    df_r2scan[Key.spacegroup], title="r2SCAN spacegroup histogram", log=True
)
fig.show()
save_fig(fig, "matpes-r2scan-spacegroup-hist.pdf")


# %%
df_pbe[Key.spacegroup] = [
    struct.get_space_group_info()[1]
    for struct in tqdm(df_pbe[Key.structure], desc="pbe spacegroups")
]
