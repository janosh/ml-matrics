# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz import count_elements, ptable_heatmap, spacegroup_sunburst, structure_2d
from pymatviz.enums import Key


"""Stats for the matbench_perovskites dataset.

Input: Pymatgen Structure of the material.
Target variable: Heat of formation of the entire 5-atom perovskite cell in eV
    as calculated by RPBE GGA-DFT. Note the reference state for oxygen was
    computed from oxygen's chemical potential in water vapor, not as oxygen
    molecules, to reflect the application which these perovskites were studied for.
Entries: 18,928

Matbench v0.1 dataset for predicting formation energy from crystal structure.
Adapted from an original dataset generated by Castelli et al.

https://ml.materialsproject.org/projects/matbench_perovskites
"""

# %%
df_perov = load_dataset("matbench_perovskites")

df_perov[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_perov[Key.structure])
]
df_perov[Key.volume] = df_perov[Key.structure].map(lambda struct: struct.volume)

df_perov[Key.formula] = df_perov[Key.structure].map(lambda cryst: cryst.formula)

df_perov[Key.crystal_system] = df_perov[Key.spg_num].map(
    pmv.utils.crystal_sys_from_spg_num,
)


# %%
ax = structure_2d(df_perov[Key.structure].iloc[:12])
pmv.save_fig(ax, "perovskite-structures-2d.pdf")


# %%
ax = df_perov.hist(column="e_form", bins=50)
pmv.save_fig(ax, "perovskites-e_form-hist.pdf")


# %%
fig = ptable_heatmap(
    count_elements(df_perov[Key.formula]),
    log=True,
    return_type="figure",
)
fig.suptitle("Elements in Matbench Perovskites dataset")
pmv.save_fig(fig, "perovskites-ptable-heatmap.pdf")


# %%
ax = df_perov[Key.crystal_system].value_counts().plot.bar(rot="horizontal")
ax.set(title="Crystal systems in Matbench Perovskites")
pmv.powerups.annotate_bars(ax, v_offset=250)

pmv.save_fig(ax, "perovskites-crystal-system-counts.pdf")


# %%
df_perov.plot.scatter(x=Key.volume, y="e_form", c=Key.spg_num, colormap="viridis")


# %%
fig = spacegroup_sunburst(df_perov[Key.spg_num], show_counts="percent")
fig.layout.title = "Matbench Perovskites spacegroup sunburst"
fig.write_image("perovskite-spacegroup-sunburst.pdf")
fig.show()
