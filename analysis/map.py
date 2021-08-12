#!/usr/bin/python3 -W ignore

import gzip
import math
import pickle
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from shapely.ops import transform


def create_colormap(name, colors, alphas=None, xs=None):
    def get_rgb1(c, alpha=1):
        return tuple(list(matplotlib.colors.hex2color(c)) + [alpha])

    if str(matplotlib.__version__).startswith("2"):
        get_rgb = matplotlib.colors.to_rgba
    else:
        get_rgb = get_rgb1
    if alphas is None:
        colors = [get_rgb(c) for c in colors]
    else:
        colors = [get_rgb(c, alpha=a) for c, a in zip(colors, alphas)]
    if xs is None:
        xs = np.linspace(0, 1, len(colors))
    res = LinearSegmentedColormap(
        name,
        {
            channel: tuple(
                (x, float(c[channel_id]), float(c[channel_id]))
                for c, x in zip(colors, xs)
            )
            for channel_id, channel in enumerate(["red", "green", "blue", "alpha"])
        },
        N=2048,
    )
    res.set_under(colors[0])
    res.set_over(colors[-1])
    return res


def make_map(
        patchespickle_file,
        regions,
        data,
        show_cbar=True,
        cm=None,
        outfile=None,
        ax=None,
        cax=None,
        extend_c="both",
        ignore_regions=None,
        invalid_edgecolor="lightgrey",
        invalid_facecolor="lightgrey",
        linewidth=0.1,
        norm_color=None,
        numbering=None,
        numbering_fontsize=10,
        rasterize=True,
        title=None,
        title_fontsize=10,
        valid_edgecolor="black",
        y_label=None,
        y_label_fontsize=10,
        y_ticks=None,
        y_tick_labels=None,
        y_ticks_fontsize=8,
        lims=None,
        only_usa=False,
        v_limits=None
):
    if ignore_regions is None:
        ignore_regions = ["ATA"]
    if cm is None:
        cm = create_colormap("custom", ["red", "white", "blue"], xs=[0, 0.5, 1])

    patchespickle = pickle.load(gzip.GzipFile(patchespickle_file, "rb"))
    patches = patchespickle["patches"]
    projection_name = patchespickle["projection"]

    if y_ticks is None:
        vmin = np.min(data)
        vmax = np.max(data)
    else:
        vmin = y_ticks[0]
        vmax = y_ticks[-1]

    if v_limits is not None:
        (vmin, vmax) = v_limits

    if norm_color is None:
        norm_color = Normalize(vmin=vmin, vmax=vmax)

    def EmptyPatch():
        return PathPatch(Path([(0, 0)], [Path.MOVETO]))

    def my_transform(scale, t, trans, x, y):
        p = trans(x, y)
        return (p[0] * scale + t[0], p[1] * scale + t[1])

    def get_projection(to, scale=1, translate=(0, 0)):
        return partial(
            my_transform,
            scale,
            translate,
            partial(
                pyproj.transform,
                pyproj.Proj("+proj=lonlat +datum=WGS84 +no_defs"),
                pyproj.Proj(f"+proj={to} +datum=WGS84 +no_defs"),
            ),
        )

    projection = get_projection(projection_name)
    if lims is None:
        miny, maxy, minx, maxx = -58, 89, -156, 170
    else:
        miny, maxy, minx, maxx = lims
    minx = transform(projection, Point(minx, 0)).x
    maxx = transform(projection, Point(maxx, 0)).x
    miny = transform(projection, Point(0, miny)).y
    maxy = transform(projection, Point(0, maxy)).y

    width_ratios = [1]  # , 0.005, 0.03] TODO
    if isinstance(outfile, str):
        # figure widths: 2.25 inches (1 column) or 4.75 inches (2 columns)
        fig = plt.figure(figsize=(4.75, 3))
        gs_base = plt.GridSpec(1, len(width_ratios), width_ratios=width_ratios, wspace=0)
    elif ax is None:
        fig = outfile.get_gridspec().figure
        gs_base = outfile.subgridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=0)
    if ax is None:
        ax = fig.add_subplot(gs_base[:, 0])
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    invpatches = []
    validpatches = []
    regions_with_data = set([])

    for r, d in zip(regions, data):
        if r in patches:
            level, subregions, patch = patches[r]
            if only_usa:
                if r in 'US.AK':
                    patch.set_transform(patch.get_transform() + matplotlib.transforms.Affine2D().scale(
                        0.4) + matplotlib.transforms.ScaledTranslation(transform(projection, Point(-25, 0)).x,
                                                                       transform(projection, Point(0, -14)).y,
                                                                       patch.get_transform()))  #

                elif r == 'US.HI':
                    patch.set_transform(
                        patch.get_transform() + matplotlib.transforms.ScaledTranslation(
                            transform(projection, Point(-15, 0)).x,
                            transform(projection, Point(0, -18)).y,
                            patch.get_transform()))
            if math.isnan(d):
                validpatches.append(EmptyPatch())
                invpatches.append(patch)
                print('NAN data for region {}'.format(r))
            elif r in ignore_regions:
                validpatches.append(EmptyPatch())
                invpatches.append(patch)
                print('Ignore region {}'.format(r))
            else:
                validpatches.append(patch)
            regions_with_data.update(subregions)
        else:
            validpatches.append(EmptyPatch())

    # for r, (level, subregions, patch) in patches.items():
    #     if not level and (r not in ignore_regions and subregions.isdisjoint(regions_with_data)):
    #         invpatches.append(patch)

    ax.add_collection(
        PatchCollection(
            invpatches,
            hatch="///",
            facecolors=invalid_facecolor,
            edgecolors=invalid_edgecolor,
            linewidths=linewidth,
            rasterized=rasterize,
        )
    )

    if numbering is not None:
        ax.text(
            0.0, 1.0, numbering, fontsize=numbering_fontsize, transform=ax.transAxes, fontweight='bold'
        )

    region_collection = ax.add_collection(
        PatchCollection(
            validpatches,
            edgecolors=valid_edgecolor,
            facecolors="black",
            linewidths=linewidth,
            rasterized=rasterize,
        )
    )

    if show_cbar:
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = matplotlib.colorbar.ColorbarBase(
            cax,
            cmap=cm,
            norm=norm_color,
            ticks=y_ticks,
            orientation="vertical",
            spacing="proportional",
            extend=extend_c,
        )
        cbar.minorticks_on()
        if y_tick_labels is not None:
            cbar.ax.set_yticklabels(y_tick_labels)
        if y_label is not None:
            cax.set_ylabel(y_label, fontsize=y_label_fontsize)
        cax.tick_params(axis="y", labelsize=y_ticks_fontsize)

    # region_collection.set_facecolors('r')
    region_collection.set_facecolors(cm(norm_color(data)))

    if isinstance(outfile, str):
        # plt.subplots_adjust(bottom=0.02, top=0.98, left=0.05, right=0.9)
        plt.tight_layout()
        fig.savefig(outfile, dpi=300)


cm = create_colormap("custom", ["red", "white", "blue"], xs=[0, 0.6667, 1])


def do_plot(d, label, numbering, fig, ax, cax, cm=None):
    # d = pd.read_csv(filename)
    regions = d['region'].array
    data = d["consumption_deviation"].array
    # if min(data) < 0:
    #     cm = create_colormap("custom", ["red", "white", "blue"], xs=[0, -min(data) / (max(data) - min(data)), 1])
    # else:
    #     cm = create_colormap("custom", ["white", "blue"], xs=[0, 1])
    if cm is None:
        cm = create_colormap("custom", ["red", "white", "blue"], xs=[0, 0.6667, 1])

    make_map(
        patchespickle_file="../data/external/maps/map_robinson_0.1simplified.pkl.gz",
        regions=regions,
        data=data,
        y_ticks=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5],
        y_label=label,
        numbering=numbering,
        extend_c="both",
        ax=fig.add_subplot(ax),
        cax=fig.add_subplot(cax),
        cm=cm,
    )
