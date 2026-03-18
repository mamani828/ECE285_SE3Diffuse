# make_conditional_unet_diagram.py
#
# Generates an architecture diagram for the ConditionalTemporalUNet
# from your diffusion-control model.
#
# Output:
#   conditional_unet_architecture.png
#   conditional_unet_architecture.svg
#
# Install:
#   pip install graphviz
#
# You also need the Graphviz system package installed:
#   macOS:   brew install graphviz
#   Ubuntu:  sudo apt-get install graphviz
#   Windows: https://graphviz.org/download/

from graphviz import Digraph


def add_box(dot, name, label, fill="#EAF2FF", shape="box"):
    dot.node(
        name,
        label,
        shape=shape,
        style="rounded,filled",
        fillcolor=fill,
        color="#3B4A6B",
        fontname="Helvetica",
        fontsize="11",
        margin="0.12,0.08",
    )


def add_edge(dot, a, b, label=None, color="#4A5568", style="solid"):
    kwargs = {
        "color": color,
        "penwidth": "1.4",
        "style": style,
        "arrowsize": "0.8",
    }
    if label is not None:
        kwargs["label"] = label
        kwargs["fontsize"] = "10"
        kwargs["fontname"] = "Helvetica"
    dot.edge(a, b, **kwargs)


def build_conditional_unet_diagram(
    out_name="conditional_unet_architecture",
    base_channels=64,
    cond_dim=128,
    time_emb_dim=128,
    pose_emb_dim=64,
    map_emb_dim=128,
    control_dim=2,
    map_in_ch=1,
):
    dot = Digraph("ConditionalTemporalUNet", format="png")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.4", ranksep="0.6")
    dot.attr(
        labelloc="t",
        label=(
            "ConditionalTemporalUNet Architecture\n"
            f"control_dim={control_dim}, base_channels={base_channels}, cond_dim={cond_dim}"
        ),
        fontsize="16",
        fontname="Helvetica-Bold",
    )

    # Global styles
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    # ============================================================
    # Inputs
    # ============================================================
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(
            label="Inputs",
            color="#9AA5B1",
            style="rounded",
            penwidth="1.2",
        )
        add_box(c, "x_in", f"Noisy controls x_t\n(B, T, {control_dim})", fill="#D6EAF8")
        add_box(c, "t_in", "Diffusion step t\n(B,)", fill="#FCF3CF")
        add_box(c, "pose_in", "Pose condition\n(B, 8)\n[start, goal]", fill="#D5F5E3")
        add_box(c, "map_in", f"Map tensor\n(B, {map_in_ch}, H, W)", fill="#FADBD8")

    # ============================================================
    # Conditioning branch
    # ============================================================
    with dot.subgraph(name="cluster_cond") as c:
        c.attr(
            label="Conditioning Branch",
            color="#9AA5B1",
            style="rounded",
            penwidth="1.2",
        )

        add_box(c, "time_emb", f"SinusoidalTimeEmbedding\n-> ({time_emb_dim})", fill="#F9E79F")
        add_box(
            c,
            "time_mlp",
            f"Time MLP\nLinear -> SiLU -> Linear\n-> ({cond_dim})",
            fill="#FDEBD0",
        )

        add_box(
            c,
            "pose_enc",
            f"PoseEncoder\nLinear(8,128) -> SiLU -> Linear(128,{pose_emb_dim})",
            fill="#D4EFDF",
        )
        add_box(c, "pose_proj", f"Pose proj\nLinear({pose_emb_dim},{cond_dim})", fill="#D4EFDF")

        add_box(
            c,
            "map_enc",
            (
                "MapEncoder\n"
                "Conv2d -> SiLU\n"
                "Conv2d -> SiLU\n"
                "Conv2d -> SiLU\n"
                "Conv2d -> SiLU\n"
                "AdaptiveAvgPool2d(1)\n"
                f"-> Linear(128,{map_emb_dim})"
            ),
            fill="#F5CBA7",
        )
        add_box(c, "map_proj", f"Map proj\nLinear({map_emb_dim},{cond_dim})", fill="#F5CBA7")

        add_box(
            c,
            "cond_fuse",
            (
                "Condition Fusion\n"
                "Concat[time, pose, map]\n"
                f"Linear({cond_dim*3},{cond_dim}) -> SiLU -> Linear({cond_dim},{cond_dim})"
            ),
            fill="#E8DAEF",
        )

    # ============================================================
    # U-Net branch
    # ============================================================
    ch = base_channels
    with dot.subgraph(name="cluster_unet") as c:
        c.attr(
            label="1D Conditional Temporal U-Net",
            color="#9AA5B1",
            style="rounded",
            penwidth="1.2",
        )

        add_box(c, "in_proj", f"Input projection\nConv1d({control_dim},{ch}, k=3,p=1)", fill="#AED6F1")

        add_box(c, "down1", f"Down Block 1\nResBlock1D\n{ch} -> {ch}", fill="#A9DFBF")
        add_box(c, "ds1", f"Downsample1D\nConv1d({ch},{ch}, k=4,s=2,p=1)", fill="#ABEBC6")

        add_box(c, "down2", f"Down Block 2\nResBlock1D\n{ch} -> {2*ch}", fill="#82E0AA")
        add_box(c, "ds2", f"Downsample1D\nConv1d({2*ch},{2*ch}, k=4,s=2,p=1)", fill="#82E0AA")

        add_box(c, "mid1", f"Mid Block 1\nResBlock1D\n{2*ch} -> {4*ch}", fill="#F8C471")
        add_box(c, "mid2", f"Mid Block 2\nResBlock1D\n{4*ch} -> {4*ch}", fill="#F5B041")

        add_box(c, "us1", f"Upsample1D\ninterp x2 + Conv1d({4*ch},{4*ch})", fill="#F9E79F")
        add_box(c, "cat1", f"Concat skip\n[{4*ch} + {2*ch}] = {6*ch}", fill="#FCF3CF")
        add_box(c, "up1", f"Up Block 1\nResBlock1D\n{6*ch} -> {2*ch}", fill="#85C1E9")

        add_box(c, "us2", f"Upsample1D\ninterp x2 + Conv1d({2*ch},{2*ch})", fill="#D6EAF8")
        add_box(c, "cat2", f"Concat skip\n[{2*ch} + {ch}] = {3*ch}", fill="#EBF5FB")
        add_box(c, "up2", f"Up Block 2\nResBlock1D\n{3*ch} -> {ch}", fill="#5DADE2")

        add_box(
            c,
            "out_proj",
            f"Output head\nGroupNorm -> SiLU -> Conv1d({ch},{control_dim}, k=3,p=1)",
            fill="#D2B4DE",
        )
        add_box(c, "noise_out", f"Predicted noise eps_hat\n(B, T, {control_dim})", fill="#E8DAEF")

    # ============================================================
    # Flow edges
    # ============================================================
    add_edge(dot, "t_in", "time_emb")
    add_edge(dot, "time_emb", "time_mlp")
    add_edge(dot, "pose_in", "pose_enc")
    add_edge(dot, "pose_enc", "pose_proj")
    add_edge(dot, "map_in", "map_enc")
    add_edge(dot, "map_enc", "map_proj")

    add_edge(dot, "time_mlp", "cond_fuse")
    add_edge(dot, "pose_proj", "cond_fuse")
    add_edge(dot, "map_proj", "cond_fuse")

    add_edge(dot, "x_in", "in_proj")
    add_edge(dot, "in_proj", "down1")
    add_edge(dot, "down1", "ds1")
    add_edge(dot, "ds1", "down2")
    add_edge(dot, "down2", "ds2")
    add_edge(dot, "ds2", "mid1")
    add_edge(dot, "mid1", "mid2")
    add_edge(dot, "mid2", "us1")
    add_edge(dot, "us1", "cat1")
    add_edge(dot, "cat1", "up1")
    add_edge(dot, "up1", "us2")
    add_edge(dot, "us2", "cat2")
    add_edge(dot, "cat2", "up2")
    add_edge(dot, "up2", "out_proj")
    add_edge(dot, "out_proj", "noise_out")

    # Skip connections
    add_edge(dot, "down2", "cat1", label="skip", color="#7D3C98", style="dashed")
    add_edge(dot, "down1", "cat2", label="skip", color="#7D3C98", style="dashed")

    # Conditioning injection into each ResBlock
    for block in ["down1", "down2", "mid1", "mid2", "up1", "up2"]:
        add_edge(dot, "cond_fuse", block, label="cond", color="#C0392B", style="dashed")

    # Small note
    dot.node(
        "note",
        "Inside each ResBlock1D:\nGroupNorm -> SiLU -> Conv1d\n+ cond_proj(cond)\nGroupNorm -> SiLU -> Conv1d\n+ skip connection",
        shape="note",
        style="filled",
        fillcolor="#FEF9E7",
        color="#B7950B",
        fontname="Helvetica",
        fontsize="10",
    )
    add_edge(dot, "cond_fuse", "note", color="#B7950B", style="dotted")

    # Render PNG
    dot.render(out_name, cleanup=True)

    # Render SVG too
    dot_svg = dot.copy()
    dot_svg.format = "svg"
    dot_svg.render(out_name, cleanup=True)

    print(f"Saved: {out_name}.png")
    print(f"Saved: {out_name}.svg")


if __name__ == "__main__":
    build_conditional_unet_diagram(
        out_name="conditional_unet_architecture",
        base_channels=64,
        cond_dim=128,
        time_emb_dim=128,
        pose_emb_dim=64,
        map_emb_dim=128,
        control_dim=2,
        map_in_ch=1,   # set to 2 if map_mode == "sdf_occupancy"
    )