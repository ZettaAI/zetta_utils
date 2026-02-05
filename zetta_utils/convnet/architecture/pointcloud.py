from __future__ import annotations

import pointnet

from zetta_utils import builder

builder.register("pointnet.PointNetCls")(pointnet.PointNetCls)
builder.register("pointnet.PointNetSeg")(pointnet.PointNetSeg)
builder.register("pointnet.PointNet2ClsSSG")(pointnet.PointNet2ClsSSG)
builder.register("pointnet.PointNet2SegSSG")(pointnet.PointNet2SegSSG)
builder.register("pointnet.PointNet2ClsMSG")(pointnet.PointNet2ClsMSG)
builder.register("pointnet.PointNet2SegMSG")(pointnet.PointNet2SegMSG)
builder.register("pointnet.STN")(pointnet.STN)
