from __future__ import annotations

import io
import json
import os
import struct
from collections.abc import Sequence

import attrs
import fsspec
import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.backend_base import Backend
from zetta_utils.layer.volumetric import VolumetricIndex

from .contact import SegContact, read_info


@attrs.define
class SegContactLayerBackend(Backend[VolumetricIndex, Sequence[SegContact], Sequence[SegContact]]):
    """Backend for reading/writing seg_contact data in chunked format."""

    path: str
    resolution: Vec3D[int]  # voxel size in nm
    voxel_offset: Vec3D[int]  # dataset start in voxels
    size: Vec3D[int]  # dataset dimensions in voxels
    chunk_size: Vec3D[int]  # chunk dimensions in voxels
    max_contact_span: int  # in voxels
    enforce_chunk_aligned_writes: bool = True
    local_point_clouds: list[tuple[int, int]] | None = None  # [(radius_nm, n_points), ...]

    @property
    def name(self) -> str:
        return self.path

    def with_changes(self, **kwargs) -> SegContactLayerBackend:
        return attrs.evolve(self, **kwargs)

    @classmethod
    def from_path(cls, path: str) -> SegContactLayerBackend:
        """Load backend from existing info file."""
        info = read_info(path)
        return cls(
            path=path,
            resolution=Vec3D(*info["resolution"]),
            voxel_offset=Vec3D(*info["voxel_offset"]),
            size=Vec3D(*info["size"]),
            chunk_size=Vec3D(*info["chunk_size"]),
            max_contact_span=info["max_contact_span"],
        )

    def write_info(self) -> None:
        """Write info file to disk."""
        info = {
            "format_version": "1.0",
            "type": "seg_contact",
            "resolution": list(self.resolution),
            "voxel_offset": list(self.voxel_offset),
            "size": list(self.size),
            "chunk_size": list(self.chunk_size),
            "max_contact_span": self.max_contact_span,
        }
        info_path = f"{self.path}/info"
        fs, fs_path = fsspec.core.url_to_fs(info_path)
        fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
        with fs.open(fs_path, "wb") as f:
            f.write(json.dumps(info, indent=2).encode("utf-8"))

    def read_info(self) -> dict:
        """Read the info file as dict."""
        return read_info(self.path)

    def get_pointcloud_configs(self) -> list[tuple[float, int]]:
        """Get list of (radius_nm, n_points) configs from info file."""
        info = self.read_info()
        configs = info.get("local_point_clouds", [])
        return [(cfg["radius_nm"], cfg["n_points"]) for cfg in configs]

    def read(self, idx: VolumetricIndex) -> Sequence[SegContact]:
        """Read contacts whose COM falls within the given index."""
        # bbox.start and bbox.end are already in nm
        bbox = idx.bbox
        start_nm = bbox.start
        end_nm = bbox.end

        # Find which chunks to read
        start_chunk = self.com_to_chunk_idx(start_nm)
        end_chunk = self.com_to_chunk_idx(
            Vec3D(end_nm[0] - 0.001, end_nm[1] - 0.001, end_nm[2] - 0.001)
        )

        result = []
        for gx in range(start_chunk[0], end_chunk[0] + 1):
            for gy in range(start_chunk[1], end_chunk[1] + 1):
                for gz in range(start_chunk[2], end_chunk[2] + 1):
                    contacts = self.read_chunk((gx, gy, gz))
                    # Filter by COM within bbox
                    for c in contacts:
                        if (
                            start_nm[0] <= c.com[0] < end_nm[0]
                            and start_nm[1] <= c.com[1] < end_nm[1]
                            and start_nm[2] <= c.com[2] < end_nm[2]
                        ):
                            result.append(c)
        return result

    def get_contact_counts(self, idx: VolumetricIndex) -> dict[tuple[int, int, int], int]:
        """Get count of contacts per chunk within the given bounds.

        Returns dict mapping chunk_idx -> count of contacts with COM in bounds.
        """
        bbox = idx.bbox
        start_nm = bbox.start
        end_nm = bbox.end

        start_chunk = self.com_to_chunk_idx(start_nm)
        end_chunk = self.com_to_chunk_idx(
            Vec3D(end_nm[0] - 0.001, end_nm[1] - 0.001, end_nm[2] - 0.001)
        )

        fs, _ = fsspec.core.url_to_fs(self.path)

        result: dict[tuple[int, int, int], int] = {}
        for gx in range(start_chunk[0], end_chunk[0] + 1):
            for gy in range(start_chunk[1], end_chunk[1] + 1):
                for gz in range(start_chunk[2], end_chunk[2] + 1):
                    chunk_idx = (gx, gy, gz)
                    chunk_path = self.get_chunk_path(chunk_idx)
                    _, fs_path = fsspec.core.url_to_fs(chunk_path)
                    if not fs.exists(fs_path):
                        result[chunk_idx] = 0
                        continue
                    with fs.open(fs_path, "rb") as f:
                        content = f.read()
                    count = self._count_contacts_in_bounds(content, start_nm, end_nm)
                    result[chunk_idx] = count

        return result

    def _count_contacts_in_bounds(self, chunk_data: bytes, start_nm: Vec3D, end_nm: Vec3D) -> int:
        """Parse chunk binary data and count contacts with COM in bounds."""
        count = 0
        with io.BytesIO(chunk_data) as f:
            n_contacts = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_contacts):
                # Skip id, seg_a, seg_b (24 bytes)
                f.read(24)
                # Read COM
                com = struct.unpack("<fff", f.read(12))
                # Skip n_faces and contact_faces
                n_faces = struct.unpack("<I", f.read(4))[0]
                f.read(n_faces * 4 * 4)
                # Skip metadata
                metadata_len = struct.unpack("<I", f.read(4))[0]
                if metadata_len > 0:
                    f.read(metadata_len)

                # Skip representative_points (6 floats)
                f.read(24)

                # Check if COM is in bounds
                if (
                    start_nm[0] <= com[0] < end_nm[0]
                    and start_nm[1] <= com[1] < end_nm[1]
                    and start_nm[2] <= com[2] < end_nm[2]
                ):
                    count += 1

        return count

    def write(self, idx: VolumetricIndex, data: Sequence[SegContact]) -> None:
        """Write contacts to appropriate chunks based on their COM."""
        # Group contacts by chunk
        chunk_contacts: dict[tuple[int, int, int], list[SegContact]] = {}
        for contact in data:
            chunk_idx = self.com_to_chunk_idx(contact.com)
            if chunk_idx not in chunk_contacts:
                chunk_contacts[chunk_idx] = []
            chunk_contacts[chunk_idx].append(contact)

        # Write each chunk
        for chunk_idx, contacts in chunk_contacts.items():
            self.write_chunk(chunk_idx, contacts)

    def get_chunk_path(self, chunk_idx: tuple[int, int, int]) -> str:
        """Get file path for a contacts chunk given grid indices."""
        return os.path.join(self.path, "contacts", self.get_chunk_name(chunk_idx))

    def get_pointcloud_chunk_path(
        self, chunk_idx: tuple[int, int, int], radius_nm: float, n_points: int
    ) -> str:
        """Get file path for a pointcloud chunk given grid indices and config."""
        config_dir = f"{int(radius_nm)}nm_{n_points}pts"
        return os.path.join(
            self.path, "local_point_clouds", config_dir, self.get_chunk_name(chunk_idx)
        )

    def get_merge_decision_chunk_path(
        self, chunk_idx: tuple[int, int, int], authority: str
    ) -> str:
        """Get file path for a merge decision chunk given grid indices and authority."""
        return os.path.join(
            self.path, "merge_decisions", authority, self.get_chunk_name(chunk_idx)
        )

    def get_merge_probability_chunk_path(
        self, chunk_idx: tuple[int, int, int], authority: str
    ) -> str:
        """Get file path for a merge probability chunk given grid indices and authority."""
        return os.path.join(
            self.path, "merge_probabilities", authority, self.get_chunk_name(chunk_idx)
        )

    def get_representative_supervoxels_chunk_path(self, chunk_idx: tuple[int, int, int]) -> str:
        """Get file path for representative supervoxels chunk."""
        return os.path.join(
            self.path, "representative_supervoxels", self.get_chunk_name(chunk_idx)
        )

    def get_chunk_name(self, chunk_idx: tuple[int, int, int]) -> str:
        """Get chunk filename in precomputed format."""
        gx, gy, gz = chunk_idx
        x_start = self.voxel_offset[0] + gx * self.chunk_size[0]
        x_end = x_start + self.chunk_size[0]
        y_start = self.voxel_offset[1] + gy * self.chunk_size[1]
        y_end = y_start + self.chunk_size[1]
        z_start = self.voxel_offset[2] + gz * self.chunk_size[2]
        z_end = z_start + self.chunk_size[2]
        return f"{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}"

    def com_to_chunk_idx(self, com_nm: Vec3D[float]) -> tuple[int, int, int]:
        """Convert COM in nanometers to chunk grid index."""
        # Convert COM from nm to voxels
        com_vx = Vec3D(
            com_nm[0] / self.resolution[0],
            com_nm[1] / self.resolution[1],
            com_nm[2] / self.resolution[2],
        )
        # Subtract offset and divide by chunk size
        gx = int((com_vx[0] - self.voxel_offset[0]) // self.chunk_size[0])
        gy = int((com_vx[1] - self.voxel_offset[1]) // self.chunk_size[1])
        gz = int((com_vx[2] - self.voxel_offset[2]) // self.chunk_size[2])
        return (gx, gy, gz)

    def get_chunk_bounds_nm(self, chunk_idx: tuple[int, int, int]) -> tuple[Vec3D, Vec3D]:
        """Get chunk bounds in nanometers.

        Returns (start_nm, end_nm) tuple where both are Vec3D in nm.
        """
        gx, gy, gz = chunk_idx
        # Chunk bounds in voxels
        x_start_vx = self.voxel_offset[0] + gx * self.chunk_size[0]
        y_start_vx = self.voxel_offset[1] + gy * self.chunk_size[1]
        z_start_vx = self.voxel_offset[2] + gz * self.chunk_size[2]
        x_end_vx = x_start_vx + self.chunk_size[0]
        y_end_vx = y_start_vx + self.chunk_size[1]
        z_end_vx = z_start_vx + self.chunk_size[2]
        # Convert to nm
        start_nm = Vec3D(
            x_start_vx * self.resolution[0],
            y_start_vx * self.resolution[1],
            z_start_vx * self.resolution[2],
        )
        end_nm = Vec3D(
            x_end_vx * self.resolution[0],
            y_end_vx * self.resolution[1],
            z_end_vx * self.resolution[2],
        )
        return start_nm, end_nm

    def normalize_points(
        self, points: np.ndarray, com: tuple[float, float, float], radius_nm: float
    ) -> np.ndarray:
        """Normalize points to [-1, 1] range centered on COM.

        Args:
            points: Array of shape [..., 3] with xyz coordinates in nm.
            com: Center of mass (x, y, z) in nm - becomes origin after normalization.
            radius_nm: Radius in nm - points at this distance from COM become ±1.

        Returns:
            Normalized points with xyz in [-1, 1] range, same shape as input.
        """
        com_np = np.array(com, dtype=np.float32)
        return (points - com_np) / radius_nm

    def write_chunk(self, chunk_idx: tuple[int, int, int], contacts: Sequence[SegContact]) -> None:
        """Write contacts to chunk files (contacts, pointclouds, merge_decisions)."""
        self._write_contacts_chunk(chunk_idx, contacts)

        pointclouds_by_config = self._collect_pointclouds(contacts)
        for config_tuple, entries in pointclouds_by_config.items():
            self._write_pointcloud_chunk(chunk_idx, config_tuple, entries)

        decisions_by_authority = self._collect_merge_decisions(contacts)
        for authority, decisions in decisions_by_authority.items():
            self._write_merge_decision_chunk(chunk_idx, authority, decisions)

        probabilities_by_authority = self._collect_merge_probabilities(contacts)
        for authority, probabilities in probabilities_by_authority.items():
            self._write_merge_probability_chunk(chunk_idx, authority, probabilities)

        supervoxels = self._collect_representative_supervoxels(contacts)
        if supervoxels:
            self._write_representative_supervoxels_chunk(chunk_idx, supervoxels)

    def _collect_pointclouds(
        self, contacts: Sequence[SegContact]
    ) -> dict[tuple[int, int], list[tuple[int, int, int, np.ndarray, np.ndarray]]]:
        """Collect pointclouds from contacts grouped by config."""
        result: dict[tuple[int, int], list[tuple[int, int, int, np.ndarray, np.ndarray]]] = {}
        for contact in contacts:
            if contact.local_pointclouds is None:
                continue
            for config_tuple, seg_points in contact.local_pointclouds.items():
                seg_a_pts = seg_points.get(contact.seg_a)
                seg_b_pts = seg_points.get(contact.seg_b)
                if seg_a_pts is None or seg_b_pts is None:
                    continue
                if config_tuple not in result:
                    result[config_tuple] = []
                result[config_tuple].append(
                    (contact.id, contact.seg_a, contact.seg_b, seg_a_pts, seg_b_pts)
                )
        return result

    def _collect_merge_decisions(
        self, contacts: Sequence[SegContact]
    ) -> dict[str, list[tuple[int, bool]]]:
        """Collect merge decisions from contacts grouped by authority."""
        result: dict[str, list[tuple[int, bool]]] = {}
        for contact in contacts:
            if contact.merge_decisions is None:
                continue
            for authority, should_merge in contact.merge_decisions.items():
                if authority not in result:
                    result[authority] = []
                result[authority].append((contact.id, should_merge))
        return result

    def _collect_merge_probabilities(
        self, contacts: Sequence[SegContact]
    ) -> dict[str, list[tuple[int, float]]]:
        """Collect merge probabilities from contacts grouped by authority."""
        result: dict[str, list[tuple[int, float]]] = {}
        for contact in contacts:
            if contact.merge_probabilities is None:
                continue
            for authority, probability in contact.merge_probabilities.items():
                if authority not in result:
                    result[authority] = []
                result[authority].append((contact.id, probability))
        return result

    def _collect_representative_supervoxels(
        self, contacts: Sequence[SegContact]
    ) -> list[tuple[int, int, int, int, int]]:
        """Collect representative supervoxels from contacts.

        Returns list of (contact_id, seg_a, seg_b, supervoxel_a, supervoxel_b).
        """
        result: list[tuple[int, int, int, int, int]] = []
        for contact in contacts:
            if contact.representative_supervoxels is None:
                continue
            sv_a = contact.representative_supervoxels.get(contact.seg_a)
            sv_b = contact.representative_supervoxels.get(contact.seg_b)
            if sv_a is not None and sv_b is not None:
                result.append((contact.id, contact.seg_a, contact.seg_b, sv_a, sv_b))
        return result

    def _write_contacts_chunk(
        self, chunk_idx: tuple[int, int, int], contacts: Sequence[SegContact]
    ) -> None:
        """Write core contact data to contacts/ directory."""
        chunk_path = self.get_chunk_path(chunk_idx)

        buf = io.BytesIO()
        # Header: n_contacts
        buf.write(struct.pack("<I", len(contacts)))

        for contact in contacts:
            # id, seg_a, seg_b
            buf.write(struct.pack("<qqq", contact.id, contact.seg_a, contact.seg_b))
            # com (3 floats)
            buf.write(struct.pack("<fff", contact.com[0], contact.com[1], contact.com[2]))
            # n_faces
            n_faces = contact.contact_faces.shape[0]
            buf.write(struct.pack("<I", n_faces))
            # contact_faces
            buf.write(contact.contact_faces.astype(np.float32).tobytes())

            # partner_metadata
            if contact.partner_metadata is not None:
                metadata_bytes = json.dumps(contact.partner_metadata).encode("utf-8")
                buf.write(struct.pack("<I", len(metadata_bytes)))
                buf.write(metadata_bytes)
            else:
                buf.write(struct.pack("<I", 0))

            # representative_points: 6 floats (point_a xyz, point_b xyz)
            pt_a = contact.representative_points[contact.seg_a]
            pt_b = contact.representative_points[contact.seg_b]
            buf.write(struct.pack("<ffffff", pt_a[0], pt_a[1], pt_a[2], pt_b[0], pt_b[1], pt_b[2]))

        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
        with fs.open(fs_path, "wb") as f:
            f.write(buf.getvalue())

    def _write_pointcloud_chunk(
        self,
        chunk_idx: tuple[int, int, int],
        config_tuple: tuple[int, int],
        entries: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    ) -> None:
        """Write pointcloud data to local_point_clouds/{config}/ directory.

        Format per DESIGN.md:
        - n_entries: uint32
        - Per entry:
          - contact_id: int64
          - seg_a_points: float32[n_points, 3]
          - seg_b_points: float32[n_points, 3]
        """
        radius_nm, n_points = config_tuple

        chunk_path = self.get_pointcloud_chunk_path(chunk_idx, radius_nm, n_points)

        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(entries)))

        for contact_id, _seg_a, _seg_b, seg_a_pts, seg_b_pts in entries:
            buf.write(struct.pack("<q", contact_id))
            buf.write(seg_a_pts.astype(np.float32).tobytes())
            buf.write(seg_b_pts.astype(np.float32).tobytes())

        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
        with fs.open(fs_path, "wb") as f:
            f.write(buf.getvalue())

    def _write_merge_decision_chunk(
        self,
        chunk_idx: tuple[int, int, int],
        authority: str,
        decisions: list[tuple[int, bool]],
    ) -> None:
        """Write merge decisions to merge_decisions/{authority}/ directory.

        Format per DESIGN.md:
        - n_decisions: uint32
        - Per decision:
          - contact_id: int64
          - should_merge: uint8 (0 or 1)
        """
        chunk_path = self.get_merge_decision_chunk_path(chunk_idx, authority)

        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(decisions)))

        for contact_id, should_merge in decisions:
            buf.write(struct.pack("<q", contact_id))
            buf.write(struct.pack("<B", 1 if should_merge else 0))

        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
        with fs.open(fs_path, "wb") as f:
            f.write(buf.getvalue())

    def _write_merge_probability_chunk(
        self,
        chunk_idx: tuple[int, int, int],
        authority: str,
        probabilities: list[tuple[int, float]],
    ) -> None:
        """Write merge probabilities to merge_probabilities/{authority}/ directory.

        Format:
        - n_entries: uint32
        - Per entry:
          - contact_id: int64
          - probability: float32
        """
        chunk_path = self.get_merge_probability_chunk_path(chunk_idx, authority)

        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(probabilities)))

        for contact_id, probability in probabilities:
            buf.write(struct.pack("<q", contact_id))
            buf.write(struct.pack("<f", probability))

        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
        with fs.open(fs_path, "wb") as f:
            f.write(buf.getvalue())

    def _write_representative_supervoxels_chunk(
        self,
        chunk_idx: tuple[int, int, int],
        entries: list[tuple[int, int, int, int, int]],
    ) -> None:
        """Write representative supervoxels to representative_supervoxels/ directory.

        Format:
        - n_entries: uint32
        - Per entry:
          - contact_id: int64
          - seg_a: int64
          - seg_b: int64
          - supervoxel_a: uint64
          - supervoxel_b: uint64
        """
        if not entries:
            return

        chunk_path = self.get_representative_supervoxels_chunk_path(chunk_idx)

        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(entries)))

        for contact_id, seg_a, seg_b, sv_a, sv_b in entries:
            buf.write(struct.pack("<q", contact_id))
            buf.write(struct.pack("<q", seg_a))
            buf.write(struct.pack("<q", seg_b))
            buf.write(struct.pack("<Q", sv_a))  # uint64
            buf.write(struct.pack("<Q", sv_b))  # uint64

        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
        with fs.open(fs_path, "wb") as f:
            f.write(buf.getvalue())

    def read_chunk(self, chunk_idx: tuple[int, int, int]) -> Sequence[SegContact]:
        """Read contacts from chunk files (contacts, pointclouds, merge_decisions)."""
        contacts_data = self._read_contacts_chunk(chunk_idx)
        if not contacts_data:
            return []

        contacts = [
            SegContact(
                id=c["id"],
                seg_a=c["seg_a"],
                seg_b=c["seg_b"],
                com=c["com"],
                contact_faces=c["contact_faces"],
                local_pointclouds=None,
                merge_decisions=None,
                partner_metadata=c["partner_metadata"],
                representative_points=c["representative_points"],
            )
            for c in contacts_data
        ]
        contact_lookup = {c.id: c for c in contacts}

        info = self._read_info_if_exists()
        if info is not None:
            self._attach_pointclouds(chunk_idx, contact_lookup, info)
            self._attach_merge_decisions(chunk_idx, contact_lookup, info)
            self._attach_merge_probabilities(chunk_idx, contact_lookup, info)

        # Attach supervoxels unconditionally (file existence check is in the method)
        self._attach_representative_supervoxels(chunk_idx, contact_lookup)

        return contacts

    def _read_info_if_exists(self) -> dict | None:
        """Read info file if it exists."""
        info_path = f"{self.path}/info"
        fs, fs_path = fsspec.core.url_to_fs(info_path)
        if not fs.exists(fs_path):
            return None
        with fs.open(fs_path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))

    def _attach_pointclouds(
        self, chunk_idx: tuple[int, int, int], contact_lookup: dict[int, SegContact], info: dict
    ) -> None:
        """Attach pointcloud data to contacts."""
        if self.local_point_clouds is not None:
            configs = [{"radius_nm": r, "n_points": n} for r, n in self.local_point_clouds]
        else:
            configs = info.get("local_point_clouds", [])

        for cfg in configs:
            radius_nm = int(cfg["radius_nm"])
            n_points = int(cfg["n_points"])
            pc_data = self._read_pointcloud_chunk(chunk_idx, radius_nm, n_points)
            config_tuple = (radius_nm, n_points)
            for contact_id, seg_a_pts, seg_b_pts in pc_data:
                if contact_id not in contact_lookup:
                    continue
                c = contact_lookup[contact_id]
                if c.local_pointclouds is None:
                    c.local_pointclouds = {}
                c.local_pointclouds[config_tuple] = {c.seg_a: seg_a_pts, c.seg_b: seg_b_pts}

    def _attach_merge_decisions(
        self, chunk_idx: tuple[int, int, int], contact_lookup: dict[int, SegContact], info: dict
    ) -> None:
        """Attach merge decision data to contacts."""
        for authority in info.get("merge_decisions", []):
            decisions = self._read_merge_decision_chunk(chunk_idx, authority)
            for contact_id, should_merge in decisions:
                if contact_id not in contact_lookup:
                    continue
                c = contact_lookup[contact_id]
                if c.merge_decisions is None:
                    c.merge_decisions = {}
                c.merge_decisions[authority] = should_merge

    def _attach_merge_probabilities(
        self, chunk_idx: tuple[int, int, int], contact_lookup: dict[int, SegContact], info: dict
    ) -> None:
        """Attach merge probability data to contacts."""
        for authority in info.get("merge_probabilities", []):
            probabilities = self._read_merge_probability_chunk(chunk_idx, authority)
            for contact_id, probability in probabilities:
                if contact_id not in contact_lookup:
                    continue
                c = contact_lookup[contact_id]
                if c.merge_probabilities is None:
                    c.merge_probabilities = {}
                c.merge_probabilities[authority] = probability

    def _attach_representative_supervoxels(
        self, chunk_idx: tuple[int, int, int], contact_lookup: dict[int, SegContact]
    ) -> None:
        """Attach representative supervoxel data to contacts."""
        supervoxels = self._read_representative_supervoxels_chunk(chunk_idx)
        for contact_id, seg_a, seg_b, sv_a, sv_b in supervoxels:
            if contact_id not in contact_lookup:
                continue
            c = contact_lookup[contact_id]
            c.representative_supervoxels = {seg_a: sv_a, seg_b: sv_b}

    def _read_contacts_chunk(self, chunk_idx: tuple[int, int, int]) -> list[dict]:
        """Read core contact data from contacts/ directory."""
        chunk_path = self.get_chunk_path(chunk_idx)
        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        if not fs.exists(fs_path):
            return []

        contacts = []
        with fs.open(fs_path, "rb") as f_in:
            data = f_in.read()
        with io.BytesIO(data) as f:
            n_contacts = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_contacts):
                id_, seg_a, seg_b = struct.unpack("<qqq", f.read(24))
                com = struct.unpack("<fff", f.read(12))
                n_faces = struct.unpack("<I", f.read(4))[0]
                contact_faces = np.frombuffer(f.read(n_faces * 4 * 4), dtype=np.float32).reshape(
                    n_faces, 4
                )

                metadata_len = struct.unpack("<I", f.read(4))[0]
                if metadata_len > 0:
                    metadata_bytes = f.read(metadata_len)
                    partner_metadata = json.loads(metadata_bytes.decode("utf-8"))
                    partner_metadata = {int(k): v for k, v in partner_metadata.items()}
                else:
                    partner_metadata = None

                # Read representative_points (6 floats: point_a xyz, point_b xyz)
                pts = struct.unpack("<ffffff", f.read(24))
                representative_points: dict[int, Vec3D] = {
                    seg_a: Vec3D(pts[0], pts[1], pts[2]),
                    seg_b: Vec3D(pts[3], pts[4], pts[5]),
                }

                contacts.append(
                    {
                        "id": id_,
                        "seg_a": seg_a,
                        "seg_b": seg_b,
                        "com": Vec3D(*com),
                        "contact_faces": contact_faces.copy(),
                        "local_pointclouds": None,
                        "merge_decisions": None,
                        "partner_metadata": partner_metadata,
                        "representative_points": representative_points,
                    }
                )

        return contacts

    def _read_pointcloud_chunk(
        self, chunk_idx: tuple[int, int, int], radius_nm: float, n_points: int
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Read pointcloud data from local_point_clouds/{config}/ directory."""
        chunk_path = self.get_pointcloud_chunk_path(chunk_idx, radius_nm, n_points)
        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        if not fs.exists(fs_path):
            return []

        entries = []
        with fs.open(fs_path, "rb") as f_in:
            data = f_in.read()
        with io.BytesIO(data) as f:
            n_entries = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_entries):
                contact_id = struct.unpack("<q", f.read(8))[0]
                seg_a_pts = (
                    np.frombuffer(f.read(n_points * 3 * 4), dtype=np.float32)
                    .reshape(n_points, 3)
                    .copy()
                )
                seg_b_pts = (
                    np.frombuffer(f.read(n_points * 3 * 4), dtype=np.float32)
                    .reshape(n_points, 3)
                    .copy()
                )
                entries.append((contact_id, seg_a_pts, seg_b_pts))

        return entries

    def _read_merge_decision_chunk(
        self, chunk_idx: tuple[int, int, int], authority: str
    ) -> list[tuple[int, bool]]:
        """Read merge decisions from merge_decisions/{authority}/ directory."""
        chunk_path = self.get_merge_decision_chunk_path(chunk_idx, authority)
        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        if not fs.exists(fs_path):
            return []

        decisions = []
        with fs.open(fs_path, "rb") as f_in:
            data = f_in.read()
        with io.BytesIO(data) as f:
            n_decisions = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_decisions):
                contact_id = struct.unpack("<q", f.read(8))[0]
                should_merge = struct.unpack("<B", f.read(1))[0] == 1
                decisions.append((contact_id, should_merge))

        return decisions

    def _read_merge_probability_chunk(
        self, chunk_idx: tuple[int, int, int], authority: str
    ) -> list[tuple[int, float]]:
        """Read merge probabilities from merge_probabilities/{authority}/ directory."""
        chunk_path = self.get_merge_probability_chunk_path(chunk_idx, authority)
        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        if not fs.exists(fs_path):
            return []

        entries = []
        with fs.open(fs_path, "rb") as f_in:
            data = f_in.read()
        with io.BytesIO(data) as f:
            n_entries = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_entries):
                contact_id = struct.unpack("<q", f.read(8))[0]
                probability = struct.unpack("<f", f.read(4))[0]
                entries.append((contact_id, probability))

        return entries

    def _read_representative_supervoxels_chunk(
        self, chunk_idx: tuple[int, int, int]
    ) -> list[tuple[int, int, int, int, int]]:
        """Read representative supervoxels from representative_supervoxels/ directory.

        Returns list of (contact_id, seg_a, seg_b, supervoxel_a, supervoxel_b).
        """
        chunk_path = self.get_representative_supervoxels_chunk_path(chunk_idx)
        fs, fs_path = fsspec.core.url_to_fs(chunk_path)
        if not fs.exists(fs_path):
            return []

        entries = []
        with fs.open(fs_path, "rb") as f_in:
            data = f_in.read()
        with io.BytesIO(data) as f:
            n_entries = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_entries):
                contact_id = struct.unpack("<q", f.read(8))[0]
                seg_a = struct.unpack("<q", f.read(8))[0]
                seg_b = struct.unpack("<q", f.read(8))[0]
                sv_a = struct.unpack("<Q", f.read(8))[0]  # uint64
                sv_b = struct.unpack("<Q", f.read(8))[0]  # uint64
                entries.append((contact_id, seg_a, seg_b, sv_a, sv_b))

        return entries
