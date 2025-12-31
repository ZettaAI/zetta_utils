from __future__ import annotations

import json
import os
from collections.abc import Sequence

import attrs
import numpy as np

from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex

from .contact import Contact


@attrs.define
class ContactLayerBackend:
    """Backend for reading/writing contact data in chunked format."""

    path: str
    resolution: Vec3D[int]  # voxel size in nm
    voxel_offset: Vec3D[int]  # dataset start in voxels
    size: Vec3D[int]  # dataset dimensions in voxels
    chunk_size: Vec3D[int]  # chunk dimensions in voxels
    max_contact_span: int  # in voxels

    @classmethod
    def from_path(cls, path: str) -> ContactLayerBackend:
        """Load backend from existing info file."""
        info_path = os.path.join(path, "info")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Info file not found: {info_path}")
        with open(info_path, "r") as f:
            info = json.load(f)
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
            "type": "contact",
            "resolution": list(self.resolution),
            "voxel_offset": list(self.voxel_offset),
            "size": list(self.size),
            "chunk_size": list(self.chunk_size),
            "max_contact_span": self.max_contact_span,
        }
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "info"), "w") as f:
            json.dump(info, f, indent=2)

    def read(self, idx: VolumetricIndex) -> Sequence[Contact]:
        """Read contacts whose COM falls within the given index."""
        # Get bbox in nm
        bbox = idx.bbox
        start_nm = Vec3D(
            bbox.start[0] * idx.resolution[0],
            bbox.start[1] * idx.resolution[1],
            bbox.start[2] * idx.resolution[2],
        )
        end_nm = Vec3D(
            bbox.end[0] * idx.resolution[0],
            bbox.end[1] * idx.resolution[1],
            bbox.end[2] * idx.resolution[2],
        )

        # Find which chunks to read
        start_chunk = self.com_to_chunk_idx(start_nm)
        end_chunk = self.com_to_chunk_idx(Vec3D(end_nm[0] - 0.001, end_nm[1] - 0.001, end_nm[2] - 0.001))

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

    def write(self, idx: VolumetricIndex, data: Sequence[Contact]) -> None:
        """Write contacts to appropriate chunks based on their COM."""
        # Group contacts by chunk
        chunk_contacts: dict[tuple[int, int, int], list[Contact]] = {}
        for contact in data:
            chunk_idx = self.com_to_chunk_idx(contact.com)
            if chunk_idx not in chunk_contacts:
                chunk_contacts[chunk_idx] = []
            chunk_contacts[chunk_idx].append(contact)

        # Write each chunk
        for chunk_idx, contacts in chunk_contacts.items():
            self.write_chunk(chunk_idx, contacts)

    def get_chunk_path(self, chunk_idx: tuple[int, int, int]) -> str:
        """Get file path for a chunk given grid indices."""
        return os.path.join(self.path, "contacts", self.get_chunk_name(chunk_idx))

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

    def write_chunk(self, chunk_idx: tuple[int, int, int], contacts: Sequence[Contact]) -> None:
        """Write contacts to a specific chunk file."""
        import struct

        chunk_path = self.get_chunk_path(chunk_idx)
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)

        with open(chunk_path, "wb") as f:
            # Header: n_contacts
            f.write(struct.pack("<I", len(contacts)))

            for contact in contacts:
                # id, seg_a, seg_b
                f.write(struct.pack("<qqq", contact.id, contact.seg_a, contact.seg_b))
                # com (3 floats)
                f.write(struct.pack("<fff", contact.com[0], contact.com[1], contact.com[2]))
                # n_faces
                n_faces = contact.contact_faces.shape[0]
                f.write(struct.pack("<I", n_faces))
                # contact_faces
                f.write(contact.contact_faces.astype(np.float32).tobytes())
                # partner_metadata
                if contact.partner_metadata is not None:
                    metadata_bytes = json.dumps(contact.partner_metadata).encode("utf-8")
                    f.write(struct.pack("<I", len(metadata_bytes)))
                    f.write(metadata_bytes)
                else:
                    f.write(struct.pack("<I", 0))

    def read_chunk(self, chunk_idx: tuple[int, int, int]) -> Sequence[Contact]:
        """Read contacts from a specific chunk file."""
        import struct

        chunk_path = self.get_chunk_path(chunk_idx)
        if not os.path.exists(chunk_path):
            return []

        contacts = []
        with open(chunk_path, "rb") as f:
            # Header: n_contacts
            n_contacts = struct.unpack("<I", f.read(4))[0]

            for _ in range(n_contacts):
                # id, seg_a, seg_b
                id_, seg_a, seg_b = struct.unpack("<qqq", f.read(24))
                # com
                com = struct.unpack("<fff", f.read(12))
                # n_faces
                n_faces = struct.unpack("<I", f.read(4))[0]
                # contact_faces
                contact_faces = np.frombuffer(f.read(n_faces * 4 * 4), dtype=np.float32).reshape(
                    n_faces, 4
                )
                # partner_metadata
                metadata_len = struct.unpack("<I", f.read(4))[0]
                if metadata_len > 0:
                    metadata_bytes = f.read(metadata_len)
                    partner_metadata = json.loads(metadata_bytes.decode("utf-8"))
                    # Convert string keys back to int
                    partner_metadata = {int(k): v for k, v in partner_metadata.items()}
                else:
                    partner_metadata = None

                contacts.append(
                    Contact(
                        id=id_,
                        seg_a=seg_a,
                        seg_b=seg_b,
                        com=Vec3D(*com),
                        contact_faces=contact_faces.copy(),
                        partner_metadata=partner_metadata,
                    )
                )

        return contacts
