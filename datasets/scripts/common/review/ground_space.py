"""Local ground coordinates and placement geometry for one reviewed image.

The quad supplies a planar projective frame. With automatic review corners, its
units and angles describe that frame only. If the same four points are known
corners of a physical rectangle, supply its side lengths in a common unit.
Depth and apparent scale are separate image cues for flat-ground compositing.
An explicit height reference and vertical vanishing point can extend the frame
to local XYZ. Automatic review does not supply either vertical reference.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..perspective import fit_ground_homography, project_calibrated_ground
from .geometry import _row_span, estimate_ground_region


def _point(value: Sequence[float], dimensions: int) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (dimensions,) or not np.isfinite(point).all():
        raise ValueError(f"expected {dimensions} finite coordinates")
    return point


def _heading(dx: float, dy: float) -> float:
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return 0.0 if angle == 360.0 else angle


def image_ground_scale_ratio(
    cues: Mapping[str, Any], foot_xy: Sequence[float], reference_xy: Sequence[float],
) -> float | None:
    """Approximate a size ratio using a supported ground horizon in source pixels."""
    scale = cues.get("scale") or {}
    if scale.get("status") != "estimated" or scale.get("basis") != "ground_supported_vanishing_line":
        return None
    line = _point(scale["horizon_line_px"], 3)
    distances = [float(line[:2] @ _point(point, 2) + line[2]) for point in (foot_xy, reference_xy)]
    if min(distances) <= 0:
        return None
    return distances[0] / distances[1]


class GroundSpace:
    """Map a ground quad in far-left, far-right, near-right, near-left order.

    ``image_shape`` is (height, width), with an optional channel dimension.
    All image inputs and outputs use pixels in this same frame. Local (0, 0)
    is the far-left corner, +x goes right and +y goes towards the near edge.
    The default ``local_size`` uses one unit along each side of the quad.
    """

    def __init__(
        self,
        polygon: Sequence[Sequence[float]],
        *,
        image_shape: Sequence[int],
        local_size: Sequence[float] = (1.0, 1.0),
        horizon_y: float | None = None,
        far_scale: float | None = None,
    ) -> None:
        height, width = image_shape[:2]
        quad = np.asarray(polygon, dtype=float)
        if (quad.shape != (4, 2) or not np.isfinite(quad).all()
                or np.any(quad < 0) or np.any(quad > (width - 1, height - 1))):
            raise ValueError("ground quad must have four finite in-frame corners")
        local_width, local_depth = local_size
        if not all(math.isfinite(value) and value > 0 for value in local_size):
            raise ValueError("local_size must have positive finite side lengths")
        if horizon_y is not None and not math.isfinite(horizon_y):
            raise ValueError("horizon_y must be finite")
        if far_scale is not None and not 0 < far_scale <= 1:
            raise ValueError("far_scale must be in (0, 1]")

        local_quad = ((0, 0), (local_width, 0), (local_width, local_depth), (0, local_depth))
        self._to_image = fit_ground_homography(local_quad, quad)
        self._to_local = np.linalg.inv(self._to_image)
        self._quad = quad.copy()
        self._local_size = (float(local_width), float(local_depth))
        self._far_y = float(quad[:, 1].min())
        self._near_y = float(quad[:, 1].max())
        self._horizon_y = horizon_y
        self._far_scale = far_scale

    @classmethod
    def from_template(
        cls,
        template: Mapping[str, Any] | None,
        *,
        region: str = "placement_patch",
        local_size: Sequence[float] = (1.0, 1.0),
        far_scale: float | None = None,
    ) -> GroundSpace | None:
        """Open an existing perspective template in its original pixel coordinates.

        ``ground_polygon`` is the visible floor and can be requested explicitly.
        A missing requested region returns None. The template carries geometry
        independently of the background verdict and obstacle evidence.
        """
        if region not in ("placement_patch", "ground_polygon"):
            raise ValueError("expected placement_patch or ground_polygon")
        if template is None or not template.get(region):
            return None
        polygon = template[region]
        if region == "placement_patch":
            # Review stores near-left, near-right, far-right, far-left.
            polygon = polygon[::-1]
        return cls(polygon, image_shape=template["image_shape"], local_size=local_size,
                   horizon_y=template.get("horizon_y"), far_scale=far_scale)

    @property
    def polygon(self) -> tuple[tuple[float, float], ...]:
        """Ground region in image pixels, ordered from the far-left corner."""
        return tuple((float(x), float(y)) for x, y in self._quad)

    @property
    def scale_basis(self) -> str | None:
        """The reviewed horizon row or a caller-supplied far/near scale ratio."""
        if self._horizon_y is not None and self._horizon_y <= self._far_y:
            return "horizon_row"
        return "configured_far_scale" if self._far_scale is not None else None

    def _within_local(self, xy: Sequence[float], tolerance: float = 0.0) -> bool:
        normal = _point(xy, 2) / self._local_size
        return bool(np.all(normal >= -tolerance) and np.all(normal <= 1 + tolerance))

    def to_local(self, image_xy: Sequence[float]) -> tuple[float, float] | None:
        """Map an image point inside the quad to local coordinates."""
        return self._unproject(self._to_local, image_xy)

    def _unproject(self, inverse: np.ndarray,
                   image_xy: Sequence[float]) -> tuple[float, float] | None:
        image_point = _point(image_xy, 2)
        try:
            point = project_calibrated_ground(inverse, image_point)
        except ValueError:
            return None
        if not self._within_local(point, tolerance=1e-7):
            return None
        x, y = np.clip(point, (0.0, 0.0), self._local_size)
        return float(x), float(y)

    def footpoint(self, local_xy: Sequence[float], *,
                  contact_width: float = 0.0) -> tuple[float, float] | None:
        """Project a local point; optionally fit a horizontal pixel contact segment."""
        if not math.isfinite(contact_width) or contact_width < 0:
            raise ValueError("contact_width must be finite and nonnegative")
        if not self._within_local(local_xy):
            return None
        x, y = (float(value) for value in project_calibrated_ground(self._to_image, local_xy))
        y = min(max(y, self._far_y), self._near_y)
        left, right = _row_span(self._quad, y)
        half_contact = contact_width / 2
        if x - half_contact < left - 1e-6 or x + half_contact > right + 1e-6:
            return None
        return min(max(x, left), right), y

    def relative_depth(self, image_xy: Sequence[float]) -> float | None:
        """Return an image-row rank in [0 far, 1 near] within this ground quad."""
        if self.to_local(image_xy) is None:
            return None
        depth = (image_xy[1] - self._far_y) / (self._near_y - self._far_y)
        return float(np.clip(depth, 0.0, 1.0))

    def perspective_scale(self, image_xy: Sequence[float], *,
                          reference_xy: Sequence[float] | None = None) -> float | None:
        """Return an apparent size multiplier relative to the near row or reference."""
        scale = self._scale_at(image_xy)
        reference = 1.0 if reference_xy is None else self._scale_at(reference_xy)
        if scale is None or reference is None:
            return None
        return scale / reference

    def _scale_at(self, image_xy: Sequence[float]) -> float | None:
        rank = self.relative_depth(image_xy)
        if rank is None:
            return None
        if self.scale_basis == "horizon_row":
            distance = image_xy[1] - self._horizon_y
            if distance <= 0:
                return None
            return float(distance / (self._near_y - self._horizon_y))
        if self._far_scale is not None:
            return self._far_scale + rank * (1.0 - self._far_scale)
        return None

    def image_heading(self, local_xy: Sequence[float], yaw_degrees: float) -> float | None:
        """Project a local ground yaw to a pixel heading clockwise from image +x.

        Local yaw is measured from +x towards +y, in the declared local units.
        The derivative of the projection gives the heading at this specific point.
        Sprite foreshortening uses its own camera-relative yaw parameters.
        """
        if not math.isfinite(yaw_degrees):
            raise ValueError("yaw_degrees must be finite")
        if not self._within_local(local_xy):
            return None
        radians = math.radians(yaw_degrees % 360.0)
        point = self._to_image @ (*local_xy, 1.0)
        tangent = self._to_image[:, :2] @ (math.cos(radians), math.sin(radians))
        dx, dy = (tangent[:2] - point[:2] / point[2] * tangent[2]) / point[2]
        return _heading(dx, dy)

    def local_heading(self, image_start: Sequence[float],
                      image_end: Sequence[float]) -> float | None:
        """Read a ground heading from two image points in the same quad."""
        start, end = self.to_local(image_start), self.to_local(image_end)
        if start is None or end is None or start == end:
            return None
        return _heading(end[0] - start[0], end[1] - start[1])

    def with_height_reference(
        self,
        reference_xyz: Sequence[float],
        top_image_xy: Sequence[float],
        *,
        vertical_vanishing_point: Sequence[float],
    ) -> GroundProjection:
        """Fit the vertical column of a 3x4 projection with one height reference.

        ``reference_xyz`` is a point at positive height over local (x, y).
        ``top_image_xy`` is its corresponding image pixel. The vertical
        vanishing point uses homogeneous coordinates, e.g. (0,-1,0) declares
        parallel image verticals for an approximate composition. Heights use
        the supplied reference's units. Fit error is exposed in image pixels.
        """
        x, y, z = _point(reference_xyz, 3)
        if z <= 0 or not self._within_local((x, y)):
            raise ValueError("height reference needs an in-region base and positive height")
        top = _point(top_image_xy, 2)
        vertical = _point(vertical_vanishing_point, 3)
        if np.linalg.matrix_rank(np.column_stack((self._to_image[:, :2], vertical))) != 3:
            raise ValueError("vertical direction must vanish off the ground horizon")

        # P = [H_x H_y p_z H_origin]. Solve for the scale of p_z = alpha * vertical
        # by requiring H @ (x, y, 1) + z * p_z to project to the reference top.
        base = self._to_image @ (x, y, 1.0)
        direction = top * vertical[2] - vertical[:2]
        if np.linalg.norm(direction) <= 1e-12 * np.linalg.norm(vertical):
            raise ValueError("height reference top coincides with the vertical vanishing point")
        if np.linalg.norm(base[:2] / base[2] - top) <= 1e-7:
            raise ValueError("height reference top coincides with its ground contact")
        offset = base[:2] - top * base[2]
        alpha = float(np.dot(direction, offset) / np.dot(direction, direction) / z)
        if alpha == 0.0 or not math.isfinite(alpha):
            raise ValueError("height reference supplies no usable vertical displacement")
        matrix = np.column_stack((self._to_image[:, :2], alpha * vertical, self._to_image[:, 2]))
        return GroundProjection(self, matrix, (x, y, z), top)


class GroundProjection:
    """Local XYZ projection fitted by ``GroundSpace.with_height_reference``."""

    def __init__(self, ground: GroundSpace, matrix: np.ndarray,
                 reference_xyz: Sequence[float], reference_top: np.ndarray) -> None:
        self._ground = ground
        self._matrix = matrix
        projected_top = self.project_xyz(reference_xyz)
        if projected_top is None:
            raise ValueError("height reference must project finitely in front of the camera")
        self.reference_error_px = float(np.linalg.norm(projected_top - reference_top))

    def project_xyz(self, local_xyz: Sequence[float]) -> tuple[float, float] | None:
        """Project (x, y, z) above the local quad. Return pixels, including off-frame."""
        x, y, z = _point(local_xyz, 3)
        if not self._ground._within_local((x, y)):
            return None
        if z == 0:
            return self._ground.footpoint((x, y))
        point = self._matrix @ (x, y, z, 1.0)
        if not np.isfinite(point).all() or point[2] <= 1e-10:
            return None
        return float(point[0] / point[2]), float(point[1] / point[2])

    def to_local(self, image_xy: Sequence[float], *, z: float) -> tuple[float, float, float] | None:
        """Back-project an image point onto the horizontal plane at a known z."""
        if not math.isfinite(z):
            raise ValueError("z must be finite")
        plane = self._matrix[:, (0, 1, 3)].copy()
        plane[:, 2] += z * self._matrix[:, 2]
        try:
            inverse = np.linalg.inv(plane)
        except np.linalg.LinAlgError:
            return None
        local_xy = self._ground._unproject(inverse, image_xy)
        if local_xy is None or self.project_xyz((*local_xy, z)) is None:
            return None
        return (local_xy[0], local_xy[1], float(z))


def estimate_ground_space(
    image_bgr: np.ndarray,
    *,
    params: dict | None = None,
    region: str = "placement_patch",
    far_scale: float | None = None,
) -> GroundSpace | None:
    """Reuse background review's estimator for an image without stored metrics."""
    ground = estimate_ground_region(image_bgr, params=params)
    template = {
        "image_shape": image_bgr.shape,
        "ground_polygon": ground.polygon,
        "placement_patch": ground.placement_patch,
        "horizon_y": ground.horizon_y,
    } if ground.ok else None
    return GroundSpace.from_template(template, region=region, far_scale=far_scale)
