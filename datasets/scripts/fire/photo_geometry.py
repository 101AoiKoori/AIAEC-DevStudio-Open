"""Fit an approximate photo camera from known building corners and a floor plane.

Photo control points are measured observations. Intrinsics are estimated, not a
replacement for a lens calibration. OpenCV uses +Z forward and +Y down; the
renderer converts this pose to Blender's -Z forward and +Y up explicitly.
"""
import numpy as np


def fit_photo_camera(world_points, image_points, image_size):
    import cv2
    from datasets.scripts.common.perspective import fit_ground_homography
    world = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    width, height = image_size
    if world.shape != (len(pixels), 3) or pixels.shape[1:] != (2,) or len(world) < 6:
        raise ValueError('Photo camera needs at least six 3D/2D corner correspondences')
    if not np.isfinite(world).all() or not np.isfinite(pixels).all():
        raise ValueError('Camera correspondences must be finite')
    candidates = []
    # The phone EXIF supplies a physical focal length but no sensor size. Fit the
    # pixel focal length over a declared interval instead of inventing a sensor.
    for focal in np.linspace(.5 * max(image_size), 2.5 * max(image_size), 161):
        k = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
        ok, rv, tv = cv2.solvePnP(world, pixels, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rotation = cv2.Rodrigues(rv)[0]
        depths = (world @ rotation.T + tv.ravel())[:, 2]
        eye = -rotation.T @ tv.ravel()
        if np.any(depths <= 0) or eye[2] <= 0:
            continue
        projected = cv2.projectPoints(world, rv, tv, k, None)[0][:, 0]
        error = float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))
        candidates.append((error, focal, k, rotation, tv.ravel(), eye))
    if not candidates:
        raise ValueError('No positive-depth photo camera fits the control points')
    error, focal, k, rotation, translation, eye = min(candidates, key=lambda x: x[0])
    projection = k @ np.column_stack((rotation, translation))
    ground = projection[:, [0, 1, 3]]
    # Reuse the shared plane fitting contract to verify the recovered ground map.
    square = np.float64([[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]])
    homogeneous = np.column_stack((square, np.ones(4))) @ ground.T
    if np.any(homogeneous[:, 2] <= 0):
        raise ValueError('Recovered ground reference lies behind the camera')
    fitted = fit_ground_homography(square, homogeneous[:, :2]/homogeneous[:, 2:])
    return {'image_size': list(image_size), 'intrinsic_matrix': k.tolist(),
            'world_to_cv_rotation': rotation.tolist(), 'world_to_cv_translation': translation.tolist(),
            'camera_position_m': eye.tolist(), 'ground_homography': fitted.tolist(),
            'reprojection_rms_px': error, 'focal_px': float(focal),
            'focal_search_px': [.5*max(image_size), 2.5*max(image_size)],
            'method': 'known_building_corners_pnp_focal_grid',
            'calibration_status': 'estimated_from_approximate_photo_corners',
            'distortion': 'unknown; zero-distortion pinhole approximation'}


def choose_photo_placement(review, bounds, rng):
    """Fit a complete 3D body and its footprint to the observed photo floor."""
    import math
    camera=review['camera'];k=np.asarray(camera['intrinsic_matrix']);rotation=np.asarray(camera['world_to_cv_rotation'])
    translation=np.asarray(camera['world_to_cv_translation']);inverse=np.linalg.inv(camera['ground_homography'])
    ground_h=np.asarray(camera['ground_homography']);eye=np.asarray(camera['camera_position_m'])
    polygon=np.asarray(review['ground_polygon_px']);size=np.asarray(camera['image_size'])
    corners=np.array([[x,y,z] for x in bounds[0] for y in bounds[1] for z in bounds[2]])
    feet=np.array([[bounds[0][0],bounds[1][0],0],[bounds[0][1],bounds[1][0],0],
                   [bounds[0][1],bounds[1][1],0],[bounds[0][0],bounds[1][1],0]])
    for attempt in range(96):
        anchor=np.asarray(review['ground_anchor_px'],dtype=float)+[rng.uniform(-j,j) for j in review['anchor_jitter_px']]
        xy=inverse@[*anchor,1];xy=xy[:2]/xy[2]
        scale=rng.uniform(*review['model_scale_range']);toward=eye[:2]-xy
        yaw=math.atan2(toward[0],-toward[1])+math.radians(rng.uniform(*review['yaw_offset_deg']))
        cos,sin=math.cos(yaw),math.sin(yaw);body_rotation=np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])
        matrix=np.eye(4);matrix[:3,:3]=body_rotation*scale;matrix[:2,3]=xy
        points=corners@matrix[:3,:3].T+matrix[:3,3]
        cv=points@rotation.T+translation
        if np.any(cv[:,2]<=0):continue
        pixels=cv@k.T;pixels=pixels[:,:2]/pixels[:,2:]
        low,high=pixels.min(0),pixels.max(0)
        if np.any(low<3) or np.any(high>size-3):continue
        footprint=feet@matrix[:3,:3].T+matrix[:3,3]
        projected=np.column_stack((footprint[:,:2],np.ones(4)))@ground_h.T
        projected=projected[:,:2]/projected[:,2:]
        crosses=[]
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            edge=b-a;delta=projected-a;crosses.append(edge[0]*delta[:,1]-edge[1]*delta[:,0])
        signs=np.asarray(crosses)
        if not (np.all(signs>=0) or np.all(signs<=0)):continue
        return {'matrix':matrix.tolist(),'ground_anchor_px':anchor.tolist(),'ground_position_m':xy.tolist(),
                'model_scale':scale,'model_yaw_rad':yaw,'footprint_photo_px':projected.tolist(),
                'body_bbox_photo_px':[*low.tolist(),*high.tolist()],'placement_attempts':attempt+1}
    raise ValueError(f'No complete 3D body/footprint fits photo floor {review["id"]}')


def blend_render(background_bgr, foreground_bgra):
    """Composite straight PNG alpha in linear light, including catcher shadows."""
    if background_bgr.shape[:2] != foreground_bgra.shape[:2] or foreground_bgra.shape[2] != 4:
        raise ValueError('Hybrid foreground must be matching-size BGRA')
    def linear(x):
        x = x.astype(np.float32)/255
        return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
    alpha = foreground_bgra[:, :, 3:4].astype(np.float32)/255
    color = linear(foreground_bgra[:, :, :3])*alpha + linear(background_bgr)*(1-alpha)
    encoded = np.where(color <= .0031308, color*12.92, 1.055*np.maximum(color, 0)**(1/2.4)-.055)
    return np.clip(np.rint(encoded*255), 0, 255).astype(np.uint8)
