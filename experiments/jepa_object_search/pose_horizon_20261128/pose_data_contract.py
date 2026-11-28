"""Source-camera frame: +x right, +z forward, target camera yaw positive to the left."""
FEATURES = ("dx_right_m","dz_forward_m","height_m","height_delta_m","sin_yaw","cos_yaw",
            "sin_pitch","cos_pitch","source_box_cx","source_box_cy","source_box_w","source_box_h",
            "source_box_median_depth_m","projected_x","projected_y","optical_z_m","source_ray_support",
            "source_ray_occluder_fraction","forward_path_observed_fraction","forward_path_clear_fraction")
POSE_ONLY = FEATURES[:8]
EXTRINSICS = ((1.31, 0.), (.85, 0.), (1.65, 0.), (1.31, -20.), (1.31, 20.),
              (.85, -20.), (.85, 20.), (1.65, -20.), (1.65, 20.))
