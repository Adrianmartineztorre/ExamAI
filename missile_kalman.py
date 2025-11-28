from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    R: np.ndarray
    z: np.ndarray

def kalman_filter(s: KalmanState):
    # Prediction
    s.x = s.F @ s.x
    s.P = s.F @ s.P @ s.F.T

    # Update
    K = s.P @ np.linalg.inv(s.P + s.R)
    
    s.x = s.x + K @ (s.z - s.x)
    s.P = s.P - K @ s.P
    return s, K





def missile(seed=None):
    rng = np.random.default_rng(seed)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect("equal")


    # Allow closing with keyboard (q or escape) and detect window close
    def _on_key(event):
        if event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    ################################# Parameters ##############################################
    missile_speed = 0.1
    missile_height = 10
    missile_distance = -20
    sigma = 0.5
    bullet_speed = 0.1
    animation_pause = 0.05

    ###########################################################################################

    x = missile_distance
    y = missile_height
    bx, by = 0.0, 0.0

    draw_anti_aircraft_gun(ax, scale=5.0)
    missile_marker, = ax.plot(x, y, 'ro', markersize=6)
    update_axes(ax, x, y, bx, by)

    # Initial noisy measurement and a second measurement for a velocity estimate
    z = np.array([y + rng.normal(0, sigma), x + rng.normal(0, sigma), np.nan])

    x += missile_speed
    missile_marker.set_data([x], [y])
    update_axes(ax, x, y, bx, by)

    # second measurement (used to estimate velocity)
    z = np.array([y + rng.normal(0, sigma), x + rng.normal(0, sigma), z[1]])
    z[2] = z[1] - z[2]

    sensitivity = 1e-6
    kalman_boost = 5

    # Initialize KalmanState and assign matrices directly to s.F, s.R, s.P
    s = KalmanState(x=z.copy(), P=np.zeros((3, 3)), F=np.eye(3), R=np.zeros((3, 3)), z=z.copy())
    s.F = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    s.R = np.diag([sigma**2, sigma**2, 2 * sigma**2])
    s.P = kalman_boost * s.R.copy()

    # Create a small HUD panel to the right of the main axes and render
    # the status strings there so they remain visible outside the data area.
    # Shrink the main axes to leave room on the right.
    fig.subplots_adjust(right=0.76)
    # Compute the main axes position and place the HUD to the right so its
    # top lines up with the top of the plot (axes) rather than the figure.
    ax_pos = ax.get_position()  # Bbox in figure coordinates
    gap = 0.01
    hud_left = ax_pos.x1 + gap
    hud_bottom = ax_pos.y0
    hud_width = max(0.18, 1.0 - hud_left - 0.02)
    hud_height = ax_pos.height
    hud_ax = fig.add_axes([hud_left, hud_bottom, hud_width, hud_height])
    hud_ax.axis("off")

    # Place text lines in the HUD axes using axes coordinates
    # Reduced vertical gaps so strings appear closer together
    # Position the top HUD line aligned with the top of the main axes.
    height_text = hud_ax.text(0.02, 1.0, "", transform=hud_ax.transAxes, fontsize=11, va="top")
    distance_text = hud_ax.text(0.02, 0.95, "", transform=hud_ax.transAxes, fontsize=11, va="top")
    speed_text = hud_ax.text(0.02, 0.90, "", transform=hud_ax.transAxes, fontsize=11, va="top")

    # Accumulate radar signature history so we can keep them in-frame
    radar_points = []

    while s.P[2, 2] > sensitivity:
        s, _ = kalman_filter(s)
        x += missile_speed
        missile_marker.set_data([x], [y])

        s.z = np.array([
            y + rng.normal(0, sigma),
            x + rng.normal(0, sigma),
            s.z[1],
        ])
        ax.plot(s.z[1], s.z[0], "b.")
        s.z[2] = s.z[1] - s.z[2]

        # Store the radar signature so it remains part of the scene
        radar_points.append((s.z[1], s.z[0]))

        height_text.set_text(f"Height:   {s.x[0]:.2f}")
        distance_text.set_text(f"Distance: {s.x[1]:.2f}")
        speed_text.set_text(f"Speed:    {s.x[2]:.4f}")

        # If user closed the figure window, exit cleanly
        if not plt.fignum_exists(fig.number):
            print("Figure closed — stopping tracking loop")
            plt.close('all')
            return

        # Update axes: expansion only happens if missile/bullet leave frame;
        # radar_points are preserved and included when we expand.
        update_axes(ax, x, y, bx, by, radar_points=radar_points)
        plt.draw()
        plt.pause(animation_pause)

    a = s.x[1] * bullet_speed
    b = -s.x[0] * bullet_speed
    c = s.x[0] * s.x[2]

    discriminant = a**2 + b**2 - c**2
    if discriminant <= 0:
        theta = np.nan
    else:
        angle_term = np.degrees(np.arctan(c / np.sqrt(discriminant)))
        theta = angle_term - np.degrees(np.arctan(a / b))

    if np.isnan(theta):
        print("✗ No real firing solution; bullet will not be fired")
        plt.show()
        return
    bvx = bullet_speed * np.sin(np.radians(theta))
    bvy = bullet_speed * np.cos(np.radians(theta))
    print(f"Initial bullet velocity: vx={bvx:.3f}, vy={bvy:.3f}")
    print(f"Speed: {np.sqrt(bvx**2 + bvy**2):.3f}")

    proximity = np.linalg.norm([x - bx, y - by])
    time_step = 0

    # Remove the static missile point from the tracking phase so it doesn't linger
    if missile_marker:
        missile_marker.remove()
    missile_marker = None
    bullet_marker = None
    bullet_trail_x, bullet_trail_y = [], []
    trail_line = None
    max_steps = 2500

    while proximity > 0.2 and by >= -1 and time_step < max_steps:
        x += missile_speed

        if missile_marker:
            missile_marker.remove()
        missile_marker, = ax.plot(x, y, 'ro', markersize=6)

        bx += bvx
        by += bvy

        bullet_trail_x.append(bx)
        bullet_trail_y.append(by)

        if bullet_marker:
            bullet_marker.remove()
        bullet_marker, = ax.plot(bx, by, 'bo', markersize=4)

        if trail_line is None:
            (trail_line,) = ax.plot(bullet_trail_x, bullet_trail_y, 'b-', linewidth=1, alpha=0.5)
        else:
            trail_line.set_data(bullet_trail_x, bullet_trail_y)

        # Break if figure was closed by the user
        if not plt.fignum_exists(fig.number):
            print("Figure closed — stopping firing loop")
            plt.close('all')
            return

        update_axes(ax, x, y, bx, by)
        plt.draw()
        plt.pause(animation_pause)

        proximity = np.linalg.norm([x - bx, y - by])
        time_step += 1

    if time_step >= max_steps:
        print("Timeout during firing animation")

    if proximity <= 0.2:
        impact_x, impact_y = (x + bx) / 2, (y + by) / 2
        ax.plot(
            impact_x,
            impact_y,
            marker="*",
            color="red",
            markersize=25,
            markeredgecolor="orange",
            markeredgewidth=2,
        )
        ax.plot(impact_x, impact_y, marker="*", color="yellow", markersize=15)
        print(f"✓ INTERCEPT SUCCESSFUL at ({impact_x:.2f}, {impact_y:.2f})")
    else:
        print(f"✗ Intercept failed - missed by {proximity:.3f} units")

    plt.show()


def draw_anti_aircraft_gun(ax, scale=5.0):
    """Render a simple anti-aircraft gun at the origin."""
    wheel_y = -0.13 * scale
    wheel_radius = 0.03 * scale
    inner_radius = 0.015 * scale
    for wheel_x in [-0.075 * scale, 0.0, 0.075 * scale]:
        wheel = plt.Circle((wheel_x, wheel_y), wheel_radius, facecolor="darkgray", edgecolor="black", linewidth=1, zorder=2)
        ax.add_patch(wheel)
        inner_wheel = plt.Circle((wheel_x, wheel_y), inner_radius, facecolor="gray", zorder=2)
        ax.add_patch(inner_wheel)

    tank_body_x = [c * scale for c in (-0.125, 0.125, 0.1, -0.1, -0.125)]
    tank_body_y = [c * scale for c in (-0.13, -0.13, -0.03, -0.03, -0.13)]
    ax.fill(tank_body_x, tank_body_y, color="green", edgecolor="darkgreen", linewidth=1.5, zorder=3)


def update_axes(ax, missile_x, missile_y, bullet_x, bullet_y, radar_points=None):
    """Maintain persistent axes limits that only expand when missile or bullet
    go out of frame. Radar signatures are preserved (never excluded) but do not
    trigger expansion by themselves.

    Parameters
    - ax: matplotlib axes
    - missile_x, missile_y, bullet_x, bullet_y: current coords
    - radar_points: optional iterable of (x, y) radar signatures collected so far
    """
    margin = 1.5

    # Compute required extents based on missile and bullet (these can trigger expansion)
    required_xs = [0.0, float(missile_x), float(bullet_x)]
    required_ys = [0.0, float(missile_y), float(bullet_y)]

    req_xmin = min(required_xs) - margin
    req_xmax = max(required_xs) + margin
    req_ymin = min(required_ys) - margin
    req_ymax = max(required_ys) + margin

    # Ensure radar lists are in numeric form if provided
    radar_xs, radar_ys = [], []
    if radar_points:
        for px, py in radar_points:
            try:
                radar_xs.append(float(px))
                radar_ys.append(float(py))
            except Exception:
                continue

    # Initialize persistent limits on the axes if not present
    if not hasattr(ax, "_persistent_xlim"):
        # initial limits should include missile, bullet and any radar points
        all_xs = [req_xmin, req_xmax]
        all_ys = [req_ymin, req_ymax]
        if radar_xs:
            all_xs.extend([min(radar_xs) - margin, max(radar_xs) + margin])
        if radar_ys:
            all_ys.extend([min(radar_ys) - margin, max(radar_ys) + margin])
        ax._persistent_xlim = [min(all_xs), max(all_xs)]
        ax._persistent_ylim = [min(all_ys), max(all_ys)]
        ax.set_xlim(ax._persistent_xlim)
        ax.set_ylim(ax._persistent_ylim)
        return

    # Expand only if missile or bullet goes out of the current persistent frame
    cur_xmin, cur_xmax = ax._persistent_xlim
    cur_ymin, cur_ymax = ax._persistent_ylim

    expand = False
    if req_xmin < cur_xmin or req_xmax > cur_xmax or req_ymin < cur_ymin or req_ymax > cur_ymax:
        expand = True

    if not expand:
        # Do nothing (never reduce size)
        return

    # Compute new limits: ensure radar signatures are also inside after expansion
    new_xmin = min(cur_xmin, req_xmin)
    new_xmax = max(cur_xmax, req_xmax)
    new_ymin = min(cur_ymin, req_ymin)
    new_ymax = max(cur_ymax, req_ymax)

    if radar_xs:
        new_xmin = min(new_xmin, min(radar_xs) - margin)
        new_xmax = max(new_xmax, max(radar_xs) + margin)
    if radar_ys:
        new_ymin = min(new_ymin, min(radar_ys) - margin)
        new_ymax = max(new_ymax, max(radar_ys) + margin)

    ax._persistent_xlim = [new_xmin, new_xmax]
    ax._persistent_ylim = [new_ymin, new_ymax]
    ax.set_xlim(ax._persistent_xlim)
    ax.set_ylim(ax._persistent_ylim)



if __name__ == "__main__":
    missile_kalman()
