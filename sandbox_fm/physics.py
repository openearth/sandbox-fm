import numpy as np
import matplotlib.colors
import matplotlib.cm
import cv2


def apply_hillshade(z, vertical_exageration=1.0):
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    hillshaded = ls.shade(z, vert_exag=vertical_exageration, blend_mode='hsv', cmap=matplotlib.cm.gist_earth)
    return hillshaded

def create_wave(data):
    """generate a wave in the dataset"""
    n_segments = 100
    wave_x = np.linspace(data['box'][0][0], data['box'][1][0], num=n_segments + 1)
    wave_y = np.zeros(n_segments + 1) + 10.0 # y-pixels
    wave_xy = np.c_[wave_x, wave_y]
    segments = []               # preallocate segments structure
    for i in range(n_segments):
        segments.append([
            wave_xy[i],         # from
            wave_xy[i+1]        # to
        ])
    wave = matplotlib.collections.LineCollection(segments, color='white')
    return wave



def warp_waves(waves, flow, data, wave_height_img, dissipation_img):
    """advect the waves"""

    N = matplotlib.colors.Normalize(0, 2000, clip=True)
    for wave in waves:
        # segments x 2(from, to) x 2 (x,y)
        segments = wave.get_segments()
        wave_idx = np.round(segments).astype('int')
        # we only have velocities inside the domain, use nearest
        wave_idx[:, :, 0] = np.clip(wave_idx[:, :, 0], 0, flow.shape[1] - 1)
        wave_idx[:, :, 1] = np.clip(wave_idx[:, :, 1], 0, flow.shape[0] - 1)
        # segments x 2(from, to) x 2 (u, v)
        flow_per_segment = flow[wave_idx[:, :, 1], wave_idx[:, :, 0], :]
        new_segments = segments + (flow_per_segment * data.get('wave.scale', 1.0))
        # compute average wave height per segment
        wave_height_per_segment = np.mean(wave_height_img[wave_idx[:, :, 1], wave_idx[:, :, 0]], axis=1)
        dissipation_per_segment = np.mean(dissipation_img[wave_idx[:, :, 1], wave_idx[:, :, 0]], axis=1)

        wave.set_segments(new_segments)
        wave.set_linewidths(wave_height_per_segment)
        dissipation_per_segment_normalized = N(dissipation_per_segment)
        dissipation_per_segment_color = np.ones((len(segments), 4))
        dissipation_per_segment_color[:, 3] = dissipation_per_segment_normalized
        wave.set_color(dissipation_per_segment_color)

    return waves


def warp_particles(particles, flow, data):
    """advect the particles"""
    # segments x 2(from, to) x 2 (x,y)
    points = particles.get_data()

    points_idx = np.round(points).astype('int')
    # we only have velocities inside the domain, use nearest
    points_idx[:, 0] = np.clip(points_idx[:, 0], 0, flow.shape[1] - 1)
    points_idx[:, 1] = np.clip(points_idx[:, 1], 0, flow.shape[0] - 1)
    # segments x 2(from, to) x 2 (u, v)
    flow_per_point = flow[points_idx[:, 1], points_idx[:, 0], :]
    new_points = points + (flow_per_point * data.get('particles.scale', 1.0))

    # compute average wave height per segment
    particles.set_xdata(new_points[:, 0])
    particles.set_ydata(new_points[:, 1])

def warp_flow(img, flow):
    """transform image with flow field"""
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR,
                    borderValue=(1.0, 1.0, 1.0, 0.0))
    return res
